"""Extract full RGB frames with a visible annotated target; preserve video groups."""
from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .common import ROOT, META, OUT, setup, protect_frozen, config, rows, write_rows, read_json, save_json, resolve, relative, digest
from .image_metadata import image_record
from .video_annotations import ANNOTATIONS


def extract_frames():
    protect_frozen()
    setup()
    import cv2

    cfg = config()
    spec = cfg["halmstad"]
    requested_fps = float(spec["sample_fps"])
    cap_per_video = int(spec["max_frames_per_video"])
    if requested_fps <= 0 or cap_per_video <= 0:
        raise ValueError("抽帧频率和每视频帧数必须为正")
    videos = rows(META / "halmstad_videos.csv")
    status = {r["video_id"]: r for r in rows(META / "halmstad_annotation_status.csv")}
    if not videos or any(status.get(v["video_id"], {}).get("status") != "needs_visual_review" for v in videos):
        raise RuntimeError("必须先完成所有已下载视频的01b标注转换")
    output_dir = ROOT / "data" / "raw" / "public_halmstad_frames"
    preview_dir = OUT / "halmstad_annotation_previews"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    all_records, extraction_rows = [], []

    for row in videos:
        annotations = read_json(ANNOTATIONS / (row["video_id"] + ".json"))
        video = resolve(row["video_path"])
        if annotations["video_sha256"] != digest(video) or annotations["labels_sha256"] != digest(resolve(row["labels_path"])):
            raise ValueError("标注转换后原文件发生变化：" + row["video_id"])
        stride = max(1, round(float(annotations["fps"]) / requested_fps))
        candidates = [f["frame_index"] for f in annotations["frames"]
                      if f["frame_index"] % stride == 0 and f["bbox_xywh_matlab"] is not None]
        if len(candidates) > cap_per_video:
            positions = [round(i*(len(candidates)-1)/max(cap_per_video-1, 1)) for i in range(cap_per_video)]
            candidates = [candidates[i] for i in positions]
        if not candidates:
            raise ValueError("按既定间隔没有采到带目标帧，需检查协议：" + row["video_id"])
        chosen = set(candidates)
        preview_indices = {candidates[0], candidates[len(candidates)//2], candidates[-1]}
        previews = []
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        saved = 0
        try:
            if not cap.isOpened():
                raise ValueError("无法打开视频：" + str(video))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = decoded
                decoded += 1
                if frame_index not in chosen:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im = Image.fromarray(rgb)
                sample_id = "halmstad_" + row["video_id"] + f"_f{frame_index:06d}"
                path = output_dir / (sample_id + ".png")
                im.save(path)
                bbox = annotations["frames"][frame_index]["bbox_xywh_matlab"]
                record = image_record(path, "public", int(row["candidate_label"]),
                    "https://github.com/DroneDetectionThesis/Drone-detection-dataset/blob/" + row["revision"] + "/" + row["upstream_video"],
                    sample_id, {"source_dataset": "halmstad_rgb", "video_id": row["video_id"],
                    "frame_index": frame_index, "time_seconds": frame_index / float(annotations["fps"]),
                    "video_path": row["video_path"], "group_hint": "halmstad_" + row["video_id"],
                    "bbox_x": bbox[0], "bbox_y": bbox[1], "bbox_w": bbox[2], "bbox_h": bbox[3],
                    "revision": row["revision"], "license": "See original repository LICENSE", "image_scope": "full_frame"})
                all_records.append(record)
                saved += 1
                if frame_index in preview_indices:
                    display_image = im.copy()
                    draw = ImageDraw.Draw(display_image)
                    x, y, width, height = bbox
                    draw.rectangle([x, y, x+width, y+height], outline="red", width=2)
                    previews.append((frame_index, ImageOps.contain(display_image, (600, 400))))
        finally:
            cap.release()
        if decoded != annotations["frame_count"] or saved != len(candidates):
            raise RuntimeError(f"{row['video_id']} 实际解码帧数/采样数不符，停止写入正式图片清单")
        sheet = Image.new("RGB", (600, len(previews)*430), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (index, im) in enumerate(previews):
            sheet.paste(im, (0, j*430))
            draw.text((5, j*430+405), f"{row['video_id']} frame={index} candidate={row['candidate_label']}", fill="black")
        sheet.save(preview_dir / (row["video_id"] + ".jpg"), quality=90)
        extraction_rows.append({"video_id": row["video_id"], "class_name": row["class_name"],
                                "decoded_frames": decoded, "sampled_frames": saved, "stride": stride})
        write_rows(META / "halmstad_extraction_status.csv", extraction_rows)
        print(row["video_id"], "抽取", saved, "帧", flush=True)

    if {int(r["candidate_label"]) for r in all_records} != {0, 1}:
        raise ValueError("抽帧后必须有两类样本")
    manifest = META / "public_raw.csv"
    if manifest.exists() and any(r.get("source_dataset") != "halmstad_rgb" for r in rows(manifest)):
        backup = META / ("public_raw_previous_" + digest(manifest)[:12] + ".csv")
        if not backup.exists():
            shutil.copy2(manifest, backup)
    write_rows(manifest, all_records)
    save_json(OUT / "halmstad_extraction_summary.json", {
        "videos": dict(Counter(r["class_name"] for r in videos)),
        "sampled_frames": dict(Counter(str(r["candidate_label"]) for r in all_records)),
        "sample_fps": requested_fps, "max_frames_per_video": cap_per_video,
        "video_groups_preserved": True, "visual_review_required": True,
        "note": "Frame counts are not independent scene counts. No missing-box frame is relabeled as bird.",
    })
    print("图片清单：", manifest)
    print("标注叠加预览：", preview_dir)
    print("帧级标签仍需03审核。不要将同视频帧随机拆到训练/测试两侧。")
