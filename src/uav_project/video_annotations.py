"""Convert per-frame Halmstad MCOS annotations into an auditable JSON format.

The third-party decoder documents IR compatibility only. RGB conversion here
is provisional: frame counts, geometry and preview overlays must be reviewed.
"""
from __future__ import annotations

import importlib.metadata
import math
from pathlib import Path

from .common import ROOT, META, OUT, setup, protect_frozen, rows, save_json, write_rows, resolve, relative, digest, append_log

ANNOTATIONS = ROOT / "data" / "annotations" / "halmstad"


def convert_annotations():
    protect_frozen()
    setup()
    import cv2
    from mcos_decoder import load_groundtruth

    videos = rows(META / "halmstad_videos.csv")
    if not videos:
        raise RuntimeError("先运行01下载原始视频和标注；不能从旧图片清单推断视频标签。")
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    summary = []
    failures = []
    for row in videos:
        cap = None
        try:
            video = resolve(row["video_path"])
            labels = resolve(row["labels_path"])
            if digest(video) != row["video_sha256"] or digest(labels) != row["labels_sha256"]:
                raise ValueError("原视频或标注发生变化")
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise ValueError("OpenCV无法打开视频")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if frame_count <= 0 or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
                raise ValueError("视频帧数、尺寸或帧率无效")
            boxes = load_groundtruth(labels)
            if len(boxes) != frame_count:
                raise ValueError(f"RGB解码未通过帧数核对：标注{len(boxes)}，视频{frame_count}。不得补齐或截断掩盖问题。")
            frames = []
            for frame_index, box in enumerate(boxes):
                converted = None
                if box is not None:
                    if len(box) != 4:
                        raise ValueError("解码器未返回单个[x,y,w,h]框")
                    converted = [float(v) for v in box]
                    x, y, w, h = converted
                    if not all(math.isfinite(v) for v in converted) or w <= 0 or h <= 0:
                        raise ValueError(f"第{frame_index}帧目标框不是有效数值")
                    if x < -1 or y < -1 or x+w > width+1 or y+h > height+1:
                        raise ValueError(f"第{frame_index}帧框超出视频尺寸，需要检查坐标或解码")
                frames.append({"frame_index": frame_index, "bbox_xywh_matlab": converted})
            present = sum(f["bbox_xywh_matlab"] is not None for f in frames)
            if present == 0:
                raise ValueError("没有解码出目标框，不能把该视频自动标成鸟类/无人机样本")
            target = ANNOTATIONS / (row["video_id"] + ".json")
            save_json(target, {"video_id": row["video_id"], "video_sha256": row["video_sha256"],
                "labels_sha256": row["labels_sha256"], "revision": row["revision"],
                "class_name_from_filename": row["class_name"], "candidate_label": int(row["candidate_label"]),
                "width": width, "height": height, "fps": fps, "frame_count": frame_count,
                "decoder": "mcos-decoder", "decoder_version": importlib.metadata.version("mcos-decoder"),
                "validation": "frame_count_and_geometry_checked; RGB_semantics_require_visual_review",
                "coordinate_note": "Original MATLAB xywh retained; images are not cropped using these coordinates.",
                "frames": frames})
            summary.append({"video_id": row["video_id"], "frames": frame_count,
                "frames_with_box": present, "status": "needs_visual_review", "json_path": relative(target)})
            print(row["video_id"], frame_count, "帧；有框", present, flush=True)
        except Exception as exc:
            failures.append(row["video_id"])
            summary.append({"video_id": row["video_id"], "frames": "", "frames_with_box": "",
                            "status": "failed", "json_path": ""})
            append_log("halmstad_annotation_errors", {"video_id": row["video_id"], "error": repr(exc)})
        finally:
            if cap is not None:
                cap.release()
        write_rows(META / "halmstad_annotation_status.csv", summary)
    if failures:
        raise RuntimeError(f"{len(failures)} 个视频的标注转换未完成。回传 halmstad_annotation_errors.jsonl；不跳过失败序列训练。")
    print("数值转换完成，不等于RGB标注语义已验证。01c会生成框预览，再进行人工审核。")
