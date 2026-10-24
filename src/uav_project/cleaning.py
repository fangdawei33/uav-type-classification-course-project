"""Seven-class cleaning: prepare evidence first, apply only explicit reviewed decisions.

Raw images and annotations are immutable inputs. Perceptual similarity is a
candidate-review signal, never proof of duplicate identity or source flight.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .common import ROOT, META, OUT, config, digest, protect_frozen, read_json, resolve, relative, rows, save_json, setup, write_rows
from .review import REVIEW_PATH, image_number

QUALITY = META / "drone_type_set_quality.csv"
PREPARED = OUT / "drone_type_set_cleaning_preparation.json"
IMAGE_MANIFEST = META / "drone_type_set_images.csv"
BOX_SELECTIONS = META / "drone_type_set_box_selections.csv"


def boxes_for(row):
    boxes = []
    for line in resolve(row["annotation_path"]).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5 or int(fields[0]) != int(row["class_id"]):
            raise ValueError("标注类别或格式不一致")
        values = tuple(float(x) for x in fields[1:])
        if not all(math.isfinite(x) for x in values):
            raise ValueError("非有限坐标")
        cx, cy, width, height = values
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError("归一化坐标范围不合法")
        boxes.append(values)
    return boxes


def pixel_box(box, width, height):
    cx, cy, bw, bh = box
    return (max(0, math.floor((cx-bw/2)*width)), max(0, math.floor((cy-bh/2)*height)),
            min(width, math.ceil((cx+bw/2)*width)), min(height, math.ceil((cy+bh/2)*height)))


def difference_hash(gray):
    array = np.asarray(gray.resize((9, 8)), dtype=np.int16)
    return int("".join("1" if v else "0" for v in (array[:, 1:] > array[:, :-1]).ravel()), 2)


def source_flags():
    path = OUT / "drone_type_set_visual_observations.json"
    result = defaultdict(list)
    if path.exists():
        for item in read_json(path)["observations"]:
            for filename in item["examples"]:
                # Names are only used to attach recorded human observations, never as model input.
                result[(item["class_name"].casefold(), filename)].extend(item["flags"])
    return result


def prepare_cleaning():
    protect_frozen()
    setup()
    cfg = config()["cleaning_review"]
    status = read_json(OUT / "drone_type_set_download_summary.json")
    if not status.get("listed_files_download_complete"):
        raise RuntimeError("先完成当前数据清单的导入")
    images = sorted(rows(IMAGE_MANIFEST), key=lambda r: (int(r["class_id"]), image_number(r)))
    receipts = {r["path"]: r for r in rows(META / "drone_type_set_download_log.csv")}
    notes = source_flags()
    folder = OUT / "cleaning_candidates"
    folder.mkdir(parents=True, exist_ok=True)
    quality, thumbnails = [], []
    pages = defaultdict(list)
    for index, row in enumerate(images, 1):
        original = resolve(row["path"])
        file_hash = digest(original)
        if file_hash != receipts[row["path"]]["sha256"]:
            raise ValueError("原图在导入后改变：" + row["path"])
        flags = list(notes.get((row["class_name"].casefold(), original.name), []))
        with Image.open(original) as source:
            orientation = source.getexif().get(274, 1)
            # Do not transpose without transforming the source coordinates as well.
            image = source.convert("RGB")
            width, height = image.size
            gray = image.convert("L")
            small = np.asarray(gray.resize((32, 32)), dtype=np.int16)
            thumb128 = np.asarray(gray.resize((128, 128)), dtype=np.float32)
            laplacian = (thumb128[:-2, 1:-1] + thumb128[2:, 1:-1] + thumb128[1:-1, :-2]
                         + thumb128[1:-1, 2:] - 4*thumb128[1:-1, 1:-1])
            if orientation != 1:
                flags.append("exif_orientation_requires_review")
            boxes = boxes_for(row)
            if len(boxes) != 1:
                flags.append("not_single_target")
            crop = None
            rect = None
            if len(boxes) == 1:
                rect = pixel_box(boxes[0], width, height)
                left, top, right, bottom = rect
                if right <= left or bottom <= top:
                    flags.append("empty_pixel_box")
                else:
                    crop = image.crop(rect)
                    if min(crop.size) < cfg["small_target_pixels"]:
                        flags.append("small_target_review_not_auto_reject")
                    if left == 0 or top == 0 or right == width or bottom == height:
                        flags.append("box_touches_image_edge")
            if float(np.mean(small < 16)) > cfg["dark_pixel_fraction"]:
                flags.append("mostly_dark_background_review")
            # Derived visualizations only: never used as training images.
            marked = image.copy()
            painter = ImageDraw.Draw(marked)
            for box in boxes:
                painter.rectangle(pixel_box(box, width, height), outline="red", width=3)
            card = Image.new("RGB", (720, 310), "white")
            card.paste(ImageOps.contain(marked, (420, 250)), (0, 0))
            if crop is not None:
                card.paste(ImageOps.contain(crop, (285, 250)), (430, 0))
            draw = ImageDraw.Draw(card)
            draw.text((5, 254), f"class={row['class_id']} {row['class_name']} | {original.name}", fill="black")
            draw.text((5, 272), "Left: original + source box. Right: proposed object crop.", fill="black")
            draw.text((5, 290), "REVIEW ONLY. Not an approved training sample.", fill="black")
            card_path = folder / (row["sample_id"] + ".jpg")
            card.save(card_path, quality=85)
            pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        values = {
            "sample_id": row["sample_id"], "path": row["path"], "annotation_path": row["annotation_path"],
            "class_id": row["class_id"], "class_name": row["class_name"], "sha256": file_hash,
            "annotation_sha256": digest(resolve(row["annotation_path"])), "pixel_sha256": pixel_hash,
            "dhash64": str(difference_hash(gray)), "width": width, "height": height,
            "box_count": len(boxes), "box_left": rect[0] if rect else "", "box_top": rect[1] if rect else "",
            "box_right": rect[2] if rect else "", "box_bottom": rect[3] if rect else "",
            "brightness": round(float(small.mean()), 3), "contrast": round(float(small.std()), 3),
            "laplacian_variance_128": round(float(laplacian.var()), 3),
            "flags": "|".join(sorted(set(flags))), "exif_orientation": orientation,
            "preview_path": relative(card_path), "review_status": "pending",
        }
        quality.append(values)
        thumbnails.append(small)
        pages[int(row["class_id"])].append(values)
        if index % 500 == 0:
            print("数据质量特征与目标框预览：", index, "/", len(images), flush=True)

    # Similar backgrounds and small targets may collide: keep pairs as hypotheses only.
    near_pairs = []
    hash_values = [int(r["dhash64"]) for r in quality]
    for i, current in enumerate(quality):
        for j in range(i):
            other = quality[j]
            hamming = (hash_values[i] ^ hash_values[j]).bit_count()
            if hamming > cfg["candidate_hamming_distance"]:
                continue
            distance = float(np.abs(thumbnails[i] - thumbnails[j]).mean())
            if distance <= cfg["candidate_gray_mean_absolute_difference"]:
                near_pairs.append({
                    "sample_a": other["sample_id"], "sample_b": current["sample_id"],
                    "class_a": other["class_id"], "class_b": current["class_id"],
                    "hamming": hamming, "gray_mean_absolute_difference": round(distance, 3),
                    "pixel_identical": other["pixel_sha256"] == current["pixel_sha256"],
                    "decision": "pending_similarity_not_identity",
                })
    write_rows(QUALITY, quality)
    write_rows(META / "drone_type_set_near_duplicate_candidates.csv", near_pairs,
               ["sample_a", "sample_b", "class_a", "class_b", "hamming", "gray_mean_absolute_difference", "pixel_identical", "decision"])
    # Source groups remain unfilled. A new image ID is not evidence of a new flight.
    groups_path = META / "drone_type_set_source_groups.csv"
    previous_groups = {r["sample_id"]: r for r in rows(groups_path)}
    group_rows = []
    for row in quality:
        group_rows.append(previous_groups.get(row["sample_id"], {
            "sample_id": row["sample_id"], "path": row["path"], "class_name": row["class_name"],
            "source_group": "", "source_evidence": "", "status": "unverified",
        }))
    write_rows(groups_path, group_rows)

    class_links = []
    for class_id, members in sorted(pages.items()):
        filename = f"class_{class_id}.html"
        parts = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>数据清洗复核</title>",
                 "<style>body{font-family:system-ui;margin:24px;background:#f5f6f8}article{background:white;padding:12px;margin:16px 0;max-width:760px}img{max-width:100%}code{overflow-wrap:anywhere}.flags{color:#9b3315}</style>",
                 f"<h1>{html.escape(members[0]['class_name'])}</h1><p>红框来自原始标注；右侧为候选裁剪。未自动批准、未删除原图、未划分训练集。</p>",
                 "<p><a href='index.html'>返回类别目录</a></p>"]
        for row in members:
            parts.append("<article><strong>" + html.escape(Path(row["path"]).name) + "</strong>"
                         + "<p><code>" + html.escape(row["sample_id"]) + "</code></p>"
                         + "<img loading='lazy' src='" + html.escape(row["sample_id"] + ".jpg", quote=True) + "'>"
                         + "<p class='flags'>" + html.escape(row["flags"] or "暂无自动标记；仍需审核") + "</p></article>")
        parts.append("</html>")
        (folder / filename).write_text("\n".join(parts), encoding="utf-8")
        class_links.append(f"<li><a href='{filename}'>{html.escape(members[0]['class_name'])}</a>：{len(members)}张</li>")
    (folder / "index.html").write_text(
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>七类清洗预览</title>"
        "<h1>七类目标框与裁剪预览</h1><p>仅供审核，不是清洗后的训练集。</p><ul>" + "".join(class_links) + "</ul></html>", encoding="utf-8")
    exact_pixels = Counter(r["pixel_sha256"] for r in quality)
    flag_counts = Counter(flag for r in quality for flag in r["flags"].split("|") if flag)
    result = {
        "status": "review_candidates_prepared", "image_count": len(quality), "classes": dict(Counter(r["class_name"] for r in quality)),
        "manifest_sha256": digest(IMAGE_MANIFEST), "quality_csv_sha256": digest(QUALITY),
        "preparation_parameters": cfg, "pixel_exact_duplicate_excess": sum(n-1 for n in exact_pixels.values()),
        "near_duplicate_candidate_pairs": len(near_pairs), "flag_counts": dict(flag_counts),
        "images_with_flags": sum(bool(r["flags"]) for r in quality),
        "source_groups_verified": False, "human_review_complete": False,
        "raw_images_deleted": 0, "clean_dataset_created": False, "training_ready": False,
        "candidate_warning": "Perceptual similarity can reflect a shared background; do not delete or assign source identity automatically.",
        "crop_protocol_note": "Using ground-truth crops later defines given-region classification, not end-to-end detection.",
    }
    save_json(PREPARED, result)
    print("清洗候选准备完成：", PREPARED)
    print("查看预览：", folder / "index.html")
    print("近重复候选对：", len(near_pairs), "；未自动删除或合并来源组。")


def apply_cleaning(view):
    protect_frozen()
    setup()
    if view not in {"full_frame", "target_crop"}:
        raise ValueError("应用清洗前需明确--view full_frame或target_crop")
    prepared = read_json(PREPARED)
    if prepared["manifest_sha256"] != digest(IMAGE_MANIFEST) or prepared["quality_csv_sha256"] != digest(QUALITY):
        raise ValueError("清洗候选生成后输入已变化，请重新准备")
    review = {r["sample_id"]: r for r in rows(REVIEW_PATH)}
    images = rows(IMAGE_MANIFEST)
    quality = {r["sample_id"]: r for r in rows(QUALITY)}
    source_groups = {r["sample_id"]: r for r in rows(META / "drone_type_set_source_groups.csv")}
    classes = read_json(META / "drone_type_set_classes.json")["classes"]
    # Explicit annotation choices repair false extra boxes without rewriting labels.
    selection_rows = rows(BOX_SELECTIONS)
    selections = {r["sample_id"]: r for r in selection_rows}
    if len(selections) != len(selection_rows):
        raise ValueError("目标框选择表包含重复样本")
    if set(selections) - {r["sample_id"] for r in images}:
        raise ValueError("目标框选择表包含未知样本")
    rectangles, selected_indices = {}, {}
    # Complete preflight prevents partly approved datasets from being declared ready.
    for image in images:
        r = review.get(image["sample_id"])
        if not r or r["decision"] not in {"keep", "reject"}:
            raise ValueError("仍有待审核样本；不自动把pending改为keep")
        if r["decision"] == "keep":
            group = source_groups.get(image["sample_id"], {})
            if group.get("status") != "confirmed" or not group.get("source_group") or not group.get("source_evidence"):
                raise ValueError("保留样本尚未确认来源组与证据：" + image["sample_id"])
            if r["group_id"] != group["source_group"]:
                raise ValueError("审核表group_id与来源表不一致")
            if int(r["label"]) not in range(len(classes)):
                raise ValueError("类别ID超出范围")
            q = quality[image["sample_id"]]
            if digest(resolve(image["path"])) != q["sha256"] or digest(resolve(image["annotation_path"])) != q["annotation_sha256"]:
                raise ValueError("原图或标注在审核后改变")
            if int(q["exif_orientation"]) != 1:
                raise ValueError("EXIF方向需独立处理，不能直接套原坐标")
            if view == "target_crop":
                boxes = boxes_for(image)
                choice = selections.get(image["sample_id"])
                if choice:
                    if (choice.get("status") != "reviewed" or not choice.get("reason", "").strip()
                            or not choice.get("reviewer", "").strip()
                            or not choice.get("evidence", "").strip()
                            or not resolve(choice["evidence"]).is_file()):
                        raise ValueError("目标框选择缺少审核证据")
                    if choice["sha256"] != q["sha256"] or choice["annotation_sha256"] != q["annotation_sha256"]:
                        raise ValueError("目标框选择对应的原始文件已变化")
                    if digest(resolve(choice["evidence"])) != choice["evidence_sha256"]:
                        raise ValueError("目标框审核图发生变化，需要重新核查")
                    index = int(choice["selected_box_index"])
                elif len(boxes) == 1:
                    index = 0
                else:
                    raise ValueError("多个标注框需明确审核选择，不能自动选最大框")
                if not 0 <= index < len(boxes):
                    raise ValueError("目标框选择索引越界")
                rect = pixel_box(boxes[index], int(q["width"]), int(q["height"]))
                if rect[2] <= rect[0] or rect[3] <= rect[1]:
                    raise ValueError("选定的目标框为空")
                rectangles[image["sample_id"]] = rect
                selected_indices[image["sample_id"]] = index
    destination = ROOT / "data" / "processed" / "drone_type_set" / view
    destination.mkdir(parents=True, exist_ok=True)
    kept, events, seen_pixels = [], [], {}
    for image in images:
        r = review[image["sample_id"]]
        q = quality[image["sample_id"]]
        original = resolve(image["path"])
        if digest(original) != q["sha256"] or digest(resolve(image["annotation_path"])) != q["annotation_sha256"]:
            raise ValueError("原图或标注在审核后改变")
        if r["decision"] == "reject":
            events.append({"sample_id": image["sample_id"], "reason": "manual_reject", "detail": r["note"]})
            continue
        if q["pixel_sha256"] in seen_pixels:
            canonical = seen_pixels[q["pixel_sha256"]]
            if int(canonical["label"]) != int(r["label"]):
                raise ValueError("像素完全相同的保留图存在标签冲突")
            events.append({"sample_id": image["sample_id"], "reason": "exact_duplicate", "detail": canonical["sample_id"]})
            continue
        if int(q["exif_orientation"]) != 1:
            raise ValueError("EXIF方向需独立处理，不能直接套原坐标")
        with Image.open(original) as source:
            result = source.convert("RGB")
            if view == "target_crop":
                result = result.crop(rectangles[image["sample_id"]])
            target = destination / (image["sample_id"] + ".png")
            result.save(target)
        record = {**image, "label": int(r["label"]), "class_name": classes[int(r["label"])],
                  "group_id": r["group_id"], "domain": "public", "clean_path": relative(target),
                  "clean_sha256": digest(target), "view": view, "review_note": r["note"],
                  "selected_annotation_index": selected_indices.get(image["sample_id"], ""),
                  "crop_xyxy": ",".join(map(str, rectangles.get(image["sample_id"], ())))}
        kept.append(record)
        seen_pixels[q["pixel_sha256"]] = {"sample_id": image["sample_id"], "label": r["label"]}
    write_rows(META / "drone_type_set_clean_manifest.csv", kept)
    write_rows(META / "drone_type_set_cleaning_events.csv", events, ["sample_id", "reason", "detail"])
    save_json(OUT / "drone_type_set_cleaning_summary.json", {
        "status": "reviewed_clean_dataset_created", "view": view, "raw_images": len(images),
        "kept": len(kept), "events": dict(Counter(r["reason"] for r in events)),
        "review_sha256": digest(REVIEW_PATH), "source_groups_sha256": digest(META / "drone_type_set_source_groups.csv"),
        "box_selections_sha256": digest(BOX_SELECTIONS) if BOX_SELECTIONS.exists() else None,
        "split_created": False, "raw_images_deleted": 0,
    })
    print("清洗结果已另存；原图保留，尚未划分数据集。")


def clean_data(apply=False, view=None):
    if apply:
        apply_cleaning(view)
    else:
        prepare_cleaning()
