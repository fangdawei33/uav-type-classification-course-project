"""Materialize reviewed target crops without granting training/source approval."""
from collections import Counter
from pathlib import Path
from PIL import Image
from .common import ROOT, META, OUT, rows, resolve, digest, relative, write_rows, save_json, now
from .cleaning import boxes_for, pixel_box


def export_candidate_crops():
    cohort_path = META / "drone_type_set_visual_cohort.csv"
    quality = {r["sample_id"]: r for r in rows(META / "drone_type_set_quality.csv")}
    cohort = rows(cohort_path)
    selections = {r["sample_id"]: r for r in rows(META / "drone_type_set_box_selections.csv")}
    destination = ROOT / "data/processed/drone_type_set/visual_candidate_crops"
    manifest_path = META / "drone_type_set_candidate_crops.csv"
    signature = {"cohort": digest(cohort_path), "quality": digest(META / "drone_type_set_quality.csv"),
                 "box_selections": digest(META / "drone_type_set_box_selections.csv")}
    receipt_path = OUT / "drone_type_set_candidate_crops.json"
    if receipt_path.exists():
        from .common import read_json
        if read_json(receipt_path)["signature"] != signature:
            raise RuntimeError("候选审核已变化；另建版本，不能覆盖既有候选裁剪。")
    destination.mkdir(parents=True, exist_ok=True)
    prepared = []
    # Validate every source/box before writing any derived image.
    for row in cohort:
        if row["visual_decision"] != "keep":
            continue
        sid = row["sample_id"]
        q = quality[sid]
        if digest(resolve(row["path"])) != row["image_sha256"]:
            raise ValueError("原图哈希变化：" + sid)
        if digest(resolve(row["annotation_path"])) != row["annotation_sha256"]:
            raise ValueError("标注哈希变化：" + sid)
        if int(q["exif_orientation"]) != 1:
            raise ValueError("EXIF方向需先审核：" + sid)
        boxes = boxes_for({**row, "class_id": row["label"]})
        choice = selections.get(sid)
        if choice:
            if choice["status"] != "reviewed" or not choice.get("reason") or not choice.get("reviewer"):
                raise ValueError("框选择尚未审核：" + sid)
            if choice["sha256"] != row["image_sha256"] or choice["annotation_sha256"] != row["annotation_sha256"]:
                raise ValueError("框选择哈希不匹配：" + sid)
            if digest(resolve(choice["evidence"])) != choice["evidence_sha256"]:
                raise ValueError("框选择证据变化：" + sid)
            index = int(choice["selected_box_index"])
        elif len(boxes) == 1:
            index = 0
        else:
            raise ValueError("多框样本缺少明确选择：" + sid)
        rect = pixel_box(boxes[index], int(row["width"]), int(row["height"]))
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError("空裁剪：" + sid)
        prepared.append((row, index, rect))
    records = []
    for n, (row, index, rect) in enumerate(prepared, 1):
        target = destination / (row["sample_id"] + ".png")
        with Image.open(resolve(row["path"])) as image:
            crop = image.convert("RGB").crop(rect)
            if target.exists():
                with Image.open(target) as existing:
                    if existing.size != crop.size or existing.convert("RGB").tobytes() != crop.tobytes():
                        raise ValueError("已有裁剪内容不同，停止覆盖：" + str(target))
            else:
                temporary = target.with_suffix(".tmp.png")
                crop.save(temporary)
                temporary.replace(target)
        records.append({"sample_id": row["sample_id"], "label": row["label"], "class_name": row["class_name"],
                        "clean_path": relative(target), "clean_sha256": digest(target),
                        "parent_path": row["path"], "parent_sha256": row["image_sha256"],
                        "selected_annotation_index": index, "crop_xyxy": ",".join(map(str, rect)),
                        "conservative_constraint_group": row["conservative_constraint_group"],
                        "source_verified": False, "training_approved": False,
                        "role": "reviewed_visual_candidate_only_not_a_split"})
        if n % 400 == 0:
            print(f"候选裁剪 {n}/{len(prepared)}", flush=True)
    write_rows(manifest_path, records)
    save_json(receipt_path, {"created_at": now(), "signature": signature, "images": len(records),
                            "class_counts": dict(Counter(r["class_name"] for r in records)),
                            "manifest_sha256": digest(manifest_path), "training_ready": False,
                            "formal_cleaning_replaced": False, "raw_files_modified": 0,
                            "note": "Reusable reviewed crops; no source confirmation, dataset split or independence claim."})
    print("候选目标裁剪已保存：", manifest_path, "；尚未批准训练划分。")

