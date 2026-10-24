"""Audit the imported seven-class image dataset; do not delete or train anything."""
from collections import Counter, defaultdict
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageOps

from .common import META, OUT, digest, now, protect_frozen, read_json, relative, resolve, rows, save_json, setup, write_rows

REVIEW_PATH = META / "drone_type_set_review.csv"


def image_number(row):
    match = re.search(r"\((\d+)\)\.[^.]+$", row["path"])
    return int(match.group(1)) if match else 0


def make_review():
    protect_frozen()
    setup()
    status = read_json(OUT / "drone_type_set_download_summary.json")
    if not status.get("listed_files_download_complete"):
        raise RuntimeError("原图尚未完整导入当前清单，先完成01数据获取")
    images = rows(META / "drone_type_set_images.csv")
    labels = read_json(META / "drone_type_set_classes.json")["classes"]
    receipts = {r["path"]: r for r in rows(META / "drone_type_set_download_log.csv")}
    previous = {r["sample_id"]: r for r in rows(REVIEW_PATH)}
    by_hash = defaultdict(list)
    sizes = Counter()
    tiny = []
    review = []
    counts = Counter()
    for row in images:
        label = int(row["class_id"])
        if label < 0 or label >= len(labels):
            raise ValueError("未知机型类别ID")
        receipt = receipts[row["path"]]
        by_hash[receipt["sha256"]].append(row)
        width, height = int(receipt["width"]), int(receipt["height"])
        sizes[(width, height)] += 1
        counts[labels[label]] += 1
        if min(width, height) < 64:
            tiny.append({"sample_id": row["sample_id"], "path": row["path"], "width": width, "height": height})
        review.append(previous.get(row["sample_id"], {
            "sample_id": row["sample_id"], "decision": "pending", "label": label,
            "class_name": labels[label], "group_id": "", "note": "",
        }))
    duplicate_rows = []
    for file_hash, members in by_hash.items():
        if len(members) < 2:
            continue
        conflicting = len({r["class_id"] for r in members}) > 1
        for row in members:
            duplicate_rows.append({"sha256": file_hash, "sample_id": row["sample_id"],
                "path": row["path"], "class_id": row["class_id"], "class_name": row["class_name"],
                "group_size": len(members), "cross_class_conflict": conflicting})
    write_rows(REVIEW_PATH, review, ["sample_id", "decision", "label", "class_name", "group_id", "note"])
    write_rows(META / "drone_type_set_exact_duplicates.csv", duplicate_rows,
               ["sha256", "sample_id", "path", "class_id", "class_name", "group_size", "cross_class_conflict"])
    write_rows(META / "drone_type_set_small_images.csv", tiny, ["sample_id", "path", "width", "height"])
    write_rows(META / "drone_type_set_resolutions.csv", [{"width": w, "height": h, "count": n} for (w, h), n in sizes.most_common()])

    folder = OUT / "review" / "drone_type_set"
    folder.mkdir(parents=True, exist_ok=True)
    samples = []
    for class_id, class_name in enumerate(labels):
        subset = sorted((r for r in images if int(r["class_id"]) == class_id), key=image_number)
        if not subset:
            raise ValueError("缺少机型样本：" + class_name)
        indices = sorted({round(j*(len(subset)-1)/5) for j in range(6)})
        sheet = Image.new("RGB", (1200, 660), "white")
        draw = ImageDraw.Draw(sheet)
        for j, index in enumerate(indices):
            row = subset[index]
            x, y = (j % 3)*400, (j // 3)*330
            with Image.open(resolve(row["path"])) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                thumbnail = ImageOps.contain(rgb, (390, 270))
                sheet.paste(thumbnail, (x, y))
            # Overview images remain unmodified. Ground-truth box inspection is separate.
            draw.text((x+4, y+275), f"class={class_id} {class_name}", fill="black")
            draw.text((x+4, y+292), Path(row["path"]).name, fill="black")
            draw.text((x+4, y+309), "source label; not human-approved", fill="black")
            samples.append({"sample_id": row["sample_id"], "path": row["path"], "class_name": class_name,
                            "selection": "six_evenly_spaced_source_numbers_per_class"})
        sheet.save(folder / f"class_{class_id}_overview.jpg", quality=90)
    write_rows(META / "drone_type_set_preview_samples.csv", samples)
    result = {
        "image_count": len(images), "classes": dict(counts), "class_count": len(labels),
        "unique_image_sha256": len(by_hash),
        "exact_duplicate_excess": len(images)-len(by_hash),
        "exact_duplicate_groups": sum(len(v)>1 for v in by_hash.values()),
        "cross_class_exact_duplicate_groups": sum(len(v)>1 and len({r['class_id'] for r in v})>1 for v in by_hash.values()),
        "small_images_below_64px": len(tiny),
        "annotation_syntax_issues": status.get("annotation_issues"),
        "max_min_class_ratio": max(counts.values())/min(counts.values()),
        "preview_images": len(samples), "human_review_complete": False,
        "near_duplicates_checked": False, "source_groups_verified": False, "training_ready": False,
        "source_scope": status.get("scope"),
        "note": "Exact file hashes only. This does not detect resized/compressed duplicates or prove labels correct.",
    }
    save_json(OUT / "drone_type_set_data_audit.json", result)
    print("原图数量：", len(images), "；SHA256不同的图片：", len(by_hash))
    print("完全重复的多余图片：", result["exact_duplicate_excess"])
    print("标注格式/类别映射问题：", result["annotation_syntax_issues"])
    print("数据审核摘要：", OUT / "drone_type_set_data_audit.json")
    print("七类抽样预览：", folder)
    print("仅生成统计与候选审核表；未删除原图、未完成逐图人审、未划分训练集。")


def apply_review(csv_path):
    protect_frozen()
    labels = read_json(META / "drone_type_set_classes.json")["classes"]
    current = {r["sample_id"]: r for r in rows(REVIEW_PATH)}
    if not current:
        raise RuntimeError("请先生成当前七类审核表")
    for incoming in rows(csv_path):
        if incoming["sample_id"] not in current:
            raise ValueError("审核表包含其他数据集样本")
        if incoming["decision"] not in {"pending", "keep", "reject"}:
            raise ValueError("非法审核状态")
        label = int(incoming["label"])
        if label not in range(len(labels)):
            raise ValueError("机型标签应为0至6")
        # Visual acceptance and source verification are separate decisions.
        # apply_cleaning still requires a confirmed, evidenced source group.
        if incoming["decision"] in {"keep", "reject"} and not incoming.get("note", "").strip():
            raise ValueError("保留或剔除决定必须记录审核依据")
        current[incoming["sample_id"]].update({
            "decision": incoming["decision"], "label": label, "class_name": labels[label],
            "group_id": incoming.get("group_id", ""), "note": incoming.get("note", ""),
        })
    write_rows(REVIEW_PATH, current.values())
    print("已导入审核记录，未改动原图。")
    summarize_review()


def summarize_review():
    """Refresh metadata counts after a real review import; do not approve images."""
    review = rows(REVIEW_PATH)
    if not review:
        raise RuntimeError("尚无公开数据审核表")
    images = {r["sample_id"]: r for r in rows(META / "drone_type_set_images.csv")}
    labels = read_json(META / "drone_type_set_classes.json")["classes"]
    # Evidence snapshots record actual inspection. Preview generation alone is
    # never counted, and overlapping review rounds are counted only once.
    inspected = set()
    evidence_files = []
    for path in sorted(OUT.glob("drone_type_set_visual_review*.json")):
        snapshot = read_json(path)
        ids = {r["sample_id"] for r in snapshot.get("records", []) if r.get("sample_id")}
        unknown = ids - images.keys()
        if unknown:
            raise ValueError("视觉审核证据含未知样本：" + str(path))
        inspected.update(ids)
        evidence_files.append({"path": relative(path), "records": len(ids), "sha256": digest(path)})
    decisions = Counter(r["decision"] for r in review)
    by_class = []
    for class_id, name in enumerate(labels):
        subset = [r for r in review if int(images[r["sample_id"]]["class_id"]) == class_id]
        counts = Counter(r["decision"] for r in subset)
        by_class.append({"class_id": class_id, "class_name": name, "total": len(subset),
                         "keep": counts["keep"], "reject": counts["reject"], "pending": counts["pending"],
                         "evidence_recorded_images": sum(r["sample_id"] in inspected for r in subset)})
    boxes = Counter(r["status"] for r in rows(META / "drone_type_set_box_selections.csv"))
    sources = rows(META / "drone_type_set_source_groups.csv")
    confirmed = {r["source_group"] for r in sources if r["status"] == "confirmed"
                 and r.get("source_group") and r.get("source_evidence")}
    constraints = rows(META / "drone_type_set_reviewed_correlation_constraints.csv")
    result = {
        "date": now()[:10], "updated_at": now(), "review_rows": len(review),
        "decision_counts": dict(decisions), "by_class": by_class,
        "visually_inspected_unique_with_recorded_evidence": len(inspected),
        "inspected_but_pending": sum(r["sample_id"] in inspected and r["decision"] == "pending" for r in review),
        "visual_evidence_files": evidence_files,
        "box_choices_reviewed": boxes["reviewed"], "box_choices_proposed": boxes["proposed"],
        "confirmed_source_groups": len(confirmed),
        "reviewed_correlation_constraint_groups": len({r["constraint_group"] for r in constraints}),
        "correlation_constraint_images": len({r["sample_id"] for r in constraints}),
        "correlation_constraints_are_verified_flights": False,
        "review_csv_sha256": digest(REVIEW_PATH),
        "review_method": "AI-assisted contact-sheet visual inspection; not independent human double annotation",
        "summary_scope": "Current review metadata and recorded evidence coverage; this command does not inspect pixels, rerun image integrity checks, or verify source identities.",
    }
    save_json(OUT / "drone_type_set_review_progress.json", result)
    print(f"当前审核：keep {decisions['keep']}，reject {decisions['reject']}，pending {decisions['pending']}；总数 {len(review)}。")
    print(f"已有逐图视觉证据 {len(inspected)} 张；正式来源确认 {len(confirmed)} 组。")
    print("当前进度：", OUT / "drone_type_set_review_progress.json")
    return result
