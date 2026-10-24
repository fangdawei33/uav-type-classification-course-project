"""Dated transcription of nine already inspected article candidates (not a classifier).

Default: prepare target-box evidence only. After viewing those files, use
--record-inspected to preserve these specific observations. Never auto-approve
new acquisitions; historical outputs refuse overwriting.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from PIL import Image, ImageDraw, ImageOps
from uav_project.common import META, OUT, rows, write_rows, resolve, relative, digest, save_json, now, protect_frozen

DAY = "2026-10-22"
FOLDER = OUT / "review_external_boxes"
SNAPSHOT = OUT / f"drone_type_external_visual_review_supplement_{DAY}.json"
CARDS = OUT / f"review_batches/external_box_supplement_{DAY}.csv"
# Coordinates refer to EXIF-transposed display pixels; right/bottom are exclusive.
OBSERVATIONS = {
    "dta_7cb76e67775a112d57ef": ("candidate_keep", "top_view", "白色折叠机俯视，机身完整可辨；来源正文支持19HW，作者类别对应仍待确认。", (206, 12, 830, 572)),
    "dta_95c539e9571da219e312": ("candidate_keep", "controller_outside_proposed_target", "左侧遥控器不作目标，提议框只取右侧无人机。", (397, 24, 1010, 422)),
    "dta_f69f1fc2268315f3ce90": ("candidate_keep", "closeup|natural_truncation", "正面相机/中央机壳可辨，近景机体截断；保留困难条件标记。", (0, 0, 1024, 510)),
    "dta_d15f60a269f8c5b51672": ("candidate_keep", "angled_view", "斜视整机及灯光可见；同文章桌面相关，不能计成独立飞行。", (65, 44, 962, 577)),
    "dta_c5ce0034d049bfe9f6ff": ("candidate_keep", "multiple_targets|foreground_instance_only", "前景白红原版EZ Pilot为候选目标；背景有其他无人机，不能对整图赋单目标标签。", (0.323, 0.557, 0.716, 0.895)),
    "dta_4696dbbbca48a8391ecb": ("candidate_keep", "ezpilot_to_dataset_emax_mapping_pending", "白红整机侧后视可辨；正文为原版EZ Pilot，不能把品牌直接等同作者Emax类别。", (0.14, 0.206, 0.889, 0.790)),
    "dta_c301d4045fec684aa774": ("candidate_keep", "ezpilot_to_dataset_emax_mapping_pending", "白红整机正面可辨；保留原文图注含Tinyhawk的歧义，身份依正文/图像联合核实。", (0.089, 0.255, 0.974, 0.888)),
    "dta_8c58b02c9dc9a9e09770": ("pending", "disassembled|operational_scope_unresolved", "上壳打开，内部电子器件外露；是拆机状态，尚不批准作正常运行机体测试样本。", None),
    "dta_3e8a5ef93dd0a0c3e1ab": ("pending", "unboxing|propellers_unmounted|operational_scope_unresolved", "Intel官方开箱图，中央机体/碳臂可见，桨未安装且有包装/遥控器；身份线索有用，场景准入与最终框未决。", None),
}
GROUPS = {
    "halfchrome_yh19hw_spark_clone": ("ext_halfchrome_yh19hw_tabletop_article", "同一19HW测评文章及相似桌面布景，仅施加保守相关约束。"),
    "halfchrome_ezpilot_original": ("ext_halfchrome_ezpilot_tabletop_article", "同一原版EZ Pilot测评文章及桌面布景，含拆机图；不是四次独立飞行。"),
    "intel_official_about_rtf": ("ext_intel_official_rtf_unboxing", "同一官方Intel Aero开箱图片来源；不是已确认的独立飞行。"),
}


def prepare():
    protect_frozen()
    if CARDS.exists():
        raise FileExistsError("Historical card manifest exists; do not overwrite: " + str(CARDS))
    raw = {r["sample_id"]: r for r in rows(META / "drone_type_article_raw.csv")}
    FOLDER.mkdir(parents=True, exist_ok=True)
    cards = []
    for sid, (_, _, _, bounds) in OBSERVATIONS.items():
        if bounds is None:
            continue
        r = raw[sid]
        with Image.open(resolve(r["path"])) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
        box = tuple(round(v * (rgb.width if i % 2 == 0 else rgb.height)) for i, v in enumerate(bounds)) if max(bounds) <= 1 else bounds
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= rgb.width and 0 <= y1 < y2 <= rgb.height):
            raise ValueError((sid, box, rgb.size))
        marked = rgb.copy()
        d = ImageDraw.Draw(marked)
        d.rectangle((x1, y1, x2-1, y2-1), outline="red", width=max(2, rgb.width // 250))
        card = Image.new("RGB", (1200, 460), "white")
        for offset, image in ((0, marked), (600, rgb.crop(box))):
            thumb = ImageOps.contain(image, (590, 400))
            card.paste(thumb, (offset + (600-thumb.width)//2, 25 + (400-thumb.height)//2))
        d = ImageDraw.Draw(card)
        d.text((8, 4), sid + " | PROPOSED, NOT FINAL | source / target crop", fill="black")
        d.text((8, 433), f"display {rgb.width}x{rgb.height}; exclusive xyxy={box}; review visualization only", fill="black")
        path = FOLDER / (sid + ".jpg")
        if path.exists():
            raise FileExistsError(path)
        card.save(path, quality=95)
        cards.append({"sample_id": sid, "class_id": r["candidate_label"], "x_min": x1, "y_min": y1,
            "x_max_exclusive": x2, "y_max_exclusive": y2,
            "coordinate_space": "EXIF-transposed RGB display pixels", "image_sha256": r["sha256"],
            "status": "proposed_not_final", "reviewer": "Codex_AI_assisted_visual_review",
            "evidence": relative(path), "evidence_sha256": digest(path)})
    write_rows(CARDS, cards)
    for label in (3, 6):
        part = [r for r in cards if int(r["class_id"]) == label]
        sheet = Image.new("RGB", (1200, 460 * len(part)), "white")
        for index, r in enumerate(part):
            with Image.open(resolve(r["evidence"])) as card:
                sheet.paste(card, (0, 460 * index))
        sheet.save(FOLDER / f"supplement_class_{label}_{DAY}.jpg", quality=95)
    print("Prepared seven proposed-box cards; no decisions imported.")


def record():
    protect_frozen()
    if SNAPSHOT.exists():
        raise FileExistsError("Historical observation snapshot exists: " + str(SNAPSHOT))
    raw = {r["sample_id"]: r for r in rows(META / "drone_type_article_raw.csv")}
    sheet = {r["sample_id"]: r for r in rows(OUT / "review_external/manifest.csv")}
    cards = {r["sample_id"]: r for r in rows(CARDS)}
    current = rows(META / "drone_type_external_review.csv")
    by_id = {r["sample_id"]: r for r in current}
    groups = rows(META / "drone_type_external_reviewed_groups.csv")
    proposals = rows(META / "drone_type_external_box_proposals.csv")
    updates, observations, refs = [], [], []
    for sid, (decision, flags, note, bounds) in OBSERVATIONS.items():
        r, m = raw[sid], sheet[sid]
        group, basis = GROUPS[r["source_id"]]
        evidence = m["sheet"]
        updated = {**by_id[sid], "visual_decision": decision,
            "label_status": "article_caption_supported_class_mapping_pending",
            "box_status": "proposed_not_final" if bounds else "pending",
            "source_group": group, "source_group_status": "reviewed_correlation_constraint",
            "flags": flags, "note": note, "reviewer": "Codex_AI_assisted_visual_review",
            "evidence": evidence + "; position=" + m["position"]}
        updates.append(updated)
        observations.append({**updated, "sheet": evidence, "sheet_position": m["position"],
            "evidence_sha256": digest(resolve(evidence)), "image_sha256": r["sha256"],
            "original_image_inspected": True, "original_display_may_be_downscaled": int(r["width"]) > 1600,
            "detail_snapshot": r["detail_snapshot"], "detail_sha256": r["detail_sha256"],
            "box_evidence": cards.get(sid, {}).get("evidence", ""),
            "box_evidence_sha256": cards.get(sid, {}).get("evidence_sha256", "")})
        groups.append({"sample_id": sid, "source_group": group, "status": "reviewed_correlation_constraint",
            "basis": basis, "evidence": evidence, "source_url": r["source_url"],
            "detail_snapshot": r["detail_snapshot"], "independent_from_public": "unverified", "original_flight_id": ""})
        if bounds:
            proposals.append(cards[sid])
    for r in raw.values():
        if r.get("sample_scope") == "reference_only":
            refs.append({"sample_id": r["sample_id"], "scope": "reference_only", "source_url": r["source_url"],
                "image_evidence": r["path"], "image_sha256": r["sha256"],
                "detail_snapshot": r["detail_snapshot"], "detail_sha256": r["detail_sha256"],
                "original_image_inspected": True,
                "observation": "已查看Tinyhawk整机/配件/底面照片；与作者Emax具体产品对应未确证，不进入实验候选。"})
    # Preserve the prior current tables as history before transcribing observations.
    for name, data in (("review", current), ("groups", rows(META / "drone_type_external_reviewed_groups.csv")),
                       ("boxes", rows(META / "drone_type_external_box_proposals.csv"))):
        backup = OUT / f"review_batches/external_before_supplement_{name}_{DAY}.csv"
        if backup.exists():
            raise FileExistsError(backup)
        write_rows(backup, data)
    update_by_id = {r["sample_id"]: r for r in updates}
    write_rows(META / "drone_type_external_review.csv", [update_by_id.get(r["sample_id"], r) for r in current])
    write_rows(META / "drone_type_external_reviewed_groups.csv", groups)
    write_rows(META / "drone_type_external_box_proposals.csv", proposals)
    write_rows(OUT / f"review_batches/external_supplement_updates_{DAY}.csv", updates)
    save_json(SNAPSHOT, {"recorded_at": now(), "scope": "nine new candidates and six reference-only images",
        "review_type": "AI-assisted visual observations, not independent human authentication",
        "records": observations, "reference_only_inspections": refs,
        "final_labels_or_boxes_approved": False, "source_independence_confirmed": False,
        "software_tests_run": False, "training_or_model_evaluation_run": False})
    print("Recorded nine candidate observations, seven box proposals and six reference-only inspections.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-inspected", action="store_true")
    args = parser.parse_args()
    record() if args.record_inspected else prepare()
