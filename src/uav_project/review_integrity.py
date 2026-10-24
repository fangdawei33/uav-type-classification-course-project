"""Reconcile saved visual-review evidence with actual data files and decisions.

This is a data integrity audit. It does not approve labels, inspect target
semantics, verify source independence, or run a model experiment.
"""
from collections import Counter
from PIL import Image

from .common import META, OUT, digest, now, read_json, relative, resolve, rows, save_json


def audit_review_integrity():
    images = {r["sample_id"]: r for r in rows(META / "drone_type_set_images.csv")}
    quality = {r["sample_id"]: r for r in rows(META / "drone_type_set_quality.csv")}
    reviews = {r["sample_id"]: r for r in rows(META / "drone_type_set_review.csv")}
    issues, snapshots, latest, cached_hashes, decoded = [], [], {}, {}, set()

    def check_hash(path, expected, context):
        if not path or not expected:
            issues.append({"context": context, "issue": "missing_path_or_hash"})
            return
        try:
            actual_path = resolve(path)
            if path not in cached_hashes:
                cached_hashes[path] = digest(actual_path)
            if cached_hashes[path] != expected:
                issues.append({"context": context, "path": path, "issue": "sha256_mismatch"})
        except (OSError, ValueError) as exc:
            issues.append({"context": context, "path": path, "issue": str(exc)})

    ordered = []
    for path in sorted(OUT.glob("drone_type_set_visual_review*.json")):
        snapshot = read_json(path)
        ordered.append((snapshot.get("recorded_at", snapshot.get("date", "")), path.name, path, snapshot))
    # Later snapshots intentionally supersede earlier opinions, while all
    # historical files and hashes remain auditable.
    for _, _, path, snapshot in sorted(ordered):
        records = snapshot.get("records", [])
        snapshots.append({"path": relative(path), "records": len(records), "sha256": digest(path)})
        seen = set()
        for record in records:
            sid = record["sample_id"]
            if sid in seen:
                issues.append({"sample_id": sid, "issue": "duplicate_record_in_snapshot", "snapshot": relative(path)})
            seen.add(sid)
            if sid not in images or sid not in quality or sid not in reviews:
                issues.append({"sample_id": sid, "issue": "unknown_or_missing_sample"})
                continue
            image, q = images[sid], quality[sid]
            if record.get("path", image["path"]) != image["path"]:
                issues.append({"sample_id": sid, "issue": "image_path_mismatch"})
            image_hash = record.get("image_sha256", record.get("sha256"))
            check_hash(image["path"], image_hash, sid + ":image")
            check_hash(q["annotation_path"], record.get("annotation_sha256"), sid + ":annotation")
            check_hash(record.get("evidence"), record.get("evidence_sha256"), sid + ":evidence")
            for extra in record.get("additional_evidence", []):
                check_hash(extra.get("path"), extra.get("sha256"), sid + ":additional_evidence")
            if sid not in decoded:
                try:
                    with Image.open(resolve(image["path"])) as im:
                        im.load()
                        if im.size != (int(q["width"]), int(q["height"])):
                            issues.append({"sample_id": sid, "issue": "image_size_mismatch"})
                    decoded.add(sid)
                except (OSError, ValueError) as exc:
                    issues.append({"sample_id": sid, "issue": "image_decode: " + str(exc)})
            latest[sid] = (record, relative(path))
    for sid, (record, path) in latest.items():
        if record["decision"] != reviews[sid]["decision"]:
            issues.append({"sample_id": sid, "issue": "latest_decision_not_imported", "snapshot": path})
        if "label" in record and int(record["label"]) != int(reviews[sid]["label"]):
            issues.append({"sample_id": sid, "issue": "latest_label_not_imported"})

    constraints = rows(META / "drone_type_set_reviewed_correlation_constraints.csv")
    constraint_ids = set()
    for record in constraints:
        sid = record["sample_id"]
        if sid in constraint_ids:
            issues.append({"sample_id": sid, "issue": "duplicate_correlation_constraint"})
        constraint_ids.add(sid)
        if sid not in latest:
            issues.append({"sample_id": sid, "issue": "constraint_without_visual_record"})
        if sid not in images:
            issues.append({"sample_id": sid, "issue": "unknown_constraint_image"})
            continue
        check_hash(images[sid]["path"], record.get("image_sha256"), sid + ":constraint_image")
        check_hash(record.get("evidence"), record.get("evidence_sha256"), sid + ":constraint_evidence")
        if record.get("source_identity_status") != "unverified" or record.get("flight_id"):
            issues.append({"sample_id": sid, "issue": "visual_constraint_claims_verified_source"})

    box_records = rows(META / "drone_type_set_box_selections.csv")
    for record in box_records:
        sid = record["sample_id"]
        if sid not in images or sid not in quality:
            issues.append({"sample_id": sid, "issue": "unknown_box_image"})
            continue
        check_hash(images[sid]["path"], record.get("sha256"), sid + ":box_image")
        check_hash(quality[sid]["annotation_path"], record.get("annotation_sha256"), sid + ":box_annotation")
        check_hash(record.get("evidence"), record.get("evidence_sha256"), sid + ":box_evidence")
        if not 0 <= int(record["selected_box_index"]) < int(quality[sid]["box_count"]):
            issues.append({"sample_id": sid, "issue": "box_index_out_of_bounds"})

    result = {
        "recorded_at": now(),
        "scope": "Actual image decoding, image/annotation/evidence hashes and latest review-import reconciliation; not software testing or model evaluation.",
        "visually_recorded_unique_images_checked": len(latest),
        "unique_images_decoded": len(decoded),
        "visual_snapshots_checked": snapshots,
        "correlation_constraint_images_checked": len(constraint_ids),
        "correlation_groups": dict(Counter(r["constraint_group"] for r in constraints)),
        "box_selection_records_checked": len(box_records),
        "current_decisions": dict(Counter(r["decision"] for r in reviews.values())),
        "review_csv_sha256": digest(META / "drone_type_set_review.csv"),
        "issues": issues, "all_checked_files_and_decisions_match": not issues,
        "does_not_prove_label_or_source_identity": True,
        "latest_snapshot_supersedes_historical_decision": True,
    }
    output = OUT / ("drone_type_set_review_integrity_" + now()[:10] + ".json")
    save_json(output, result)
    print(f"Data integrity: {len(latest)} recorded images, {len(decoded)} decoded; {len(issues)} issues.")
    print(output)
    if issues:
        raise ValueError("Review integrity issues recorded in " + str(output))
    return result
