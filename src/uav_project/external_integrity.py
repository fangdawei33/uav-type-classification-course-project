"""Inspect acquired files and recorded review evidence; never run model/software tests."""
from collections import Counter, defaultdict
from PIL import Image, ImageOps
from .common import META, OUT, rows, digest, resolve, relative, read_json, save_json, now
from .external_review import collected_external_candidates


def audit_external_integrity():
    candidates = collected_external_candidates()
    references = [r for r in rows(META / "drone_type_article_raw.csv") if r.get("sample_scope") == "reference_only"]
    acquired = candidates + references
    reviews = rows(META / "drone_type_external_review.csv")
    review_by_id = {r["sample_id"]: r for r in reviews}
    quality = {r["sample_id"]: r for r in rows(META / "drone_type_external_quality.csv")}
    groups = rows(META / "drone_type_external_reviewed_groups.csv")
    proposals = rows(META / "drone_type_external_box_proposals.csv")
    issues, decoded, source_checks, evidence_checks = [], 0, 0, 0
    sizes, files = {}, defaultdict(list)
    candidate_ids = {r["sample_id"] for r in candidates}
    reference_ids = {r["sample_id"] for r in references}
    if len(candidate_ids) != len(candidates) or len(review_by_id) != len(reviews):
        issues.append({"issue": "duplicate_candidate_or_review_id"})
    if set(review_by_id) != candidate_ids:
        issues.append({"issue": "review_candidate_id_mismatch"})
    if reference_ids & candidate_ids:
        issues.append({"issue": "reference_in_candidate_pool"})
    raw_by_id = {r["sample_id"]: r for r in acquired}
    for r in acquired:
        sid = r["sample_id"]
        try:
            path = resolve(r["path"])
            actual = digest(path)
            if actual != r["sha256"]:
                raise ValueError("image hash mismatch")
            files[actual].append(sid)
            with Image.open(path) as image:
                image.load()
                stored_size = image.size
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                sizes[sid] = rgb.size
            # Collection CSV keeps stored dimensions; box/quality tables use EXIF display pixels.
            if stored_size != (int(r["width"]), int(r["height"])):
                raise ValueError("stored dimensions differ from collection record")
            decoded += 1
            if digest(resolve(r["detail_snapshot"])) != r["detail_sha256"]:
                raise ValueError("source snapshot hash mismatch")
            source_checks += 1
            if sid in candidate_ids:
                q = quality[sid]
                if q["sha256"] != actual or sizes[sid] != (int(q["display_width"]), int(q["display_height"])):
                    raise ValueError("candidate quality record mismatch")
        except Exception as exc:
            issues.append({"sample_id": sid, "issue": str(exc)})
    # Compare every current candidate decision with its most recent saved observation.
    latest = {}
    snapshot_paths = list(OUT.glob("drone_type_external_visual_review*.json")) + list(OUT.glob("drone_type_article_visual_review*.json"))
    snapshots = [(read_json(path), path) for path in snapshot_paths]
    for data, path in sorted(snapshots, key=lambda item: (item[0].get("recorded_at", ""), item[1].name)):
        for record in data.get("records", []):
            sid = record["sample_id"]
            latest[sid] = (record, relative(path))
    for sid in candidate_ids:
        try:
            record, snapshot = latest[sid]
            review = review_by_id[sid]
            if review["visual_decision"] != record.get("visual_decision", record.get("decision")):
                raise ValueError("current decision differs from saved observation")
            if record["image_sha256"] != raw_by_id[sid]["sha256"]:
                raise ValueError("observation image hash mismatch")
            evidence = record.get("sheet", record["evidence"]).split(";")[0]
            if digest(resolve(evidence)) != record["evidence_sha256"]:
                raise ValueError("observation evidence hash mismatch")
            if review["evidence"].split(";")[0] != evidence:
                raise ValueError("current review points to different evidence")
            evidence_checks += 1
        except Exception as exc:
            issues.append({"sample_id": sid, "issue": str(exc)})
    for p in proposals:
        try:
            sid = p["sample_id"]
            w, h = sizes[sid]
            x1, y1, x2, y2 = (int(p[k]) for k in ("x_min", "y_min", "x_max_exclusive", "y_max_exclusive"))
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ValueError("invalid proposal bounds")
            if p["image_sha256"] != raw_by_id[sid]["sha256"] or digest(resolve(p["evidence"])) != p["evidence_sha256"]:
                raise ValueError("proposal image/evidence hash mismatch")
        except Exception as exc:
            issues.append({"sample_id": p["sample_id"], "issue": str(exc)})
    for g in groups:
        sid = g["sample_id"]
        if sid not in candidate_ids or review_by_id[sid]["source_group"] != g["source_group"]:
            issues.append({"sample_id": sid, "issue": "correlation group differs from current review"})
        elif g["source_url"] != raw_by_id[sid]["source_url"] or g["detail_snapshot"] != raw_by_id[sid]["detail_snapshot"]:
            issues.append({"sample_id": sid, "issue": "correlation group source differs from collection"})
    ref_observed = {}
    for path in sorted(OUT.glob("drone_type_external_visual_review*.json")):
        for record in read_json(path).get("reference_only_inspections", []):
            ref_observed[record["sample_id"]] = record
    for sid in reference_ids:
        record = ref_observed.get(sid, {})
        if not record.get("original_image_inspected") or record.get("image_sha256") != raw_by_id[sid]["sha256"]:
            issues.append({"sample_id": sid, "issue": "missing reference inspection evidence"})
    public_hashes = {r["sha256"] for r in rows(META / "drone_type_set_quality.csv")}
    timestamp = now()
    status = {"recorded_at": timestamp, "candidate_images": len(candidates), "reference_only_images": len(references),
        "total_saved_images": len(acquired), "decoded_and_size_checked": decoded,
        "source_snapshots_checked": source_checks, "candidate_visual_evidence_checked": evidence_checks,
        "reference_visual_inspections_checked": len(reference_ids), "box_proposals_checked": len(proposals),
        "correlation_records_checked": len(groups), "problems": issues,
        "all_checked_files_and_decisions_match": not issues,
        "exact_file_duplicate_groups": sum(len(v) > 1 for v in files.values()),
        "exact_public_file_matches": sum(k in public_hashes for k in files),
        "current_decisions": dict(Counter(r["visual_decision"] for r in reviews)),
        "review_csv_sha256": digest(META / "drone_type_external_review.csv"),
        "collection_manifest_sha256": {relative(p): digest(p) for p in (META / "drone_type_external_raw.csv", META / "drone_type_article_raw.csv")},
        "software_tests_run": False, "model_training_or_evaluation_run": False,
        "source_independence_verified": False, "final_labels_and_boxes_verified": False,
        "limitations": "File/evidence inspection only; exact hashes miss crops, transforms and shared origins. Source independence and label truth are not proven."}
    archived = OUT / ("drone_type_external_integrity_" + timestamp.replace(":", "").replace("+", "_") + ".json")
    save_json(archived, status)
    save_json(OUT / "drone_type_external_integrity_current.json", {**status, "archived_audit": relative(archived)})
    print(f"External integrity: {decoded}/{len(acquired)} acquired images decoded; {evidence_checks} candidate records; {len(issues)} issues.")
    if issues:
        raise ValueError("External data integrity findings: see outputs/drone_type_external_integrity_current.json")
    return status
