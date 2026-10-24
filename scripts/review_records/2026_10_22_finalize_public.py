"""Archive already observed review decisions; does not inspect or classify new images.

The explicit sample-number rules below transcribe contact-sheet inspection and
44 native-pixel detail cards viewed on 2026-10-22. They are not automatic image
quality heuristics, source identities, or model predictions. Import the produced
CSV with scripts/03_review_data.py --review-file after reviewing this record.
"""
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from uav_project.common import META, OUT, digest, now, read_json, relative, resolve, rows, save_json, write_rows
from uav_project.review import image_number

POLICY = "visual-primary-cohort-v1-2026-10-22"
DEST = OUT / "review_batches" / "final_review_updates_2026-10-22.csv"


def record_review():
    if DEST.exists():
        raise SystemExit("Dated record already exists; preserve history. Re-import the saved CSV if necessary.")
    images = rows(META / "drone_type_set_images.csv")
    quality = {r["sample_id"]: r for r in rows(META / "drone_type_set_quality.csv")}
    previous = {r["sample_id"]: r for r in rows(META / "drone_type_set_review.csv")}
    details = {r["sample_id"]: r for r in rows(OUT / "review_batches/final_detail_2026-10-22/manifest.csv")}
    constraints = rows(META / "drone_type_set_reviewed_correlation_constraints.csv")
    constraint_fields = list(constraints[0])
    stamp = now()
    hashes = {}

    def hashed(path):
        if path not in hashes:
            hashes[path] = digest(resolve(path))
        return hashes[path]

    def span(a, b):
        return set(range(a, b + 1))

    rules = {
        0: [
            ({14}, "wrong_target_box", "Source box selects tree/brick background; the flying drone is left outside the box."),
            ({366}, "misaligned_body_crop", "Source box is shifted right and omits much of the central black fuselage; a red nose and partial body remain. Not an absent-drone claim."),
            ({16, 17, 242, 281, 282, 414, 415}, "crossfade", "Visible crossfade/double exposure superimposes scene content on or immediately around the target."),
            (span(19, 37) | span(295, 306), "postproduction_overlay", "Lower-third model/source graphics or residual animation bars intersect the candidate crop; exclude the contaminated sequence before modeling."),
        ],
        2: [
            ({139, 140, 141}, "exact_duplicate", "Exact copies of image 138, also in the excluded promotional title-card sequence; retain the prior duplicate finding."),
            (span(1, 18) | span(115, 138), "promotional_title_card", "Black-backdrop promotional title-card domain with model text in the source frame. Text lies outside the crop in some images; exclusion is a declared field-image cohort choice, not a claim of in-crop text in every image."),
            (span(80, 87) | span(463, 470) | span(564, 585), "rendered_target_graphics", "Rendered sensor cones/lines or white flight-path graphics intersect the target/candidate crop."),
            ({241, 242}, "body_out_of_frame", "Only motors/propellers/arms remain at the bottom edge; no identifiable central shell is present in the candidate crop."),
            ({280}, "misaligned_body_crop", "Small source box selects a left motor/hand/partial shell edge and omits most central body; not absence of the drone from the original frame."),
        ],
        5: [
            ({12, 13, 14}, "postproduction_overlay", "Postproduction 'Quadcopter courtesy of TomTop.com' credit overlaps the body/crop; this is a source-credit overlay, not a model-answer subtitle."),
            ({101}, "wrong_target_box", "Source box targets a detached accessory/grabber; the drone body is partly visible to the left outside the box."),
            (span(118, 122), "wrong_target_box", "Source box covers the red H landing pad while the drone body is outside the frame; image 119 had already been rejected."),
            (span(288, 291), "body_out_of_frame", "Only propellers/guards/arm roots remain at the lower boundary; central body is outside the frame."),
        ],
    }
    scene_basis = {
        "corr_bebop_sunny_park_digitaltrends": "Repeated sunny park/lawn/baseball-diamond setting, playground/fence and operator; noncontiguous numbers and resolutions revisit the scene. Conservatively constrained together, not verified as one original video.",
        "corr_phantom4_promotional_montage": "Repeated promotional montage sequences including noncontiguous numbering; montage shots are not independent source units.",
        "corr_phantom4_fisheye_grass_handheld": "Repeated fisheye grass setting and handheld drone views; conservatively constrained together.",
        "corr_phantom4_autumn_fence_yard": "Recurring autumn fence/yard scene across noncontiguous numbering; conservatively constrained together.",
        "corr_phantom4_pale_meadow": "Three adjacent pale-meadow views; conservatively constrained together. Independence from other groups is unverified.",
        "corr_mambo_sandy_field_handheld_red_pad": "Repeated sandy/dry field, handheld white/black Mambo, operator hands/watch and red H pad; hundreds of views are not hundreds of independent collections.",
    }

    def scene(class_id, n):
        if class_id == 0:
            return "corr_bebop_sunny_park_digitaltrends"
        if class_id == 5:
            return "corr_mambo_sandy_field_handheld_red_pad"
        if 142 <= n <= 445:
            return "corr_phantom4_fisheye_grass_handheld"
        if 446 <= n <= 460 or 474 <= n <= 514 or 647 <= n <= 656:
            return "corr_phantom4_autumn_fence_yard"
        if 515 <= n <= 517:
            return "corr_phantom4_pale_meadow"
        return "corr_phantom4_promotional_montage"

    def keep_detail(c, n):
        if c == 0:
            if n == 18:
                return "Transition/graphic bars are outside the candidate crop; visible body retained."
            if n in {65, 147, 148, 149, 411, 412}:
                return "Strong natural truncation but identifiable body remains; retained as a difficult candidate."
            if 390 <= n <= 395:
                return "Glare is visible but the target outline remains; retained as a difficult candidate."
        if c == 2:
            if n == 79:
                return "Red illumination but no obvious rendered sensor cone in the native crop; retained."
            if n == 88:
                return "Blue rings are below the legs/outside the candidate body region; body is visible. Not included in the cone-overlay exclusion."
            if n in {102, 210, 237, 239, 240, 335} or 322 <= n <= 336:
                return "Natural/partial truncation with some central white shell still visible; retained as a difficult candidate."
            if 536 <= n <= 546:
                return "Small low-contrast target over sea/ice has a coherent body/landing-gear outline; no crossfade asserted."
            if any(a <= n <= b for a, b in [(528,535),(547,556),(594,605),(612,621),(635,646)]):
                return "Small/blurred target outline is visible; retained as a difficult candidate."
        if c == 5:
            if n in {86, 88, 292}:
                return "Strong lower-edge truncation but a central white/black body portion remains; retained."
            if n in {578, 579}:
                return "Low exposure; no evident double-image crossfade established. Retained as a difficult candidate."
            if n == 225 or 230 <= n <= 239 or n in {258,259}:
                return "Source crop contains only part of the central body and omits the lower torso; retained as partial-body crop, not a complete-body view."
        return "Target body is visible in the candidate crop at review scale. Natural blur, small size, backlight and partial occlusion alone do not cause exclusion."

    updates, transitions, reason_records = [], [], []
    write_rows(OUT / "review_batches/review_before_finalization_2026-10-22.csv", previous.values())
    for c in (0, 2, 5):
        manifest = {r["sample_id"]: r for r in rows(OUT / f"review_batches/class_{c}/manifest.csv")}
        records = []
        for image in sorted((r for r in images if int(r["class_id"]) == c), key=image_number):
            sid, n = image["sample_id"], image_number(image)
            q, evidence = quality[sid], manifest[sid]
            hits = [(reason, note) for numbers, reason, note in rules[c] if n in numbers]
            if len(hits) > 1:
                raise ValueError("Overlapping primary exclusion reasons: " + sid)
            decision = "reject" if hits else "keep"
            reason, observation = hits[0] if hits else ("visual_candidate", keep_detail(c, n))
            group = scene(c, n)
            note = observation + " " + scene_basis[group] + " Author label retained, not independently authenticated. Visual keep is not training/source approval. Policy: " + POLICY
            flags = set(filter(None, q.get("flags", "").split("|")))
            flags.update({"reviewed_visual_correlation_constraint", reason})
            extras = []
            if sid in details:
                d = details[sid]
                extras.append({"path": d["evidence"], "sha256": d["evidence_sha256"], "kind": "native_crop_detail_card", "position": int(d["position"])})
                native_path = f"outputs/review_batches/final_detail_2026-10-22/class_{c}_{n:03d}_box_0.png"
                extras.append({"path": native_path, "sha256": hashed(native_path), "kind": "source_crop_pixels_not_generated"})
            records.append({
                "sample_id": sid, "path": image["path"], "image_number": n,
                "decision": decision, "label": c, "label_basis": "author_label_not_independently_verified",
                "reviewer": "Codex_AI_assisted_visual_review", "note": note,
                "flags": "|".join(sorted(flags)), "primary_reason": reason, "policy_version": POLICY,
                "evidence": evidence["sheet"], "evidence_position": int(evidence["position"]),
                "evidence_sha256": hashed(evidence["sheet"]), "image_sha256": q["sha256"],
                "annotation_sha256": q["annotation_sha256"], "individual_original_viewed": False,
                "native_detail_card_viewed": sid in details, "additional_evidence": extras,
                "constraint_group": group, "previous_decision": previous[sid]["decision"],
            })
            constraints.append({
                "sample_id": sid, "class_id": c, "constraint_group": group,
                "status": "reviewed_conservative_visual_correlation_constraint", "basis": scene_basis[group],
                "evidence": evidence["sheet"], "evidence_position": evidence["position"],
                "evidence_sha256": hashed(evidence["sheet"]), "image_sha256": q["sha256"],
                "reviewer": "Codex_AI_assisted_visual_review", "original_source_url": "", "flight_id": "",
                "source_identity_status": "unverified", "independent_from_other_groups": "unverified",
                "must_not_split_across_partitions": "true",
            })
        name = {0: "bebop", 2: "phantom4", 5: "mambo"}[c]
        save_json(OUT / f"drone_type_set_visual_review_{name}_2026-10-22.json", {
            "recorded_at": stamp, "class_id": c, "class_name": next(r["class_name"] for r in images if int(r["class_id"]) == c),
            "images_inspected": len(records), "evidence_pages_inspected": len({r["evidence"] for r in records}),
            "decision_counts": dict(Counter(r["decision"] for r in records)), "policy_version": POLICY,
            "reviewer": "Codex_AI_assisted_visual_review_not_two_independent_human_annotators",
            "source_identity_verified": False, "formal_cleaning_or_splits_created": False,
            "software_tests_run": False, "model_experiments_run": False, "records": records,
        })
        reason_records.extend(records)

    # Resolve prior observed problems by the same pretraining exclusion policy.
    # Historical records (including genuine original-image inspections) survive.
    prior_round = read_json(OUT / "drone_type_set_visual_review_round2.json")
    prior_yh = read_json(OUT / "drone_type_set_visual_review_yh_2026-10-22.json")
    id_to_image = {r["sample_id"]: r for r in images}
    resolved = []
    prior_records = {r["sample_id"]: r for r in prior_round["records"] + prior_yh["records"]}
    for old in prior_records.values():
        image = id_to_image[old["sample_id"]]
        c, n = int(image["class_id"]), image_number(image)
        if c not in {1, 6} or old["decision"] != "pending":
            continue
        record = dict(old)
        if c == 1:
            reason = "burned_in_detector_overlay" if n <= 93 else "inset_intersects_crop"
        elif n in {1, 39}:
            reason = "wrong_target_box"
        elif n in {73, 74}:
            reason = "crossfade"
        elif n == 210:
            reason = "unverifiable_tiny_target"
        else:
            reason = "inset_intersects_crop"
        observation = old.get("note", old.get("findings", ""))
        note = "Exclude from the primary cohort under " + POLICY + ": " + reason + ". Prior observed evidence retained: " + observation
        record.update({"decision": "reject", "label": c, "image_number": n,
                       "previous_decision": "pending", "primary_reason": reason,
                       "policy_version": POLICY, "note": note,
                       "decision_basis": "policy_resolution_of_previously_observed_problem_not_a_new_pixel_inspection"})
        resolved.append(record)
    if len(resolved) != 109:
        raise ValueError("Prior unresolved set changed; inspect before importing.")
    save_json(OUT / "drone_type_set_visual_review_policy_resolution_2026-10-22.json", {
        "recorded_at": stamp, "policy_version": POLICY, "records": resolved,
        "scope": "Policy resolution of 109 already inspected cases, not 109 newly inspected images.",
        "yh210_box_selection_remains_proposed": True, "source_identity_verified": False,
        "software_tests_run": False, "model_experiments_run": False,
    })
    reason_records.extend(resolved)
    for record in reason_records:
        sid = record["sample_id"]
        old = previous[sid]
        image = id_to_image[sid]
        updates.append({"sample_id": sid, "decision": record["decision"], "label": record["label"],
                        "class_name": image["class_name"], "group_id": old.get("group_id", ""), "note": record["note"]})
        transitions.append({"sample_id": sid, "path": image["path"], "class_id": image["class_id"],
                            "previous_decision": old["decision"], "new_decision": record["decision"],
                            "primary_reason": record["primary_reason"], "policy_version": POLICY,
                            "decision_changed": old["decision"] != record["decision"], "recorded_at": stamp})
    write_rows(META / "drone_type_set_reviewed_correlation_constraints.csv", constraints, constraint_fields)
    write_rows(DEST, updates, ["sample_id", "decision", "label", "class_name", "group_id", "note"])
    write_rows(OUT / "review_batches/final_review_transitions_2026-10-22.csv", transitions)
    save_json(OUT / "drone_type_set_visual_policy_2026-10-22.json", {
        "recorded_at": stamp, "version": POLICY, "defined_before_any_model_training_or_scores": True,
        "excluded": ["wrong/insufficient main-body crop", "target-content unverifiable on native pixels", "in-crop postproduction detector/inset/source/model graphics", "obvious crossfade", "promotional title-card domain", "exact duplicate excess"],
        "retained_when_body_visible": ["natural blur", "small target", "backlight/glare", "partial occlusion", "partial natural truncation"],
        "outside_crop_graphics_alone_cause_reject": False,
        "title_card_exception": "Promotional model-title cards excluded as a declared cohort-domain choice even if text is outside crop; not a text-in-crop assertion.",
        "raw_files_modified": False, "generative_repair_used": False,
        "keep_is_source_or_training_approval": False,
        "rejected_cases_may_be_reconsidered_only_with_new_evidence_and_new_version": True,
        "reviewer": "AI-assisted visual review; not independent human double annotation",
        "updated_rows": len(updates), "changed_decisions": sum(r["decision_changed"] for r in transitions),
        "constraint_images": len(constraints), "constraint_groups": len({r["constraint_group"] for r in constraints}),
        "source_independence_verified": False,
    })
    print(f"Saved {len(updates)} explicit updates, {len(resolved)} historical policy resolutions, {len(constraints)} correlation constraints.")
    print(DEST)


if __name__ == "__main__":
    record_review()
