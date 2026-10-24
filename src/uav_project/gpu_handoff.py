"""Build an explicit, credential-free file list for the T4 representation run.

This is a transfer manifest, not a monolithic runner or archive. Existing image
identity checks are referenced; image bytes are not redundantly rehashed here.
"""
import csv
from pathlib import Path
from .common import ROOT, META, OUT, rows, read_json, digest, relative, resolve, save_json, write_rows, now
from .representation_protocol import load_protocol, SETTINGS


def prepare_gpu_handoff(include_candidates=True):
    folder, protocol, frame = load_protocol("auxiliary")
    destination = OUT / "gpu_handoff" / ("t4_representation_v1" if include_candidates else "t4_auxiliary_only_v1")
    destination.mkdir(parents=True, exist_ok=True)
    records = {}

    def add(path, role, expected=None):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        key = relative(path)
        record = {"path": key, "bytes": path.stat().st_size, "role": role,
                  "sha256": expected or digest(path),
                  "hash_basis": "prior_acquisition_or_processing_receipt_not_recomputed_in_handoff" if expected else "computed_for_handoff"}
        if key in records and records[key]["sha256"] != record["sha256"]:
            raise ValueError("Contradictory transfer identity: " + key)
        records[key] = record

    scripts = ["_bootstrap.py", "05b_prepare_representation.py", "06b_extract_representation.py",
               "07b_train_representation.py", "09b_freeze_representation.py",
               "10b_evaluate_representation.py", "11b_export_representation.py"]
    modules = ["__init__.py", "common.py", "representation_protocol.py", "representation_features.py",
               "representation_training.py", "representation_evaluation.py", "features.py", "splitting.py", "colab_support.py"]
    for name in scripts:
        add(ROOT / "scripts" / name, "modular_python_entrypoint")
    for name in modules:
        add(ROOT / "src/uav_project" / name, "python_module")
    for path in [ROOT / "configs/experiment.json", SETTINGS, ROOT / "requirements-representation.txt",
                 ROOT / "docs/NEXT_STEPS.md"]:
        add(path, "configuration_or_instructions")
    for path, expected in protocol["input_hashes"].items():
        add(resolve(path.replace("\\", "/")), "auxiliary_source_metadata", expected)
    for name in ["manifest.csv", "protocol.json", "cv_folds.json"]:
        add(folder / name, "prepared_auxiliary_protocol")
    for row in frame.itertuples():
        add(resolve(row.clean_path), "auxiliary_development_image" if row.split == "development" else "auxiliary_holdout_image", row.clean_sha256)
    if include_candidates:
        candidate_path = META / "drone_type_set_candidate_crops.csv"
        add(candidate_path, "unapproved_seven_class_candidate_manifest")
        for row in rows(candidate_path):
            if row["training_approved"].strip().lower() != "false":
                raise ValueError("Unexpected candidate approval state")
            add(resolve(row["clean_path"]), "candidate_crop_cache_only_not_training", row["clean_sha256"])
    ordered = [records[key] for key in sorted(records)]
    write_rows(destination / "transfer_manifest.csv", ordered,
               ["path", "bytes", "role", "sha256", "hash_basis"])
    (destination / "files_to_transfer.txt").write_text("\n".join(r["path"] for r in ordered) + "\n", encoding="utf-8")
    summary = {"created_at": now(), "target_gpu": "Colab auto-detected GPU; A100 40GB observed, T4 compatible", "batch_size": 16, "inference_precision": "FP16",
               "included_candidate_crops": include_candidates, "files": len(ordered),
               "bytes": sum(row["bytes"] for row in ordered),
               "role_counts": {role: sum(r["role"] == role for r in ordered) for role in sorted({r["role"] for r in ordered})},
               "protocol_manifest_sha256": protocol["manifest_hash"],
               "files_list_sha256": digest(destination / "files_to_transfer.txt"),
               "transfer_executed": False, "models_run": False, "software_tests_run": False,
               "hash_caveat": "Image SHA256 references existing acquisition/processing checks; this step checks presence and sizes only. GPU extraction verifies selected input bytes before inference.",
               "local_venv_or_tokens_included": False}
    save_json(destination / "summary.json", summary)
    print("T4 transfer file list:", destination / "files_to_transfer.txt", flush=True)
    print("Files:", len(ordered), "Bytes:", summary["bytes"], "Candidate crops:", include_candidates, flush=True)
    return summary


