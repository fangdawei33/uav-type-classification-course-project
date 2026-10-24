"""Fixed, group-aware 12-label diagnostic protocol for the v3 crop release."""
from collections import Counter, defaultdict
import json
import random

import pandas as pd

from .common import ROOT, digest, read_json, save_json, now, resolve

CONFIG = ROOT / "configs/uav12_experiment.json"


def settings():
    return read_json(CONFIG)


def run_dir():
    return resolve(settings()["run_directory"])


def _assign_development_folds(frame, folds, seed):
    """Place whole constraint groups into folds; singleton groups stay train-only."""
    rng = random.Random(seed)
    fold_counts = [Counter() for _ in range(folds)]
    assignment = {}
    group_rows = defaultdict(list)
    for row in frame.to_dict("records"):
        group_rows[row["group_id"]].append(row)
    group_label = {}
    for group_id, members in group_rows.items():
        labels = {int(member["label"]) for member in members}
        if len(labels) != 1:
            raise ValueError(f"Constraint group has multiple labels: {group_id}")
        group_label[group_id] = labels.pop()
    for label in sorted(set(group_label.values())):
        groups = [group for group, value in group_label.items() if value == label]
        if len(groups) < 2:
            for group in groups:
                assignment[group] = None
            continue
        rng.shuffle(groups)
        groups.sort(key=lambda group: len(group_rows[group]), reverse=True)
        for group in groups:
            fold = min(range(folds), key=lambda index: (fold_counts[index][label], index))
            assignment[group] = fold
            fold_counts[fold][label] += len(group_rows[group])
    return assignment


def prepare_protocol(refresh=False):
    """Promote the already audited draft to a versioned diagnostic experiment."""
    cfg = settings()
    destination = run_dir()
    if (destination / "protocol.json").exists():
        if not refresh:
            raise FileExistsError(f"Protocol exists; preserving it: {destination}")
        protected = [destination / "features", destination / "models", destination / "freeze.json",
                     destination / "evaluation"]
        if any(path.exists() for path in protected):
            raise RuntimeError("Cannot refresh protocol after feature/model artifacts have been created")
    release = resolve(cfg["dataset_release"])
    draft_dir = resolve(cfg["split_draft"])
    clean_path = release / "clean_manifest.csv"
    class_map_path = release / "class_map.json"
    draft_path = draft_dir / "manifest.csv"
    draft_receipt_path = draft_dir / "receipt.json"
    receipt_path = release / "receipt.json"
    for path in [clean_path, class_map_path, draft_path, draft_receipt_path, receipt_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
    release_receipt = read_json(receipt_path)
    draft_receipt = read_json(draft_receipt_path)
    if digest(clean_path) != release_receipt["clean_manifest_sha256"]:
        raise ValueError("Clean data manifest differs from the frozen v3 release")
    if not release_receipt["clean_manifest_matches_frozen_v3"]:
        raise ValueError("Delivery manifest is not bound to the frozen v3 dataset")
    if digest(draft_path) != draft_receipt["draft_manifest_sha256"]:
        raise ValueError("Split feasibility draft changed after it was recorded")
    clean = pd.read_csv(clean_path, dtype={"sample_id": str, "group_id": str}, low_memory=False).fillna("")
    draft = pd.read_csv(draft_path, dtype={"sample_id": str, "group_id": str}, low_memory=False).fillna("")
    class_map = read_json(class_map_path)["classes"]
    names = [item["name"] for item in sorted(class_map, key=lambda item: int(item["label"]))]
    if len(names) != 12 or len(clean) != 20010 or clean.sample_id.duplicated().any():
        raise ValueError("Unexpected v3 sample count, class count or repeated sample id")
    if len(draft) != len(clean) or draft.sample_id.duplicated().any():
        raise ValueError("Split draft does not map one-to-one to the clean dataset")
    if set(draft.sample_id) != set(clean.sample_id):
        raise ValueError("Split draft sample ids differ from the released manifest")
    split_by_id = draft.set_index("sample_id").draft_split.to_dict()
    if set(split_by_id.values()) != {"development", "holdout"}:
        raise ValueError("Unexpected split names")
    expected = clean.set_index("sample_id")
    draft_indexed = draft.set_index("sample_id")
    for column in ["label", "class_name", "group_id"]:
        if not expected[column].astype(str).equals(draft_indexed[column].reindex(expected.index).astype(str)):
            raise ValueError(f"Split draft changed the clean data field: {column}")
    clean["split"] = clean.sample_id.map(split_by_id)
    clean["clean_path"] = clean.image_path
    clean["clean_sha256"] = clean.sha256
    clean["label"] = clean.label.astype(int)
    clean["group_id"] = clean.group_id.astype(str)
    clean = clean.sort_values(["split", "label", "sample_id"]).reset_index(drop=True)
    if set(clean.label) != set(range(12)):
        raise ValueError("Class ids must be the frozen contiguous range 0..11")
    if clean.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("A conservative constraint group crosses development/holdout")
    class_lists = {split: sorted(map(int, part.label.unique())) for split, part in clean.groupby("split")}
    holdout_labels = class_lists["holdout"]
    if len(holdout_labels) != int(cfg["holdout"]["expected_class_coverage"]):
        raise ValueError(f"Expected partial holdout coverage changed: {holdout_labels}")
    development = clean[clean.split.eq("development")].reset_index(drop=True)
    fold_assignment = _assign_development_folds(development, int(cfg["cv_folds"]), int(cfg["seed"]))
    class_group_counts = development.groupby("label").group_id.nunique().to_dict()
    cv_labels = sorted(int(label) for label, count in class_group_counts.items() if count >= 2)
    cv_splits = []
    for fold in range(int(cfg["cv_folds"])):
        validation_mask = development.group_id.map(fold_assignment).eq(fold)
        train_ids = [int(index) for index in development.index[~validation_mask]]
        validation_ids = [int(index) for index in development.index[validation_mask]]
        train_labels = set(development.iloc[train_ids].label)
        validation_labels = set(development.iloc[validation_ids].label)
        if train_labels != set(range(12)) or not set(cv_labels).issubset(validation_labels):
            raise ValueError(f"Group fold {fold} lacks required training/validation labels")
        train_groups = set(development.iloc[train_ids].group_id)
        validation_groups = set(development.iloc[validation_ids].group_id)
        if train_groups & validation_groups:
            raise ValueError(f"Constraint group leak in fold {fold}")
        train_group_ids = sorted(group for group, assigned in fold_assignment.items()
                                 if assigned is None or assigned != fold)
        validation_group_ids = sorted(group for group, assigned in fold_assignment.items()
                                      if assigned == fold)
        cv_splits.append({"fold": fold, "train_group_ids": train_group_ids,
                          "validation_group_ids": validation_group_ids,
                          "train_samples": len(train_ids), "validation_samples": len(validation_ids),
                          "validation_labels": sorted(map(int, validation_labels)),
                          "validation_classes_scored": cv_labels,
                          "train_only_labels": sorted(set(range(12)) - set(cv_labels))})
    # log_stage creates the run directory before calling this function so that
    # failures are recorded. Reuse that directory only when no protocol has
    # been written yet; the guard at the top still prevents overwriting a
    # completed experiment.
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.csv"
    clean.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    counts = clean.groupby(["split", "label", "class_name"], dropna=False).size().reset_index(name="images")
    counts.to_csv(destination / "class_split_counts.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "created_at": now(), "version": cfg["version"], "dataset": "uav_type_v3",
        "classes": names, "label_ids": list(range(12)), "seed": cfg["seed"],
        "holdout_rule": "Precomputed v3 constraint-group draft nearest to 30%; labels/counts only; no model scores.",
        "counts": {name: int(count) for name, count in clean.split.value_counts().items()},
        "holdout_fraction": float(clean.split.eq("holdout").mean()),
        "holdout_labels": holdout_labels, "holdout_class_names": [names[index] for index in holdout_labels],
        "development_cv_kind": "two_fold_label_stratified_constraint_group_cv",
        "development_cv_labels": cv_labels,
        "development_cv_class_names": [names[index] for index in cv_labels],
        "development_train_only_labels": sorted(set(range(12)) - set(cv_labels)),
        "development_train_only_class_names": [names[index] for index in sorted(set(range(12)) - set(cv_labels))],
        "folds": cv_splits,
        "independent_source_groups_authenticated": False,
        "all_12_class_holdout_supported": False,
        "task": "classification of author-box target crops; no end-to-end detection",
        "input_hashes": {"configs/uav12_experiment.json": digest(CONFIG),
                         "data/delivery/uav_dataset_v3/clean_manifest.csv": digest(clean_path),
                         "data/delivery/uav_dataset_v3/class_map.json": digest(class_map_path),
                         "data/delivery/uav_dataset_v3/receipt.json": digest(receipt_path),
                         "outputs/expansion/processing_v3/split_draft/manifest.csv": digest(draft_path),
                         "outputs/expansion/processing_v3/split_draft/receipt.json": digest(draft_receipt_path)},
        "limitations": cfg["limitations"],
        "models_or_software_tests_run": False,
    }
    protocol["manifest_sha256"] = digest(manifest_path)
    save_json(destination / "protocol.json", protocol)
    save_json(destination / "cv_folds.json", {"manifest_sha256": protocol["manifest_sha256"],
                                                "folds": cv_splits})
    print(f"Protocol created: {destination}")
    print(f"Development / holdout: {protocol['counts']}; holdout coverage: {len(holdout_labels)}/12 labels")
    print(f"Group CV scores {len(cv_labels)}/12 labels; all 12 remain in every model's training folds.")
    print("No model fitting or evaluation was run.")


def load_protocol(verify_inputs=True):
    folder = run_dir()
    protocol_path = folder / "protocol.json"
    manifest_path = folder / "manifest.csv"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run scripts/05d_prepare_uav12_protocol.py first")
    protocol = read_json(protocol_path)
    if digest(CONFIG) != protocol["input_hashes"]["configs/uav12_experiment.json"]:
        raise ValueError("Experiment configuration changed after protocol creation")
    if digest(manifest_path) != protocol["manifest_sha256"]:
        raise ValueError("Experiment manifest changed after protocol creation")
    if verify_inputs:
        for relative_path, expected_hash in protocol["input_hashes"].items():
            if digest(resolve(relative_path)) != expected_hash:
                raise ValueError(f"Protocol input changed: {relative_path}")
    frame = pd.read_csv(manifest_path, dtype={"sample_id": str, "group_id": str}, low_memory=False).fillna("")
    if len(frame) != 20010 or frame.sample_id.duplicated().any():
        raise ValueError("Experiment manifest sample inventory changed")
    if frame.groupby("group_id").split.nunique().max() != 1:
        raise ValueError("A constraint group crosses development and holdout")
    fold_record = read_json(folder / "cv_folds.json")
    if (fold_record["manifest_sha256"] != protocol["manifest_sha256"] or
            fold_record["folds"] != protocol["folds"]):
        raise ValueError("Recorded CV folds differ from the fixed protocol")
    return folder, protocol, frame


def log_stage(name, action):
    folder = run_dir()
    log_dir = folder / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "stages.jsonl"
    started = {"time": now(), "stage": name, "status": "started"}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(started, ensure_ascii=False) + "\n")
    try:
        result = action()
    except Exception as exc:
        import traceback
        record = {"time": now(), "stage": name, "status": "failed", "error": repr(exc),
                  "traceback": traceback.format_exc()}
        with (log_dir / "errors.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        raise
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), "stage": name, "status": "completed"},
                                ensure_ascii=False) + "\n")
    return result
