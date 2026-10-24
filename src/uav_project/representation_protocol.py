"""Dataset protocols for the active multiclass representation experiment.

The auxiliary diagnostic preserves the source train/val division. Its image-level
CV does NOT establish render/source independence. Seven-class work requires the
formal cleaned/source-approved data; visual candidates cannot bypass that gate.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from .common import ROOT, META, OUT, digest, resolve, rows, read_json, save_json, now

SETTINGS = ROOT / "configs/representation_experiment.json"


def run_dir(dataset):
    if dataset not in {"auxiliary", "seven"}:
        raise ValueError("Unknown dataset scope")
    return OUT / "representation_v1" / dataset


def _validate_parts(frame, classes, grouped):
    required = {"sample_id", "clean_path", "clean_sha256", "label", "split"}
    if not required.issubset(frame.columns) or frame.empty or frame.sample_id.duplicated().any():
        raise ValueError("Invalid or duplicated sample manifest")
    if set(frame.split) != {"development", "holdout"}:
        raise ValueError("Protocol must contain development and holdout")
    labels = set(range(len(classes)))
    for split, part in frame.groupby("split"):
        if set(part.label.astype(int)) != labels:
            raise ValueError("Missing class in " + split)
    if frame.groupby("clean_sha256").split.nunique().max() > 1:
        raise ValueError("Exact file duplicate crosses partitions")
    if "pixel_sha256" in frame and frame.groupby("pixel_sha256").split.nunique().max() > 1:
        raise ValueError("Exact pixel duplicate crosses partitions")
    if grouped and (frame.group_id.eq("").any() or frame.groupby("group_id").split.nunique().max() > 1):
        raise ValueError("Missing group or group crosses partitions")


def cv_indices(frame, protocol):
    cfg = read_json(SETTINGS)
    if protocol["cv_kind"] == "stratified_group":
        splitter = StratifiedGroupKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
        folds = list(splitter.split(frame, frame.label, frame.group_id))
    else:
        splitter = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=cfg["seed"])
        folds = list(splitter.split(frame, frame.label))
    expected = set(range(len(protocol["classes"])))
    for train, val in folds:
        if set(frame.iloc[train].label.astype(int)) != expected or set(frame.iloc[val].label.astype(int)) != expected:
            raise ValueError("CV fold missing a class; cannot report comparable macro-F1")
        if protocol["cv_kind"] == "stratified_group" and set(frame.iloc[train].group_id) & set(frame.iloc[val].group_id):
            raise ValueError("Group crosses CV fold")
    return folds


def prepare_protocol(dataset):
    cfg = read_json(SETTINGS)
    folder = run_dir(dataset)
    if (folder / "freeze.json").exists():
        raise RuntimeError("Frozen experiment cannot be rewritten")
    inputs = {}
    if dataset == "auxiliary":
        label_path = META / "cranfield_synthetic_classification_auxiliary_classes.json"
        mapping = read_json(label_path)["source_labels"]
        classes = [k for k, v in sorted(mapping.items(), key=lambda item: item[1])]
        inputs[label_path.relative_to(ROOT).as_posix()] = digest(label_path)
        parts = []
        for name, split in [("train", "development"), ("val", "holdout")]:
            path = META / f"cranfield_synthetic_classification_auxiliary_{name}.csv"
            part = pd.read_csv(path).fillna("")
            if not part.seven_class_label.eq("").all():
                raise ValueError("Auxiliary labels must remain separate from seven-class labels")
            part["label"] = part.source_label.map(mapping)
            part["class_name"] = part.source_label
            part["clean_path"] = part.path
            part["clean_sha256"] = part.sha256
            part["split"] = split
            part["group_id"] = ""  # Unknown parent IDs are not invented.
            parts.append(part)
            inputs[path.relative_to(ROOT).as_posix()] = digest(path)
        frame = pd.concat(parts, ignore_index=True)
        cv_kind = "stratified_image_diagnostic"
        limitation = cfg["auxiliary"]["independence"]
    else:
        path = META / "drone_type_set_clean_manifest.csv"
        if not path.exists():
            raise RuntimeError("七类正式来源审核未通过：缺少drone_type_set_clean_manifest.csv。候选裁剪不是训练批准。")
        frame = pd.read_csv(path).fillna("")
        classes = read_json(META / "drone_type_set_classes.json")["classes"]
        sources = {r["sample_id"]: r for r in rows(META / "drone_type_set_source_groups.csv")}
        constraints_path = META / "drone_type_set_reviewed_correlation_constraints.csv"
        for row in frame.itertuples():
            source = sources.get(row.sample_id, {})
            if source.get("status") != "confirmed" or source.get("source_group") != row.group_id or not source.get("source_evidence"):
                raise ValueError("Unconfirmed source: " + row.sample_id)
        minimum = int(frame.groupby("label").group_id.nunique().min())
        if minimum < cfg["cv_folds"] + 1:
            raise ValueError("Not enough groups per class for holdout plus grouped CV")
        k = min(10, minimum)
        splits = StratifiedGroupKFold(k, shuffle=True, random_state=cfg["seed"])
        assignment = np.zeros(len(frame), dtype=int)
        for i, (_, ids) in enumerate(splits.split(frame, frame.label, frame.group_id)):
            assignment[ids] = i
        target = cfg["public_test_fraction"]
        best = None
        for count in sorted({max(1, int(k * target)), min(k - 1, int(np.ceil(k * target)))}):
            for selected in combinations(range(k), count):
                mask = np.isin(assignment, selected)
                if frame.loc[mask, "label"].nunique() != len(classes) or frame.loc[~mask, "label"].nunique() != len(classes):
                    continue
                try:
                    cv_indices(frame.loc[~mask].reset_index(drop=True), {"classes": classes, "cv_kind": "stratified_group"})
                except ValueError:
                    continue
                proportions = frame.groupby("label").size()
                held = frame.loc[mask].groupby("label").size().reindex(proportions.index, fill_value=0)
                penalty = float(np.abs(held / proportions - target).mean())
                if best is None or penalty < best[0]:
                    best = penalty, mask
        if best is None:
            raise ValueError("No feasible grouped holdout with complete grouped CV")
        frame["split"] = np.where(best[1], "holdout", "development")
        membership = frame.set_index("sample_id").split.to_dict()
        correlation_parts = {}
        for row in rows(constraints_path):
            if row["sample_id"] in membership:
                correlation_parts.setdefault(row["constraint_group"], set()).add(membership[row["sample_id"]])
        if any(len(parts) > 1 for parts in correlation_parts.values()):
            raise ValueError("Conservative correlated scenes cross holdout; repair source group unions")
        # Confirm that the same conservative constraints also stay within CV folds.
        dev = frame[frame.split.eq("development")].reset_index(drop=True)
        corr = {r["sample_id"]: r["constraint_group"] for r in rows(constraints_path)}
        for train, val in cv_indices(dev, {"classes": classes, "cv_kind": "stratified_group"}):
            left = {corr[s] for s in dev.iloc[train].sample_id if s in corr}
            right = {corr[s] for s in dev.iloc[val].sample_id if s in corr}
            if left & right:
                raise ValueError("Conservative correlated scenes cross CV folds")
        cv_kind = "stratified_group"
        limitation = "Given author target regions; source-isolated internal evaluation, not end-to-end detection. External real test is a separate unapproved dataset."
        for source_path in [path, META / "drone_type_set_source_groups.csv", constraints_path]:
            inputs[source_path.relative_to(ROOT).as_posix()] = digest(source_path)
    frame = frame.sort_values(["split", "sample_id"]).reset_index(drop=True)
    _validate_parts(frame, classes, cv_kind == "stratified_group")
    protocol = {"created_at": now(), "dataset": dataset, "classes": classes, "cv_kind": cv_kind,
                "limitations": limitation, "input_hashes": inputs, "config_hash": digest(SETTINGS),
                "seed": cfg["seed"], "main_seven_class_result": dataset == "seven",
                "counts": frame.groupby(["split", "class_name"]).size().to_dict()}
    protocol["counts"] = {str(k): int(v) for k, v in protocol["counts"].items()}
    dev = frame[frame.split.eq("development")].reset_index(drop=True)
    folds = cv_indices(dev, protocol)
    # Manifest preparation checks file identity only; no model fitting or scores.
    for row in frame.itertuples():
        if digest(resolve(row.clean_path)) != row.clean_sha256:
            raise ValueError("Input image changed: " + row.sample_id)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "manifest.csv"
    if manifest.exists():
        old = pd.read_csv(manifest).fillna("")
        if old.astype(str).to_dict("records") != frame.astype(str).to_dict("records"):
            raise RuntimeError("Existing protocol differs; use a new version directory")
    frame.to_csv(manifest, index=False, encoding="utf-8-sig")
    protocol["manifest_hash"] = digest(manifest)
    save_json(folder / "protocol.json", protocol)
    save_json(folder / "cv_folds.json", {"development_sample_ids": dev.sample_id.tolist(),
              "folds": [{"train": a.tolist(), "validation": b.tolist()} for a, b in folds]})
    print("Protocol prepared:", folder, frame.groupby(["split", "class_name"]).size().to_dict())
    print("Limitation:", limitation)


def load_protocol(dataset):
    folder = run_dir(dataset)
    protocol = read_json(folder / "protocol.json")
    if protocol["config_hash"] != digest(SETTINGS) or protocol["manifest_hash"] != digest(folder / "manifest.csv"):
        raise ValueError("Protocol changed after preparation")
    for path, expected in protocol["input_hashes"].items():
        if digest(resolve(path.replace("\\", "/"))) != expected:
            raise ValueError("Protocol source manifest changed: " + path)
    frame = pd.read_csv(folder / "manifest.csv").fillna("")
    _validate_parts(frame, protocol["classes"], protocol["cv_kind"] == "stratified_group")
    return folder, protocol, frame

