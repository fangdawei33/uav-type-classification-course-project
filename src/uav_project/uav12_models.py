"""Four independent multiclass baselines, CV tuning and a single freeze gate."""
import importlib.metadata

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import ROOT, digest, read_json, save_json, now, resolve, relative
from .uav12_features import load_uav12_features
from .uav12_protocol import CONFIG, load_protocol, run_dir, settings


def _splits(protocol, development):
    result = []
    for fold in protocol["folds"]:
        validation_groups = set(fold["validation_group_ids"])
        val_mask = development.group_id.isin(validation_groups).to_numpy()
        train_idx, val_idx = np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
        if not len(train_idx) or not len(val_idx):
            raise ValueError(f"Empty group CV fold {fold['fold']}")
        train_groups = set(development.iloc[train_idx].group_id)
        val_groups = set(development.iloc[val_idx].group_id)
        if train_groups & val_groups or set(development.iloc[train_idx].label) != set(range(12)):
            raise ValueError(f"Group leak or missing training labels in fold {fold['fold']}")
        expected = set(protocol["development_cv_labels"])
        if not expected.issubset(set(development.iloc[val_idx].label)):
            raise ValueError(f"Evaluation labels absent from fold {fold['fold']}")
        result.append((train_idx, val_idx))
    return result


def _candidates(seed):
    def pipe(model, scale=True):
        steps = [("impute", SimpleImputer(strategy="median")),
                 ("variance", VarianceThreshold())]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("model", model)])

    return {
        "logistic": pipe(LogisticRegression(max_iter=3000, class_weight="balanced",
                                            random_state=seed)),
        "knn": pipe(KNeighborsClassifier(n_neighbors=5)),
        "svm": pipe(SVC(C=1, gamma="scale", class_weight="balanced",
                        decision_function_shape="ovr", random_state=seed)),
        "forest": pipe(RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                              n_jobs=1, random_state=seed), scale=False),
    }


def _training_signature(folder, protocol, representation):
    feature_path = folder / "features" / representation / "development.npz"
    feature_receipt_path = feature_path.with_suffix(".json")
    if not feature_path.is_file() or not feature_receipt_path.is_file():
        raise FileNotFoundError(f"Run 06d development feature extraction for {representation}")
    feature_receipt = read_json(feature_receipt_path)
    if digest(feature_path) != feature_receipt["feature_sha256"]:
        raise ValueError("Development feature cache changed")
    code_files = [ROOT / "src/uav_project/uav12_protocol.py",
                  ROOT / "src/uav_project/uav12_features.py",
                  ROOT / "src/uav_project/uav12_models.py",
                  ROOT / "src/uav_project/features.py"]
    return {"protocol_hash": protocol["manifest_sha256"], "config_hash": digest(CONFIG),
            "development_features_sha256": feature_receipt["feature_sha256"],
            "cv_folds_sha256": digest(folder / "cv_folds.json"),
            "code_hashes": {relative(path): digest(path) for path in code_files},
            "representation": representation,
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn")}


def train_uav12_baselines(representation, jobs=2):
    if representation not in settings()["representations"]:
        raise ValueError(f"Unknown representation: {representation}")
    folder, protocol, manifest = load_protocol()
    if (folder / "freeze.json").exists():
        raise RuntimeError("Experiment is frozen; training cannot be changed")
    X, feature_names, development = load_uav12_features(representation, "development")
    y = development.label.to_numpy(dtype=int)
    cv = _splits(protocol, development)
    signature = _training_signature(folder, protocol, representation)
    destination = folder / "models" / representation
    destination.mkdir(parents=True, exist_ok=True)
    candidates = _candidates(int(settings()["seed"]))
    labels = list(protocol["development_cv_labels"])
    scorer = make_scorer(f1_score, labels=labels, average="macro", zero_division=0)
    base_results = []
    for name, estimator in candidates.items():
        receipt_path = destination / f"{name}.json"
        model_path = destination / f"{name}.joblib"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["signature"] != signature or not model_path.exists() or digest(model_path) != receipt["model_sha256"]:
                raise ValueError(f"Existing {name} result differs from current inputs; use a new experiment version")
        else:
            scores = cross_val_score(estimator, X, y, cv=cv, scoring=scorer, n_jobs=jobs, error_score="raise")
            estimator.fit(X, y)
            joblib.dump(estimator, model_path, compress=3)
            train_report = classification_report(y, estimator.predict(X), labels=list(range(12)),
                target_names=protocol["classes"], output_dict=True, zero_division=0)
            receipt = {"model": name, "representation": representation, "signature": signature,
                "cv_metric": "macro_f1_on_group_supported_validation_classes_only",
                "cv_scored_labels": labels, "cv_scored_class_names": [protocol["classes"][i] for i in labels],
                "cv_scores": [float(value) for value in scores],
                "cv_macro_f1_mean": float(scores.mean()), "cv_macro_f1_std": float(scores.std()),
                "cv_folds": len(scores), "cv_unit": "conservative constraint group (not authenticated source)",
                "training_all_12_labels": True, "training_report": train_report,
                "model_path": relative(model_path), "model_sha256": digest(model_path), "trained_at": now()}
            save_json(receipt_path, receipt)
        base_results.append(receipt)
        print(f"{representation}/{name}: grouped-CV supported-label macro-F1="
              f"{receipt['cv_macro_f1_mean']:.4f} ± {receipt['cv_macro_f1_std']:.4f}", flush=True)

    if not (destination / "signature.json").exists():
        save_json(destination / "signature.json", signature)
    elif read_json(destination / "signature.json") != signature:
        raise ValueError("Training signature changed; preserve the existing run")
    selected = max(base_results, key=lambda item: item["cv_macro_f1_mean"])["model"]
    grids = {
        "logistic": {"model__C": [0.01, 0.1, 1.0, 10.0]},
        "knn": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
        "svm": {"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.001]},
        "forest": {"model__max_depth": [None, 20], "model__min_samples_leaf": [1, 3, 5]},
    }
    tuned_path = destination / "tuned.json"
    if tuned_path.exists():
        tuned = read_json(tuned_path)
        if (tuned["signature"] != signature or tuned["selected_family"] != selected or
                digest(resolve(tuned["model_path"])) != tuned["model_sha256"]):
            raise ValueError("Tuned classifier differs from current CV selection or data")
    else:
        grid = GridSearchCV(candidates[selected], grids[selected], cv=cv, scoring=scorer,
                            n_jobs=jobs, refit=True, error_score="raise", return_train_score=True)
        grid.fit(X, y)
        tuned_model_path = destination / "tuned.joblib"
        joblib.dump(grid.best_estimator_, tuned_model_path, compress=3)
        pd.DataFrame(grid.cv_results_).to_csv(destination / "grid_search.csv", index=False)
        train_report = classification_report(y, grid.predict(X), labels=list(range(12)),
            target_names=protocol["classes"], output_dict=True, zero_division=0)
        tuned = {"model": "tuned", "representation": representation, "selected_family": selected,
            "selection_rule": "highest mean supported-label macro-F1 on development grouped CV only",
            "signature": signature, "best_parameters": grid.best_params_,
            "cv_macro_f1_mean": float(grid.best_score_),
            "cv_macro_f1_std": float(grid.cv_results_["std_test_score"][grid.best_index_]),
            "cv_scored_labels": labels, "training_all_12_labels": True,
            "training_report": train_report, "model_path": relative(tuned_model_path),
            "model_sha256": digest(tuned_model_path), "tuned_at": now()}
        save_json(tuned_path, tuned)
        pd.DataFrame(grid.cv_results_).to_csv(destination / "grid_search.csv", index=False)
    forest = joblib.load(destination / "forest.joblib")
    support = forest.named_steps["variance"].get_support()
    importance = pd.DataFrame({"feature": np.asarray(feature_names)[support],
                               "impurity_importance": forest.named_steps["model"].feature_importances_})
    importance = importance.sort_values("impurity_importance", ascending=False)
    importance.to_csv(destination / "forest_importance.csv", index=False, encoding="utf-8-sig")
    save_json(destination / "summary.json", {"signature": signature, "baselines": base_results,
        "selected_family": selected, "tuned": tuned, "holdout_read": False,
        "cv_classes": [protocol["classes"][i] for i in labels],
        "limitations": protocol["limitations"],
        "importance_caveat": "Impurity importance is model-specific, affected by correlated features and is not causal."})
    print(f"Finished {representation} baselines and CV tuning; holdout remains unopened.", flush=True)


def _expected_uav12_artifacts():
    folder, protocol, _ = load_protocol()
    cfg = settings()
    paths = [CONFIG, folder / "manifest.csv", folder / "protocol.json", folder / "cv_folds.json",
             ROOT / "src/uav_project/uav12_protocol.py", ROOT / "src/uav_project/uav12_features.py",
             ROOT / "src/uav_project/uav12_models.py", ROOT / "src/uav_project/uav12_qwen.py",
             ROOT / "src/uav_project/uav12_evaluation.py"]
    for script in ["05d_prepare_uav12_protocol.py", "06d_extract_uav12_features.py",
                   "07d_train_uav12_baselines.py", "07q_train_qwen3vl.py",
                   "09d_freeze_uav12.py", "10d_evaluate_uav12.py", "11d_export_uav12_report.py"]:
        paths.append(ROOT / "scripts" / script)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("Missing declared source/protocol artifact: " + str(path))
    paths.extend([ROOT / "src/uav_project/features.py", ROOT / "src/uav_project/common.py",
                  ROOT / "requirements-uav12.txt", ROOT / "requirements-uav12-qwen.txt"])
    result = {relative(path): digest(path) for path in paths}
    for representation in cfg["representations"]:
        destination = folder / "models" / representation
        for name in cfg["classifiers"] + ["tuned"]:
            receipt_path = destination / f"{name}.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"Missing baseline receipt: {receipt_path}")
            receipt = read_json(receipt_path)
            model_path = resolve(receipt["model_path"])
            if not model_path.is_file() or digest(model_path) != receipt["model_sha256"]:
                raise ValueError(f"Model hash mismatch: {model_path}")
            result[relative(receipt_path)] = digest(receipt_path)
            result[relative(model_path)] = digest(model_path)
        for name in ["signature.json", "summary.json", "grid_search.csv", "forest_importance.csv"]:
            path = destination / name
            if path.is_file():
                result[relative(path)] = digest(path)
        feature_dir = folder / "features" / representation
        for name in ["development.npz", "development.json", "model_pin.json"]:
            path = feature_dir / name
            if path.is_file():
                result[relative(path)] = digest(path)
    qwen_dir = folder / "models" / "qwen3_vl"
    result[relative(qwen_dir / "model_pin.json")] = digest(qwen_dir / "model_pin.json")
    for variant in cfg["qwen3_vl"]["variants"]:
        receipt_path = qwen_dir / variant / "training_receipt.json"
        if not receipt_path.is_file():
            raise FileNotFoundError(f"Missing Qwen LoRA training receipt: {receipt_path}")
        receipt = read_json(receipt_path)
        adapter_dir = resolve(receipt["adapter_directory"])
        if not adapter_dir.is_dir():
            raise FileNotFoundError(adapter_dir)
        result[relative(receipt_path)] = digest(receipt_path)
        for item in receipt["adapter_files"]:
            path = resolve(item["path"])
            if digest(path) != item["sha256"]:
                raise ValueError(f"Qwen adapter file hash mismatch: {path}")
            result[relative(path)] = digest(path)
        for name in ["model_pin.json", "training_config.json", "training_history.csv"]:
            path = qwen_dir / variant / name
            if path.is_file():
                result[relative(path)] = digest(path)
        for path in sorted((qwen_dir / variant / "processor").rglob("*")):
            if path.is_file():
                result[relative(path)] = digest(path)
    return folder, protocol, result


def freeze_uav12():
    folder = run_dir()
    freeze_path = folder / "freeze.json"
    if freeze_path.exists():
        verify_uav12_freeze()
        print("Existing global freeze verified; holdout may be processed.", flush=True)
        return
    folder, protocol, artifacts = _expected_uav12_artifacts()
    summaries = [read_json(folder / "models" / rep / "summary.json") for rep in settings()["representations"]]
    traditional = max((summary["tuned"] for summary in summaries), key=lambda item: item["cv_macro_f1_mean"])
    freeze = {"frozen_at": now(), "protocol_hash": protocol["manifest_sha256"],
        "config_hash": digest(CONFIG), "artifact_sha256": artifacts,
        "primary_model": {"family": "qwen3_vl_lora_sft", "variant": "standard",
                           "selection": "predeclared main method; no holdout-driven selection"},
        "traditional_reference": {"representation": traditional["representation"],
            "classifier": traditional["model"], "development_cv_macro_f1": traditional["cv_macro_f1_mean"]},
        "evaluated_models_predeclared": ["qwen3_vl_zero_shot", "qwen3_vl_lora_standard",
            "qwen3_vl_lora_sqrt_class_balanced", "handcrafted_4_baselines_plus_tuned",
            "siglip2_4_baselines_plus_tuned"],
        "holdout_read": False, "independent_source_evaluation": False,
        "holdout_coverage": protocol["holdout_class_names"],
        "limitations": protocol["limitations"]}
    save_json(freeze_path, freeze)
    print("Global 12-label experiment frozen:", freeze_path, flush=True)
    print("The holdout remains sealed until all declared model variants are trained.", flush=True)


def verify_uav12_freeze():
    folder = run_dir()
    freeze_path = folder / "freeze.json"
    if not freeze_path.is_file():
        raise RuntimeError("Run all development-only models, then scripts/09d_freeze_uav12.py")
    frozen = read_json(freeze_path)
    _, protocol, current = _expected_uav12_artifacts()
    if frozen["protocol_hash"] != protocol["manifest_sha256"] or frozen["config_hash"] != digest(CONFIG):
        raise ValueError("Frozen protocol or configuration changed")
    if frozen["artifact_sha256"] != current:
        changed = sorted(set(frozen["artifact_sha256"]) ^ set(current) |
                         {key for key in set(current) & set(frozen["artifact_sha256"])
                          if current[key] != frozen["artifact_sha256"][key]})
        raise ValueError("Frozen model/code artifacts changed: " + ", ".join(changed[:10]))
    return frozen
