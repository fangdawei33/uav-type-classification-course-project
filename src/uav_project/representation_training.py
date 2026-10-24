"""Four independent multiclass classifiers, train-only CV tuning, artifact freeze."""
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from .common import ROOT, digest, read_json, save_json, now, resolve, relative
from .representation_protocol import SETTINGS, load_protocol, cv_indices
from .representation_features import load_features


def multiclass_metrics(y, pred, classes):
    labels = list(range(len(classes)))
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
            "per_class": classification_report(y, pred, labels=labels, target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}


def candidates(seed):
    def pipeline(model, scale=True):
        steps = [("impute", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps + [("model", model)])
    return {
        "logistic": pipeline(LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
        "knn": pipeline(KNeighborsClassifier(n_neighbors=5)),
        "svm": pipeline(SVC(C=1, gamma="scale", class_weight="balanced", random_state=seed)),
        "forest": pipeline(RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=1, random_state=seed), False)}


def train_classifiers(dataset, representation, jobs=2):
    folder, protocol, frame = load_protocol(dataset)
    if (folder / "freeze.json").exists():
        raise RuntimeError("Experiment frozen; training cannot overwrite artifacts")
    cfg = read_json(SETTINGS)
    X, names, dev = load_features(dataset, representation, "development")
    y = dev.label.to_numpy(dtype=int)
    cv = cv_indices(dev, protocol)
    recorded_cv = read_json(folder / "cv_folds.json")
    if recorded_cv["development_sample_ids"] != dev.sample_id.tolist() or recorded_cv["folds"] != [
            {"train": a.tolist(), "validation": b.tolist()} for a, b in cv]:
        raise ValueError("Recorded CV folds changed")
    destination = folder / "models" / representation
    destination.mkdir(parents=True, exist_ok=True)
    signature = {"protocol_hash": digest(folder / "protocol.json"), "config_hash": digest(SETTINGS),
                 "feature_hash": digest(folder / "features" / representation / "development.npz"),
                 "cv_hash": digest(folder / "cv_folds.json"),
                 "training_code": digest(ROOT / "src/uav_project/representation_training.py"),
                 "numpy": importlib.metadata.version("numpy"), "scikit_learn": importlib.metadata.version("scikit-learn")}
    old_receipt = destination / "signature.json"
    if old_receipt.exists() and read_json(old_receipt) != signature:
        raise ValueError("Training inputs/code/runtime changed; create a new experiment version")
    save_json(old_receipt, signature)
    estimators = candidates(cfg["seed"])
    results = []
    for name, model in estimators.items():
        receipt_path = destination / (name + ".json")
        model_path = destination / (name + ".joblib")
        if receipt_path.exists():
            result = read_json(receipt_path)
            if result["signature"] != signature or digest(model_path) != result["model_sha256"]:
                raise ValueError("Saved classifier differs from training receipt")
        else:
            scores = cross_val_score(model, X, y, cv=cv, scoring=cfg["selection_metric"], n_jobs=jobs, error_score="raise")
            model.fit(X, y)
            joblib.dump(model, model_path)
            result = {"model": name, "representation": representation, "signature": signature,
                      "cv_macro_f1_mean": float(scores.mean()), "cv_macro_f1_std": float(scores.std()),
                      "cv_scores": scores.tolist(), "cv_kind": protocol["cv_kind"],
                      "training": multiclass_metrics(y, model.predict(X), protocol["classes"]),
                      "model_sha256": digest(model_path), "model_path": relative(model_path)}
            save_json(receipt_path, result)
        results.append(result)
        print(representation, name, "development CV macro-F1:", result["cv_macro_f1_mean"], flush=True)
    chosen = max(results, key=lambda result: result["cv_macro_f1_mean"])["model"]
    grids = {"logistic": {"model__C": [.01, .1, 1., 10.]},
             "knn": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
             "svm": {"model__C": [.1, 1., 10.], "model__gamma": ["scale", .001]},
             "forest": {"model__max_depth": [None, 15, 30], "model__min_samples_leaf": [1, 3, 5]}}
    tuned_path = destination / "tuned.json"
    if tuned_path.exists():
        tuned = read_json(tuned_path)
        if tuned["signature"] != signature or tuned["selected_family"] != chosen or digest(resolve(tuned["model_path"])) != tuned["model_sha256"]:
            raise ValueError("Tuned artifact changed")
    else:
        grid = GridSearchCV(clone(estimators[chosen]), grids[chosen], cv=cv, scoring=cfg["selection_metric"],
                            n_jobs=jobs, refit=True, error_score="raise", return_train_score=True)
        grid.fit(X, y)
        model_path = destination / "tuned.joblib"
        joblib.dump(grid.best_estimator_, model_path)
        pd.DataFrame(grid.cv_results_).to_csv(destination / "grid_search.csv", index=False)
        tuned = {"model": "tuned", "representation": representation, "signature": signature,
                 "selected_family": chosen, "selection_rule": "development CV macro-F1 only; holdout unopened",
                 "best_parameters": grid.best_params_, "cv_macro_f1_mean": float(grid.best_score_),
                 "cv_macro_f1_std": float(grid.cv_results_["std_test_score"][grid.best_index_]),
                 "training": multiclass_metrics(y, grid.predict(X), protocol["classes"]),
                 "model_path": relative(model_path), "model_sha256": digest(model_path)}
        save_json(tuned_path, tuned)
    forest_path = destination / "forest.joblib"
    forest = joblib.load(forest_path)  # Only locally produced artifact with verified receipt.
    surviving = forest.named_steps["variance"].get_support()
    pd.DataFrame({"feature": np.asarray(names)[surviving],
                  "impurity_importance": forest.named_steps["model"].feature_importances_}).sort_values(
                      "impurity_importance", ascending=False).to_csv(destination / "forest_importance.csv", index=False)
    save_json(destination / "summary.json", {"signature": signature, "results": results, "tuned": tuned,
              "holdout_read": False, "limits": protocol["limitations"],
              "importance_caveat": "Impurity importance is model-specific, biased by correlated features and not causal; embedding dimensions have no physical names."})
    print("Training/tuning complete:", destination, "Holdout has not been evaluated.")


def _freeze_paths(folder):
    paths = [SETTINGS, folder / "manifest.csv", folder / "protocol.json", folder / "cv_folds.json"]
    paths += [ROOT / "src/uav_project" / name for name in ["representation_protocol.py", "representation_features.py",
              "representation_training.py", "representation_evaluation.py", "features.py", "common.py"]]
    cfg = read_json(SETTINGS)
    for representation in cfg["representations"]:
        destination = folder / "models" / representation
        for name in cfg["classifiers"] + ["tuned"]:
            receipt_path = destination / (name + ".json")
            receipt = read_json(receipt_path)
            path = resolve(receipt["model_path"])
            if digest(path) != receipt["model_sha256"]:
                raise ValueError("Model changed before/after freeze")
            paths.extend([receipt_path, path])
        paths.extend([destination / "summary.json", destination / "signature.json", destination / "grid_search.csv",
                      destination / "forest_importance.csv", folder / "features" / representation / "development.json",
                      folder / "features" / representation / "development.npz"])
        pin = folder / "features" / representation / "model_pin.json"
        if pin.exists():
            paths.append(pin)
    return {relative(path): digest(path) for path in paths}


def freeze_experiment(dataset):
    folder, protocol, frame = load_protocol(dataset)
    path = folder / "freeze.json"
    if path.exists():
        verify_freeze(dataset)
        print("Existing freeze verified:", path)
        return
    cfg = read_json(SETTINGS)
    summaries = [read_json(folder / "models" / rep / "summary.json") for rep in cfg["representations"]]
    selected = max((summary["tuned"] for summary in summaries), key=lambda result: result["cv_macro_f1_mean"])
    save_json(path, {"frozen_at": now(), "dataset": dataset, "file_hashes": _freeze_paths(folder),
                     "primary_model": {"representation": selected["representation"], "classifier": "tuned"},
                     "selection_rule": "development CV only; all declared baselines reported regardless of holdout rank",
                     "limitations": protocol["limitations"], "final_external_test_approved": False})
    print("Frozen:", path, "; holdout may now be extracted and evaluated.")


def verify_freeze(dataset):
    folder, _, _ = load_protocol(dataset)
    frozen = read_json(folder / "freeze.json")
    if frozen["file_hashes"] != _freeze_paths(folder):
        raise ValueError("Frozen artifacts changed; do not evaluate")
    return frozen
