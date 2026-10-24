"""Evaluate all frozen declared models once on the partial-coverage holdout."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from .common import ROOT, digest, read_json, save_json, now, relative, resolve
from .uav12_features import load_uav12_features
from .uav12_models import verify_uav12_freeze
from .uav12_protocol import CONFIG, load_protocol, run_dir, settings


def _metrics(frame, prediction, protocol):
    truth = frame.label.to_numpy(dtype=int)
    prediction = np.asarray(prediction, dtype=int)
    present = sorted(set(truth))
    all_labels = list(range(len(protocol["classes"])))
    report = classification_report(truth, prediction, labels=all_labels,
        target_names=protocol["classes"], output_dict=True, zero_division=0)
    invalid = float(np.mean(~np.isin(prediction, all_labels)))
    supported_macro = float(f1_score(truth, prediction, labels=present, average="macro", zero_division=0))
    return {"samples": int(len(frame)), "labels_present": present,
        "class_names_present": [protocol["classes"][index] for index in present],
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy_present_labels": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1_supported_labels": supported_macro,
        "macro_f1_all_12_labels": float(f1_score(truth, prediction, labels=all_labels,
                                                  average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, labels=all_labels,
                                       average="weighted", zero_division=0)),
        "invalid_prediction_rate": invalid,
        "classification_report": report,
        "confusion_matrix_all_12": confusion_matrix(truth, prediction, labels=all_labels).tolist(),
        "confusion_matrix_with_invalid": confusion_matrix(truth, prediction, labels=all_labels + [-1]).tolist()}


def _save_predictions(path, frame, prediction, extra=None):
    output = frame[["sample_id", "label", "class_name", "group_id", "clean_path"]].copy()
    output["predicted_label"] = np.asarray(prediction, dtype=int)
    if extra:
        for key, value in extra.items():
            output[key] = value
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return digest(path)


def evaluate_uav12():
    frozen = verify_uav12_freeze()
    folder, protocol, manifest = load_protocol()
    output = folder / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "measured_results.json"
    if result_path.exists():
        result = read_json(result_path)
        if result["freeze_hash"] != digest(folder / "freeze.json"):
            raise ValueError("Saved evaluation belongs to a different global freeze")
        for path, expected in result["prediction_hashes"].items():
            if digest(resolve(path)) != expected:
                raise ValueError("Saved prediction output changed: " + path)
        print("Frozen evaluation already exists:", result_path, flush=True)
        return
    holdout = manifest[manifest.split.eq("holdout")].reset_index(drop=True)
    all_results, prediction_hashes = [], {}
    for representation in settings()["representations"]:
        X, _, feature_frame = load_uav12_features(representation, "holdout")
        development_receipt = read_json(folder / "features" / representation / "development.json")
        holdout_receipt = read_json(folder / "features" / representation / "holdout.json")
        for key in ["model_id", "revision", "inference_dtype", "torch", "transformers", "feature_code_hashes"]:
            if development_receipt["signature"].get(key) != holdout_receipt["signature"].get(key):
                raise ValueError("Development and holdout feature extractors differ: " + key)
        if feature_frame.sample_id.tolist() != holdout.sample_id.tolist():
            raise ValueError("Holdout feature order differs from protocol")
        directory = folder / "models" / representation
        for classifier in settings()["classifiers"] + ["tuned"]:
            receipt = read_json(directory / f"{classifier}.json")
            model_path = resolve(receipt["model_path"])
            if digest(model_path) != receipt["model_sha256"]:
                raise ValueError("Frozen classifier hash changed")
            model = joblib.load(model_path)
            prediction = model.predict(X).astype(int)
            metrics = _metrics(holdout, prediction, protocol)
            prediction_path = output / f"{representation}_{classifier}_predictions.csv"
            prediction_hashes[relative(prediction_path)] = _save_predictions(prediction_path, holdout, prediction)
            all_results.append({"model": f"{representation}_{classifier}", "family": "traditional",
                "representation": representation, "classifier": classifier,
                "cv_macro_f1_supported_labels": receipt.get("cv_macro_f1_mean"),
                "metrics": metrics, "prediction_file": relative(prediction_path)})
            print(f"{representation}/{classifier}: holdout supported-label macro-F1="
                  f"{metrics['macro_f1_supported_labels']:.4f}", flush=True)
    from .uav12_qwen import predict_qwen_holdout
    qwen_predictions = predict_qwen_holdout()
    for name, prediction_frame in qwen_predictions.items():
        prediction = prediction_frame.predicted_label.to_numpy(dtype=int)
        extras = {"prediction_text": prediction_frame.prediction_text.tolist(),
                  "valid_label": np.isin(prediction, list(range(12)))}
        prediction_path = output / f"{name}_predictions.csv"
        prediction_hashes[relative(prediction_path)] = _save_predictions(prediction_path, holdout, prediction, extras)
        metrics = _metrics(holdout, prediction, protocol)
        metrics["valid_label_rate"] = float(np.mean(extras["valid_label"]))
        all_results.append({"model": name, "family": "qwen3_vl",
            "representation": "image_language_model", "classifier": name,
            "cv_macro_f1_supported_labels": None, "metrics": metrics,
            "prediction_file": relative(prediction_path)})
    verify_uav12_freeze()
    save_json(result_path, {"evaluated_at": now(), "freeze_hash": digest(folder / "freeze.json"),
        "dataset": "uav_type_v3", "holdout_samples": len(holdout),
        "holdout_class_coverage": protocol["holdout_class_names"],
        "primary_model": frozen["primary_model"], "results": all_results,
        "prediction_hashes": prediction_hashes,
        "test_selection_used": False, "external_real_test_evaluated": False,
        "limitations": protocol["limitations"]})
    print("Frozen holdout evaluation saved:", result_path, flush=True)


def export_uav12_report():
    folder, protocol, _ = load_protocol()
    result_path = folder / "evaluation/measured_results.json"
    if not result_path.is_file():
        raise FileNotFoundError("Run scripts/10d_evaluate_uav12.py first")
    measured = read_json(result_path)
    if measured["freeze_hash"] != digest(folder / "freeze.json"):
        raise ValueError("Evaluation is not bound to the current freeze")
    for path, expected in measured["prediction_hashes"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Prediction output changed: " + path)
    rows = []
    for item in measured["results"]:
        metrics = item["metrics"]
        rows.append({"model": item["model"], "family": item["family"],
            "representation": item["representation"], "cv_macro_f1_supported_labels": item.get("cv_macro_f1_supported_labels"),
            "samples": metrics["samples"], "labels_present": len(metrics["labels_present"]),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy_present_labels": metrics["balanced_accuracy_present_labels"],
            "macro_f1_supported_labels": metrics["macro_f1_supported_labels"],
            "macro_f1_all_12_labels": metrics["macro_f1_all_12_labels"],
            "weighted_f1": metrics["weighted_f1"], "invalid_prediction_rate": metrics["invalid_prediction_rate"]})
    table = pd.DataFrame(rows)
    table.to_csv(folder / "evaluation/model_comparison.csv", index=False, encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure_dir = folder / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    order = table.sort_values("macro_f1_supported_labels", ascending=False)
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(np.arange(len(order)), order.macro_f1_supported_labels)
    axis.set(xticks=np.arange(len(order)), xticklabels=order.model, ylim=(0, 1),
             ylabel="Macro-F1 on labels present in holdout", title="Frozen model comparison")
    axis.tick_params(axis="x", labelrotation=55)
    fig.tight_layout(); fig.savefig(figure_dir / "model_comparison.png", dpi=180); plt.close(fig)
    primary_name = "qwen3_vl_lora_standard"
    primary = next(item for item in measured["results"] if item["model"] == primary_name)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, axis = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(np.asarray(primary["metrics"]["confusion_matrix_with_invalid"]),
        display_labels=protocol["classes"] + ["Invalid output"]).plot(ax=axis, xticks_rotation=50, colorbar=False)
    axis.set_title("Primary Qwen3-VL LoRA confusion matrix (partial holdout)")
    fig.tight_layout(); fig.savefig(figure_dir / "primary_confusion_matrix.png", dpi=180); plt.close(fig)
    importance_path = folder / "models" / "handcrafted" / "forest_importance.csv"
    if importance_path.is_file():
        importance = pd.read_csv(importance_path).head(20).iloc[::-1]
        fig, axis = plt.subplots(figsize=(9, 7))
        axis.barh(importance.feature, importance.impurity_importance)
        axis.set(xlabel="Random-forest impurity importance", title="Top handcrafted feature dimensions")
        fig.tight_layout(); fig.savefig(figure_dir / "handcrafted_importance.png", dpi=180); plt.close(fig)
    lines = ["# 十二类无人机机型分类冻结评估", "",
        "数据入口：`data/delivery/uav_dataset_v3/clean_manifest.csv`。",
        f"留出集：{measured['holdout_samples']} 张，覆盖 {len(measured['holdout_class_coverage'])}/12 类；这是分组诊断留出，不是完整十二类独立来源测试。", "",
        "主指标为留出集实际出现类别上的 Macro-F1。`macro_f1_all_12_labels` 将未出现的四类按零计入，只作完整标签空间参考，不能解读为四类已完成测试。", ""]
    columns = table.columns.tolist()
    lines += ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table.itertuples(index=False, name=None):
        cells = []
        for value in row:
            cells.append("" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "![模型对比](figures/model_comparison.png)", "",
              "![主模型混淆矩阵](figures/primary_confusion_matrix.png)", "",
              "图像特征重要性来自随机森林不纯度分数，不能解释为因果影响。", "",
              "本报告不把当前数据写成端到端检测、完整十二类独立评测或真实图像-日志-文本三模态融合。"]
    (folder / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Exported report:", folder / "results.md", flush=True)
