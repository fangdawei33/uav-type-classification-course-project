"""One frozen multiclass holdout evaluation, then tables/figures from saved scores."""
import numpy as np
import pandas as pd
import joblib
from .common import ROOT, digest, read_json, save_json, relative, resolve, now
from .representation_protocol import SETTINGS, load_protocol
from .representation_features import load_features
from .representation_training import verify_freeze, multiclass_metrics


def _group_interval(frame, pred, classes, seed=42, repeats=1000):
    from sklearn.metrics import f1_score
    groups = frame.group_id.unique()
    ids = {group: np.flatnonzero(frame.group_id.to_numpy() == group) for group in groups}
    rng = np.random.default_rng(seed)
    y = frame.label.to_numpy(dtype=int)
    scores = []
    # Resample complete sources rather than pretending correlated frames independent.
    for _ in range(repeats):
        selected = rng.choice(groups, size=len(groups), replace=True)
        index = np.concatenate([ids[group] for group in selected])
        scores.append(f1_score(y[index], pred[index], labels=range(len(classes)), average="macro", zero_division=0))
    return {"macro_f1_95_percentile_interval": np.quantile(scores, [.025, .975]).tolist(),
            "bootstrap_unit": "source_group", "repeats": repeats, "unique_groups": len(groups),
            "caveat": "An interval based on few sources is unstable; bootstrap does not create new independent sources."}


def evaluate_representation(dataset):
    frozen = verify_freeze(dataset)
    folder, protocol, manifest = load_protocol(dataset)
    output = folder / "evaluation"
    output.mkdir(exist_ok=True)
    result_path = output / "measured_results.json"
    freeze_hash = digest(folder / "freeze.json")
    if result_path.exists():
        existing = read_json(result_path)
        if existing["freeze_hash"] != freeze_hash:
            raise ValueError("Results use a different freeze")
        for path, expected in existing["prediction_hashes"].items():
            if digest(resolve(path)) != expected:
                raise ValueError("Saved predictions changed")
        print("Saved frozen evaluation already exists:", result_path)
        return
    cfg = read_json(SETTINGS)
    all_results, prediction_hashes = [], {}
    for representation in cfg["representations"]:
        X, _, holdout = load_features(dataset, representation, "holdout")
        development_receipt = read_json(folder / "features" / representation / "development.json")
        holdout_receipt = read_json(folder / "features" / representation / "holdout.json")
        for key in ["representation", "config_hash", "extractor_code", "handcrafted_code", "model_id", "revision", "transformers", "torch", "scikit_image", "inference_dtype"]:
            if development_receipt["signature"].get(key) != holdout_receipt["signature"].get(key):
                raise ValueError("Development and holdout extractors differ: " + key)
        for classifier in cfg["classifiers"] + ["tuned"]:
            directory = folder / "models" / representation
            receipt = read_json(directory / (classifier + ".json"))
            path = resolve(receipt["model_path"])
            if digest(path) != receipt["model_sha256"]:
                raise ValueError("Model differs from freeze receipt")
            model = joblib.load(path)  # Locally produced model, checked against frozen hashes.
            pred = model.predict(X).astype(int)
            metrics = multiclass_metrics(holdout.label, pred, protocol["classes"])
            prediction_path = output / (representation + "_" + classifier + "_predictions.csv")
            predictions = holdout[["sample_id", "class_name", "label", "group_id"]].copy()
            predictions["predicted_label"] = pred
            predictions["predicted_class"] = [protocol["classes"][i] for i in pred]
            predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
            prediction_hashes[relative(prediction_path)] = digest(prediction_path)
            uncertainty = (_group_interval(holdout, pred, protocol["classes"], cfg["seed"])
                           if protocol["cv_kind"] == "stratified_group" else
                           {"interval": None, "reason": "Unknown render/background parents; no source-independent CI claimed."})
            all_results.append({"representation": representation, "classifier": classifier,
                                "cv_macro_f1": receipt["cv_macro_f1_mean"],
                                "training_macro_f1": receipt["training"]["macro_f1"],
                                "holdout": metrics, "uncertainty": uncertainty})
    verify_freeze(dataset)
    save_json(result_path, {"evaluated_at": now(), "freeze_hash": freeze_hash,
                           "dataset": dataset, "main_seven_class_result": dataset == "seven",
                           "classes": protocol["classes"], "primary_model": frozen["primary_model"],
                           "results": all_results, "prediction_hashes": prediction_hashes,
                           "limitations": protocol["limitations"], "external_real_test_evaluated": False})
    print("Frozen holdout results:", result_path)


def export_representation_report(dataset):
    verify_freeze(dataset)
    folder, protocol, _ = load_protocol(dataset)
    result_path = folder / "evaluation/measured_results.json"
    measured = read_json(result_path)
    if measured["freeze_hash"] != digest(folder / "freeze.json"):
        raise ValueError("Measured results do not match frozen experiment")
    for path, expected in measured["prediction_hashes"].items():
        if digest(resolve(path)) != expected:
            raise ValueError("Prediction file changed")
    records = []
    for record in measured["results"]:
        records.append({"representation": record["representation"], "classifier": record["classifier"],
                        "training_macro_f1": record["training_macro_f1"], "cv_macro_f1": record["cv_macro_f1"],
                        **{key: record["holdout"][key] for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]}})
    table = pd.DataFrame(records)
    table.to_csv(folder / "evaluation/model_comparison.csv", index=False, encoding="utf-8-sig")
    tuning = []
    for representation in read_json(SETTINGS)["representations"]:
        tuned = read_json(folder / "models" / representation / "tuned.json")
        tuning.append({"representation": representation, "selected_family": tuned["selected_family"],
                       "best_parameters": str(tuned["best_parameters"]),
                       "before_cv_macro_f1": float(table[(table.representation == representation) & (table.classifier == tuned["selected_family"])].cv_macro_f1.iloc[0]),
                       "after_cv_macro_f1": tuned["cv_macro_f1_mean"]})
    pd.DataFrame(tuning).to_csv(folder / "evaluation/tuning_comparison.csv", index=False, encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    labels = (table.representation + "/" + table.classifier).tolist()
    x = np.arange(len(table))
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(x - .2, table.cv_macro_f1, .4, label="Development CV")
    axis.bar(x + .2, table.macro_f1, .4, label="Frozen holdout")
    axis.set(xticks=x, xticklabels=labels, ylabel="Macro-F1", ylim=(0, 1), title=dataset + " classification")
    axis.tick_params(axis="x", labelrotation=45)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "model_comparison.png", dpi=180)
    plt.close(fig)
    primary = measured["primary_model"]
    selected = next(record for record in measured["results"] if record["representation"] == primary["representation"] and record["classifier"] == primary["classifier"])
    fig, axis = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(np.asarray(selected["holdout"]["confusion_matrix"]), display_labels=protocol["classes"])
    display.plot(ax=axis, xticks_rotation=35, colorbar=False)
    axis.set_title("Primary model chosen before opening holdout")
    fig.tight_layout()
    fig.savefig(figures / "primary_confusion_matrix.png", dpi=180)
    plt.close(fig)
    importance = pd.read_csv(folder / "models/handcrafted/forest_importance.csv").head(20).iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(importance.feature, importance.impurity_importance)
    axis.set(xlabel="Random forest impurity importance (not causal)", title="Top 20 handcrafted feature dimensions")
    fig.tight_layout()
    fig.savefig(figures / "handcrafted_importance.png", dpi=180)
    plt.close(fig)
    # Plain Markdown requires no PDF/Word or third-party table formatter.
    columns = table.columns.tolist()
    markdown = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table.itertuples(index=False, name=None):
        markdown.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    report = ["# 实测多分类结果：" + dataset, "", "数据范围：" + protocol["limitations"], "",
              "七类正式结果：" + str(dataset == "seven") + "；外部真实测试未实施。", "",
              "模型选择依据为开发集CV Macro-F1。测试集上所有已声明模型均报告，不能根据测试排名再次调参。", "",
              *markdown, "", "图1：开发集CV与冻结后留出集Macro-F1。", "![](figures/model_comparison.png)", "",
              "图2：预先选定主模型的混淆矩阵。", "![](figures/primary_confusion_matrix.png)", "",
              "图3：手工特征随机森林的不纯度重要性；相关维度可能互相分摊，不能解释为因果效应。",
              "![](figures/handcrafted_importance.png)", "",
              "训练/CV/留出集差距用于讨论过拟合；调参前后CV表见evaluation/tuning_comparison.csv。",
              "当前代码不声称多模态配对、合成数据独立性或论文创新性。", "",
              "实测JSON SHA256：`" + digest(result_path) + "`"]
    (folder / "results.md").write_text("\n".join(report), encoding="utf-8")
    print("Measured tables/figures/report:", folder / "results.md")

