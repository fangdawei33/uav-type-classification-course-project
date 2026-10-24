"""15. Turn imported Colab tables into report figures and deterministic examples.

Reads saved predictions only. Does not fit/load models, extract representations,
perform inference, calculate a new model score or run software tests.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone
from _bootstrap import PROJECT_ROOT


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save_table(path, records):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    root = PROJECT_ROOT
    current = load_json(root / "outputs/auxiliary_results_current.json")
    imported = root / current["import_folder"]
    receipt = load_json(imported / "import_receipt.json")
    if not receipt["all_available_checks_passed"]:
        raise ValueError("Resolve imported file/table consistency issues first")
    source = imported / "outputs/representation_v1/auxiliary"
    measured = load_json(source / "evaluation/measured_results.json")
    frozen = load_json(source / "freeze.json")
    if measured["primary_model"] != frozen["primary_model"]:
        raise ValueError("Preselected model changed")
    if measured["primary_model"] != {"representation": "siglip2", "classifier": "tuned"}:
        raise ValueError("This report recipe describes the returned auxiliary v1 run only")
    # Reports are derived outside the immutable import, separately for each ZIP.
    out = root / "outputs/report_analysis" / imported.name
    out.mkdir(parents=True, exist_ok=True)
    comparison = table(source / "evaluation/model_comparison.csv")
    models = ["logistic", "knn", "svm", "forest", "tuned"]
    names = ["Logistic", "KNN", "RBF-SVM", "Random forest", "Tuned SVM"]
    captions = {"handcrafted": "Handcrafted (1,831 dimensions)", "siglip2": "Frozen SigLIP2 (768 dimensions)"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    for ax, rep in zip(axes, captions):
        subset = {r["classifier"]: r for r in comparison if r["representation"] == rep}
        x = np.arange(5)
        for offset, field, label, color in [(-.19, "cv_macro_f1", "Development CV", "#92b6d5"), (.19, "macro_f1", "Frozen holdout", "#27648b")]:
            heights = [float(subset[m][field]) * 100 for m in models]
            bars = ax.bar(x + offset, heights, .36, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in heights], fontsize=8, rotation=90, padding=3)
        ax.set_xticks(x, names, rotation=18, ha="right")
        ax.set_title(captions[rep], fontsize=12)
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Macro-F1 (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2, frameon=False)
    fig.savefig(out / "representation_comparison.png", dpi=180)
    plt.close(fig)

    tuned = {r["representation"]: r for r in measured["results"] if r["classifier"] == "tuned"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    labels = ["Inspire", "Mavic", "Phantom", "No drone"]
    for ax, rep in zip(axes, captions):
        cm = np.array(tuned[rep]["holdout"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1000)
        for row in range(4):
            for col in range(4):
                ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > 500 else "black", fontsize=11)
        ax.set_xticks(range(4), labels, rotation=20, ha="right")
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        suffix = "Preselected before holdout" if rep == "siglip2" else "Declared comparison"
        ax.set_title(captions[rep].split(" (")[0] + " + tuned SVM\n" + suffix, fontsize=11)
    fig.savefig(out / "tuned_confusion_comparison.png", dpi=180)
    plt.close(fig)

    preds = {rep: table(source / f"evaluation/{rep}_tuned_predictions.csv") for rep in captions}
    lookups = {rep: {r["sample_id"]: r for r in records} for rep, records in preds.items()}
    if set(lookups["handcrafted"]) != set(lookups["siglip2"]):
        raise ValueError("Paired returned predictions have different samples")
    pair_counts = Counter()
    for sid, r in lookups["siglip2"].items():
        h = lookups["handcrafted"][sid]
        if h["label"] != r["label"]:
            raise ValueError("Paired truths differ")
        pair_counts[(h["label"] == h["predicted_label"], r["label"] == r["predicted_label"])] += 1
    save_table(out / "paired_prediction_outcomes.csv", [
        {"handcrafted_correct": h, "siglip2_correct": s, "images": pair_counts[(h, s)]}
        for h, s in [(True, True), (True, False), (False, True), (False, False)]])

    # Fixed sample-ID order: transparent case selection, no result-dependent tuning.
    selected = sorted((r for r in preds["siglip2"] if r["label"] != r["predicted_label"]), key=lambda r: r["sample_id"])[:12]
    manifest = {r["sample_id"]: r for r in table(root / "outputs/representation_v1/auxiliary/manifest.csv")}
    contact = Image.new("RGB", (4 * 340, 3 * 290), "white")
    draw = ImageDraw.Draw(contact)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    selection_rows = []
    for idx, r in enumerate(selected):
        item = manifest[r["sample_id"]]
        image_path = (root / item["clean_path"]).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Example outside project")
        actual_sha = sha(image_path)
        if actual_sha != item["clean_sha256"]:
            raise ValueError("Original image differs from frozen local manifest")
        x, y = (idx % 4) * 340, (idx // 4) * 290
        with Image.open(image_path) as original:
            im = ImageOps.exif_transpose(original).convert("RGB")
            im = ImageOps.contain(im, (324, 210))
            contact.paste(im, (x + (340 - im.width) // 2, y + 4 + (210 - im.height) // 2))
        draw.text((x + 8, y + 218), f"{idx+1}. {r['sample_id'][-12:]}", font=font, fill="black")
        draw.text((x + 8, y + 240), f"True: {labels[int(r['label'])]}", font=font, fill="black")
        draw.text((x + 8, y + 260), f"Pred: {labels[int(r['predicted_label'])]}", font=font, fill="#9b3134")
        selection_rows.append({"order": idx + 1, "sample_id": r["sample_id"], "true_class": r["class_name"], "predicted_class": r["predicted_class"], "original_path": item["clean_path"], "original_sha256": actual_sha, "selection_rule": "first 12 errors in ascending sample_id; preselected siglip2 tuned model"})
    contact.save(out / "preselected_error_examples.png")
    save_table(out / "error_example_selection.csv", selection_rows)
    outputs = ["representation_comparison.png", "tuned_confusion_comparison.png", "preselected_error_examples.png", "paired_prediction_outcomes.csv", "error_example_selection.csv"]
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "archive_sha256": current["report_archive_sha256"], "source_import": current["import_folder"], "primary_model": frozen["primary_model"], "source_sha256": {str(p.relative_to(root)).replace('\\','/'): sha(p) for p in [source / 'evaluation/measured_results.json', source / 'evaluation/model_comparison.csv', source / 'evaluation/handcrafted_tuned_predictions.csv', source / 'evaluation/siglip2_tuned_predictions.csv', root / 'outputs/representation_v1/auxiliary/manifest.csv']}, "output_sha256": {name: sha(out / name) for name in outputs}, "script_sha256": sha(Path(__file__)), "local_model_inference_run": False, "local_model_training_run": False, "software_tests_run": False, "scope": "Plot returned scores; count paired saved predictions; verify and display deterministic source-image examples. No new model evaluation."}
    (out / "analysis_receipt.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Report analysis exported:", out)
    print("Paired saved predictions:", dict(pair_counts))


if __name__ == "__main__":
    main()
