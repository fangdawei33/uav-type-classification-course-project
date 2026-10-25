"""Draw visually checked, correct-only examples from existing frozen predictions.

This is a selected illustration, not a representative estimate of accuracy.
No model runs, image enhancement, or changes to the evaluation split occur.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/paper_work/matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from PIL import Image

RUN = ROOT / "outputs/imported_results/uav12_a3479ced386e1dea/outputs/experiments/uav_type_v3_v1"
SELECTED = [
    "dts_10GULQi0zBqzj9sfYDJd3oaCBmRUZ5-xs",
    "dts_1Me2WHAvopux1JsNBtldw8Ugjbz0Pp_BV",
    "visio_path_6e4ea96386f687e50105",
    "visio_path_8e8ae597aef29d50e929",
]


def main():
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next(f for f in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if f in available)
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cjk, "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    pred_path = RUN / "evaluation/qwen3_vl_lora_standard_predictions.csv"
    pred = pd.read_csv(pred_path).set_index("sample_id", verify_integrity=True)
    quality = pd.read_csv(ROOT / "data/delivery/uav_dataset_v3/clean_manifest.csv", low_memory=False).set_index("sample_id", verify_integrity=True)
    records = []
    fig = plt.figure(figsize=(180 / 25.4, 126 / 25.4))
    # Equal physical panels and fixed image windows avoid unpredictable whitespace.
    for i, sample_id in enumerate(SELECTED):
        row = pred.loc[sample_id]
        if int(row.label) != int(row.predicted_label):
            raise ValueError(f"Selected prediction is incorrect: {sample_id}")
        path = ROOT / row.clean_path
        with Image.open(path) as original:
            im = original.convert("RGB")
            width, height = im.size
        x = .035 + (i % 2) * .50
        y = .575 if i < 2 else .085
        ax = fig.add_axes([x, y, .43, .365])
        # Preserve the entire crop and its aspect ratio; no synthesized details.
        ax.imshow(im, interpolation="none")
        ax.set_axis_off()
        title = row.class_name.replace("DJI phantom 4 pro", "DJI Phantom 4 Pro").replace("Mavic_Air", "Mavic Air")
        fig.text(x, y + .393, f"({chr(97 + i)})  {title}", fontsize=10, weight="bold", color="#252B32")
        fig.text(x + .215, y - .035, "真实标签 = 预测标签", ha="center", fontsize=8.3, color="#3D6858")
        records.append({
            "panel": chr(97 + i), "sample_id": sample_id,
            "class_name": row.class_name, "true_label": int(row.label),
            "predicted_label": int(row.predicted_label), "native_width": width,
            "native_height": height, "source_dataset": quality.loc[sample_id, "source_dataset"],
            "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selection": "Visually reviewed for legibility among correctly predicted frozen test samples; four distinct classes; selected illustration only",
        })
    out = ROOT / "reports/standalone_figures"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ["png", "svg"]:
        path = out / f"fig13_correct_examples_scientific.{ext}"
        fig.savefig(path, dpi=600)
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    pd.DataFrame(records).to_csv(out / "fig13_correct_examples_cases.csv", index=False, encoding="utf-8-sig")
    receipt = {
        "prediction_source": pred_path.relative_to(ROOT).as_posix(),
        "prediction_source_sha256": hashlib.sha256(pred_path.read_bytes()).hexdigest(),
        "model": "qwen3_vl_lora_standard", "models_run": False,
        "caption": "标准QLoRA的正确分类样例。四张图来自固定测试集，按视觉清晰度人工筛选，仅用于展示正确识别案例，不代表总体性能。图像保持原始裁剪内容与比例，未进行细节生成。",
        "cases": records, "files": files,
    }
    (out / "fig13_correct_examples_provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved four verified correct examples as PNG, SVG, CSV and provenance JSON.")


if __name__ == "__main__":
    main()
