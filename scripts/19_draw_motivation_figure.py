"""Draw one standalone scientific motivation figure from existing evidence.

No models run. The source report and its figures are left untouched.
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
from matplotlib import font_manager, patches
import numpy as np
import pandas as pd
from PIL import Image


def main():
    available = {font.name for font in font_manager.fontManager.ttflist}
    cjk = next((s for s in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if s in available), None)
    if not cjk:
        raise RuntimeError("A Chinese font is required")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cjk, "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "axes.unicode_minus": False,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    ink, muted, line, blue = "#22272E", "#65717D", "#CBD1D7", "#315D7D"
    fig = plt.figure(figsize=(180 / 25.4, 85 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 180), ylim=(85, 0)); ax.axis("off")

    def text(x, y, value, size=8.5, color=ink, **kwargs):
        return ax.text(x, y, value, fontsize=size, color=color, va="center", **kwargs)

    def arrow(p, q, color=muted, linewidth=.85):
        ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle": "-|>", "color": color,
                    "lw": linewidth, "mutation_scale": 8, "shrinkA": 0, "shrinkB": 0})

    # Panel labels and quiet rules replace the previous presentation cards.
    for x, end, heading in [(3, 52, "(a) 视觉证据"), (60, 114, "(b) 对照方法"), (123, 177, "(c) 研究问题")]:
        text(x, 5, heading, 9.5, weight="bold")
        ax.plot([x, end], [10, 10], color=line, lw=.65)
    for x in [56, 119]:
        ax.plot([x, x], [15, 70], color=line, lw=.55)

    # A real saved crop, not a generated or artificially degraded image.
    run = ROOT / "outputs/imported_results/uav12_a3479ced386e1dea/outputs/experiments/uav_type_v3_v1"
    pred = pd.read_csv(run / "evaluation/qwen3_vl_lora_standard_predictions.csv")
    sample_id = "dts_1-5YrBPryj37GSeCWggDO4FORDohoKNlt"
    row = pred.loc[pred.sample_id.eq(sample_id)].iloc[0]
    source = ROOT / row.clean_path
    with Image.open(source) as img:
        pixels = np.asarray(img.convert("RGB")); width, height = img.size
    # Different display scales of identical pixels; neither adds image detail.
    ax.imshow(pixels, extent=(5, 15.5, 37, 28), interpolation="nearest", aspect="auto")
    ax.imshow(pixels, extent=(27, 51.5, 43, 22), interpolation="nearest", aspect="auto")
    for x, y, w, h in [(5, 28, 10.5, 9), (27, 22, 24.5, 21)]:
        ax.add_patch(patches.Rectangle((x, y), w, h, fc="none", ec=line, lw=.55))
    arrow((17, 32.5), (24.5, 32.5))
    text(10.3, 45.5, f"{width} × {height} px", 7.1, muted, ha="center")
    text(39.3, 47.5, "最近邻放大", 7.6, muted, ha="center")
    text(27.5, 57, "放大不增加视觉细节", 8.4, ha="center")
    text(27.5, 64.5, "短边 < 24 px：6.31%", 7.8, muted, ha="center")

    # Three independent comparisons, with no implied ensemble or new module.
    method_rows = [
        (23, "手工视觉表示", "HOG / HSV / LBP", "1,831 维"),
        (40, "冻结预训练表示", "SigLIP2", "768 维"),
        (57, "视觉语言模型", "Qwen3-VL + QLoRA", "标签生成"),
    ]
    for y, heading, detail, suffix in method_rows:
        # A small baseline marker gives each row structure without a box.
        ax.plot([61, 61], [y-3.4, y+5.3], color=blue, lw=1.4)
        text(64, y-1, heading, 8.5, weight="bold")
        text(64, y+5.1, detail, 7.5, muted)
        text(112, y-1, suffix, 7.2, muted, ha="right")

    questions = [
        (21, "RQ1  数据与评价范围", "类别及来源覆盖\n支持怎样的结论？"),
        (39, "RQ2  视觉表示", "冻结预训练表示\n是否优于手工表示？"),
        (57, "RQ3  监督适配", "监督训练与类别加权\n分别带来什么变化？"),
    ]
    for y, heading, detail in questions:
        text(124, y-1, heading, 8.1, blue, weight="bold")
        text(124, y+7.2, detail, 7.9, linespacing=1.45)

    # A single line names the actual control shared by the comparisons.
    ax.plot([3, 177], [74, 74], color=line, lw=.65)
    text(90, 80, "统一评价协议：14,106 张训练图像 · 5,904 张冻结测试图像 · 相同类别映射", 8.0, muted, ha="center")

    out = ROOT / "reports/standalone_figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        path = out / f"fig01_motivation_scientific.{ext}"
        fig.savefig(path, dpi=600, bbox_inches=None)
        if ext == "svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)
    record = {
        "source_sample_id": sample_id,
        "source_dataset": "Drone-type-Set author subset",
        "source_image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_size": [width, height], "display": "same real crop; nearest-neighbor enlargement only",
        "main_dataset_size": 20010, "short_side_below_24": 1262,
        "training_samples": 14106, "test_samples": 5904, "models_run": False,
        "scope": "Standalone replacement candidate; report files unchanged",
    }
    (out / "fig01_motivation_scientific_provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(out / "fig01_motivation_scientific.png")


if __name__ == "__main__":
    main()
