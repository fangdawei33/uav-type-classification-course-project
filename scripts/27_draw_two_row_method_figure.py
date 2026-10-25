"""Draw the requested two-row traditional-vs-VLM method schematic."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/paper_work/matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, patheffects
from PIL import Image


def main():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "SimHei"] if f in available), None)
    if not font:
        raise RuntimeError("A Chinese font is required")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    source = ROOT / "data/final/uav_type_v3/images/class_02/dts_10GULQi0zBqzj9sfYDJd3oaCBmRUZ5-xs.png"
    with Image.open(source) as im:
        image = im.convert("RGB")
    fig, ax = plt.subplots(figsize=(180 / 25.4, 105 / 25.4))
    ax.set(xlim=(0, 180), ylim=(0, 105)); ax.axis("off")
    ink, muted, navy = "#1F2933", "#65717D", "#263F62"
    blue, green, orange, yellow = "#9FC6E7", "#A9D3B6", "#F5C9A9", "#F7DD8B"

    def text(x, y, s, size=9, color=ink, weight="normal", ha="center", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, linespacing=1.18)

    def shadow_box(x, y, w, h, fc, radius=.7, ec="none", lw=0):
        p = patches.FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=.012,rounding_size={radius}",
                                   facecolor=fc, edgecolor=ec, linewidth=lw)
        p.set_path_effects([patheffects.SimplePatchShadow(offset=(1.1, -1.1), alpha=.18), patheffects.Normal()])
        ax.add_patch(p)

    def arrow(x1, y1, x2, y2, color=ink, lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "mutation_scale": 10})

    def input_image(x, y, w=22, h=17):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor="white", edgecolor="#D7DEE6", lw=.7))
        ax.imshow(image, extent=(x+1, x+w-1, y+1, y+h-1), interpolation="none", zorder=3)

    # Top: traditional machine-learning pipeline.
    text(4, 95, "传统机器学习", size=11.5, color=navy, weight="bold", ha="left")
    text(5, 82, "图像", size=8.5, color=muted, ha="left")
    input_image(18, 72, 23, 17)
    arrow(41.5, 80.5, 54, 80.5, navy, 1.1)

    # Feature polygon, mirroring the reference backbone shape.
    feature = patches.Polygon([(55, 72), (65, 76), (75, 74), (75, 87), (65, 89), (55, 85)],
                               closed=True, facecolor=blue, edgecolor="none")
    feature.set_path_effects([patheffects.SimplePatchShadow(offset=(1.0, -1.0), alpha=.18), patheffects.Normal()])
    ax.add_patch(feature)
    text(65, 82, "特征提取", size=9.5, weight="bold")
    text(65, 77.7, "HOG / HSV / LBP", size=7.4)
    arrow(75, 80.5, 83, 80.5, navy)

    shadow_box(84, 74, 29, 13, green)
    text(98.5, 82.5, "传统分类器", size=9.2, weight="bold")
    text(98.5, 78, "LR · KNN · SVM · RF", size=7.2)
    arrow(113, 80.5, 122, 80.5, navy)

    # Traditional output as three small candidate labels.
    for i, (y, label, fc) in enumerate([(84, "机型 A", green), (79.5, "机型 B", yellow), (75, "机型 C", orange)]):
        shadow_box(124, y-1.6, 17, 3.2, fc, radius=.35)
        text(132.5, y, label, size=7.0)
    text(146, 80.5, "…", size=11, color=muted)
    arrow(141.5, 80.5, 151, 80.5, navy)
    text(162, 80.5, "预测机型", size=10, color=navy, weight="bold")

    # Divider between traditional ML and VLM.
    ax.plot([6, 174], [61.5, 61.5], color="#222831", lw=.9, ls=(0, (5, 3)))
    text(8, 64.2, "图像特征分类", size=7.5, color=muted, ha="left")

    # Bottom: vision-language model pipeline with image + text instruction.
    text(4, 54, "视觉语言模型", size=11.5, color=navy, weight="bold", ha="left")
    text(5, 43.3, "图像", size=8.5, color=muted, ha="left")
    input_image(18, 34, 23, 17)
    arrow(41.5, 42.5, 53, 42.5, navy, 1.1)

    shadow_box(55, 33, 50, 19, orange, radius=1)
    text(80, 46.5, "Qwen3-VL-2B", size=11, weight="bold")
    text(80, 41.6, "图像 + 分类指令", size=8.5)
    text(80, 37.5, "视觉主干冻结 · LoRA 适配", size=7.7, color=ink)
    arrow(105, 42.5, 116, 42.5, navy)

    shadow_box(118, 35, 24, 15, yellow, radius=.8)
    text(130, 45, "语言侧输出", size=9.2, weight="bold")
    text(130, 40, "LABEL_00", size=7.6)
    text(130, 37.2, "··· LABEL_11", size=7.2)
    arrow(142, 42.5, 151, 42.5, navy)
    text(162, 42.5, "预测机型", size=10, color=navy, weight="bold")

    # Instruction box under the image, connected to the VLM input.
    ax.add_patch(patches.FancyBboxPatch((48, 18), 57, 11, boxstyle="round,pad=.01,rounding_size=.6",
                                       facecolor="white", edgecolor="#5E6874", linewidth=.8,
                                       linestyle=(0, (5, 3))))
    text(76.5, 24.2, "图像 + 分类指令", size=8.8, color=ink, weight="bold")
    text(76.5, 20.7, "输出对应机型标签", size=7.8, color=muted)
    arrow(80, 29, 80, 33, navy, .95)

    # Plain caption-like bottom statement.
    ax.plot([6, 174], [12, 12], color="#DCE3EA", lw=.7)
    text(90, 7.5, "上：图像特征 + 传统分类器    下：图像指令 + 视觉语言模型", size=8.0, color=muted)

    out = ROOT / "reports/standalone_figures"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ["png", "svg"]:
        p = out / f"fig02_two_row_traditional_vs_vlm.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=.08)
        files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    (out / "fig02_two_row_traditional_vs_vlm_provenance.json").write_text(
        json.dumps({"source_image": str(source.relative_to(ROOT)), "files": files,
                    "models_run": False, "image_modified": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("Saved two-row traditional-vs-VLM figure.")


if __name__ == "__main__":
    main()
