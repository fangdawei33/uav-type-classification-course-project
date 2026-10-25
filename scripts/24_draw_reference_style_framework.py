"""Draw a compact academic framework figure in the supplied reference style."""
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
    fig, ax = plt.subplots(figsize=(180 / 25.4, 105 / 25.4))
    ax.set(xlim=(0, 180), ylim=(0, 105)); ax.axis("off")
    ink = "#20252A"
    line = "#18365C"
    process = "#667487"
    blue = "#275D91"
    green = "#3E8978"
    ochre = "#9A752A"
    white = "#FFFFFF"

    def label(x, y, s, size=9, color=ink, weight="normal", ha="center", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, linespacing=1.25)

    def box(x, y, w, h, title, lines, color=line, body_size=8.4):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=white,
                                       edgecolor=color, linewidth=1.15))
        ax.plot([x, x + w], [y + h - 8, y + h - 8], color=color, linewidth=.85)
        label(x + w / 2, y + h - 4, title, size=10.2, color=ink, weight="bold")
        label(x + w / 2, y + (h - 8) / 2 - .5, "\n".join(lines), size=body_size)

    def arrow(x1, y1, x2, y2, color=ink, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "linewidth": .95, "mutation_scale": 9,
                                "linestyle": "--" if dashed else "-"})

    # Main boxes: compact, symmetric, and intentionally text-light.
    box(3, 39, 31, 28, "数据输入", ["20,010 张裁剪", "12 种机型", "公开数据 / 爬虫", "标签统一"], line, 7.6)
    box(40, 36, 35, 34, "数据处理与划分", ["RGB / 框校验", "去重 / 尺寸", "分组划分", "14,106 / 5,904", "seed 42"], process, 7.3)

    box(84, 72, 45, 24, "手工视觉表示", ["HOG + HSV + LBP", "1,831 维特征", "LR / KNN / SVM / RF"], blue, 8.3)
    box(84, 42, 45, 24, "冻结视觉表示", ["SigLIP2", "768 维向量", "LR / KNN / SVM / RF"], green, 8.3)
    box(84, 12, 45, 24, "视觉语言模型适配", ["Qwen3-VL-2B", "视觉冻结 + LoRA", "标准 / 类别加权"], ochre, 8.3)
    box(143, 39, 34, 28, "统一测试评价", ["测试 5,904 张", "Accuracy", "Macro-F1", "混淆 / 尺度"], line, 7.7)

    # Input flow and three-way branch.
    arrow(34, 53, 40, 53, line)
    ax.plot([78, 78], [24, 84], color=ink, linewidth=.8)
    for y in [84, 54, 24]:
        arrow(75, 53, 83.5, y, line, dashed=(y == 24))
        arrow(129, y, 142.5, 53, line)

    # One unobtrusive legend, matching the reference figure.
    ax.add_patch(patches.Rectangle((62, 1.2), 56, 7.2, facecolor=white, edgecolor="#555B60", linewidth=.65))
    label(90, 6.5, "实线：数据前向路径    虚线：可训练参数 / 监督损失", size=7.2, color="#3C444A")

    out = ROOT / "reports/standalone_figures"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ["png", "svg"]:
        p = out / f"fig02_framework_reference_style.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=.08)
        files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    (out / "fig02_framework_reference_style_provenance.json").write_text(
        json.dumps({"files": files, "models_run": False, "style": "reference-inspired compact academic box-and-arrow diagram"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved reference-style framework figure.")


if __name__ == "__main__":
    main()
