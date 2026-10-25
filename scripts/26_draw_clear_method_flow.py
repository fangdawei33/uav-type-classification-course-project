"""Draw an unambiguous three-path method flow with one genuine input image."""
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
    image_path = ROOT / "data/final/uav_type_v3/images/class_02/dts_10GULQi0zBqzj9sfYDJd3oaCBmRUZ5-xs.png"
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    fig, ax = plt.subplots(figsize=(180 / 25.4, 100 / 25.4))
    ax.set(xlim=(0, 180), ylim=(0, 100)); ax.axis("off")
    ink, muted, navy = "#202A35", "#687580", "#315F8A"
    blue, green, orange = "#A9CBEA", "#B1D8BD", "#F5C9A9"

    def txt(x, y, s, size=9, color=ink, weight="normal", ha="center", va="center"):
        ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va=va, linespacing=1.2)

    def pill(x, y, w, h, title, detail, color):
        p = patches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=.9", facecolor=color,
            edgecolor="none", linewidth=0)
        p.set_path_effects([patheffects.SimplePatchShadow(offset=(1, -1), alpha=.17), patheffects.Normal()])
        ax.add_patch(p)
        txt(x + w/2, y + h*.64, title, size=9.1, weight="bold")
        txt(x + w/2, y + h*.29, detail, size=7.6, color=ink)

    def arrow(x1, y1, x2, y2, color=navy, lw=1.15):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "lw": lw, "mutation_scale": 10})

    # One real image and one shared preprocessing step.
    txt(18, 91, "输入图像", size=11, color=navy, weight="bold")
    ax.add_patch(patches.FancyBboxPatch((4, 43), 28, 42,
        boxstyle="round,pad=0.012,rounding_size=1", facecolor="white",
        edgecolor="#D3DCE4", linewidth=1.0))
    ax.imshow(image, extent=(7, 29, 57, 78), interpolation="none", zorder=3)
    txt(18, 49.5, "无人机目标裁剪", size=8.8, color=muted)
    pill(38, 54, 24, 20, "预处理", "RGB · 框校验", "#DDE8F1")
    arrow(32, 64, 37, 64)

    # Simple headers make the reading order explicit.
    txt(78, 91, "表示 / 适配", size=10.5, color=navy, weight="bold")
    txt(125, 91, "决策模块", size=10.5, color=navy, weight="bold")
    txt(163, 91, "输出", size=10.5, color=navy, weight="bold")
    ax.plot([66, 66], [9, 88], color="#D8E0E7", lw=.8, ls="--")
    ax.plot([112, 112], [9, 88], color="#D8E0E7", lw=.8, ls="--")
    ax.plot([148, 148], [9, 88], color="#D8E0E7", lw=.8, ls="--")

    rows = [(70, "A", "手工特征", "HOG · HSV · LBP", "传统分类器", blue),
            (48, "B", "冻结编码器", "SigLIP2", "线性分类器", green),
            (26, "C", "视觉语言模型", "Qwen3-VL", "LoRA 适配", orange)]
    for y, letter, feature, detail, decision, color in rows:
        # row label
        txt(68, y + 2, letter, size=11, color=navy, weight="bold")
        pill(72, y - 6, 37, 14, feature, detail, color)
        detail = "LR / KNN / SVM / RF" if letter == "A" else "预测机型"
        pill(116, y - 6, 28, 14, decision, detail, "#F3F5F6")
        arrow(62, 64, 71, y + 1, navy, lw=.95)
        arrow(109, y + 1, 115, y + 1, navy, lw=.95)
        arrow(144, y + 1, 151, y + 1, navy, lw=.95)
        pill(151, y - 6, 25, 14, "机型标签", "LABEL", "#E9EEF3")

    # Minimal plain-language note clarifies that these are alternatives.
    txt(101, 5, "三条路径分别训练与评价，不进行预测融合", size=8.2, color=muted)
    ax.plot([4, 176], [9, 9], color="#DCE3EA", lw=.7)

    out = ROOT / "reports/standalone_figures"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ["png", "svg"]:
        p = out / f"fig02_clear_method_flow.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=.08)
        files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    (out / "fig02_clear_method_flow_provenance.json").write_text(
        json.dumps({"source_image": str(image_path.relative_to(ROOT)), "files": files,
                    "models_run": False, "image_modified": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("Saved clear method flow.")


if __name__ == "__main__":
    main()
