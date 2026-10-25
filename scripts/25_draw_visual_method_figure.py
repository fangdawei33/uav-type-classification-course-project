"""Draw a visual, paper-style method schematic using one genuine UAV crop."""
from pathlib import Path
import base64
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
    ink = "#202A35"; muted = "#6D7782"; navy = "#315F8A"
    blue = "#9FC6E7"; green = "#A8D2B6"; orange = "#F4C6A5"; yellow = "#F5DD8B"

    def text(x, y, s, size=9, color=ink, weight="normal", ha="center", va="center", **kw):
        return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                       ha=ha, va=va, linespacing=1.2, **kw)

    def rounded(x, y, w, h, fc, ec="none", radius=.12, shadow=True, lw=1.0):
        p = patches.FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0.012,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw)
        if shadow:
            p.set_path_effects([patheffects.SimplePatchShadow(offset=(1.0, -1.0), alpha=.18), patheffects.Normal()])
        ax.add_patch(p)
        return p

    def arrow(x1, y1, x2, y2, color=ink, lw=1.2, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw,
                                "mutation_scale": 11, "linestyle": "--" if dashed else "-"})

    # The genuine image is the visual anchor of the whole schematic.
    rounded(5, 36, 31, 42, "white", ec="#D5DCE3", radius=.8, shadow=False, lw=.8)
    ax.imshow(image, extent=(8, 33, 49, 73), interpolation="none", zorder=3)
    text(20.5, 43.5, "真实目标裁剪", size=9, color=muted)
    text(20.5, 80.5, "输入图像", size=11, weight="bold", color=navy)
    arrow(36.5, 57, 45, 57, navy, lw=1.6)

    # Dashed research-method enclosure, with a soft central visual encoder.
    outline = patches.FancyBboxPatch((45, 14), 87, 72,
        boxstyle="round,pad=0.015,rounding_size=1.2", facecolor="#F7FAFC",
        edgecolor="#8FB5D5", linewidth=1.25, linestyle=(0, (5, 3)))
    ax.add_patch(outline)
    text(88.5, 82.5, "视觉表示与任务适配", size=11.5, weight="bold", color=navy)

    # Feature extraction hub: an organic polygon rather than a plain box.
    hub = patches.Polygon([(58, 65), (68, 72), (78, 68), (78, 46), (68, 42), (58, 49)],
                          closed=True, facecolor=blue, edgecolor="none")
    hub.set_path_effects([patheffects.SimplePatchShadow(offset=(1.3, -1.3), alpha=.18), patheffects.Normal()])
    ax.add_patch(hub)
    text(68, 58, "视觉\n编码", size=11, weight="bold")
    arrow(45, 57, 57, 57, navy, lw=1.2)

    # Three compact representation/adaptation pills.
    rounded(84, 64, 39, 13, blue, radius=.7)
    text(103.5, 72.5, "手工特征", size=10, weight="bold")
    text(103.5, 67.5, "HOG · HSV · LBP", size=8.3, color=ink)
    arrow(78, 63, 83, 70, navy)

    rounded(84, 43.5, 39, 13, green, radius=.7)
    text(103.5, 52, "冻结视觉编码器", size=10, weight="bold")
    text(103.5, 47, "SigLIP2", size=8.6, color=ink)
    arrow(78, 57, 83, 50, navy)

    rounded(84, 23, 39, 13, orange, radius=.7)
    text(103.5, 31.5, "视觉语言适配", size=10, weight="bold")
    text(103.5, 26.5, "Qwen3-VL + LoRA", size=8.3, color=ink)
    arrow(78, 51, 83, 29, navy, dashed=True)

    # Small visual feature tokens: a compact cue that features are extracted.
    for i, (x, fc) in enumerate(zip([51, 55.5, 60, 64.5], [blue, green, orange, yellow])):
        rounded(x, 27, 3.2, 4.5, fc, radius=.35, shadow=False)
    text(58, 22.5, "结构 / 颜色 / 纹理", size=7.4, color=muted)
    arrow(68, 42, 68, 32, muted, lw=.9, dashed=True)

    # Unified output at right, styled like the reference's prediction head.
    arrow(123.5, 70.5, 137, 70.5, navy)
    arrow(123.5, 50, 137, 50, navy)
    arrow(123.5, 29.5, 137, 29.5, navy)
    rounded(137, 61, 36, 19, yellow, radius=.8)
    text(155, 74, "传统分类器", size=10, weight="bold")
    text(155, 67.5, "LR · KNN · SVM · RF", size=8.1)
    text(155, 63.5, "→ 机型标签", size=8.2, color=navy)
    rounded(137, 38, 36, 19, green, radius=.8)
    text(155, 51, "视觉嵌入", size=10, weight="bold")
    text(155, 44.5, "SigLIP2 + 分类器", size=8.1)
    text(155, 40.5, "→ 机型标签", size=8.2, color=navy)
    rounded(137, 15, 36, 19, orange, radius=.8)
    text(155, 28, "语言侧输出", size=10, weight="bold")
    text(155, 21.5, "LABEL_00 ··· LABEL_11", size=8.1)
    text(155, 17.5, "→ 机型标签", size=8.2, color=navy)

    # Minimal footer, no counts and no explanatory paragraph.
    ax.plot([5, 175], [7, 7], color="#DCE3EA", lw=.7)
    text(90, 3.5, "图像输入 → 视觉表示 → 分类输出", size=8.2, color=muted)

    out = ROOT / "reports/standalone_figures"
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for ext in ["png", "svg"]:
        p = out / f"fig02_visual_method_reference_style.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight", pad_inches=.08)
        files.append({"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    plt.close(fig)
    (out / "fig02_visual_method_reference_style_provenance.json").write_text(
        json.dumps({"source_image": str(image_path.relative_to(ROOT)), "files": files,
                    "models_run": False, "image_modified": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print("Saved visual method figure with genuine input image.")


if __name__ == "__main__":
    main()
