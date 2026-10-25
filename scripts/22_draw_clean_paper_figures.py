"""Draw clean, publication-style overview and method figures."""
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
    available = {x.name for x in font_manager.fontManager.ttflist}
    cjk = next((x for x in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if x in available), None)
    if not cjk:
        raise RuntimeError("No Chinese font available")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [cjk, "DejaVu Sans"],
        "font.size": 9, "svg.fonttype": "none", "axes.unicode_minus": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    out = ROOT / "reports/research_figures"
    out.mkdir(parents=True, exist_ok=True)
    ink, muted, rule, blue, teal = "#263238", "#64727A", "#D4DBDF", "#2F6288", "#357D72"

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(out / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=.12)
        plt.close(fig)

    def box(ax, x, y, w, h, title, body, color=blue, title_size=11, body_size=9.5):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white", edgecolor=color, linewidth=1.2,
        ))
        ax.text(x + .03*w, y + h - .27*h, title, ha="left", va="center",
                fontsize=title_size, fontweight="bold", color=color)
        ax.plot([x + .03*w, x + .97*w], [y + h - .43*h, y + h - .43*h],
                color=rule, linewidth=.75)
        ax.text(x + .03*w, y + .25*h, body, ha="left", va="center",
                fontsize=body_size, color=ink, linespacing=1.35)

    def arrow(ax, x1, y1, x2, y2, color=muted, dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops={"arrowstyle": "-|>", "color": color,
                                "lw": 1.05, "mutation_scale": 10,
                                "linestyle": "--" if dashed else "-"})

    # Figure 1: research framing.
    fig, ax = plt.subplots(figsize=(180/25.4, 66/25.4))
    ax.set(xlim=(0, 180), ylim=(0, 66)); ax.axis("off")
    ax.text(0, 63, "研究问题与比较框架", fontsize=14, fontweight="bold", color=ink, va="top")
    columns = [(0, 51, "视觉任务", blue), (64, 116, "比较路径", teal), (129, 180, "评价问题", "#8A6A32")]
    for start, end, heading, color in columns:
        ax.text(start, 55, heading, fontsize=10.5, fontweight="bold", color=color, va="center")
        ax.plot([start, end], [52.5, 52.5], color=color, lw=1.0)
    ax.text(0, 43, "已知目标框裁剪", fontsize=11, color=ink)
    ax.text(0, 34, "识别 12 种无人机机型", fontsize=11, color=ink)
    ax.text(0, 25, "旋翼布局、机身比例、\n相机与支架等局部线索", fontsize=9.4, color=muted, linespacing=1.35)
    ax.text(0, 10, "低分辨率、视角变化与相似机型\n共同提高细粒度判别难度", fontsize=9.4, color=muted, linespacing=1.35)
    for y, title, body, color in [(42, "手工视觉表示", "HOG + HSV + LBP\n四类传统分类器", blue),
                                   (27, "冻结预训练表示", "SigLIP2 +\n四类传统分类器", teal),
                                   (12, "视觉语言模型适配", "Qwen3-VL + QLoRA\n标准 / 类别加权", "#8A6A32")]:
        ax.add_patch(patches.Rectangle((64, y-7), 52, 14, facecolor="white", edgecolor=color, lw=1.05))
        ax.text(66.5, y+3.2, title, fontsize=7.6, fontweight="bold", color=color, va="center")
        ax.text(66.5, y-2.8, body, fontsize=6.9, color=muted, va="center", linespacing=1.05)
    for y, q in [(42, "RQ1  哪种表示更适合机型分类？"),
                 (27, "RQ2  监督适配带来多大变化？"),
                 (12, "RQ3  目标尺度如何影响识别？")]:
        ax.text(129, y+1.2, q, fontsize=9.6, color=ink, va="center")
        ax.text(129, y-2.8, "同一冻结测试清单，统一指标比较", fontsize=8.7, color=muted, va="center")
    for y in (42, 27, 12):
        arrow(ax, 51, y, 63, y, blue)
        arrow(ax, 117, y, 128, y, teal)
    ax.plot([0, 180], [4, 4], color=rule, lw=.7)
    ax.text(0, 1.2, "研究对象：目标框已知条件下的机型分类；模型路径独立比较，不进行预测融合。", fontsize=8.5, color=muted)
    save(fig, "fig01_motivation")

    # Figure 2: simple three-lane method diagram.
    fig, ax = plt.subplots(figsize=(180/25.4, 86/25.4))
    ax.set(xlim=(0, 180), ylim=(0, 86)); ax.axis("off")
    ax.text(0, 83, "从目标裁剪到统一测试评价", fontsize=11.5, fontweight="bold", color=ink, va="top")
    ax.text(0, 74.5, "三条路径共享输入与测试样本；只有 QLoRA 路径更新可训练参数", fontsize=7.3, color=muted)
    box(ax, 0, 30, 24, 28, "输入", "目标框裁剪\nRGB 图像\n12 类标签", blue, 8.2, 6.9)
    lanes = [(61, "A  手工表示", "HOG / HSV / LBP\n1,831 维", "LR · KNN · SVM · RF", blue),
             (44, "B  冻结视觉表示", "SigLIP2\n768 维", "LR · KNN · SVM · RF", teal),
             (27, "C  视觉语言适配", "Qwen3-VL-2B\n视觉主干冻结", "4-bit NF4 + LoRA\n标准 / 加权", "#8A6A32")]
    for y, title, rep, decision, color in lanes:
        box(ax, 34, y-5.5, 43, 11, title, rep, color, 7.7, 6.4)
        box(ax, 84, y-5.5, 42, 11, "决策模块", decision, color, 7.7, 6.4)
        arrow(ax, 24, 44, 33.5, y, color)
        arrow(ax, 77.5, y, 83.5, y, color)
    box(ax, 138, 30, 42, 28, "统一评价", "固定测试 5,904 张\nAccuracy / Macro-F1\n逐类混淆\n尺度分层", blue, 8.2, 6.8)
    for y, color in [(61, blue), (44, teal), (27, "#8A6A32")]:
        arrow(ax, 126, y, 137.5, 44, color)
    ax.plot([29, 29], [13, 70], color=rule, lw=.8)
    ax.text(0, 9, "数据划分", fontsize=7.7, fontweight="bold", color=ink)
    ax.text(0, 4, "训练 14,106  |  测试 5,904  |  seed 42  |  约束组不跨集合", fontsize=6.7, color=muted)
    ax.text(95, 9, "监督信号", fontsize=7.7, fontweight="bold", color=ink)
    ax.text(95, 4, "答案 token 计算交叉熵；提示词 token 屏蔽", fontsize=6.7, color=muted)
    save(fig, "fig02_architecture")

    receipt = {"files": []}
    for name in ["fig01_motivation", "fig02_architecture"]:
        for ext in ["png", "svg"]:
            p = out / f"{name}.{ext}"
            receipt["files"].append({"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    (out / "clean_figure_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved clean fig01 and fig02.")


if __name__ == "__main__":
    main()
