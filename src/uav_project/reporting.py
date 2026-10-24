"""Render actual results into analysis inputs; never invent paper findings."""
from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import *


def export_results():
    setup()
    if not (OUT/"final_metrics.csv").exists():
        raise RuntimeError("尚无最终结果，请先运行10_evaluate.py")
    df=pd.read_csv(OUT/"final_metrics.csv")
    fig,ax=plt.subplots(figsize=(12,5))
    df.pivot(index="model",columns="test",values="f1").plot.bar(ax=ax,ylim=(0,1),rot=35)
    ax.set_ylabel("F1 (positive=drone)"); fig.tight_layout(); fig.savefig(OUT/"figures"/"model_comparison.png",dpi=180); plt.close(fig)
    imp=pd.read_csv(OUT/"feature_importance.csv").head(20).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8,6)); ax.barh(imp.feature,imp.importance)
    ax.set_title("Random forest impurity importance (descriptive, not causal)")
    fig.tight_layout(); fig.savefig(OUT/"figures"/"feature_importance.png",dpi=160); plt.close(fig)
    for p in (OUT/"models").glob("*/history.csv"):
        history=pd.read_csv(p); fig,axes=plt.subplots(1,2,figsize=(9,3))
        axes[0].plot(history.epoch,history.training_loss); axes[0].set_title("Training loss")
        axes[1].plot(history.epoch,history.validation_f1); axes[1].set_title("Validation F1")
        fig.suptitle(p.parent.name); fig.tight_layout(); fig.savefig(OUT/"figures"/(p.parent.name+"_learning.png"),dpi=150); plt.close(fig)
    summary=read_json(OUT/"deep.json")
    repeat_names=[r["name"] for r in summary["repeats"]]
    df[df.model.isin(repeat_names)].groupby("test")[["f1","accuracy"]].agg(["mean","std"]).to_csv(OUT/"deep_repeat_summary.csv")
    # CSV is embedded as a literal block to avoid an extra tabulate dependency.
    text='''# 实验结果回传材料（非最终论文）\n\n本文件只整理已运行的指标，不预先宣称提出新算法或优于现有方法。\n\n## 实验范围\n当前标签为无人机与鸟类，不支持推论为所有非无人机目标，也不完成目标框定位。\n公开训练集取自配置中列出的基准；外部候选由 Wikimedia 公共页面采集，经人工审核。\n外部来源不同不自动保证不存在预训练重叠；应在论文局限性中说明。\n\n## 数据与划分\n详见 dataset_counts.csv、data_protocol.json、cleaning_events.csv 和 manifest.csv。\n公开数据按组近似7:3划分开发集/测试集；开发集再分训练/验证；记录实际比例。\n\n## 结果\n```csv\n'''+df.to_csv(index=False)+'''```\n\n## 待基于证据完成的讨论\n- 对比绝对外部F1、召回率和组bootstrap区间，不能只比较性能下降。\n- 分析重复种子结果、无增强与冻结主干消融。\n- 检查错误图片是否与背景、尺度、遮挡有关；未标注因素不能直接当成已验证结论。\n- 分析交叉验证、训练/验证差距及过拟合，保留失败实验日志。\n- 四个传统模型和深度网络同时改变了特征表示与分类器，差异不能全归因于分类器。\n\n## 可复现材料\n见 environment.json、配置、固定划分、网格搜索表、训练历史和protocol_lock.json。\n\n## 图表\nfigures/model_comparison.png：两类测试集的模型F1对比。\nfigures/feature_importance.png：随机森林特征重要性，非因果解释。\n各模型对应混淆矩阵、ROC、PR和错误样本图。\n'''
    (OUT/"results_for_paper.md").write_text(text,encoding="utf-8")
    print("报告分析材料：", OUT / "results_for_paper.md")
