"""Publication figures from frozen CSV/JSON evidence, without model execution.

All reported scores remain unchanged. Scale stratification is descriptive,
post-hoc analysis of saved predictions, not a new experiment or a causal test.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


COLORS = {"handcrafted": "#3D678F", "siglip2": "#4B8A78", "qwen": "#BF783E"}
INK = "#263443"
GREY = "#667584"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def build_figures(root, run):
    os.environ.setdefault("MPLCONFIGDIR", str(root / "outputs/paper_work/matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, patches
    from matplotlib.colors import Normalize

    # Use a real CJK font; never silently ship missing Chinese glyphs.
    available = {font.name for font in font_manager.fontManager.ttflist}
    fonts = [name for name in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "SimSun"] if name in available]
    if not fonts:
        raise RuntimeError("Install a CJK font (Microsoft YaHei / Noto Sans CJK SC) before rendering figures")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": fonts + ["DejaVu Sans"],
                         "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.edgecolor": "#AAB5BF",
                         "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "svg.fonttype": "path",
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    out = root / "reports/paper_figures"
    out.mkdir(parents=True, exist_ok=True)
    measured_path = run / "evaluation/measured_results.json"
    measured = read(measured_path)
    results = {r["model"]: r for r in measured["results"]}
    manifest = pd.read_csv(run / "manifest.csv", low_memory=False)
    clean = pd.read_csv(root / "data/delivery/uav_dataset_v3/clean_manifest.csv", low_memory=False)
    protocol = read(run / "protocol.json")
    test = manifest.loc[manifest.split.eq("holdout")].copy()
    ids = ["handcrafted_tuned", "siglip2_tuned", "qwen3_vl_lora_standard"]
    shortnames = {"handcrafted_tuned": "手工＋调优 KNN", "siglip2_tuned": "SigLIP2＋调优 KNN",
                  "qwen3_vl_lora_standard": "Qwen 标准 LoRA"}
    def category(model):
        return "handcrafted" if model.startswith("handcrafted") else "siglip2" if model.startswith("siglip2") else "qwen"
    def color(model): return COLORS[category(model)]
    def style(ax, axis="x"):
        ax.set_axisbelow(True)
        ax.grid(axis=axis, color="#E4E9EE", linewidth=.65)
        ax.tick_params(length=0, pad=5)
    files = []
    def save(fig, name):
        for ext in ["png", "svg"]:
            path = out / (name + "." + ext)
            fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=.13)
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
            files.append({"path": str(path.relative_to(root)).replace("\\", "/"),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)
    def box(ax, xy, wh, title, text, c=COLORS["handcrafted"], fs=11):
        x,y=xy; w,h=wh
        ax.add_patch(patches.FancyBboxPatch((x,y), w,h, boxstyle="round,pad=0.006,rounding_size=0.020",
                                          facecolor=c+"10", edgecolor=c, linewidth=1.1))
        ax.text(x+.020, y+h-.045, title, fontsize=fs, weight="bold", va="top", color=c)
        ax.text(x+.020, y+h-.14, text, fontsize=fs-.5, va="top", linespacing=1.6)
    def arrow(ax, start, end, c=GREY, **kwargs):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle":"-|>","color":c,"lw":1.2, **kwargs})

    # 1. Explain the actual parallel comparisons; do not suggest model fusion.
    fig, ax=plt.subplots(figsize=(9.4,5.8)); ax.set(xlim=(-.012,1.012),ylim=(0,1)); ax.axis("off")
    box(ax,(.01,.36),(.18,.28),"图像输入","标注框裁剪\n固定清单与标签\n12 个作者类别")
    branches=[(.69,"A  手工视觉表示","HOG / HSV / LBP\n1,831 维 → 四类分类器",COLORS["handcrafted"]),
              (.37,"B  冻结视觉表示","SigLIP2 图像编码器\n768 维 → 四类分类器",COLORS["siglip2"]),
              (.05,"C  视觉语言模型","Qwen3-VL-2B\n零样本 / 标准与加权 QLoRA",COLORS["qwen"])]
    for y,title,txt,c in branches:
        box(ax,(.28,y),(.40,.24),title,txt,c)
        arrow(ax,(.19,.50),(.28,y+.12),c)
        arrow(ax,(.68,y+.12),(.77,.50),c)
    box(ax,(.77,.33),(.22,.35),"冻结后统一测试","相同 5,904 张图像\nAccuracy / Macro-F1\n混淆矩阵与错误分析")
    ax.text(.29,.985,"训练集内选参／按预设参数训练；三条路径独立比较",fontsize=10,va="top",color=GREY)
    save(fig,"fig01_framework")

    # 2. Keep web weak labels and synthetic material visibly outside main training.
    fig,ax=plt.subplots(figsize=(9.4,5.6)); ax.set(xlim=(-.012,1.012),ylim=(0,1)); ax.axis("off")
    box(ax,(.01,.65),(.24,.28),"公开作者数据","Drone-type-Set：2,973\nVisioDECT：20,924",fs=10)
    box(ax,(.33,.65),(.28,.28),"主清单审核与裁剪","原七类保留 2,720 张\n新增五类保留 17,290 张\n合计 20,010 张真实裁剪")
    box(ax,(.70,.67),(.28,.26),"固定测试集","5,904 张，约 29.5%\n覆盖 8 类；冻结后评价",COLORS["qwen"])
    box(ax,(.70,.28),(.28,.28),"训练集与内部验证","14,106 张，约 70.5%\n传统模型：两折分组验证\nQwen：无独立验证集")
    arrow(ax,(.25,.80),(.33,.80)); arrow(ax,(.61,.80),(.70,.80)); arrow(ax,(.61,.70),(.70,.43))
    box(ax,(.01,.17),(.27,.29),"独立采集材料","网页爬虫：179 原图\n135 弱标签裁剪＋4 身份参考\n未进入主监督训练与测试",GREY,9.5)
    ax.text(.35,.49,"按完整约束组分配\n随机种子 42\nCV 计分 6 类／测试 8 类",fontsize=9,va="top",linespacing=1.8)
    ax.text(.35,.20,"约束组尚未认证为独立飞行或独立拍摄来源。",fontsize=9,color=GREY)
    ax.text(.02,.04,"第三方 8,001 张合成原图为早期独立辅助材料，不并入主实验。",fontsize=9,color=GREY)
    save(fig,"fig02_data_protocol")

    # 3. Category balance and original resolution describe different data issues.
    fig,(a,b)=plt.subplots(1,2,figsize=(10.3,4.1),gridspec_kw={"width_ratios":[1.45,1]},layout="constrained")
    g=manifest.groupby(["label","split"]).size().unstack(fill_value=0).reindex(range(12),fill_value=0)
    pos=np.arange(12)
    a.bar(pos,g.development,color=COLORS["handcrafted"],label="训练集",width=.68)
    a.bar(pos,g.holdout,bottom=g.development,color="#B7CADB",label="测试集",width=.68)
    a.set_xticks(pos,[f"C{i:02}" for i in pos],rotation=45,ha="right")
    a.set(ylabel="样本数",xlabel="类别编号（名称见表8）",title="(a) 类别与集合分布")
    a.legend(frameon=False,ncol=2,loc="upper left"); style(a,"y")
    short=clean[["width","height"]].min(axis=1)
    edges=[0,24,32,48,64,96,np.inf]; bn=["<24","24–31","32–47","48–63","64–95","≥96"]
    counts=pd.cut(short,edges,right=False,labels=bn).value_counts(sort=False)
    b.bar(range(6),counts.values,color=COLORS["siglip2"],width=.65)
    for i,n in enumerate(counts): b.text(i,n+80,f"{n:,}",ha="center",fontsize=8)
    b.set_xticks(range(6),bn,rotation=30,ha="right")
    b.set(ylabel="样本数",xlabel="原生裁剪短边 / 像素",title="(b) 目标尺寸分布",ylim=(0,counts.max()*1.18)); style(b,"y")
    save(fig,"fig03_dataset_profile")

    # 4. Human-readable horizontal labels and consistent family colors.
    names={"logistic":"逻辑回归","knn":"KNN","svm":"RBF-SVM","forest":"随机森林","tuned":"调优 KNN"}
    labels={f"{r}_{k}":f"{'手工' if r=='handcrafted' else 'SigLIP2'}＋{v}" for r in ["handcrafted","siglip2"] for k,v in names.items()}
    labels.update({"qwen3_vl_zero_shot":"Qwen 零样本","qwen3_vl_lora_standard":"Qwen 标准 LoRA",
                   "qwen3_vl_lora_sqrt_class_balanced":"Qwen 加权 LoRA"})
    order=sorted(results,key=lambda m:results[m]["metrics"]["macro_f1_supported_labels"],reverse=True)
    fig,axes=plt.subplots(1,2,figsize=(9.4,5.4),sharey=True,layout="constrained")
    for ax,key,title in zip(axes,["accuracy","macro_f1_supported_labels"],["(a) 测试准确率","(b) 测试 Macro-F1（八类）"]):
        values=[results[m]["metrics"][key]*100 for m in order]
        ax.barh(np.arange(len(order)),values,color=[color(m) for m in order],height=.62)
        for i,value in enumerate(values): ax.text(value+1,i,f"{value:.1f}",va="center",fontsize=9)
        ax.set(xlim=(0,86),xlabel="分数 / %",title=title); style(ax)
        ax.set_yticks(np.arange(len(order)),[labels[m] for m in order])
    axes[0].invert_yaxis()
    save(fig,"fig04_model_comparison")

    # 5. Show exactly the eight combinations evaluated, with a shared colorbar.
    grids={rep:pd.read_csv(run/"models"/rep/"grid_search.csv") for rep in ["handcrafted","siglip2"]}
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.3),layout="constrained")
    low=min(x.mean_test_score.min() for x in grids.values())*100
    high=max(x.mean_test_score.max() for x in grids.values())*100
    for ax,(rep,grid),title in zip(axes,grids.items(),["(a) 手工表示","(b) SigLIP2 表示"]):
        mat=grid.pivot(index="param_model__weights",columns="param_model__n_neighbors",values="mean_test_score").reindex(["uniform","distance"])*100
        im=ax.imshow(mat,cmap="Blues",vmin=low-1,vmax=high+1,aspect="auto")
        for i in range(2):
            for j in range(4):
                v=mat.iloc[i,j]
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=10,color="white" if v>(low+high)/2 else INK)
        best=np.unravel_index(np.argmax(mat.to_numpy()),mat.shape)
        ax.add_patch(patches.Rectangle((best[1]-.49,best[0]-.49),.98,.98,fill=False,edgecolor=COLORS["qwen"],lw=2))
        ax.set_xticks(range(4),[3,5,9,15]); ax.set_yticks(range(2),["均匀投票","距离加权"])
        ax.set(xlabel="邻居数 k",title=title)
    fig.colorbar(im,ax=axes,label="两折 CV Macro-F1 / %",shrink=.85)
    save(fig,"fig05_grid_search")

    # 6. Fine-tuning comparisons and real logged training loss, not validation loss.
    variants=["qwen3_vl_zero_shot","qwen3_vl_lora_standard","qwen3_vl_lora_sqrt_class_balanced"]
    fig,(a,b)=plt.subplots(1,2,figsize=(10.3,3.8),layout="constrained")
    x=np.arange(3)
    for off,key,c,label in [(-.18,"accuracy",COLORS["handcrafted"],"Accuracy"),(.18,"macro_f1_supported_labels",COLORS["qwen"],"Macro-F1")]:
        val=[results[m]["metrics"][key]*100 for m in variants]
        a.bar(x+off,val,width=.33,color=c,label=label)
        for px,v in zip(x+off,val):a.text(px,v+1,f"{v:.1f}",ha="center",fontsize=9)
    a.set_xticks(x,["零样本","标准 LoRA","加权 LoRA"])
    a.set(ylabel="测试分数 / %",ylim=(0,60),title="(a) 适配策略对照"); a.legend(frameon=False); style(a,"y")
    for variant,c,label in [("standard",COLORS["handcrafted"],"标准 LoRA"),("sqrt_class_balanced",COLORS["qwen"],"加权 LoRA")]:
        history=pd.read_csv(run/"models/qwen3_vl"/variant/"training_history.csv").dropna(subset=["loss"])
        b.plot(history.step,history.loss,lw=1.3,color=c,label=label,alpha=.9)
    b.set(xlabel="优化步",ylabel="日志训练损失",title="(b) 单轮训练过程"); style(b,"y"); b.legend(frameon=False)
    save(fig,"fig06_adaptation")

    # 7. Only rows with test support, retaining every possible predicted class.
    fig,axes=plt.subplots(2,1,figsize=(9.0,7.0),layout="constrained")
    present=sorted(test.label.unique())
    for ax,model,title in zip(axes,[ids[0],ids[2]],["(a) 手工＋调优 KNN","(b) Qwen 标准 LoRA"]):
        m=results[model]["metrics"]
        full=np.asarray(m["confusion_matrix_with_invalid"])
        matrix=full[present,:]
        percentages=matrix/matrix.sum(axis=1,keepdims=True)*100
        im=ax.imshow(percentages,cmap="Blues",vmin=0,vmax=100,aspect="auto")
        for i in range(len(present)):
            for j in range(13):
                v=percentages[i,j]
                if matrix[i,j]:ax.text(j,i,f"{v:.0f}" if v>=1 else "<1",fontsize=9,ha="center",va="center",color="white" if v>=55 else INK)
        ax.set_yticks(range(len(present)),[f"C{v:02}  (n={int(matrix[i].sum()):,})" for i,v in enumerate(present)])
        ax.set_xticks(range(13),[f"C{i:02}" for i in range(12)]+["无效"])
        ax.set(title=title,xlabel="预测类别",ylabel="真实类别")
    fig.colorbar(im,ax=axes,label="占该真实类别的比例 / %",shrink=.8,pad=.02)
    save(fig,"fig07_confusion_comparison")

    # 8. Per-class metrics show where average performance comes from.
    fig,ax=plt.subplots(figsize=(10.3,2.9),layout="constrained")
    values=np.array([[results[m]["metrics"]["classification_report"][protocol["classes"][c]]["f1-score"]*100 for c in present] for m in ids])
    im=ax.imshow(values,cmap="Blues",vmin=0,vmax=100,aspect="auto")
    for i in range(3):
        for j in range(len(present)):
            ax.text(j,i,f"{values[i,j]:.1f}",ha="center",va="center",fontsize=10,color="white" if values[i,j]>55 else INK)
    ax.set_yticks(range(3),[shortnames[m] for m in ids]); ax.set_xticks(range(len(present)),[f"C{c:02}" for c in present])
    ax.set_xlabel("测试中有真实样本的类别（名称见表8）")
    fig.colorbar(im,ax=ax,label="逐类 F1 / %",shrink=.9)
    save(fig,"fig08_class_f1")

    # 9. No inference: join frozen predictions with native dimensions.
    scale_rows=[]
    test["native_short_side"]=test[["width","height"]].min(axis=1)
    test["size_bin"]=pd.cut(test.native_short_side,edges,right=False,labels=bn)
    counts=test.size_bin.value_counts(sort=False)
    fig,(a,b)=plt.subplots(2,1,figsize=(8.6,4.8),sharex=True,gridspec_kw={"height_ratios":[2.4,1]},layout="constrained")
    for model in ids:
        prediction=pd.read_csv(run/"evaluation"/(model+"_predictions.csv"))
        joined=test[["sample_id","label","size_bin"]].merge(prediction[["sample_id","label","predicted_label"]],on=["sample_id","label"],validate="one_to_one")
        if len(joined)!=len(test):raise ValueError("Scale analysis lost saved test predictions")
        joined["correct"]=joined.label.eq(joined.predicted_label)
        score=joined.groupby("size_bin",observed=False).correct.mean().reindex(bn)
        a.plot(range(6),score*100,marker="o",ms=4,lw=1.5,color=color(model),label=shortnames[model])
        for label,n,v in zip(bn,counts,score):scale_rows.append({"model":model,"short_side_bin":label,"samples":int(n),"accuracy":float(v)})
    a.set(ylabel="测试准确率 / %",ylim=(0,105),title="(a) 固定测试集的尺度分层诊断");a.legend(frameon=False,ncol=1,fontsize=9);style(a,"y")
    b.bar(range(6),counts.values,color="#AFBDC9",width=.55)
    for i,n in enumerate(counts):b.text(i,n+15,str(n),ha="center",fontsize=8)
    b.set_xticks(range(6),bn);b.set(ylabel="样本数",xlabel="原生目标短边 / 像素",ylim=(0,counts.max()*1.3),title="(b) 各区间测试支持数");style(b,"y")
    save(fig,"fig10_scale_diagnostics")
    pd.DataFrame(scale_rows).to_csv(out/"scale_diagnostics.csv",index=False,encoding="utf-8-sig")

    # 10. The importance belongs to the forest, not to the best KNN model.
    importance=pd.read_csv(run/"models/handcrafted/forest_importance.csv").head(12).iloc[::-1]
    fig,ax=plt.subplots(figsize=(8.4,4.3),layout="constrained")
    ax.barh(importance.feature,importance.impurity_importance*100,color=COLORS["handcrafted"],height=.65)
    ax.set(xlabel="随机森林不纯度重要性 / %",xlim=(0,importance.impurity_importance.max()*118))
    for i,v in enumerate(importance.impurity_importance*100):ax.text(v+.025,i,f"{v:.2f}",va="center",fontsize=8)
    style(ax);save(fig,"fig09_forest_importance")

    receipt={"generated_from_saved_evidence_only":True,"models_run":False,
             "freeze_hash":measured["freeze_hash"],"test_samples":len(test),
             "metric_source_sha256":hashlib.sha256(measured_path.read_bytes()).hexdigest(),
             "figure_font":fonts[0],"figure_groups":10,"files":files,
             "scale_analysis":"Post-hoc descriptive, class/source distributions vary by size; not causal, no filtering or tuning."}
    (out/"figure_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return receipt
