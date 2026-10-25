"""Research diagrams, real adaptation ablations, and saved-prediction case studies.

No model is loaded. Diagram illustrations are schematics, not new observations.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .paper_figures import build_figures as build_base


def build_research_figures(root, run):
    out=root/"reports/research_figures"
    receipt=build_base(root,run,"reports/research_figures",font_scale=1.35,skip_diagrams=True)
    import matplotlib.pyplot as plt
    from matplotlib import patches, patheffects
    from PIL import Image
    plt.rcParams.update({"font.size":13,"axes.labelsize":13,"axes.titlesize":14,
                         "xtick.labelsize":12,"ytick.labelsize":12})
    blue="#2878B5";green="#399678";orange="#E4A23B";red="#C95659";ink="#253345";muted="#66768A"
    measured=json.loads((run/"evaluation/measured_results.json").read_text(encoding="utf-8"))
    results={x["model"]:x["metrics"] for x in measured["results"]}
    frame=pd.read_csv(run/"manifest.csv",low_memory=False)
    protocol=json.loads((run/"protocol.json").read_text(encoding="utf-8"))
    def save(fig,name):
        for ext in ["png","svg"]:
            path=out/(name+"."+ext);fig.savefig(path,dpi=360,bbox_inches="tight",pad_inches=.12)
            if ext=="svg":path.write_text("\n".join(x.rstrip() for x in path.read_text(encoding="utf-8").splitlines())+"\n",encoding="utf-8")
            receipt["files"].append({"path":path.relative_to(root).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        plt.close(fig)
    def canvas(w=1600,h=850):
        f,a=plt.subplots(figsize=(10.5,10.5*h/w));a.set(xlim=(-12,w+12),ylim=(-15,h+15));a.axis("off");return f,a
    def label(a,x,y,t,size=12,**kw):
        max_width=kw.pop("max_width",None)
        artist=a.text(x,y,t,fontsize=size,color=kw.pop("color",ink),**kw)
        if max_width:
            a.figure.canvas.draw()
            renderer=a.figure.canvas.get_renderer()
            permitted=a.transData.transform((max_width,0))[0]-a.transData.transform((0,0))[0]
            actual=artist.get_window_extent(renderer).width
            if actual>permitted:artist.set_fontsize(size*permitted/actual)
        return artist
    def card(a,x,y,w,h,title,c=blue,subtitle=None):
        p=patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=17",lw=1.3,ec=c,fc="white")
        p.set_path_effects([patheffects.withSimplePatchShadow(offset=(1.4,-1.4),alpha=.09),patheffects.Normal()]);a.add_patch(p)
        a.add_patch(patches.FancyBboxPatch((x+1,y+h-52),w-2,51,boxstyle="round,pad=0,rounding_size=16",lw=0,fc=c+"17"))
        label(a,x+w/2,y+h-26,title,12.5,color=c,weight="bold",ha="center",va="center",max_width=w-30)
        if subtitle:label(a,x+w/2,y+19,subtitle,10.5,color=muted,ha="center")
    def arrow(a,p,q,c=muted,dashed=False,lw=1.8):
        a.annotate("",xy=q,xytext=p,arrowprops={"arrowstyle":"-|>","color":c,"lw":lw,"linestyle":"--" if dashed else "-","mutation_scale":13})
    def drone(a,cx,cy,s=1,c=ink):
        for dx,dy in [(-46,-25),(46,-25),(-46,25),(46,25)]:
            a.plot([cx,cx+dx*s],[cy,cy+dy*s],c=c,lw=2.5)
            a.add_patch(patches.Ellipse((cx+dx*s,cy+dy*s),50*s,17*s,ec=c,fc="white",lw=1.5))
        a.add_patch(patches.FancyBboxPatch((cx-19*s,cy-16*s),38*s,32*s,boxstyle="round,pad=0,rounding_size=5",fc=c,ec=c))
        a.add_patch(patches.Circle((cx,cy-22*s),6*s,ec=c,fc="white",lw=1.5))
    def network(a,x,y,c=blue):
        layers=[[(-25,v) for v in [-28,0,28]],[(15,v) for v in [-42,-14,14,42]],[(55,v) for v in [-22,22]]]
        for p,q in zip(layers,layers[1:]):
            for px,py in p:
                for qx,qy in q:a.plot([x+px,x+qx],[y+py,y+qy],lw=.7,c=c,alpha=.35)
        for layer in layers:
            for dx,dy in layer:a.add_patch(patches.Circle((x+dx,y+dy),6,ec=c,fc="white",lw=1.2))
    def lock(a,x,y,c=muted):
        a.add_patch(patches.Arc((x,y+8),15,20,theta1=0,theta2=180,ec=c,lw=1.5))
        a.add_patch(patches.FancyBboxPatch((x-10,y-10),20,17,boxstyle="round,pad=0,rounding_size=2",fc="white",ec=c,lw=1.2))
    # Problem figure: an actual scientific question, without a fictitious proposed-method win.
    fig,a=canvas(1600,700)
    label(a,15,672,"低分辨率机型识别：更多像素、更多参数，是否意味着更可靠的判断？",17,weight="bold")
    card(a,20,245,425,360,"视觉证据",blue)
    drone(a,124,455,.65)
    a.add_patch(patches.Rectangle((64,398),120,110,fill=False,ec=blue,lw=1.3,ls="--"))
    arrow(a,(198,455),(257,455),blue)
    # Vector pixels explicitly depict a schematic; they are not dataset examples.
    grid=np.zeros((9,13));grid[4,3:10]=1;grid[2:7,6]=1;grid[2,2:5]=1;grid[6,8:11]=1
    a.imshow(grid,cmap="Blues",extent=(275,415,402,512),interpolation="nearest",vmin=0,vmax=1,zorder=2)
    label(a,232,338,"放大显示不补回机型细节",12,ha="center")
    label(a,232,284,"短边 < 24 px：1,262 / 20,010 张",10,ha="center",color=muted,max_width=390)
    card(a,495,245,515,360,"表示与适配",green)
    labels=[("手工局部特征",blue), ("冻结 SigLIP2",green),("Qwen + QLoRA",orange)]
    for yy,(txt,c) in zip([494,423,352],labels):
        a.add_patch(patches.FancyBboxPatch((530,yy-22),443,49,boxstyle="round,pad=0,rounding_size=8",ec=c,fc=c+"0D",lw=1))
        label(a,752,yy+2,txt,13,ha="center",va="center",color=c)
    card(a,1060,245,520,360,"需要实验回答的问题",orange)
    for yy,num,txt in [(489,"RQ1","数据与分组支持什么结论？"),(411,"RQ2","预训练表示是否优于手工特征？"),(333,"RQ3","适配与类别加权各有何作用？")]:
        label(a,1090,yy,num,12,weight="bold",color=orange);label(a,1090,yy-34,txt,11,max_width=460)
    arrow(a,(445,426),(495,426),blue);arrow(a,(1010,426),(1060,426),green)
    a.plot([25,1575],[185,185],c="#DDE4EC",lw=1)
    for x,num,txt in [(30,"01","记录原生尺度与来源"),(570,"02","统一冻结测试清单"),(1110,"03","检验实际适配组件")]:
        label(a,x,121,num,24,weight="bold",color=blue);label(a,x+100,126,txt,11,max_width=370)
    label(a,25,42,"无人机与像素图为机制示意；数量来自清单。图中不预设任何方法优胜。",10.5,color=muted)
    save(fig,"fig01_motivation")

    # Architecture: three independent paths, with actual trainable scope and answer-only loss.
    fig,a=canvas(1600,1050)
    label(a,20,1017,"评估框架与 QLoRA 训练机制",18,weight="bold")
    card(a,15,493,210,343,"目标输入",blue)
    drone(a,120,715,.95);label(a,120,610,"目标框裁剪",11,ha="center",max_width=185);label(a,120,566,"图像输入",11,ha="center",color=muted)
    # Two small comparator paths at the top.
    for y,title,c,encoder,dimension in [(793,"A  手工表示路径",blue,"HOG + HSV + LBP + q","1,831 维"),(576,"B  冻结预训练表示",green,"SigLIP2 图像编码器","768 维，L2 归一化")]:
        card(a,290,y,1005,179,title,c)
        label(a,355,y+78,encoder,12);label(a,355,y+35,dimension,10.5,color=muted)
        arrow(a,(720,y+73),(805,y+73),c)
        for k,hh in enumerate([30,60,44,72,50,23]):
            a.add_patch(patches.Rectangle((817+14*k,y+35),9,hh,ec=c,fc=c+"70",lw=.7))
        label(a,1025,y+81,"LR / KNN",11.5);label(a,1025,y+39,"SVM / RF",11.5)
    arrow(a,(225,755),(290,858),blue);arrow(a,(225,665),(290,650),green)
    # Main VLM panel.
    card(a,290,115,1005,406,"C  视觉语言模型与语言侧低秩适配",orange)
    card(a,325,309,235,132,"视觉编码器",muted);lock(a,348,328);label(a,454,345,"冻结主干",10.5,ha="center",max_width=155)
    card(a,605,309,205,132,"多模态接口",muted);label(a,707,345,"视觉 token",11,ha="center")
    card(a,855,277,300,190,"Qwen 语言模块",orange);network(a,920,347,orange);label(a,1049,343,"4-bit NF4",11,ha="center")
    arrow(a,(560,375),(605,375),orange);arrow(a,(810,375),(855,375),orange)
    a.add_patch(patches.FancyBboxPatch((893,196),222,49,boxstyle="round,pad=0,rounding_size=6",ec=red,fc="#FBEFF0",lw=1.4))
    label(a,1004,220,"ΔW = (α/r) BA",10.5,ha="center",va="center",color=red,max_width=210)
    arrow(a,(1004,245),(1004,277),red);label(a,1127,220,"仅 LoRA\n可训练",9.5,color=red,va="center")
    label(a,335,256,"监督序列",11,weight="bold")
    a.add_patch(patches.Rectangle((337,192),238,38,ec="#B3BDC8",fc="#F0F3F6",lw=1))
    label(a,455,211,"图像 / 指令 token",9,ha="center",va="center",color=muted,max_width=220)
    a.add_patch(patches.Rectangle((337,141),238,38,ec=green,fc="#EAF6F0",lw=1))
    label(a,455,160,"LABEL_xx 答案 token",9,ha="center",va="center",color=green,max_width=220)
    label(a,620,211,"屏蔽提示词损失",9.5,color=muted);label(a,620,160,"标准 / 类别加权 CE",9.5,color=green,max_width=300)
    arrow(a,(575,160),(608,160),green);arrow(a,(925,160),(1004,196),green,dashed=True)
    arrow(a,(225,565),(325,375),orange)
    card(a,1340,360,245,384,"统一冻结测试",blue)
    label(a,1462,650,"5,904 张",15,ha="center",weight="bold",color=blue)
    label(a,1462,604,"相同样本与标签",11,ha="center")
    label(a,1462,540,"Accuracy",11,ha="center");label(a,1462,493,"Macro-F1（8类）",10,ha="center",max_width=220)
    label(a,1462,414,"逐类与错误诊断",11,ha="center")
    for yy,c in [(867,blue),(649,green),(375,orange)]:arrow(a,(1295 if yy!=375 else 1155,yy),(1340,max(400,min(700,yy))),c)
    label(a,25,44,"实线：数据前向路径    虚线：训练损失作用于适配参数    锁：冻结模块    三条路径独立比较，不做预测融合",10.5,color=muted)
    save(fig,"fig02_architecture")

    # Genuine adaptation ablation: compare the already executed three configurations.
    variants=["qwen3_vl_zero_shot","qwen3_vl_lora_standard","qwen3_vl_lora_sqrt_class_balanced"]
    rows=[]
    for k in variants:
        m=results[k]
        rows.append({"model":k,"sft":k!=variants[0],"class_weighting":k==variants[2],
                     "accuracy":m["accuracy"],"macro_f1":m["macro_f1_supported_labels"]})
    pd.DataFrame(rows).to_csv(out/"adaptation_ablation.csv",index=False,encoding="utf-8-sig")
    fig,(a,b)=plt.subplots(1,2,figsize=(8.5,3.7),layout="constrained",gridspec_kw={"width_ratios":[1,1.18]})
    x=np.arange(3)
    for dx,key,c,lab in [(-.19,"accuracy",blue,"Accuracy"),(.19,"macro_f1",orange,"Macro-F1")]:
        values=[r[key]*100 for r in rows];a.bar(x+dx,values,.35,color=c,label=lab)
        for px,v in zip(x+dx,values):a.text(px,v+1,f"{v:.2f}",ha="center",fontsize=10.5)
    a.set_xticks(x,["无适配","标准适配","加权适配"]);a.set_ylim(0,65);a.set_ylabel("测试分数 / %");a.set_title("(a) 组件开关对照")
    a.legend(frameon=False,fontsize=10.5,loc="upper left");a.grid(axis="y",alpha=.18);a.set_axisbelow(True)
    present=sorted(frame.loc[frame.split.eq("holdout"),"label"].unique())
    std=results[variants[1]]["classification_report"];weighted=results[variants[2]]["classification_report"]
    delta=np.array([weighted[protocol["classes"][i]]["f1-score"]-std[protocol["classes"][i]]["f1-score"] for i in present])*100
    b.barh(range(len(present)),delta,color=[green if z>=0 else red for z in delta],height=.62)
    b.set_yticks(range(len(present)),[f"C{i:02}" for i in present]);b.invert_yaxis();b.axvline(0,c="#8390A0",lw=.9)
    for i,z in enumerate(delta):b.text(z+(.6 if z>=0 else -.6),i,f"{z:+.2f}",ha="left" if z>=0 else "right",va="center",fontsize=10)
    margin=max(5,np.max(np.abs(delta))*.24);b.set_xlim(min(0,delta.min())-margin,max(0,delta.max())+margin)
    b.set_title("(b) 加权相对标准适配的逐类变化");b.set_xlabel("F1 变化 / 百分点");b.grid(axis="x",alpha=.18);b.set_axisbelow(True)
    save(fig,"fig11_ablation")

    # Representation replacement holds each classifier family and base configuration fixed.
    clfs=["logistic","knn","svm","forest"];names=["逻辑回归","KNN","RBF-SVM","随机森林"]
    fig,a=plt.subplots(figsize=(6.5,3.6),layout="constrained")
    for offset,rep,c,lab in [(-.18,"handcrafted",blue,"手工表示"),(.18,"siglip2",green,"冻结 SigLIP2")]:
        vals=[results[rep+"_"+k]["macro_f1_supported_labels"]*100 for k in clfs]
        a.bar(np.arange(4)+offset,vals,.34,color=c,label=lab)
        for xx,v in zip(np.arange(4)+offset,vals):a.text(xx,v+.9,f"{v:.1f}",ha="center",fontsize=11)
    a.set_xticks(range(4),names);a.set_ylim(0,75);a.set_ylabel("测试 Macro-F1 / %");a.legend(frameon=False,ncol=2,fontsize=11,loc="upper left");a.grid(axis="y",alpha=.18);a.set_axisbelow(True)
    save(fig,"fig12_representation")

    fig,a=plt.subplots(figsize=(6.5,3.1),layout="constrained")
    for variant,c,lab in [("standard",blue,"标准 QLoRA"),("sqrt_class_balanced",orange,"类别加权 QLoRA")]:
        history=pd.read_csv(run/"models/qwen3_vl"/variant/"training_history.csv").dropna(subset=["loss"])
        a.plot(history.step,history.loss,c=c,lw=1.5,label=lab)
    a.set(xlabel="优化步",ylabel="日志训练损失");a.legend(frameon=False,fontsize=11);a.grid(axis="y",alpha=.18);a.set_axisbelow(True)
    save(fig,"fig14_training")

    # Qualitative results are real saved samples, selected by a fixed first-match rule.
    pred=pd.read_csv(run/"evaluation/qwen3_vl_lora_standard_predictions.csv")
    choices=[]
    for c,correct in [(2,True),(2,False),(7,False),(8,False),(9,True),(10,False)]:
        subset=pred.loc[pred.label.eq(c)&pred.predicted_label.eq(pred.label).eq(correct)]
        if subset.empty:raise ValueError("Requested qualitative stratum is empty")
        choices.append(subset.iloc[0])
    fig,axes=plt.subplots(2,3,figsize=(9.0,5.5),layout="constrained")
    records=[]
    for index,(ax,row) in enumerate(zip(axes.flat,choices),1):
        path=root/row.clean_path
        with Image.open(path) as img:
            image=np.asarray(img.convert("RGB"));width,height=img.size
        ax.imshow(image,interpolation="nearest");ax.set_xticks([]);ax.set_yticks([])
        good=int(row.label)==int(row.predicted_label);c=green if good else red
        for spine in ax.spines.values():spine.set_visible(True);spine.set_color(c);spine.set_linewidth(1.4)
        ax.set_title(f"({chr(96+index)}) {width} × {height} px",fontsize=12,pad=7)
        ax.set_xlabel(f"真实：{row.class_name}\n预测：{protocol['classes'][int(row.predicted_label)]}",fontsize=10.5,color=c,labelpad=6)
        records.append({"panel":chr(96+index),"sample_id":row.sample_id,"true_label":int(row.label),"predicted_label":int(row.predicted_label),"image_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_dataset":frame.set_index("sample_id").loc[row.sample_id,"source_dataset"],"selection":"first saved example for predeclared class/correctness stratum"})
    save(fig,"fig13_qualitative")
    pd.DataFrame(records).to_csv(out/"qualitative_cases.csv",index=False,encoding="utf-8-sig")
    receipt["figure_groups"]=len(receipt["files"])//2
    receipt["adaptation_ablation"]="Existing frozen predictions only; standard/weighted receipts differ only by variant and balance_rule in their training environment."
    receipt["qualitative_images"]="Six low-resolution analytic panels with provenance; original media are not bundled."
    (out/"figure_receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return receipt
