"""Freeze protocol and run held-out evaluation, with group bootstrap uncertainty."""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, f1_score
from PIL import Image, ImageOps, ImageDraw
from .common import *
from .splitting import load_manifest
from .features import features
from .metrics import metrics


def file_set():
    paths=[CFG,META/"manifest.csv",OUT/"baselines.json",OUT/"deep.json"]
    paths+=list((ROOT/"src"/"uav_project").glob("*.py"))
    paths+=list((OUT/"models"/"baselines").glob("*.joblib"))
    deep=read_json(OUT/"deep.json")
    for r in deep["repeats"]+deep["ablations"]: paths.append(resolve(r["weights"]))
    return {relative(p):digest(p) for p in sorted(set(paths))}


def freeze():
    if (OUT/"protocol_lock.json").exists():
        print("已有冻结方案，不覆盖。"); return
    frame=load_manifest(); cfg=config()
    base=read_json(OUT/"baselines.json"); deep=read_json(OUT/"deep.json")
    if base["manifest_hash"]!=digest(META/"manifest.csv") or base["config_hash"]!=digest(CFG):
        raise RuntimeError("基线训练后数据或配置变更，请核对 outputs/baselines.json 与当前配置")
    for r in deep["repeats"]+deep["ablations"]:
        if r["signature"]["manifest"]!=digest(META/"manifest.csv") or r["signature"]["config"]!=digest(CFG):
            raise RuntimeError("主模型训练后数据或配置变更")
    save_json(OUT/"protocol_lock.json",{"time":now(),"files":file_set(),"threshold":.5,
         "primary_metric":"F1 positive=drone", "main_model":deep["selected"]["name"],
         "external_policy":"held-out evaluation only; no adaptation", "bootstrap":"resample image groups, not individual correlated frames"})
    print("冻结清单：", OUT / "protocol_lock.json")
    print("已冻结：后续不能再修改数据、特征或模型来追逐测试集分数。")


def interval(frame,score,repeats,seed):
    rng=np.random.default_rng(seed); groups=frame.group_id.unique()
    indices=[np.where(frame.group_id.to_numpy()==g)[0] for g in groups]
    values=[]; y=frame.label.astype(int).to_numpy(); pred=score>=.5
    for _ in range(repeats):
        idx=np.concatenate([indices[j] for j in rng.integers(0,len(groups),len(groups))])
        if len(np.unique(y[idx]))<2: continue
        values.append(f1_score(y[idx],pred[idx],zero_division=0))
    return [float(v) for v in np.percentile(values,[2.5,97.5])] if values else [None,None]


def plots(name,domain,frame,score):
    folder=OUT/"figures"; folder.mkdir(exist_ok=True)
    prefix=name+"_"+domain
    for suffix,display in [("confusion",ConfusionMatrixDisplay),("roc",RocCurveDisplay),("pr",PrecisionRecallDisplay)]:
        fig,ax=plt.subplots(figsize=(5,4))
        if suffix=="confusion": display.from_predictions(frame.label,score>=.5,labels=[0,1],display_labels=["bird","drone"],ax=ax,colorbar=False)
        else: display.from_predictions(frame.label,score,ax=ax)
        ax.set_title(prefix); fig.tight_layout(); fig.savefig(folder/(prefix+"_"+suffix+".png"),dpi=160); plt.close(fig)
    wrong=frame.copy(); wrong["probability"]=score; wrong=wrong[(score>=.5)!=frame.label.astype(bool).to_numpy()]
    wrong=wrong.head(12)
    if len(wrong):
        canvas=Image.new("RGB",(800,660),"white"); draw=ImageDraw.Draw(canvas)
        for k,(_,r) in enumerate(wrong.iterrows()):
            x=(k%4)*200; y=(k//4)*220
            with Image.open(resolve(r.clean_path)) as im: canvas.paste(ImageOps.contain(im.convert("RGB"),(195,175)),(x,y))
            draw.text((x+2,y+180),str(r.sample_id),fill="black")
            draw.text((x+2,y+197),f'y={r.label} p={r.probability:.3f}',fill="black")
        canvas.save(folder/(prefix+"_errors.jpg"))


def evaluate():
    lock=read_json(OUT/"protocol_lock.json")
    if lock["files"]!=file_set(): raise RuntimeError("冻结后代码/配置/权重发生变化，停止评估")
    if (OUT/"final_metrics.csv").exists():
        print("最终评估已完成，直接复用结果，不重新选模型。"); return
    import torch
    from .deep_model import network,predict
    cfg=config(); frame=load_manifest(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _,r in frame.iterrows():
        if digest(resolve(r.clean_path))!=r.clean_sha256: raise RuntimeError("清洗图片发生修改："+r.sample_id)
    predictors={}
    for p in sorted((OUT/"models"/"baselines").glob("*.joblib")): predictors[p.stem]=("baseline",p)
    deep=read_json(OUT/"deep.json")
    for r in deep["repeats"]+deep["ablations"]: predictors[r["name"]]=("deep",resolve(r["weights"]))
    pred_dir=OUT/"predictions"; pred_dir.mkdir(exist_ok=True)
    table=[]
    for domain in ["public_test","external_test"]:
        part=frame[frame.split.eq(domain)].reset_index(drop=True); X,_=features(part)
        for name,(kind,path) in predictors.items():
            if kind=="baseline": score=joblib.load(path).predict_proba(X)[:,1]
            else:
                model=network(False).to(device); model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
                score=predict(model,part,cfg,device); del model
                if device.type=="cuda": torch.cuda.empty_cache()
            low,high=interval(part,score,cfg["bootstrap_repeats"],cfg["seed"])
            table.append({"model":name,"test":domain,"n":len(part),**metrics(part.label,score),"f1_ci_low":low,"f1_ci_high":high})
            records=part[["sample_id","label","group_id","source_url","clean_path"]].copy()
            records["probability"]=score; records["prediction"]=(score>=.5).astype(int)
            records.to_csv(pred_dir/(name+"_"+domain+".csv"),index=False)
            plots(name,domain,part,score)
            print(name,domain,table[-1]["f1"],flush=True)
    result=pd.DataFrame(table)
    # Write completion marker only after all evaluation output exists.
    result.to_csv(OUT/"final_metrics.csv",index=False)
    gap=result.pivot(index="model",columns="test",values="f1")
    gap["public_minus_external_f1"]=gap.public_test-gap.external_test
    gap.to_csv(OUT/"domain_gap.csv")
    save_json(OUT/"evaluation_complete.json",{"time":now(),"lock":digest(OUT/"protocol_lock.json"),"note":"A smaller gap alone does not imply a better classifier; inspect absolute external F1 and uncertainty."})
    print("最终测试结果：", OUT / "final_metrics.csv")

