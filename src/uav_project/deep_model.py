"""EfficientNet-B0 fine tuning with resumable checkpoints and preregistered ablations."""
from __future__ import annotations
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageOps
from .common import *
from .splitting import load_manifest
from .metrics import metrics


class Images(Dataset):
    def __init__(self,frame,augment=False):
        self.frame=frame.reset_index(drop=True)
        transform=[transforms.Resize((224,224))]
        if augment: transform += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(8), transforms.ColorJitter(.2,.2,.2)]
        self.transform=transforms.Compose(transform+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        with Image.open(resolve(r.clean_path)) as im:
            x=self.transform(ImageOps.exif_transpose(im).convert("RGB"))
        return x,torch.tensor(float(r.label))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True


def network(pretrained=True):
    model=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[1]=nn.Linear(model.classifier[1].in_features,1)
    return model


def loader(frame,augment,cfg,shuffle=False):
    return DataLoader(Images(frame,augment),batch_size=cfg["batch_size"],shuffle=shuffle,num_workers=cfg["workers"],pin_memory=torch.cuda.is_available())


def predict(model,frame,cfg,device):
    model.eval(); values=[]
    with torch.inference_mode():
        for x,_ in loader(frame,False,cfg): values.extend(model(x.to(device)).squeeze(1).sigmoid().cpu().tolist())
    return np.asarray(values)


def train_run(name,train,val,cfg,lr,seed,augment=True,frozen=False):
    folder=OUT/"models"/name; folder.mkdir(exist_ok=True)
    signature={"manifest":digest(META/"manifest.csv"),"config":digest(CFG),"lr":lr,"seed":seed,"augment":augment,"frozen":frozen}
    if (folder/"result.json").exists():
        result=read_json(folder/"result.json")
        if result["signature"] != signature: raise RuntimeError("训练配置已改变，不能复用旧结果；请回传日志，由我整理新实验版本。")
        return result
    seed_all(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=network().to(device)
    if frozen:
        for p in model.features.parameters(): p.requires_grad=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",patience=2,factor=.3)
    scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    counts=train.label.astype(int).value_counts()
    loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts[0]/counts[1],device=device))
    history=[]; best=-1.; wait=0; start=0
    last=folder/"last.pt"
    if last.exists():
        checkpoint=torch.load(last,map_location="cpu",weights_only=False) # only our locally produced checkpoint
        if checkpoint["signature"] != signature: raise RuntimeError("断点配置不一致，停止以免混用实验")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for k,v in state.items():
                if torch.is_tensor(v): state[k]=v.to(device)
        scheduler.load_state_dict(checkpoint["scheduler"]); scaler.load_state_dict(checkpoint["scaler"])
        history=checkpoint["history"]; best=checkpoint["best"]; wait=checkpoint["wait"]; start=checkpoint["epoch"]+1
        torch.set_rng_state(checkpoint["rng"]); random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"])
        if device.type=="cuda" and checkpoint["cuda_rng"] is not None: torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    for epoch in range(start,cfg["deep_epochs"]):
        if wait>=cfg["patience"]: break
        model.train()
        if frozen: model.features.eval() # prevent BatchNorm statistics from changing in frozen backbone
        total=0.
        for x,y in loader(train,augment,cfg,True):
            x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda"):
                loss=loss_fn(model(x).squeeze(1),y)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); total+=loss.item()*len(y)
        score=metrics(val.label,predict(model,val,cfg,device)); scheduler.step(score["f1"])
        history.append({"epoch":epoch+1,"training_loss":total/len(train),"validation_f1":score["f1"],"lr":optimizer.param_groups[0]["lr"]})
        if score["f1"]>best:
            best=score["f1"]; wait=0
            torch.save(model.state_dict(),folder/"best.tmp"); (folder/"best.tmp").replace(folder/"best.pt")
        else: wait+=1
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
                    "history":history,"best":best,"wait":wait,"epoch":epoch,"signature":signature,
                    "rng":torch.get_rng_state(),"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else None},folder/"last.tmp")
        (folder/"last.tmp").replace(last)
        pd.DataFrame(history).to_csv(folder/"history.csv",index=False)
        print(name,epoch+1,history[-1],flush=True)
    model.load_state_dict(torch.load(folder/"best.pt",map_location=device,weights_only=True))
    result={"name":name,"signature":signature,"validation":metrics(val.label,predict(model,val,cfg,device)),
            "training":metrics(train.label,predict(model,train,cfg,device)),"best_epoch":max(history,key=lambda r:r["validation_f1"])["epoch"],
            "weights":relative(folder/"best.pt")}
    save_json(folder/"result.json",result)
    del model,optimizer
    if device.type=="cuda": torch.cuda.empty_cache()
    return result


def run_deep():
    protect_frozen(); cfg=config(); frame=load_manifest()
    if not torch.cuda.is_available(): raise RuntimeError("主模型训练需要GPU；可在GPU主机或Colab运行08脚本。前面数据阶段不需要GPU。")
    train=frame[frame.split.eq("train")]; val=frame[frame.split.eq("validation")]
    searches=[]
    for i,lr in enumerate(cfg["learning_rates"]):
        searches.append(train_run(f"deep_search_{i}",train,val,cfg,lr,cfg["seed"]))
    chosen=max(searches,key=lambda r:r["validation"]["f1"]); lr=chosen["signature"]["lr"]
    repeats=[chosen]
    for seed in cfg["repeat_seeds"]:
        if seed==cfg["seed"]: continue
        repeats.append(train_run(f"deep_repeat_{seed}",train,val,cfg,lr,seed))
    ablations=[train_run("deep_no_augmentation",train,val,cfg,lr,cfg["seed"],False),
               train_run("deep_frozen_backbone",train,val,cfg,lr,cfg["seed"],True,True)]
    save_json(OUT/"deep.json",{"search":searches,"selected":chosen,"repeats":repeats,"ablations":ablations,
                             "selection":"validation F1; fixed threshold 0.5; no test data read"})
    print("主模型训练结果：", OUT / "deep.json")
