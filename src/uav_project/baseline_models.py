"""Four independent baselines, CV-based selection and tuning; no test-set access."""
from __future__ import annotations
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from .common import *
from .splitting import load_manifest
from .features import features
from .metrics import metrics


def models(seed):
    def pipeline(model, scale=True):
        steps=[("impute",SimpleImputer(strategy="median")),("variance",VarianceThreshold())]
        if scale: steps.append(("scale",StandardScaler()))
        return Pipeline(steps+[("model",model)])
    return {
        "logistic":pipeline(LogisticRegression(max_iter=3000,class_weight="balanced",random_state=seed)),
        "knn":pipeline(KNeighborsClassifier(5)),
        "svm":pipeline(SVC(C=1,probability=True,class_weight="balanced",random_state=seed)),
        "forest":pipeline(RandomForestClassifier(n_estimators=200,class_weight="balanced",n_jobs=2,random_state=seed),False)}


def run_baselines():
    protect_frozen(); cfg=config(); f=load_manifest()
    train=f[f.split.eq("train")]; val=f[f.split.eq("validation")]
    X,names=features(train); Xv,_=features(val); y=train.label.astype(int).to_numpy()
    folds=min(cfg["cv_folds"],int(train.groupby("label").group_id.nunique().min()))
    if folds<3: raise RuntimeError("训练集每类至少需要3个独立组，当前无法交叉验证")
    cv=list(StratifiedGroupKFold(folds,shuffle=True,random_state=cfg["seed"]).split(X,y,train.group_id))
    results=[]; directory=OUT/"models"/"baselines"; directory.mkdir(exist_ok=True)
    candidates=models(cfg["seed"])
    for name,model in candidates.items():
        scores=cross_val_score(model,X,y,cv=cv,scoring="f1",n_jobs=2,error_score="raise")
        model.fit(X,y); joblib.dump(model,directory/(name+".joblib"))
        results.append({"model":name,"cv_f1_mean":float(scores.mean()),"cv_f1_std":float(scores.std()),
                        "cv_folds":folds,"validation":metrics(val.label,model.predict_proba(Xv)[:,1]),
                        "training":metrics(y,model.predict_proba(X)[:,1])})
        print(name,"CV F1",scores.mean(),flush=True)
    best=max(results,key=lambda r:r["cv_f1_mean"])["model"]
    grids={"logistic":{"model__C":[.1,1,10]},"knn":{"model__n_neighbors":[3,5,9],"model__weights":["uniform","distance"]},
           "svm":{"model__C":[.1,1,10],"model__gamma":["scale",.001]},
           "forest":{"model__max_depth":[None,10,20],"model__min_samples_leaf":[1,3]}}
    search=GridSearchCV(models(cfg["seed"])[best],grids[best],cv=cv,scoring="f1",n_jobs=2,error_score="raise")
    search.fit(X,y)
    joblib.dump(search.best_estimator_,directory/"tuned.joblib")
    pd.DataFrame(search.cv_results_).to_csv(OUT/"grid_search.csv",index=False)
    save_json(OUT/"baselines.json",{"results":results,"selected_by":"training grouped CV mean F1", "selected":best,
       "tuned_params":search.best_params_,"tuned_cv_f1":search.best_score_,
       "tuned_validation":metrics(val.label,search.predict_proba(Xv)[:,1]),
       "manifest_hash":digest(META/"manifest.csv"),"config_hash":digest(CFG),"feature_names":names})
    forest=candidates["forest"]
    mask=forest.named_steps["variance"].get_support()
    pd.DataFrame({"feature":np.asarray(names)[mask],"importance":forest.named_steps["model"].feature_importances_}).sort_values("importance",ascending=False).to_csv(OUT/"feature_importance.csv",index=False)
    # Data analysis uses training samples only. No source/title/query fields enter X.
    numeric=train[["brightness","aspect_ratio","label"]].astype(float)
    numeric.corr().to_csv(OUT/"training_correlations.csv")
    print("基线结果：", OUT / "baselines.json")
