"""Shared binary classification metrics at a fixed 0.5 decision threshold."""
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

def metrics(y, score):
    y=np.asarray(y).astype(int); score=np.asarray(score); pred=(score>=.5).astype(int)
    m={"accuracy":accuracy_score(y,pred), "balanced_accuracy":balanced_accuracy_score(y,pred),
       "precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),
       "f1":f1_score(y,pred,zero_division=0),"macro_f1":f1_score(y,pred,average="macro",zero_division=0)}
    m.update(roc_auc=roc_auc_score(y,score) if len(np.unique(y))==2 else None,
             average_precision=average_precision_score(y,score) if len(np.unique(y))==2 else None)
    return {k: float(v) if v is not None else None for k,v in m.items()}
