"""Group-aware split. No model scores are used to choose the split."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from .common import *

def holdout(frame, fraction, seed):
    # Deterministic candidate selection based on labels/counts only, never model scores.
    n_groups = frame.groupby("label")["group_id"].nunique().min()
    k = min(10, int(n_groups))
    if k < 3: raise ValueError("每类独立样本组不足，不能可靠划分")
    fold_ids = np.zeros(len(frame), dtype=int)
    for fold, (_, idx) in enumerate(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(frame, frame.label, frame.group_id)):
        fold_ids[idx] = fold
    from itertools import combinations
    take = max(1, min(k-1, round(k*fraction)))
    best = None
    for comb in combinations(range(k), take):
        mask = np.isin(fold_ids, comb)
        if frame.loc[mask, "label"].nunique() != 2 or frame.loc[~mask, "label"].nunique() != 2: continue
        penalty = abs(mask.mean()-fraction) + abs(frame.loc[mask, "label"].mean()-frame.label.mean())
        if best is None or penalty < best[0]: best = (penalty, mask)
    if best is None: raise ValueError("无法得到两类齐全的分组划分")
    return frame.loc[~best[1]].copy(), frame.loc[best[1]].copy()



def split_data():
    protect_frozen()
    setup()
    cfg = config()
    cleaning = read_json(OUT / "cleaning_summary.json")
    if cleaning["config_hash"] != digest(CFG) or cleaning["review_hash"] != digest(META / "review.csv"):
        raise RuntimeError("清洗后配置或审核记录发生变化，请先重跑04清洗")
    frame = pd.read_csv(META / "clean_manifest.csv")
    dev, test = holdout(frame[frame.domain.eq("public")].reset_index(drop=True), cfg["public_test_fraction"], cfg["seed"])
    train, val = holdout(dev.reset_index(drop=True), cfg["validation_fraction_of_development"], cfg["seed"]+1)
    external = frame[frame.domain.eq("external")].copy()
    parts = []
    for name, part in [("train",train), ("validation",val), ("public_test",test), ("external_test",external)]:
        part["split"] = name; parts.append(part)
    final = pd.concat(parts).sort_values(["split", "sample_id"])
    if final.groupby("group_id").split.nunique().max() != 1: raise AssertionError("group leakage")
    final.to_csv(META / "manifest.csv", index=False, encoding="utf-8-sig")
    final.groupby(["split", "label"]).size().rename("n").to_csv(OUT / "dataset_counts.csv")
    save_json(OUT / "data_protocol.json", {"manifest_sha256": digest(META / "manifest.csv"),
        "raw": cleaning["raw"], "clean":len(final), "events": cleaning["events"],
        "public_test_actual_fraction": len(test)/(len(dev)+len(test)),
        "course_split": "public: ~70% development / ~30% test; development has internal validation",
        "task_scope":cfg["task"], "note":"Public benchmark native splits are recorded in original_path; this is a newly declared course group split, not the official benchmark score."})
    print(final.groupby(["split","label"]).size().to_string())
    print("固定划分已保存：", META / "manifest.csv")


def load_manifest():
    import pandas as pd
    p = META / "manifest.csv"
    if not p.exists(): raise RuntimeError("请先完成01至05数据脚本")
    f = pd.read_csv(p).fillna("")
    if f.groupby("group_id").split.nunique().max() != 1: raise ValueError("分组跨集合泄露")
    return f
