import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, accuracy_score
)


def build_triples(split: List[Dict[str,Any]], user2id: Dict[str,int], item2id: Dict[str,int]) -> List[Tuple[int,int,float]]:
    triples = []
    for r in split:
        u = int(user2id[str(r['reviewerID'])])
        i = int(item2id[str(r['asin'])])
        y = float(r['overall'])
        triples.append((u,i,y))
    return triples


def regression_and_cls_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_threshold: float = 4.0):
    # clamp predictions for reporting (not for loss)
    y_pred_c = np.clip(y_pred, 1.0, 5.0)
    # Regression
    rmse = float(np.sqrt(np.mean((y_true - y_pred_c)**2))) if len(y_true) else float('nan')
    mae  = float(np.mean(np.abs(y_true - y_pred_c))) if len(y_true) else float('nan')
    # Classification
    if len(y_true):
        y_bin = (y_true >= positive_threshold).astype(np.int32)
        try: auc = float(roc_auc_score(y_bin, y_pred_c))
        except: auc = float('nan')
        try: pr_auc = float(average_precision_score(y_bin, y_pred_c))
        except: pr_auc = float('nan')
        yhat = (y_pred_c >= positive_threshold).astype(np.int32)
        P,R,F1,_ = precision_recall_fscore_support(y_bin, yhat, average='binary', zero_division=0)
        acc = float(accuracy_score(y_bin, yhat))
    else:
        auc = pr_auc = P = R = F1 = acc = float('nan')
    return {
        'rmse': rmse, 'mae': mae,
        'auc': auc, 'pr_auc': pr_auc,
        'precision': float(P), 'recall': float(R), 'f1': float(F1), 'accuracy': acc,
        'n': int(len(y_true))
    }

# ---- slice helpers (head/mid/tail, cold/warm) ----
def _bins_from_counts(counts: Dict[int,int]):
    if not counts:
        return {}, {'q20':0.0, 'q80':0.0}
    vals = np.array(list(counts.values()))
    q20, q80 = float(np.quantile(vals, 0.2)), float(np.quantile(vals, 0.8))
    assign = {}
    for k,v in counts.items():
        assign[k] = 'tail' if v <= q20 else ('head' if v >= q80 else 'mid')
    return assign, {'q20':q20, 'q80':q80}

def slice_and_cold_metrics(y: np.ndarray, p: np.ndarray, users: np.ndarray, items: np.ndarray,
                           train_triples: List[Tuple[int,int,float]], positive_threshold: float):
    item_counts, user_counts, seen_items = {}, {}, set()
    for u,i,_ in train_triples:
        item_counts[i] = item_counts.get(i,0)+1
        user_counts[u] = user_counts.get(u,0)+1
        seen_items.add(i)
    item_bin, ith = _bins_from_counts(item_counts)
    user_bin, uth = _bins_from_counts(user_counts)

    def mfor(mask):
        idx = np.where(mask)[0]
        if idx.size == 0:
            return regression_and_cls_metrics(np.array([]), np.array([]), positive_threshold)
        return regression_and_cls_metrics(y[idx], p[idx], positive_threshold)

    item_slices = {lab: mfor(np.array([item_bin.get(int(i), None) == lab for i in items]))
                   for lab in ['head','mid','tail']}
    user_slices = {lab: mfor(np.array([user_bin.get(int(u), None) == lab for u in users]))
                   for lab in ['head','mid','tail']}
    cold_mask = np.array([int(i) not in seen_items for i in items])
    warm_mask = ~cold_mask
    cold = mfor(cold_mask); warm = mfor(warm_mask)
    return {
        'item_slices': {'thresholds': ith, 'metrics': item_slices},
        'user_slices': {'thresholds': uth, 'metrics': user_slices},
        'cold_start':  {'cold': cold, 'warm': warm}
    }