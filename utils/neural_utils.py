from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
from utils.neural_data_utils import CTRExample


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str,float]:
    out = {}
    try: out['auc'] = float(roc_auc_score(y_true, y_prob))
    except: out['auc'] = float('nan')
    try: out['pr_auc'] = float(average_precision_score(y_true, y_prob))
    except: out['pr_auc'] = float('nan')
    y_hat = (y_prob >= thr).astype(np.int32)
    P, R, F1, _ = precision_recall_fscore_support(y_true, y_hat, average='binary', zero_division=0)
    out.update({
        'precision': float(P), 'recall': float(R), 'f1': float(F1),
        'accuracy': float(accuracy_score(y_true, y_hat))
    })
    return out

def slice_bins_from_counts(counts: Dict[int, int]):
    if not counts:
        return {}, {"q20": 0.0, "q80": 0.0}
    vals = np.array(list(counts.values()))
    q20, q80 = np.quantile(vals, 0.2), np.quantile(vals, 0.8)
    id2bin = {}
    for k, v in counts.items():
        id2bin[k] = "tail" if v <= q20 else ("head" if v >= q80 else "mid")
    return id2bin, {"q20": float(q20), "q80": float(q80)}


def compute_slice_and_coldstart_metrics(y, p, users, items, train_tr: List[CTRExample]):
    item_counts: Dict[int, int] = {}
    user_counts: Dict[int, int] = {}
    seen_items = set()
    for ex in train_tr:
        item_counts[ex.item] = item_counts.get(ex.item, 0) + 1
        user_counts[ex.user] = user_counts.get(ex.user, 0) + 1
        seen_items.add(ex.item)
    item_bin, ith = slice_bins_from_counts(item_counts)
    user_bin, uth = slice_bins_from_counts(user_counts)

    def mfor(mask):
        if mask.sum() == 0:
            return {"auc": float("nan"), "pr_auc": float("nan"), "precision": float("nan"),
                    "recall": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "n": 0}
        m = compute_metrics(y[mask], p[mask])
        m["n"] = int(mask.sum())
        return m

    item_slices = {lab: mfor(np.array([item_bin.get(int(it), None) == lab for it in items])) for lab in ["head", "mid", "tail"]}
    user_slices = {lab: mfor(np.array([user_bin.get(int(u), None) == lab for u in users])) for lab in ["head", "mid", "tail"]}
    cold_mask = np.array([int(it) not in seen_items for it in items])
    warm_mask = ~cold_mask
    return {
        "item_slices": {"thresholds": ith, "metrics": item_slices},
        "user_slices": {"thresholds": uth, "metrics": user_slices},
        "cold_start": {"cold": mfor(cold_mask), "warm": mfor(warm_mask)},
    }

def extract_brand_cat_vocabs(metadata: Dict[str, Dict[str, Any]]) -> Tuple[int,int,Dict[str,int],Dict[str,int]]:
    """
    Given metadata dict {asin: {brand, categories, ...}}, return:
      (brand_vocab_size, cat_vocab_size, brand2id, cat2id)

    - brand_vocab_size = #unique non-empty brands
    - cat_vocab_size   = #unique category tokens (flattened)
    - brand2id, cat2id = mappings for consistent indexing
    """
    brand2id, cat2id = {}, {}

    # Collect brands
    for asin, m in metadata.items():
        b = m.get('brand')
        if b:
            if b not in brand2id:
                brand2id[b] = len(brand2id)

    # Collect categories
    for asin, m in metadata.items():
        cs = m.get('categories') or []
        if cs and isinstance(cs[0], list):
            # Flatten paths if nested
            for path in cs:
                for c in path:
                    if c not in cat2id:
                        cat2id[c] = len(cat2id)
        else:
            for c in cs:
                if c not in cat2id:
                    cat2id[c] = len(cat2id)

    return len(brand2id), len(cat2id), brand2id, cat2id


