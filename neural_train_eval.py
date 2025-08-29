from typing import List, Tuple, Dict, Any, Optional
import json
import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.common_utils import set_seed, temporal_split_per_user
from utils.neural_data_utils import CTRDataset, collate_baseline, collate_semantic, build_item_sem_codes, build_triples
from utils.neural_utils import compute_metrics, compute_slice_and_coldstart_metrics, extract_brand_cat_vocabs
from src.neural_models import WDL, DeepFM, DLRM, evaluate

def run_neural_model(train_data, val_data, test_data,
            user2id, item2id, id2item,
            metadata: Optional[Dict[str,Dict[str,Any]]] = None,
            item2semantic_id: Optional[Dict[str,List[int]]] = None,
            model_name: str = 'WDL',
            mode: str = 'baseline',
            positive_threshold: float = 4.0,
            brand_vocab:int=1000,
            cat_vocab:int=500,
            epochs: int = 3,
            batch_size: int = 1024,
            device: str = 'cuda'):
    set_seed(42)
    device = torch.device(device if (device=='mps' or torch.cuda.is_available()) else 'cpu')
    
    tr = build_triples(train_data, user2id, item2id, positive_threshold)
    va = build_triples(val_data,   user2id, item2id, positive_threshold)
    te = build_triples(test_data,  user2id, item2id, positive_threshold)
    
    collate = collate_semantic if mode=='semantic' else collate_baseline
    print(f'Using collate: {collate}')
    
    ds_tr = CTRDataset(tr, mode, id2item, metadata, item2semantic_id)
    ds_va = CTRDataset(va, mode, id2item, metadata, item2semantic_id)
    ds_te = CTRDataset(te, mode, id2item, metadata, item2semantic_id)
    
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    num_users = max(user2id.values())+1; num_items = max(item2id.values())+1
    if model_name == 'DLRM':
        model = DLRM(num_users, num_items, mode, brand_vocab=brand_vocab, cat_vocab=cat_vocab).to(device)
    elif model_name == 'DeepFM':
        model = DeepFM(num_users, num_items, mode, brand_vocab=brand_vocab, cat_vocab=cat_vocab).to(device)
    else: # default WDL model
        model = WDL(num_users, num_items, mode, brand_vocab=brand_vocab, cat_vocab=cat_vocab).to(device)
    print(f"Using model: {model_name} with {sum(p.numel() for p in model.parameters())} trainable parameters")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    bce = nn.BCELoss()
    best_auc, best_state = -1.0, None
    for ep in range(1, epochs+1):
        # train epoch
        model.train(); total = 0.0
        for batch in dl_tr:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            p = model(batch)
            loss = bce(p, batch['label'].view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
        loss = total / max(1, len(dl_tr))

        # val + selection
        m = evaluate(model, dl_va, device, "VAL", return_raw=False)
        print(f"[{model_name} {mode}] Epoch {ep} | loss={loss:.4f} | VAL AUC={m['auc']:.4f} F1={m['f1']:.4f}")
        if m['auc'] > best_auc:
            best_auc, best_state = m['auc'], {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)

    test_metrics, y, p, users, items = evaluate(model, dl_te, device, "TEST", return_raw=True)
    slices = compute_slice_and_coldstart_metrics(y, p, users, items, tr)
    out = {"overall": test_metrics, **slices}
    return out



if __name__ == "__main__":
    metadata_path = "./beauty/meta.json.gz"
    with gzip.open(metadata_path, "rt") as f:
        metadata = {}
        for line_count, line in enumerate(f):
            line = eval(line.strip())
            metadata[line['asin']] = line
    print(f"Loaded metadata for {len(metadata)} items")

    rating_splits = pd.read_pickle("./beauty/rating_splits_augmented.pkl")
    print(f"Loaded rating splits with keys: {list(rating_splits.keys())}")
    ALL_DATA = rating_splits['train'] + rating_splits['val'] + rating_splits['test']
    df = pd.DataFrame(ALL_DATA)
    df = df[['reviewerID', 'asin', 'overall', 'unixReviewTime']]
    print(f"Overall data: {len(ALL_DATA)} ratings, {df['user'].nunique()} users, {df['item'].nunique()} items")

    train_data, val_data, test_data = temporal_split_per_user(df)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    with open("./datamaps.json") as f:
        datamaps = json.load(f)
    
    user2id = datamaps['user2id']
    item2id = datamaps['item2id']
    id2user = datamaps['id2user']
    id2item = datamaps['id2item']
    print(f"Loaded data maps: {len(user2id)} users, {len(item2id)} items")
    
    with open("./beauty/semantic_ids_fixed.json") as g:
        semantic_ids_map = json.load(g)
    item2semantic_id = semantic_ids_map['semantic_ids']
    print(f"Loaded semantic ids for {len(item2semantic_id)} items")

    Bv, Cv, brand2id, cat2id = extract_brand_cat_vocabs(metadata)
    print("Brand vocab size:", Bv)
    print("Category vocab size:", Cv)

    positive_threshold = 4.0
    batch_size = 1024
    epochs = 10
    device = 'cuda'
    results = {}
    for model_name in ['WDL', 'DeepFM', 'DLRM']:
        for mode in ['baseline', 'semantic']:
            print(f"\n\nTraining {model_name} with mode={mode}")
            r = run_neural_model(train_data, val_data, test_data,
                        user2id, item2id, id2item,
                        metadata=metadata,
                        item2semantic_id=item2semantic_id,
                        model_name=model_name,
                        mode=mode,
                        positive_threshold=positive_threshold,
                        brand_vocab=Bv,
                        cat_vocab=Cv,
                        epochs=epochs,
                        batch_size=batch_size,
                        device=device)
            results[f"{model_name}_{mode}"] = r
            print(f"Results for {model_name} with mode={mode}:")
            print(r['overall'])
    with open(f"neural_results_pos{int(positive_threshold)}_ep{epochs}.json", "w") as f:
        json.dump(results, f, indent=2)
    