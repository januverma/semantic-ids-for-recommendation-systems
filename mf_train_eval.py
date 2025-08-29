from typing import List, Tuple, Dict
import json
import gzip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.common_utils import set_seed, temporal_split_per_user
from utils.mf_data_utils import MFIDDataset, MFSemDataset, collate_id, collate_sem, build_item_sem_codes
from utils.mf_utils import build_triples
from src.mf_models import MF_ID_Torch, MF_Semantic_Torch, train_epochs, eval_reg_and_slices


def run_mf_id_torch(train_data, val_data, test_data,
                    user2id: Dict[str,int], item2id: Dict[str,int],
                    id2item: Dict[int,str],
                    k:int=32, epochs:int=5, batch_size:int=4096,
                    lr:float=5e-3, weight_decay:float=1e-4,
                    device:str='cuda', positive_threshold: float = 4.0):
    set_seed(42)
    device = torch.device(device if (device=='mps' or torch.cuda.is_available()) else 'cpu')
    tr = build_triples(train_data, user2id, item2id)
    va = build_triples(val_data,   user2id, item2id)
    te = build_triples(test_data,  user2id, item2id)

    ds_tr = MFIDDataset(tr); ds_va = MFIDDataset(va); ds_te = MFIDDataset(te)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=collate_id)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate_id)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_id)

    num_users = max(int(v) for v in user2id.values()) + 1
    num_items = max(int(v) for v in item2id.values()) + 1
    model = MF_ID_Torch(num_users, num_items, k=k).to(device)
    # set global mean from train
    mu = float(np.mean([r for _,_,r in tr])) if tr else 0.0
    model.set_global_mean(mu)

    model = train_epochs(model, dl_tr, dl_va, device, epochs=epochs, lr=lr, weight_decay=weight_decay,
                         positive_threshold=positive_threshold, train_triples=tr)
    # test
    report = eval_reg_and_slices(model, dl_te, device, tr, positive_threshold)
    return report


def run_mf_semantic_torch(train_data, val_data, test_data,
                          user2id: Dict[str,int], item2id: Dict[str,int],
                          id2item: Dict[int,str], item2semantic_id: Dict[str, List[int]],
                          k:int=32, epochs:int=5, batch_size:int=4096,
                          lr:float=5e-3, weight_decay:float=1e-4,
                          device:str='cuda', positive_threshold: float = 4.0):
    set_seed(42)
    device = torch.device(device if (device=='mps' or torch.cuda.is_available()) else 'cpu')
    tr = build_triples(train_data, user2id, item2id)
    va = build_triples(val_data,   user2id, item2id)
    te = build_triples(test_data,  user2id, item2id)

    # Prepare per-item semantic codes & code sizes
    L = 4
    item_sem_codes = build_item_sem_codes(id2item, item2semantic_id, L=L)
    # infer code sizes from mapping
    code_sizes = [1]*L
    for codes in item_sem_codes:
        for l in range(L):
            code_sizes[l] = max(code_sizes[l], int(codes[l])+1)

    ds_tr = MFSemDataset(tr, item_sem_codes, L=L)
    ds_va = MFSemDataset(va, item_sem_codes, L=L)
    ds_te = MFSemDataset(te, item_sem_codes, L=L)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  collate_fn=collate_sem)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate_sem)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, collate_fn=collate_sem)

    num_users = max(int(v) for v in user2id.values()) + 1
    model = MF_Semantic_Torch(num_users, code_sizes, k=k).to(device)
    mu = float(np.mean([r for _,_,r in tr])) if tr else 0.0
    model.set_global_mean(mu)

    model = train_epochs(model, dl_tr, dl_va, device, epochs=epochs, lr=lr, weight_decay=weight_decay,
                         positive_threshold=positive_threshold, train_triples=tr)
    report = eval_reg_and_slices(model, dl_te, device, tr, positive_threshold)
    return report


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

    with open("./beauty/datamaps.json") as f:
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

    mf_id_report = run_mf_id_torch(
        train_data, val_data, test_data,
        user2id, item2id, id2item,
        k=32, epochs=5, positive_threshold=4.0
    )
    print("MF ID Done")
    mf_sem_report = run_mf_semantic_torch(
        train_data, val_data, test_data,
        user2id, item2id, id2item, item2semantic_id,
        k=32, epochs=5, positive_threshold=4.0
    )
    print("MF Semantic Done")

    ALL_RESULTS = {
        'mf_id': mf_id_report,
        'mf_sem': mf_sem_report
    }

    with open('./results/mf_results.json', 'w') as f:
        json.dump(ALL_RESULTS, f)


