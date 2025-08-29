import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

class MFIDDataset(Dataset):
    def __init__(self, triples: List[Tuple[int,int,float]]):
        self.tr = triples
    def __len__(self): return len(self.tr)
    def __getitem__(self, idx):
        u,i,y = self.tr[idx]
        return {'user': u, 'item': i, 'rating': y}

def collate_id(batch):
    return {
        'user': torch.tensor([b['user'] for b in batch], dtype=torch.long),
        'item': torch.tensor([b['item'] for b in batch], dtype=torch.long),
        'rating': torch.tensor([b['rating'] for b in batch], dtype=torch.float32)
    }

def build_item_sem_codes(id2item, item2semantic_id: Dict[str, List[int]], L: int = 4) -> List[List[int]]:
    """
    Returns a list `sem` where sem[item_id] = [c0,c1,c2,c3].
    Works when id2item is:
      - dict: {item_id (int or str) -> asin}
      - list/tuple: index is item_id, value is asin
    """
    # Helper to iterate (item_id:int, asin:str)
    pairs = []
    if isinstance(id2item, dict):
        for k, asin in id2item.items():
            try:
                i = int(k)
            except Exception:
                # If the key can't be cast, skip it (shouldn't happen in your setup)
                continue
            pairs.append((i, asin))
        max_item = max(i for i, _ in pairs) if pairs else -1
        size = max_item + 1
    else:
        # sequence: enumerate
        size = len(id2item)
        pairs = list(enumerate(id2item))

    sem = [[0]*L for _ in range(size)]
    for i, asin in pairs:
        if asin is None:
            continue
        codes = item2semantic_id.get(asin)
        if not codes:
            continue
        for l in range(min(L, len(codes))):
            try:
                c = int(codes[l])
                if c >= 0:
                    sem[i][l] = c
            except Exception:
                # ignore malformed codes
                pass
    return sem


class MFSemDataset(Dataset):
    def __init__(self, triples: List[Tuple[int,int,float]], item_sem_codes: List[List[int]], L: int = 4):
        self.tr = triples
        self.codes = item_sem_codes
        self.L = L
    def __len__(self): return len(self.tr)
    def __getitem__(self, idx):
        u,i,y = self.tr[idx]
        return {'user': u, 'item': i, 'sem_codes': self.codes[i][:self.L], 'rating': y}

def collate_sem(batch):
    return {
        'user': torch.tensor([b['user'] for b in batch], dtype=torch.long),
        'item': torch.tensor([b['item'] for b in batch], dtype=torch.long),
        'sem_codes': torch.tensor([b['sem_codes'] for b in batch], dtype=torch.long),
        'rating': torch.tensor([b['rating'] for b in batch], dtype=torch.float32)
    }