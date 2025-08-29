import math
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.common_utils import label_from_rating

class CTRExample:
    def __init__(self, user:int, item:int, label:int):
        self.user = user; self.item = item; self.label = label

def build_triples(split: List[Dict[str,Any]], user2id: Dict[str,int], item2id: Dict[str,int], thr: float=4.0):
    triples = []
    for r in split:
        u = int(user2id[str(r['reviewerID'])])
        i = int(item2id[str(r['asin'])])
        y = label_from_rating(r['overall'], thr)
        triples.append(CTRExample(u,i,y))
    return triples


class CTRDataset(Dataset):
    """
    mode='baseline': returns user, item, (brand, cats, sales_rank, title_desc), label
    mode='semantic': returns user, item, sem_codes, label
    """
    def __init__(self,
                 triples: List[CTRExample],
                 mode: str,
                 id2asin: List[Optional[str]],
                 metadata: Optional[Dict[str,Dict[str,Any]]],
                 item2semantic_id: Optional[Dict[str, List[int]]]):
        self.triples = triples
        self.mode = mode
        self.id2asin = id2asin
        self.metadata = metadata if (mode=='baseline') else None
        self.item2semantic_id = item2semantic_id if (mode=='semantic') else None

    def __len__(self): return len(self.triples)

    @staticmethod
    def _tokenize(text: str, max_tokens: int = 64) -> List[str]:
        s = (text or "").lower()
        toks = [t for t in s.replace("/", " ").replace("-", " ").split() if t]
        return toks[:max_tokens]

    @staticmethod
    def _norm_salesrank(sr):
        if sr is None: return None
        if isinstance(sr, dict):
            try: sr = list(sr.values())[0]
            except: return None
        try: return math.log1p(max(float(sr), 0.0))
        except: return None

    def __getitem__(self, idx: int):
        ex = self.triples[idx]
        asin = self.id2asin[str(ex.item)]
        if self.mode == 'semantic':
            codes = [0,0,0,0]
            if asin and self.item2semantic_id is not None:
                cc = self.item2semantic_id.get(asin)
                if cc and len(cc)>=4: codes = [int(c) for c in cc[:4]]
            return {
                'user': ex.user, 'item': ex.item,
                'sem_codes': codes, 'label': float(ex.label)
            }
        else:
            # baseline with metadata (gracefully handle missing metadata)
            brand, cats, sr, text = None, [], None, ""
            if asin and self.metadata is not None:
                m = self.metadata.get(asin, {})
                brand = m.get('brand')
                cats = m.get('categories') or []
                if cats and isinstance(cats[0], list):  # flatten if nested
                    cats = list({c for path in cats for c in path})
                sr = self._norm_salesrank(m.get('salesRank'))
                text = (m.get('title') or '') + ' ' + (m.get('description') or '')
            return {
                'user': ex.user, 'item': ex.item,
                'brand': brand, 'cats': cats, 'sales_rank': sr, 'title_desc': text,
                'label': float(ex.label)
            }

def collate_baseline(batch):
    return {
        'user': torch.tensor([b['user'] for b in batch], dtype=torch.long),
        'item': torch.tensor([b['item'] for b in batch], dtype=torch.long),
        'brand': [b.get('brand') for b in batch],
        'cats': [b.get('cats', []) for b in batch],
        'sales_rank': [b.get('sales_rank') for b in batch],
        'title_desc': [b.get('title_desc', '') for b in batch],
        'label': torch.tensor([b['label'] for b in batch], dtype=torch.float32),
    }

def collate_semantic(batch):
    return {
        'user': torch.tensor([b['user'] for b in batch], dtype=torch.long),
        'item': torch.tensor([b['item'] for b in batch], dtype=torch.long),
        'sem_codes': torch.tensor([b['sem_codes'] for b in batch], dtype=torch.long),
        'label': torch.tensor([b['label'] for b in batch], dtype=torch.float32),
    }