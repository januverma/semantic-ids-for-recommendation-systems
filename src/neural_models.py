from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.neural_data_utils import CTRDataset
from utils.neural_utils import compute_metrics

# ---------- small helpers for metadata hashing ----------
class HashEmbeddingBag(nn.Module):
    def __init__(self, num_buckets: int, dim: int, mode: str = 'mean'):
        super().__init__()
        self.emb = nn.Embedding(num_buckets, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.01)
        self.mode = mode
    def forward(self, tokens_lists: List[List[str]]):
        device = self.emb.weight.device
        B = len(tokens_lists)
        out = torch.zeros((B, self.emb.embedding_dim), device=device)
        for i, toks in enumerate(tokens_lists):
            if len(toks)==0: continue
            idx = torch.tensor([hash(t)%self.emb.num_embeddings for t in toks],
                               dtype=torch.long, device=device)
            v = self.emb(idx)
            out[i] = v.mean(dim=0) if self.mode=='mean' else v.sum(dim=0)
        return out
# ------------------------------------------------------

class WDL(nn.Module):
    def __init__(self,
                 num_users:int, num_items:int, mode:str,
                 brand_vocab:int=1000, cat_vocab:int=500, text_buckets:int=100000,
                 sem_codebook:int=256, sem_levels:int=4,
                 emb_dim:int=16, mlp=(128,64), dropout:float=0.1,
                 cross_buckets:int=200000):
        super().__init__()
        self.mode = mode
        self.user_emb = nn.Embedding(num_users, emb_dim)

        if mode=='semantic':
            self.sem_embs = nn.ModuleList([nn.Embedding(sem_codebook, emb_dim) for _ in range(sem_levels)])
            deep_fields = 1 + sem_levels
        else:
            self.item_emb = nn.Embedding(num_items, emb_dim)
            self.brand_emb = nn.Embedding(brand_vocab, emb_dim)
            self.cat_emb   = nn.Embedding(cat_vocab, emb_dim)
            self.text_bag  = HashEmbeddingBag(text_buckets, emb_dim)
            self.srank_lin = nn.Linear(1, emb_dim)
            deep_fields = 1 + 1 + 1 + 1 + 1 + 1  # user,item,brand,cats,text,srank

        self.cross_buckets = cross_buckets
        self.wide = nn.Embedding(cross_buckets, 1)
        nn.init.zeros_(self.wide.weight)

        # MLP head
        layers = []
        inp = deep_fields*emb_dim
        for h in mlp:
            layers += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]
            inp = h
        layers += [nn.Linear(inp, 1)]
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def _hash_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = a.to(torch.int64) * 0x9E3779B185EBCA87 + b.to(torch.int64)
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB
        x = x ^ (x >> 31)
        return (x.abs() % self.cross_buckets).long()

    def forward(self, batch):
        device = self.user_emb.weight.device
        B = batch['user'].shape[0]
        feats = [ self.user_emb(batch['user']) ]

        if self.mode=='semantic':
            codes = batch['sem_codes']
            for l, emb in enumerate(self.sem_embs):
                idx = codes[:, l].clamp(min=0, max=emb.num_embeddings-1).long()  # not in-place
                feats.append(emb(idx))
            deep_x = torch.cat(feats, dim=1)
            # wide user x each level
            wide_sum = torch.zeros((B,1), device=device)
            for l, emb in enumerate(self.sem_embs):
                idx = codes[:, l].clamp(min=0, max=emb.num_embeddings-1).long()
                h = self._hash_pair(batch['user'], idx)
                wide_sum = wide_sum + self.wide(h)
        else:
            it = self.item_emb(batch['item']); feats.append(it)
            # brand (hash string -> id)
            brands = batch.get('brand', [None]*B)
            bidx = torch.tensor([hash(b)%self.brand_emb.num_embeddings if b else 0 for b in brands],
                                dtype=torch.long, device=device)
            feats.append(self.brand_emb(bidx))
            # cats (avg pooled)
            cvec = torch.zeros((B, self.cat_emb.embedding_dim), device=device)
            for i, catlist in enumerate(batch.get('cats', [[]]*B)):
                if catlist:
                    idx = torch.tensor([hash(c)%self.cat_emb.num_embeddings for c in catlist],
                                       dtype=torch.long, device=device)
                    cvec[i] = self.cat_emb(idx).mean(dim=0)
            feats.append(cvec)
            # text bag (title+desc)
            tokens = []
            for t in batch.get('title_desc', ['']*B):
                tokens.append(CTRDataset._tokenize(t, 64))
            feats.append(self.text_bag(tokens))
            # sales rank
            sr = torch.tensor([x if x is not None else 0.0 for x in batch.get('sales_rank', [0.0]*B)],
                              dtype=torch.float32, device=device).view(-1,1)
            feats.append(self.srank_lin(sr))
            deep_x = torch.cat(feats, dim=1)
            # wide user x item
            h = self._hash_pair(batch['user'], batch['item'])
            wide_sum = self.wide(h)

        logits = self.mlp(deep_x) + wide_sum
        return self.sigmoid(logits).view(-1)
# ------------------------------------------------------
# ------------------------------------------------------
class DeepFM(nn.Module):
    def __init__(self,
                 num_users:int, num_items:int, mode:str,
                 brand_vocab:int=1000, cat_vocab:int=500, text_buckets:int=100000,
                 sem_codebook:int=256, sem_levels:int=4,
                 k:int=16, mlp=(128,64), dropout:float=0.1):
        super().__init__()
        self.mode = mode; self.k = k
        # shared emb for FM & Deep
        self.user_emb = nn.Embedding(num_users, k)
        if mode=='semantic':
            self.sem_embs = nn.ModuleList([nn.Embedding(sem_codebook, k) for _ in range(sem_levels)])
            fields = 1 + sem_levels
            self.sales_proj = None
        else:
            self.item_emb = nn.Embedding(num_items, k)
            self.brand_emb = nn.Embedding(brand_vocab, k)
            self.cat_emb   = nn.Embedding(cat_vocab, k)
            self.text_bag  = HashEmbeddingBag(text_buckets, k)
            self.sales_proj = nn.Linear(1, k)
            fields = 1 + 1 + 1 + 1 + 1 + 1

        # first-order (linear)
        self.user_lin = nn.Embedding(num_users, 1)
        if mode=='semantic':
            self.sem_lin = nn.ModuleList([nn.Embedding(sem_codebook,1) for _ in range(sem_levels)])
        else:
            self.item_lin  = nn.Embedding(num_items, 1)
            self.brand_lin = nn.Embedding(brand_vocab, 1)
            self.cat_lin   = nn.Embedding(cat_vocab, 1)
            self.text_lin  = nn.Embedding(text_buckets, 1)
            self.sales_lin = nn.Linear(1,1)

        # deep tower
        inp = fields*k
        layers = []
        for h in mlp:
            layers += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]
            inp = h
        layers += [nn.Linear(inp, 1)]
        self.deep = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def _sum_token_bias(self, tokens_lists: List[List[str]], emb: nn.Embedding, device) -> torch.Tensor:
        B = len(tokens_lists); out = torch.zeros((B,1), device=device)
        for i, toks in enumerate(tokens_lists):
            if not toks: continue
            idx = torch.tensor([hash(t)%emb.num_embeddings for t in toks], dtype=torch.long, device=device)
            out[i] = emb(idx).sum(dim=0)
        return out

    def forward(self, batch):
        device = self.user_emb.weight.device
        B = batch['user'].shape[0]
        v = [ self.user_emb(batch['user']) ]
        first = self.user_lin(batch['user'])

        if self.mode=='semantic':
            codes = batch['sem_codes']
            for l, emb in enumerate(self.sem_embs):
                idx = codes[:, l].clamp(min=0, max=emb.num_embeddings-1).long()
                v.append(emb(idx))
                first = first + self.sem_lin[l](idx)
            deep_x = torch.cat(v, dim=1)
        else:
            it = self.item_emb(batch['item']); v.append(it); first = first + self.item_lin(batch['item'])
            brands = batch.get('brand', [None]*B)
            bidx = torch.tensor([hash(b)%self.brand_emb.num_embeddings if b else 0 for b in brands],
                                dtype=torch.long, device=device)
            v.append(self.brand_emb(bidx)); first = first + self.brand_lin(bidx)
            # cats
            cvec = torch.zeros((B, self.k), device=device); clin = torch.zeros((B,1), device=device)
            for i, catlist in enumerate(batch.get('cats', [[]]*B)):
                if catlist:
                    idx = torch.tensor([hash(c)%self.cat_emb.num_embeddings for c in catlist],
                                       dtype=torch.long, device=device)
                    cvec[i] = self.cat_emb(idx).mean(dim=0)
                    clin[i] = self.cat_lin(idx).sum(dim=0)
            v.append(cvec); first = first + clin
            # text
            tokens = [CTRDataset._tokenize(t,64) for t in batch.get('title_desc', ['']*B)]
            tvec = self.text_bag(tokens); tlin = self._sum_token_bias(tokens, self.text_lin, device)
            v.append(tvec); first = first + tlin
            # sales rank
            sr = torch.tensor([x if x is not None else 0.0 for x in batch.get('sales_rank', [0.0]*B)],
                              dtype=torch.float32, device=device).view(-1,1)
            v.append(self.sales_proj(sr)); first = first + self.sales_lin(sr)
            deep_x = torch.cat(v, dim=1)

        # FM second order
        stack = torch.stack(v, dim=1)            # [B,F,k]
        sum_vec = stack.sum(dim=1)               # [B,k]
        sum_sq  = (sum_vec*sum_vec).sum(dim=1, keepdim=True)
        sq_sum  = sum((t*t).sum(dim=1, keepdim=True) for t in v)
        fm2 = 0.5*(sum_sq - sq_sum)

        logits = first + fm2 + self.deep(deep_x)
        return self.sigmoid(logits).view(-1)
# ------------------------------------------------------
# ------------------------------------------------------
class DLRM(nn.Module):
    def __init__(self,
                 num_users:int, num_items:int, mode:str,
                 brand_vocab:int=1000, cat_vocab:int=500, text_buckets:int=100000,
                 sem_codebook:int=256, sem_levels:int=4,
                 d:int=16, bottom_mlp=(64,), top_mlp=(128,64), dropout:float=0.1):
        super().__init__()
        self.mode = mode; self.d = d
        self.user_emb = nn.Embedding(num_users, d)
        if mode=='semantic':
            self.sem_embs = nn.ModuleList([nn.Embedding(sem_codebook, d) for _ in range(sem_levels)])
            self.num_cat = 1 + sem_levels
            self.bottom = None
        else:
            self.item_emb = nn.Embedding(num_items, d)
            self.brand_emb = nn.Embedding(brand_vocab, d)
            self.cat_emb   = nn.Embedding(cat_vocab, d)
            self.text_bag  = HashEmbeddingBag(text_buckets, d)
            # dense (sales rank) bottom mlp
            layers = []; inp = 1
            for h in bottom_mlp:
                layers += [nn.Linear(inp, h), nn.ReLU(), nn.Dropout(dropout)]; inp = h
            self.bottom = nn.Sequential(*layers)
            self.num_cat = 2 + 1 + 1 + 1  # user,item,brand,cats,text

        # top MLP
        dense_out = bottom_mlp[-1] if self.bottom is not None else 0
        self.dense_proj = nn.Linear(dense_out, d) if dense_out > 0 else None
        C = self.num_cat + (1 if dense_out>0 else 0)
        interact_dim = C*(C-1)//2
        top_in = interact_dim + (dense_out if dense_out>0 else 0)
        layers = []
        for h in top_mlp:
            layers += [nn.Linear(top_in, h), nn.ReLU(), nn.Dropout(dropout)]
            top_in = h
        layers += [nn.Linear(top_in, 1)]
        self.top = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def _dot_interact(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [B,F,d]
        Z = torch.bmm(feats, feats.transpose(1,2))  # [B,F,F]
        iu = torch.triu_indices(Z.size(1), Z.size(2), offset=1, device=feats.device)
        return Z[:, iu[0], iu[1]]                   # [B, F*(F-1)/2]

    def forward(self, batch):
        device = self.user_emb.weight.device
        feats = [ self.user_emb(batch['user']) ]    # list of [B,d]
        dense_vec = None
        if self.mode=='semantic':
            codes = batch['sem_codes']
            for l, emb in enumerate(self.sem_embs):
                idx = codes[:, l].clamp(min=0, max=emb.num_embeddings-1).long()
                feats.append(emb(idx))
        else:
            it = self.item_emb(batch['item']); feats.append(it)
            brands = batch.get('brand', [None]*batch['user'].shape[0])
            bidx = torch.tensor([hash(b)%self.brand_emb.num_embeddings if b else 0 for b in brands],
                                dtype=torch.long, device=device)
            feats.append(self.brand_emb(bidx))
            # cats
            B = batch['user'].shape[0]
            cvec = torch.zeros((B, self.d), device=device)
            for i, catlist in enumerate(batch.get('cats', [[]]*B)):
                if catlist:
                    idx = torch.tensor([hash(c)%self.cat_emb.num_embeddings for c in catlist],
                                       dtype=torch.long, device=device)
                    cvec[i] = self.cat_emb(idx).mean(dim=0)
            feats.append(cvec)
            # text
            tokens = [CTRDataset._tokenize(t,64) for t in batch.get('title_desc', ['']*B)]
            feats.append(self.text_bag(tokens))
            # dense bottom
            if self.bottom is not None:
                sr = torch.tensor([x if x is not None else 0.0 for x in batch.get('sales_rank', [0.0]*B)],
                                  dtype=torch.float32, device=device).view(-1,1)
                dense_vec = self.bottom(sr)

        cat = torch.stack(feats, dim=1)             # [B,F,d]
        if dense_vec is not None:
            dense_use = self.dense_proj(dense_vec)  # shape [B, d]
            feats_all = torch.cat([cat, dense_use.unsqueeze(1)], dim=1)
        else:
            feats_all = cat
        inter = self._dot_interact(feats_all)       # [B, *]
        top_in = inter if dense_vec is None else torch.cat([inter, dense_vec], dim=1)
        logits = self.top(top_in)
        return self.sigmoid(logits).view(-1)
# ------------------------------------------------------
# ------------------------------------------------------
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, split: str, return_raw=True):
    model.eval()
    ys, ps, us, it = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            p = model(batch)
            ys.append(batch["label"].detach().cpu().numpy())
            ps.append(p.detach().cpu().numpy())
            us.append(batch["user"].detach().cpu().numpy())
            it.append(batch["item"].detach().cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    users = np.concatenate(us)
    items = np.concatenate(it)
    metrics = compute_metrics(y, p)
    if return_raw:
        return metrics, y, p, users, items
    return metrics
