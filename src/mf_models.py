import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from utils.mf_utils import regression_and_cls_metrics, slice_and_cold_metrics

class MF_ID_Torch(nn.Module):
    """ Biased MF with user/item embeddings and biases. """
    def __init__(self, num_users:int, num_items:int, k:int=32):
        super().__init__()
        self.user_f = nn.Embedding(num_users, k)
        self.item_f = nn.Embedding(num_items, k)
        self.user_b = nn.Embedding(num_users, 1)
        self.item_b = nn.Embedding(num_items, 1)
        # init
        nn.init.normal_(self.user_f.weight, std=0.01)
        nn.init.normal_(self.item_f.weight, std=0.01)
        nn.init.zeros_(self.user_b.weight); nn.init.zeros_(self.item_b.weight)
        self.register_buffer('mu', torch.tensor(0.0))  # global mean buffer

    def set_global_mean(self, mu: float):
        self.mu.fill_(float(mu))

    def forward(self, batch):
        u = batch['user']; i = batch['item']
        p = (self.mu
             + self.user_b(u).squeeze(1)
             + self.item_b(i).squeeze(1)
             + (self.user_f(u) * self.item_f(i)).sum(dim=1))
        return p  # raw preds (not clamped)


class MF_Semantic_Torch(nn.Module):
    """ User emb + compositional item from L semantic code tables; per-code biases; learned level weights. """
    def __init__(self, num_users:int, code_sizes: List[int], k:int=32):
        super().__init__()
        self.L = len(code_sizes); self.k = k
        self.user_f = nn.Embedding(num_users, k)
        self.user_b = nn.Embedding(num_users, 1)
        nn.init.normal_(self.user_f.weight, std=0.01); nn.init.zeros_(self.user_b.weight)
        # per-level code embeddings & biases
        self.sem_embs = nn.ModuleList([nn.Embedding(Sl, k) for Sl in code_sizes])
        self.sem_bias = nn.ModuleList([nn.Embedding(Sl, 1) for Sl in code_sizes])
        for emb, b in zip(self.sem_embs, self.sem_bias):
            nn.init.normal_(emb.weight, std=0.01); nn.init.zeros_(b.weight)
        # learned global logits for levels
        self.level_logits = nn.Parameter(torch.zeros(self.L))
        self.register_buffer('mu', torch.tensor(0.0))

    def set_global_mean(self, mu: float):
        self.mu.fill_(float(mu))

    def forward(self, batch):
        u = batch['user']; codes = batch['sem_codes']  # [B, L]
        B = u.size(0); device = u.device
        w = torch.softmax(self.level_logits, dim=0)  # [L]

        # Compose item vector and bias: sum_l w_l * E_l[c_l]
        v_sum = torch.zeros(B, self.k, device=device)
        b_sum = torch.zeros(B, 1, device=device)
        for l, (emb, bias) in enumerate(zip(self.sem_embs, self.sem_bias)):
            idx = codes[:, l].clamp(min=0, max=emb.num_embeddings-1).long()
            v_sum = v_sum + w[l] * emb(idx)
            b_sum = b_sum + w[l] * bias(idx)

        p = self.mu + self.user_b(u).squeeze(1) + b_sum.squeeze(1) + (self.user_f(u) * v_sum).sum(dim=1)
        return p  # raw preds


@torch.no_grad()
def eval_reg_and_slices(model, loader, device, train_triples: List[Tuple[int,int,float]], positive_threshold: float):
    model.eval()
    ys, ps, us, it = [], [], [], []
    for batch in loader:
        batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
        y = batch['rating'].detach().cpu().numpy()
        p = model(batch).detach().cpu().numpy()
        ys.append(y); ps.append(p)
        us.append(batch['user'].detach().cpu().numpy())
        it.append(batch['item'].detach().cpu().numpy())
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    users = np.concatenate(us) if us else np.array([])
    items = np.concatenate(it) if it else np.array([])
    overall = regression_and_cls_metrics(y, p, positive_threshold)
    slices = slice_and_cold_metrics(y, p, users, items, train_triples, positive_threshold)
    return {'overall': overall, **slices}

def train_epochs(model, dl_train, dl_val, device, epochs=5, lr=5e-3, weight_decay=1e-4,
                 positive_threshold: float = 4.0, train_triples: Optional[List[Tuple[int,int,float]]] = None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    best_rmse = float('inf'); best_state = None
    for ep in range(1, epochs+1):
        model.train(); total = 0.0
        for batch in dl_train:
            batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
            opt.zero_grad(set_to_none=True)
            preds = model(batch)
            loss = mse(preds, batch['rating'])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
        # val
        val_report = eval_reg_and_slices(model, dl_val, device, train_triples or [], positive_threshold)
        rmse_v = val_report['overall']['rmse']
        print(f"Epoch {ep} | train_mse={total/max(1,len(dl_train)):.4f} | VAL RMSE={rmse_v:.4f}")
        if rmse_v < best_rmse:
            best_rmse = rmse_v
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model