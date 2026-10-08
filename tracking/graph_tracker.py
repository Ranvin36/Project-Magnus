"""
Physics-aware graph tracker (prototype).

Input: one clip's frames in candidates-file format. The clip is cut into windows of
WINDOW frames. Each window is a graph with two kinds of node:
  candidate nodes  one per detector candidate (position, sub-pixel centre, score, size)
  frame nodes      one per frame (detector visibility prob, time in window)
and two kinds of edge:
  motion edges     candidate(t) <-> candidate(t+d), d = 1..MAX_DT, carrying the implied
                   velocity (dx/d, dy/d), distance and time gap -- the "physics" signal
  frame edges      frame(t) <-> frame(t+d) for d in DILATIONS (1, 2, 4, 8, 16), plus
                   candidate <-> its own frame -- the dilated links let information
                   cross detector-blind runs longer than a few frames

After a few rounds of message passing, each frame node outputs:
  presence  is there a visible ball in this frame?
  select    which candidate is the ball, or "none of them" (index K)
  position  a predicted ball position, used when the ball is present but no candidate
            is right -- this is what lets the tracker fill detector-blind runs

Windows overlap; per-frame outputs are averaged over the windows covering a frame.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracking.evaluate import TOL

W_IMG, H_IMG = 512.0, 288.0
WINDOW = 32
STRIDE = 8
DILATIONS = (1, 2, 4, 8, 16)   # frame-node links, so information crosses long gaps
K = 8
MAX_DT = 3
N_CAND_FEAT, N_FRAME_FEAT = 7, 3


# ------------------------------------------------------------------ data -> tensors
def clip_windows(frames, with_labels=True):
    """Cut a clip into overlapping windows -> list of (start, dict of numpy arrays)."""
    n = len(frames)
    starts = list(range(0, max(n - WINDOW, 0) + 1, STRIDE))
    if starts[-1] + WINDOW < n:
        starts.append(n - WINDOW)
    out = []
    for s in starts:
        cf = np.zeros((WINDOW, K, N_CAND_FEAT), np.float32)
        cm = np.zeros((WINDOW, K), bool)
        ff = np.zeros((WINDOW, N_FRAME_FEAT), np.float32)
        fm = np.zeros(WINDOW, bool)
        pres = np.zeros(WINDOW, np.float32)
        sel = np.full(WINDOW, K, np.int64)
        pos = np.zeros((WINDOW, 2), np.float32)
        for w in range(WINDOW):
            t = s + w
            if t >= n:
                break
            f = frames[t]
            fm[w] = True
            cs = f["candidates"][:K]
            for k, c in enumerate(cs):
                cf[w, k] = [c["x"] / W_IMG, c["y"] / H_IMG, c["cx"] / W_IMG, c["cy"] / H_IMG,
                            c["score"], np.log1p(c["area"]) / 5.0, 1.0]
                cm[w, k] = True
            ff[w] = [f["vis"], float(bool(cs)), (w - WINDOW / 2) / WINDOW]
            if with_labels and f["visible"] and f.get("gt") is not None:
                gx, gy = f["gt"]
                pres[w] = 1.0
                pos[w] = [gx / W_IMG, gy / H_IMG]
                d = [np.hypot(c["x"] - gx, c["y"] - gy) for c in cs]
                if d and min(d) <= TOL:
                    sel[w] = int(np.argmin(d))
        out.append((s, dict(cf=cf, cm=cm, ff=ff, fm=fm, pres=pres, sel=sel, pos=pos)))
    return out


def collate(items):
    return {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in items[0]}


# ------------------------------------------------------------------ model
def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


class GraphTracker(nn.Module):
    def __init__(self, d=64, rounds=3):
        super().__init__()
        self.cand_in, self.frame_in = mlp(N_CAND_FEAT, d), mlp(N_FRAME_FEAT, d)
        self.rounds = rounds
        self.edge_msg = nn.ModuleList([mlp(2 * d + 4, d) for _ in range(rounds)])
        self.cand_upd = nn.ModuleList([mlp(4 * d, d) for _ in range(rounds)])
        self.frame_upd = nn.ModuleList([mlp(4 * d, d) for _ in range(rounds)])
        self.frame_prev = nn.ModuleList([nn.ModuleList([nn.Linear(d, d) for _ in DILATIONS]) for _ in range(rounds)])
        self.frame_next = nn.ModuleList([nn.ModuleList([nn.Linear(d, d) for _ in DILATIONS]) for _ in range(rounds)])
        self.pres_head, self.null_head, self.pos_head = mlp(d, 1), mlp(d, 1), mlp(d, 2)
        self.sel_head = mlp(2 * d, 1)

    def forward(self, b):
        cf, cm, ff, fm = b["cf"], b["cm"], b["ff"], b["fm"]
        B, T, Kc, _ = cf.shape
        h = self.cand_in(cf) * cm[..., None]                 # (B,T,K,d)
        g = self.frame_in(ff) * fm[..., None]                # (B,T,d)
        xy = cf[..., 0:2]
        neg = torch.finfo(h.dtype).min
        for r in range(self.rounds):
            into_later, into_earlier = [], []
            for dt in range(1, MAX_DT + 1):
                if dt >= T:
                    break
                src, dst = h[:, :-dt], h[:, dt:]               # forward edges t -> t+dt
                dxy = (xy[:, dt:, None, :, :] - xy[:, :-dt, :, None, :]) / dt   # (B,T-dt,Ksrc,Kdst,2)
                e = torch.cat([dxy, dxy.norm(dim=-1, keepdim=True),
                               torch.full_like(dxy[..., :1], dt / MAX_DT)], -1)
                pair = torch.cat([src[:, :, :, None, :].expand(-1, -1, -1, Kc, -1),
                                  dst[:, :, None, :, :].expand(-1, -1, Kc, -1, -1), e], -1)
                m = self.edge_msg[r](pair)                     # (B,T-dt,Ksrc,Kdst,d)
                valid = (cm[:, :-dt, :, None] & cm[:, dt:, None, :])[..., None]
                m = m.masked_fill(~valid, neg)
                # pad back to T frames (no in-place writes, so autograd can differentiate)
                into_later.append(F.pad(m.max(dim=2).values, (0, 0, 0, 0, dt, 0), value=neg))
                into_earlier.append(F.pad(m.max(dim=3).values, (0, 0, 0, 0, 0, dt), value=neg))
            agg_f = torch.stack(into_later).max(dim=0).values
            agg_b = torch.stack(into_earlier).max(dim=0).values
            agg_f = torch.where(agg_f == neg, torch.zeros_like(agg_f), agg_f)
            agg_b = torch.where(agg_b == neg, torch.zeros_like(agg_b), agg_b)
            h = (h + self.cand_upd[r](torch.cat([h, agg_f, agg_b, g[:, :, None, :].expand(-1, -1, Kc, -1)], -1))) * cm[..., None]
            pooled = h.masked_fill(~cm[..., None], neg).max(dim=2).values
            pooled = torch.where(pooled == neg, torch.zeros_like(pooled), pooled)
            g_prev = sum(lin(F.pad(g, (0, 0, dd, 0))[:, :T]) for dd, lin in zip(DILATIONS, self.frame_prev[r]))
            g_next = sum(lin(F.pad(g, (0, 0, 0, dd))[:, dd:]) for dd, lin in zip(DILATIONS, self.frame_next[r]))
            g = (g + self.frame_upd[r](torch.cat([g, pooled, g_prev, g_next], -1))) * fm[..., None]
        pres = self.pres_head(g).squeeze(-1)
        sel = self.sel_head(torch.cat([h, g[:, :, None, :].expand(-1, -1, Kc, -1)], -1)).squeeze(-1)
        sel = sel.masked_fill(~cm, -1e4)
        sel = torch.cat([sel, self.null_head(g)], -1)        # (B,T,K+1)
        pos = torch.sigmoid(self.pos_head(g))                  # normalised (x, y)
        return pres, sel, pos


def loss_fn(out, b, w_sel=1.0, w_pos=5.0):
    pres, sel, pos = out
    fm = b["fm"]
    l_pres = F.binary_cross_entropy_with_logits(pres[fm], b["pres"][fm])
    vis = fm & (b["pres"] > 0.5)
    l_sel = F.cross_entropy(sel[vis], b["sel"][vis]) if vis.any() else pres.sum() * 0
    l_pos = (pos[vis] - b["pos"][vis]).abs().sum(-1).mean() if vis.any() else pres.sum() * 0
    return l_pres + w_sel * l_sel + w_pos * l_pos


# ------------------------------------------------------------------ inference
@torch.no_grad()
def track_clip(model, frames, device, presence_threshold=0.5):
    """Per-frame output for one clip: None or (x, y) in 512x288 pixels."""
    n = len(frames)
    wins = clip_windows(frames, with_labels=False)
    b = {k: v.to(device) for k, v in collate([w for _, w in wins]).items()}
    pres, sel, pos = model(b)
    pres, sel, pos = torch.sigmoid(pres).cpu().numpy(), torch.softmax(sel, -1).cpu().numpy(), pos.cpu().numpy()
    acc_p, acc_s, acc_x, cnt = np.zeros(n), np.zeros((n, K + 1)), np.zeros((n, 2)), np.zeros(n)
    for wi, (s, _) in enumerate(wins):
        for w in range(WINDOW):
            t = s + w
            if t >= n:
                break
            wt = 1.0 - abs(w - (WINDOW - 1) / 2) / WINDOW      # trust window centres more
            acc_p[t] += wt * pres[wi, w]; acc_s[t] += wt * sel[wi, w]; acc_x[t] += wt * pos[wi, w]; cnt[t] += wt
    out = []
    for t in range(n):
        p, s, x = acc_p[t] / cnt[t], acc_s[t] / cnt[t], acc_x[t] / cnt[t]
        if p <= presence_threshold:
            out.append(None); continue
        k = int(np.argmax(s))
        cs = frames[t]["candidates"]
        if k < min(K, len(cs)):
            out.append((float(cs[k]["x"]), float(cs[k]["y"])))
        else:
            out.append((float(x[0] * W_IMG), float(x[1] * H_IMG)))
    return out
