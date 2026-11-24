"""TinyPAD (proposed) and baselines for multimodal wearable health-anomaly detection.

All deep models share the optimiser, epochs, batch size, class weighting, early stopping on held-out validation subjects
and the same input modalities. Only TinyPAD uses the label-free per-person calibration segment (first 5 min at rest);
every method is evaluated on the same non-calibration test windows.
"""
import copy, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VEC_MAX_LEN = 8          # modalities with length <= 8 are feature vectors (e.g. RR-interval features)


# ===================================================================================================== data
class Data:
    """Holds modality arrays (N,C,L), labels, subjects and calibration flags."""

    def __init__(self, path):
        z = np.load(path)
        self.mods = sorted(k[2:] for k in z.files if k.startswith("M_"))
        self.X = {m: z["M_" + m] for m in self.mods}
        self.y, self.ycls, self.subj, self.calib, self.t = z["y"], z["ycls"], z["subj"], z["calib"], z["t"]
        self.split = z["split"] if "split" in z.files else None

    def is_vec(self, m):
        return self.X[m].shape[2] <= VEC_MAX_LEN


def fit_stats(X, idx):
    x = X[idx]
    return x.mean(axis=(0, 2), keepdims=True)[0], x.std(axis=(0, 2), keepdims=True)[0] + 1e-6


class Normaliser:
    """Global z-scoring (statistics of the training subjects), optionally preceded by personal calibration:
    signal modalities are z-scored with the person's own calibration-segment statistics, and the four RR-interval
    features are divided by the person's mean calibration RR interval. Calibration windows carry no labels."""

    def __init__(self, data, train_idx, personal):
        self.d, self.personal, self.cache = data, personal, {}
        self.ncal = None          # use only the first ncal calibration windows (calibration-length study)
        self.g = {}
        for m in data.mods:
            x = self._subj(m, train_idx) if personal else data.X[m][train_idx]
            self.g[m] = (x.mean(axis=(0, 2), keepdims=True)[0], x.std(axis=(0, 2), keepdims=True)[0] + 1e-6)

    def _stats(self, s, m):
        key = (s, m)
        if key not in self.cache:
            d = self.d
            ci = np.where((d.subj == s) & d.calib)[0]
            if self.ncal is not None:
                ci = ci[np.argsort(d.t[ci])][:self.ncal]
            if d.is_vec(m):
                self.cache[key] = float(d.X[m][ci][:, 0, :4].mean()) if len(ci) else None
            else:
                self.cache[key] = fit_stats(d.X[m], ci) if len(ci) >= 2 else None
        return self.cache[key]

    def _subj(self, m, idx):
        d = self.d
        x = d.X[m][idx].astype(np.float32).copy()
        for s in np.unique(d.subj[idx]):
            sel = d.subj[idx] == s
            st = self._stats(s, m)
            if st is None:
                continue
            if d.is_vec(m):
                x[sel, :, :4] /= max(st, 1e-3)
            else:
                x[sel] = (x[sel] - st[0]) / st[1]
        return x

    def transform(self, idx):
        out = {}
        for m in self.d.mods:
            x = self._subj(m, idx) if self.personal else self.d.X[m][idx].astype(np.float32)
            mu, sd = self.g[m]
            out[m] = np.clip((x - mu) / sd, -20, 20).astype(np.float32)
        return out


def to_t(xd, idx=None):
    return {m: torch.from_numpy(v if idx is None else v[idx]).to(DEV) for m, v in xd.items()}


# ===================================================================================================== models
def stem_stride(L):
    return 4 if L >= 2000 else 2 if L >= 600 else 1


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=2, k=7):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, stride=stride, padding=k // 2, groups=cin, bias=False)
        self.b1 = nn.BatchNorm1d(cin)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.b2 = nn.BatchNorm1d(cout)

    def forward(self, x):
        return F.relu(self.b2(self.pw(F.relu(self.b1(self.dw(x))))))


class SigEncoder(nn.Module):
    """Tiny depthwise-separable 1D-CNN encoder for one signal modality."""

    def __init__(self, C, L, d=32, width=(16, 24, 32)):
        super().__init__()
        s = stem_stride(L)
        self.stem = nn.Sequential(nn.Conv1d(C, width[0], 7, stride=s, padding=3, bias=False), nn.BatchNorm1d(width[0]), nn.ReLU())
        blocks, c = [], width[0]
        for w in list(width[1:]) + [d]:
            blocks.append(DSBlock(c, w)); c = w
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(-1)


class VecEncoder(nn.Module):
    def __init__(self, C, L, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(C * L, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())

    def forward(self, x):
        return self.net(x)


def make_encoder(C, L, d=32):
    return VecEncoder(C, L, d) if L <= VEC_MAX_LEN else SigEncoder(C, L, d)


class TinyPAD(nn.Module):
    """Per-modality tiny encoders + modality-attention fusion (masking-aware) + small head."""

    def __init__(self, shapes, d=32, p_drop_mod=0.25, attention=True):
        super().__init__()
        self.mods = sorted(shapes)
        self.enc = nn.ModuleDict({m: make_encoder(*shapes[m], d) for m in self.mods})
        self.att = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 1))
        self.head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 2))
        self.p_drop, self.attention = p_drop_mod, attention

    def embed(self, xd, mask=None):
        E = torch.stack([self.enc[m](xd[m]) for m in self.mods], 1)            # B x M x d
        B, M, _ = E.shape
        if mask is None:
            mask = torch.ones(B, M, device=E.device)
        if self.training and self.p_drop > 0:
            keep = (torch.rand(B, M, device=E.device) > self.p_drop).float()
            keep[keep.sum(1) == 0, torch.randint(0, M, (1,)).item()] = 1.0
            mask = mask * keep
        if self.attention:
            a = self.att(E).squeeze(-1).masked_fill(mask == 0, -1e9)
            w = torch.softmax(a, 1)
        else:
            w = mask / mask.sum(1, keepdim=True).clamp_min(1)
        return (w.unsqueeze(-1) * E).sum(1), w

    def forward(self, xd, mask=None):
        z, _ = self.embed(xd, mask)
        return self.head(z)


class LateFusionCNN(nn.Module):
    """Same encoders, concatenated embeddings (no attention, no modality dropout)."""

    def __init__(self, shapes, d=32):
        super().__init__()
        self.mods = sorted(shapes)
        self.enc = nn.ModuleDict({m: make_encoder(*shapes[m], d) for m in self.mods})
        self.head = nn.Sequential(nn.Linear(d * len(self.mods), 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, xd, mask=None):
        E = [self.enc[m](xd[m]) * (1.0 if mask is None else mask[:, i:i + 1]) for i, m in enumerate(self.mods)]
        return self.head(torch.cat(E, 1))


class EarlyFusionBase(nn.Module):
    """Resamples every signal modality to a common length and stacks channels; vector modalities join at the head."""

    def __init__(self, shapes, Lc):
        super().__init__()
        self.sig = sorted(m for m in shapes if shapes[m][1] > VEC_MAX_LEN)
        self.vec = sorted(m for m in shapes if shapes[m][1] <= VEC_MAX_LEN)
        self.Cin = sum(shapes[m][0] for m in self.sig)
        self.Vin = sum(shapes[m][0] * shapes[m][1] for m in self.vec)
        self.Lc = Lc

    def stack(self, xd, mask=None):
        parts = []
        for i, m in enumerate(self.sig):
            x = F.interpolate(xd[m], size=self.Lc, mode="linear", align_corners=False) if xd[m].shape[-1] != self.Lc else xd[m]
            if mask is not None:
                x = x * mask[:, sorted(self.sig + self.vec).index(m)].view(-1, 1, 1)
            parts.append(x)
        v = torch.cat([xd[m].flatten(1) for m in self.vec], 1) if self.vec else None
        return torch.cat(parts, 1), v


class EarlyFusionCNN(EarlyFusionBase):
    def __init__(self, shapes, Lc=256):
        super().__init__(shapes, Lc)
        self.enc = SigEncoder(self.Cin, Lc, d=64, width=(32, 48, 64))
        self.head = nn.Sequential(nn.Linear(64 + self.Vin, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xd, mask=None):
        x, v = self.stack(xd, mask)
        z = self.enc(x)
        return self.head(torch.cat([z, v], 1) if v is not None else z)


class CNNLSTM(EarlyFusionBase):
    def __init__(self, shapes, Lc=256):
        super().__init__(shapes, Lc)
        self.conv = nn.Sequential(nn.Conv1d(self.Cin, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                                  nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                                  nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU())
        self.lstm = nn.LSTM(32, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(32 + self.Vin, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xd, mask=None):
        x, v = self.stack(xd, mask)
        h, _ = self.lstm(self.conv(x).transpose(1, 2))
        z = h[:, -1]
        return self.head(torch.cat([z, v], 1) if v is not None else z)


class TinyTransformer(EarlyFusionBase):
    def __init__(self, shapes, Lc=256, patch=8, d=32):
        super().__init__(shapes, Lc)
        self.patch = nn.Conv1d(self.Cin, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, Lc // patch, d))
        layer = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.1, batch_first=True)
        self.tr = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.Linear(d + self.Vin, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, xd, mask=None):
        x, v = self.stack(xd, mask)
        h = self.tr(self.patch(x).transpose(1, 2) + self.pos).mean(1)
        return self.head(torch.cat([h, v], 1) if v is not None else h)


class ConvAE(EarlyFusionBase):
    """Unsupervised convolutional auto-encoder trained on normal windows; anomaly score = reconstruction error."""

    def __init__(self, shapes, Lc=256):
        super().__init__(shapes, Lc)
        C = self.Cin
        self.e = nn.Sequential(nn.Conv1d(C, 32, 7, 2, 3), nn.ReLU(), nn.Conv1d(32, 16, 7, 2, 3), nn.ReLU(), nn.Conv1d(16, 8, 7, 2, 3))
        self.dd = nn.Sequential(nn.ConvTranspose1d(8, 16, 4, 2, 1), nn.ReLU(), nn.ConvTranspose1d(16, 32, 4, 2, 1), nn.ReLU(),
                                nn.ConvTranspose1d(32, C, 4, 2, 1))
        self.ve = nn.Sequential(nn.Linear(max(self.Vin, 1), 8), nn.ReLU(), nn.Linear(8, max(self.Vin, 1))) if self.Vin else None

    def score(self, xd, mask=None):
        x, v = self.stack(xd, mask)
        err = ((self.dd(self.e(x)) - x) ** 2).mean((1, 2))
        if v is not None:
            err = err + ((self.ve(v) - v) ** 2).mean(1)
        return err


DEEP = {"EF-CNN": EarlyFusionCNN, "LF-CNN": LateFusionCNN, "CNN-LSTM": CNNLSTM, "TinyTransformer": TinyTransformer}


# ===================================================================================================== helpers
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_macs(model, xd):
    """Multiply-accumulate operations of one forward pass (Conv1d, Linear, LSTM, attention matmuls approximated)."""
    macs = [0]

    def conv_hook(m, i, o):
        macs[0] += o.numel() // o.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m, i, o):
        macs[0] += (o.numel() // o.shape[0]) * m.in_features

    def lstm_hook(m, i, o):
        T = i[0].shape[1]
        macs[0] += T * 4 * m.hidden_size * (m.input_size + m.hidden_size)

    def mha_hook(m, i, o):
        T, d = i[0].shape[1], m.embed_dim
        macs[0] += 2 * T * T * d   # QK^T and AV (projections counted by Linear hooks)

    hs = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.ConvTranspose1d)):
            hs.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hs.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.LSTM):
            hs.append(mod.register_forward_hook(lstm_hook))
        elif isinstance(mod, nn.MultiheadAttention):
            hs.append(mod.register_forward_hook(mha_hook))
            T = None
    model.eval()
    with torch.no_grad():
        one = {m: v[:1] for m, v in xd.items()}
        model(one) if not isinstance(model, ConvAE) else model.score(one)
    for h in hs:
        h.remove()
    return macs[0]


def batches(n, bs, rng, shuffle=True):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict(model, xd, bs=512, mask=None):
    model.eval()
    n = len(next(iter(xd.values())))
    out = []
    for i in range(0, n, bs):
        sl = {m: v[i:i + bs] for m, v in xd.items()}
        mk = None if mask is None else mask[i:i + bs]
        out.append(model(sl, mk).float())
    return torch.cat(out).cpu().numpy()


def logit_margin(lg):
    return lg[:, 1] - lg[:, 0]


def train_deep(model, xtr, ytr, xva, yva, seed, epochs=40, bs=64, lr=1e-3, patience=8, ae=False):
    import os
    epochs = int(os.environ.get("HL_EPOCHS", epochs))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    yt = torch.from_numpy(ytr).to(DEV)
    w = torch.tensor([1.0, (ytr == 0).sum() / max(1, (ytr == 1).sum())], device=DEV).float() if not ae else None
    n = len(ytr)
    best, best_state, bad, hist = -float("inf"), None, 0, []
    tr_idx = np.where(ytr == 0)[0] if ae else np.arange(n)
    for ep in range(epochs):
        model.train()
        for b in batches(len(tr_idx), bs, rng):
            bi = torch.from_numpy(tr_idx[b]).to(DEV)
            xb = {m: v[bi] for m, v in xtr.items()}
            loss = model.score(xb).mean() if ae else F.cross_entropy(model(xb), yt[bi], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        # validation
        if ae:
            s = score_ae(model, xva); score = -float(np.mean(s[yva == 0]))
        else:
            p = predict(model, xva); pr = (logit_margin(p) > 0).astype(int)
            score = macro_f1(yva, pr)
        hist.append(score)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, dict(epochs_run=ep + 1, best_val=best)


@torch.no_grad()
def score_ae(model, xd, bs=512):
    model.eval()
    n = len(next(iter(xd.values())))
    return torch.cat([model.score({m: v[i:i + bs] for m, v in xd.items()}) for i in range(0, n, bs)]).cpu().numpy()


@torch.no_grad()
def adabn(model, xcal, n_pass=2):
    """Label-free re-estimation of BatchNorm statistics on a person's calibration windows."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm1d)]
    if not bns or len(next(iter(xcal.values()))) < 2:
        return model
    saved = {id(b): (b.momentum,) for b in bns}
    for b in bns:
        b.reset_running_stats(); b.momentum = None
    model.train()
    p_saved = getattr(model, "p_drop", 0)
    if hasattr(model, "p_drop"):
        model.p_drop = 0
    for _ in range(n_pass):
        model(xcal)
    if hasattr(model, "p_drop"):
        model.p_drop = p_saved
    for b in bns:
        b.momentum = saved[id(b)][0]
    model.eval()
    return model


@torch.no_grad()
def embeddings(model, xd, bs=512):
    model.eval()
    n = len(next(iter(xd.values())))
    Z = []
    for i in range(0, n, bs):
        z, _ = model.embed({m: v[i:i + bs] for m, v in xd.items()})
        Z.append(z)
    return torch.cat(Z).cpu().numpy()


def deviation(Z, Zcal):
    mu, var = Zcal.mean(0), Zcal.var(0) + 1e-2
    return np.log1p(((Z - mu) ** 2 / var).mean(1))


# ===================================================================================================== metrics
def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, average="macro", zero_division=0)


def binary_metrics(y, score, thr=0.0):
    from sklearn.metrics import f1_score, accuracy_score
    p = (score > thr).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    out = dict(acc=accuracy_score(y, p), f1=f1_score(y, p, zero_division=0), macro_f1=macro_f1(y, p),
               sens=tp / max(1, tp + fn), spec=tn / max(1, tn + fp), ppv=tp / max(1, tp + fp), tp=tp, tn=tn, fp=fp, fn=fn)
    out["auc"] = roc_auc_score(y, score) if 0 < y.sum() < len(y) else float("nan")
    out["bacc"] = 0.5 * (out["sens"] + out["spec"])
    return out


# ===================================================================================================== hand-crafted features (RF baseline)
def hcf(xd):
    feats = []
    for m in sorted(xd):
        x = xd[m]
        if x.shape[2] <= VEC_MAX_LEN:
            feats.append(x.reshape(len(x), -1)); continue
        dx = np.diff(x, axis=2)
        spec = np.abs(np.fft.rfft(x - x.mean(2, keepdims=True), axis=2)) ** 2
        k = spec.shape[2]
        bands = [spec[:, :, 1:max(2, k // 8)].sum(2), spec[:, :, k // 8:k // 2].sum(2), spec[:, :, k // 2:].sum(2)]
        tot = sum(bands) + 1e-9
        tt = np.arange(x.shape[2])
        slope = ((x - x.mean(2, keepdims=True)) * (tt - tt.mean())).sum(2) / ((tt - tt.mean()) ** 2).sum()
        f = [x.mean(2), x.std(2), x.min(2), x.max(2), np.percentile(x, 10, axis=2), np.percentile(x, 90, axis=2), slope,
             np.abs(dx).mean(2), dx.std(2)] + [b / tot for b in bands]
        feats.append(np.concatenate(f, 1))
    return np.concatenate(feats, 1)
