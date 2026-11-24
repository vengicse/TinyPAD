"""Experiment runner. One job = (dataset, method/variant, seed, fold); results -> results/<exp>/<name>.json

  python run_health.py --exp main     # all methods, WESAD LOSO x 3 seeds, MIT-BIH DS1->DS2 x 5 seeds
  python run_health.py --exp ablation # TinyPAD variants
"""
import argparse, json, os, sys, time, itertools, traceback
from multiprocessing import get_context
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CHEAP = {"wesad": ["wrist_acc", "wrist_eda_temp"], "mitbih": ["rhythm"]}     # always-on, low-power modalities
EXPENSIVE_HOLD_S = 60          # expensive sensors stay on for 60 s after a stage-1 flag (WESAD)
DEFAULT_MODDROP = {"wesad": 0.25, "mitbih": 0.10}   # chosen on validation people only (dev_select2.py, seeds 100-102)
METHODS = ["RF-HCF", "CAE", "EF-CNN", "LF-CNN", "CNN-LSTM", "TinyTransformer", "TinyPAD"]
ABLATIONS = {   # name: (method, cfg)
    "no_personal_norm": ("TinyPAD", dict(personal_norm=False)),
    "no_attention": ("TinyPAD", dict(attention=False)),
    "no_moddrop": ("TinyPAD", dict(moddrop=0.0)),
    "concat_fusion": ("LF-CNN+PN", {}),
    "plus_adabn": ("TinyPAD", dict(adabn=True)),
    "plus_deviation_stack": ("TinyPAD", dict(deviation=True)),
}
_D = {}


def data(ds, ddir):
    import healthlib as H
    if ds not in _D:
        _D[ds] = H.Data(os.path.join(ddir, f"{ds}.npz"))
    return _D[ds]


def folds(ds, D, seed):
    rng = np.random.default_rng([seed, 17])
    if ds == "wesad":
        subs = sorted(set(D.subj.tolist()))
        for s in subs:
            rest = [x for x in subs if x != s]
            val = list(rng.choice(rest, 2, replace=False))
            yield s, [x for x in rest if x not in val], val, [s]
    else:
        ds1 = sorted(set(D.subj[D.split == 1].tolist())); ds2 = sorted(set(D.subj[D.split == 2].tolist()))
        val = list(rng.choice(ds1, 4, replace=False))
        yield "DS2", [x for x in ds1 if x not in val], val, ds2


def idx_of(D, subs, exclude_calib=False):
    m = np.isin(D.subj, subs)
    if exclude_calib:
        m &= ~D.calib
    return np.where(m)[0]


# ---------------------------------------------------------------------------------------------------- one job
def run_job(args):
    ds, method, cfg, seed, ddir, out = args
    if os.path.exists(out):
        return out, "skip", 0
    import torch, healthlib as H
    torch.set_num_threads(2)
    t0 = time.time()
    D = data(ds, ddir)
    shapes = {m: D.X[m].shape[1:] for m in D.mods}
    res = dict(dataset=ds, method=method, cfg=cfg, seed=seed, folds=[])
    try:
        for fold, tr_s, va_s, te_s in folds(ds, D, seed):
            res["folds"].append(run_fold(H, D, ds, method, cfg, seed, shapes, fold, tr_s, va_s, te_s))
    except Exception as e:
        res["error"] = traceback.format_exc()
        print(res["error"], flush=True)
    res["wall_sec"] = time.time() - t0
    tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
    return out, "ok" if "error" not in res else "ERROR", time.time() - t0


def run_fold(H, D, ds, method, cfg, seed, shapes, fold, tr_s, va_s, te_s):
    import torch
    itr, iva, ite = idx_of(D, tr_s), idx_of(D, va_s, True), idx_of(D, te_s, True)
    ytr, yva, yte = D.y[itr], D.y[iva], D.y[ite]
    rec = dict(fold=str(fold), n_train=len(itr), n_test=len(ite))
    is_pad = method.startswith("TinyPAD")
    personal = (is_pad and cfg.get("personal_norm", True)) or method == "LF-CNN+PN"
    NZ = H.Normaliser(D, itr, personal)
    if method == "RF-HCF":
        Xtr, Xte = H.hcf(NZ.transform(itr)), H.hcf(NZ.transform(ite))
        rf = H.RandomForestClassifier(300, class_weight="balanced_subsample", n_jobs=2, random_state=seed, min_samples_leaf=2).fit(Xtr, ytr)
        score = rf.predict_proba(Xte)[:, 1] - 0.5
        rec.update(params=None, macs=None)
    else:
        xtr, xva, xte = H.to_t(NZ.transform(itr)), H.to_t(NZ.transform(iva)), H.to_t(NZ.transform(ite))
        if method == "CAE":
            model = H.ConvAE(shapes)
            model, info = H.train_deep(model, xtr, ytr, xva, yva, seed, ae=True)
            sva = H.score_ae(model, xva); thr = np.percentile(sva[yva == 0], 95)
            score = H.score_ae(model, xte) - thr
        elif is_pad:
            model = H.TinyPAD(shapes, p_drop_mod=cfg.get("moddrop", DEFAULT_MODDROP[ds]), attention=cfg.get("attention", True))
            model, info = H.train_deep(model, xtr, ytr, xva, yva, seed)
            score, extra = personalise_and_score(H, D, model, NZ, iva, ite, cfg)
            rec.update(extra)
            if cfg.get("full_eval", False):
                rec["missing"] = missing_modality(H, D, model, NZ, iva, ite, cfg, yte)
                rec["calib_len"] = calib_length(H, D, model, NZ, iva, ite, cfg, yte, ds)
                rec["int8"] = int8_eval(H, D, model, NZ, iva, ite, cfg, yte)
                rec["cascade"] = cascade(H, D, ds, shapes, NZ, itr, iva, ite, cfg, seed, score, model, xtr)
        else:
            model = H.DEEP[method.replace("+PN", "")](shapes)
            model, info = H.train_deep(model, xtr, ytr, xva, yva, seed)
            score = H.logit_margin(H.predict(model, xte))
            if method in ("LF-CNN", "EF-CNN") and cfg.get("full_eval", False):
                rec["missing"] = missing_modality_plain(H, D, model, xte, yte)
        rec.update(train_info=info, params=H.count_params(model), macs=H.count_macs(model, xte))
    rec["pooled"] = H.binary_metrics(yte, score)
    rec["per_subject"] = {}
    for s in te_s:
        sel = D.subj[ite] == s
        if sel.sum():
            rec["per_subject"][str(s)] = H.binary_metrics(yte[sel], score[sel])
    rec["score"] = np.round(score, 4).tolist(); rec["y"] = yte.tolist(); rec["subj"] = D.subj[ite].tolist()
    if ds == "mitbih":
        rec["ycls"] = D.ycls[ite].tolist()
    return rec


# ---------------------------------------------------------------------------------------------------- personalisation
def subject_view(H, D, model, NZ, idx, s, use_adabn, ncal=None):
    """Return (model personalised to subject s, embeddings/logits of its windows in idx, calibration embeddings)."""
    import copy
    ci = np.where((D.subj == s) & D.calib)[0]
    if ncal is not None:
        ci = ci[:ncal]
    m = copy.deepcopy(model)
    if use_adabn and len(ci) >= 2:
        H.adabn(m, H.to_t(NZ.transform(ci)))
    xs = H.to_t(NZ.transform(idx))
    lg = H.logit_margin(H.predict(m, xs))
    Z = H.embeddings(m, xs)
    Zc = H.embeddings(m, H.to_t(NZ.transform(ci))) if len(ci) >= 2 else None
    return lg, Z, Zc


def personal_features(H, D, model, NZ, idx, cfg, ncal=None):
    """Logit margin and embedding deviation of every window in idx, each standardised by the person's own
    calibration-window distribution (label-free person-specific score calibration, cfg['score_norm'])."""
    lg = np.zeros(len(idx)); dev = np.zeros(len(idx))
    for s in np.unique(D.subj[idx]):
        sel = np.where(D.subj[idx] == s)[0]
        ci = np.where((D.subj == s) & D.calib)[0]
        if ncal is not None:
            ci = ci[:ncal]
        both = np.concatenate([idx[sel], ci])
        l, Z, Zc = subject_view(H, D, model, NZ, both, s, cfg.get("adabn", False), ncal)
        n = len(sel)
        dv = H.deviation(Z, Zc) if Zc is not None else np.zeros(len(both))
        if cfg.get("score_norm", False) and len(ci) >= 2:
            l = (l - l[n:].mean()) / (l[n:].std() + 0.5)
            dv = (dv - dv[n:].mean()) / (dv[n:].std() + 0.05)
        lg[sel], dev[sel] = l[:n], dv[:n]
    return lg, dev


def personalise_and_score(H, D, model, NZ, iva, ite, cfg, ncal=None):
    lg_te, dv_te = personal_features(H, D, model, NZ, ite, cfg, ncal)
    if not cfg.get("deviation", False):          # final TinyPAD: personal normalisation, no stacking (see dev_select)
        return lg_te, dict(stack=None)
    lg_va, dv_va = personal_features(H, D, model, NZ, iva, cfg)
    yva = D.y[iva]
    lr = H.LogisticRegression(class_weight="balanced", C=1.0).fit(np.c_[lg_va, dv_va], yva)
    score = lr.decision_function(np.c_[lg_te, dv_te])
    return score, dict(stack=dict(coef=lr.coef_[0].tolist(), intercept=float(lr.intercept_[0])))


# ---------------------------------------------------------------------------------------------------- extra evaluations
def missing_modality(H, D, model, NZ, iva, ite, cfg, yte):
    """Drop one modality at test time (masked out of the attention fusion); personal normalisation kept."""
    import torch
    out = {}
    xs = H.to_t(NZ.transform(ite))
    for k, m in enumerate(model.mods):
        mask = torch.ones(len(ite), len(model.mods), device=H.DEV); mask[:, k] = 0
        out[m] = H.binary_metrics(yte, H.logit_margin(H.predict(model, xs, mask=mask)))
    return out


def missing_modality_plain(H, D, model, xte, yte):
    import torch
    out = {}
    mods = sorted(xte)
    for k, m in enumerate(mods):
        mask = torch.ones(len(yte), len(mods), device=H.DEV); mask[:, k] = 0
        xs = {mm: (v * 0 if mm == m else v) for mm, v in xte.items()}
        out[m] = H.binary_metrics(yte, H.logit_margin(H.predict(model, xs, mask=mask)))
    return out


def calib_length(H, D, model, NZ, iva, ite, cfg, yte, ds):
    """Personal normalisation computed from only the first 1, 2, 3 or 5 minutes of the test person's rest segment."""
    import copy
    out = {}
    for minutes in [0.5, 1, 2, 3, 5]:
        ncal = max(2, int(minutes * (12 if ds == "wesad" else 75)))   # WESAD: 12 windows/min (5-s stride); MIT-BIH ~75 beats/min
        nz = copy.copy(NZ); nz.cache = {k: v for k, v in NZ.cache.items() if k[0] not in set(D.subj[ite].tolist())}; nz.ncal = ncal
        sc = H.logit_margin(H.predict(model, H.to_t(nz.transform(ite))))
        out[str(minutes)] = H.binary_metrics(yte, sc)
    return out


def int8_eval(H, D, model, NZ, iva, ite, cfg, yte):
    import torch, copy
    q = copy.deepcopy(model)
    with torch.no_grad():
        for mod in q.modules():
            if isinstance(mod, (torch.nn.Conv1d, torch.nn.Linear)):
                w = mod.weight; s = w.abs().max().clamp_min(1e-12) / 127
                mod.weight.copy_(torch.round(w / s).clamp(-127, 127) * s)
    sc, _ = personalise_and_score(H, D, q, NZ, iva, ite, cfg)
    n = H.count_params(q)
    return dict(metrics=H.binary_metrics(yte, sc), fp32_kb=n * 4 / 1024, int8_kb=n / 1024)


def cascade(H, D, ds, shapes, NZ, itr, iva, ite, cfg, seed, score_full, model_full, xtr):
    """Stage 1: TinyPAD on cheap modalities only (always on). Stage 2: full TinyPAD, run only when stage 1 flags."""
    import torch
    cheap = CHEAP[ds]
    sub = {m: shapes[m] for m in cheap}
    m1 = H.TinyPAD(sub, p_drop_mod=cfg.get("moddrop", DEFAULT_MODDROP[ds]))

    def tf(idx):
        return {m: v for m, v in NZ.transform(idx).items() if m in cheap}
    x1tr, x1va = H.to_t(tf(itr)), H.to_t(tf(iva))
    m1, _ = H.train_deep(m1, x1tr, D.y[itr], x1va, D.y[iva], seed)

    class NZsub:
        def transform(self, idx):
            return tf(idx)
    s1_va, _ = personalise_and_score(H, D, m1, NZsub(), iva, iva, cfg)
    s1_te, _ = personalise_and_score(H, D, m1, NZsub(), iva, ite, cfg)
    yva, yte = D.y[iva], D.y[ite]
    macs1, macs2 = H.count_macs(m1, H.to_t(tf(ite[:1]))), H.count_macs(model_full, H.to_t(NZ.transform(ite[:1])))
    out = dict(macs_stage1=macs1, macs_stage2=macs2, stage1_alone=H.binary_metrics(yte, s1_te), sweep=[])
    pos = np.sort(s1_va[yva == 1])
    for target in [0.90, 0.95, 0.98, 0.99, 1.0]:
        tau = pos[int((1 - target) * (len(pos) - 1))] if len(pos) else 0.0
        flag = s1_te >= tau
        if ds == "wesad":
            run2, on_frac = sensor_gating(D, ite, flag)
        else:
            run2, on_frac = flag, 1.0                         # ECG always on for MIT-BIH; gating saves compute only
        final = np.where(run2, score_full, np.minimum(s1_te, score_full.min()))
        out["sweep"].append(dict(target_recall=target, tau=float(tau), stage2_rate=float(run2.mean()), sensor_on_frac=float(on_frac),
                                 rel_compute=float((macs1 + run2.mean() * macs2) / macs2), metrics=H.binary_metrics(yte, final)))
    return out


def _union(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _inter_len(A, B):
    tot, i, j = 0.0, 0, 0
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        tot += max(0.0, hi - lo)
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return tot


def sensor_gating(D, ite, flag):
    """Expensive sensors switch on when a stage-1 window flags and stay on EXPENSIVE_HOLD_S seconds; stage 2 can run on
    window w only if the sensors were on for its whole 30-s span (causal: flag decided at the end of the flagged window).
    On-time fraction = time with sensors on within the MONITORED time (union of evaluated windows), per subject pooled."""
    run2 = np.zeros(len(ite), bool); on_time = 0.0; total = 0.0
    for s in np.unique(D.subj[ite]):
        sel = np.where(D.subj[ite] == s)[0]
        order = sel[np.argsort(D.t[ite][sel])]
        t = D.t[ite][order]; f = flag[order]
        ends = t + 30.0
        on_iv = _union([(e, e + EXPENSIVE_HOLD_S) for e, ff in zip(ends, f) if ff])
        for j, st in enumerate(t):
            run2[order[j]] = any(a <= st and b >= st + 30.0 for a, b in on_iv)
        mon = _union([(a, a + 30.0) for a in t])
        on_time += _inter_len(on_iv, mon); total += sum(b - a for a, b in mon)
    return run2, on_time / max(total, 1e-9)


# ---------------------------------------------------------------------------------------------------- job lists
def build(exp, ddir, rdir, datasets, seeds_w, seeds_m):
    jobs = []
    d = os.path.join(rdir, exp); os.makedirs(d, exist_ok=True)
    for ds in datasets:
        seeds = seeds_w if ds == "wesad" else seeds_m
        if exp == "main":
            for m, s in itertools.product(METHODS, seeds):
                cfg = dict(full_eval=True) if m in ("TinyPAD", "LF-CNN", "EF-CNN") else {}
                jobs.append((ds, m, cfg, s, ddir, os.path.join(d, f"{ds}_{m}_s{s}.json")))
        elif exp == "ablation":
            for (name, (meth, cfg)), s in itertools.product(ABLATIONS.items(), seeds[:3]):
                jobs.append((ds, meth, cfg, s, ddir, os.path.join(d, f"{ds}_{name}_s{s}.json")))
    return jobs


def latency(ddir, rdir):
    """Single-thread CPU inference latency per window and model size for every deep model (untrained weights suffice)."""
    import torch, healthlib as H
    torch.set_num_threads(1)
    out = {}
    for ds in ["wesad", "mitbih"]:
        D = data(ds, ddir); shapes = {m: D.X[m].shape[1:] for m in D.mods}
        one = {m: torch.from_numpy(D.X[m][:1].astype(np.float32)) for m in D.mods}
        res = {}
        models = {"TinyPAD": H.TinyPAD(shapes), "TinyPAD stage-1": H.TinyPAD({m: shapes[m] for m in CHEAP[ds]}),
                  "CAE": H.ConvAE(shapes), **{k: v(shapes) for k, v in H.DEEP.items()}}
        for name, mdl in models.items():
            mdl.eval()
            x = {m: one[m] for m in getattr(mdl, "mods", D.mods)} if name == "TinyPAD stage-1" else one
            f = (lambda: mdl.score(x)) if name == "CAE" else (lambda: mdl(x))
            with torch.no_grad():
                for _ in range(20):
                    f()
                t0 = time.perf_counter(); n = 300
                for _ in range(n):
                    f()
                ms = (time.perf_counter() - t0) / n * 1e3
            mdl.to("cpu")
            res[name] = dict(ms=ms, params=H.count_params(mdl), macs=H.count_macs(mdl, x) if name != "CAE" else H.count_macs(mdl, x),
                             fp32_kb=H.count_params(mdl) * 4 / 1024, int8_kb=H.count_params(mdl) / 1024)
        out[ds] = res
    os.makedirs(rdir, exist_ok=True)
    json.dump(out, open(os.path.join(rdir, "latency.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--results", default=os.path.join(HERE, "..", "results"))
    ap.add_argument("--datasets", default="mitbih,wesad")
    ap.add_argument("--seeds_wesad", default="0,1,2")
    ap.add_argument("--seeds_mitbih", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.exp == "latency":
        latency(a.data, a.results); sys.exit()
    jobs = build(a.exp, a.data, a.results, a.datasets.split(","), [int(x) for x in a.seeds_wesad.split(",")],
                 [int(x) for x in a.seeds_mitbih.split(",")])
    print(f"{a.exp}: {len(jobs)} jobs, {a.workers} workers", flush=True)
    t0 = time.time()
    with get_context("spawn").Pool(a.workers) as pool:
        for i, (p, st, sec) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {st} {os.path.basename(p)} {sec:.0f}s elapsed {time.time()-t0:.0f}s", flush=True)
