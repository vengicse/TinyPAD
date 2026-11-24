"""Second design-selection round (fusion type and modality dropout) using ONLY validation people and development
seeds 100/101 (never used for reported results). Selection metric: mean validation F1 (threshold 0)."""
import json, os, sys, itertools
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
VARIANTS = {"attention_md0.25": dict(attention=True, moddrop=0.25), "attention_md0": dict(attention=True, moddrop=0.0),
            "mean_md0.25": dict(attention=False, moddrop=0.25), "mean_md0": dict(attention=False, moddrop=0.0),
            "attention_md0.10": dict(attention=True, moddrop=0.10), "mean_md0.10": dict(attention=False, moddrop=0.10)}


def job(a):
    ds, fold_no, seed, vname, ddir = a
    import torch, healthlib as H, run_health as R
    torch.set_num_threads(2)
    D = R.data(ds, ddir); shapes = {m: D.X[m].shape[1:] for m in D.mods}
    fold, tr, va, te = list(R.folds(ds, D, seed))[fold_no]
    itr, iva = R.idx_of(D, tr), R.idx_of(D, va, True)
    NZ = H.Normaliser(D, itr, True)
    xtr, xva = H.to_t(NZ.transform(itr)), H.to_t(NZ.transform(iva))
    v = VARIANTS[vname]
    m = H.TinyPAD(shapes, p_drop_mod=v["moddrop"], attention=v["attention"])
    m, _ = H.train_deep(m, xtr, D.y[itr], xva, D.y[iva], seed)
    sc = H.logit_margin(H.predict(m, xva)); y, s_of = D.y[iva], D.subj[iva]
    per = [H.binary_metrics(y[s_of == s], sc[s_of == s]) for s in np.unique(s_of) if 0 < y[s_of == s].sum() < (s_of == s).sum()]
    return ds, fold_no, seed, vname, {k: float(np.nanmean([p[k] for p in per])) for k in ["f1", "auc", "bacc"]}


if __name__ == "__main__":
    ddir, outp = sys.argv[1], sys.argv[2]
    jobs = [("wesad", f, 100, v, ddir) for f in range(15) for v in VARIANTS] + [("mitbih", 0, s, v, ddir) for s in (100, 101, 102) for v in VARIANTS]
    res = []
    with get_context("spawn").Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 6) as p:
        for r in p.imap_unordered(job, jobs):
            res.append(r); print(r, flush=True)
    summ = {}
    for ds in ["wesad", "mitbih"]:
        summ[ds] = {v: {k: float(np.mean([r[4][k] for r in res if r[0] == ds and r[3] == v])) for k in ["f1", "auc", "bacc"]} for v in VARIANTS}
    json.dump(dict(raw=res, summary=summ), open(outp, "w"), indent=1)
    print(json.dumps(summ, indent=1))
