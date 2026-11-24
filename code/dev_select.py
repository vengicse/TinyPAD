"""Design selection for TinyPAD's personal scoring, using ONLY validation people (never the test person of a fold)
and a development seed (100) that is not used for reported results. Stacking is cross-fitted across validation people."""
import json, os, sys, itertools
import numpy as np
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
VARIANTS = {"logit": dict(adabn=False, score_norm=False, stack=False), "logit+adabn": dict(adabn=True, score_norm=False, stack=False),
            "stack_raw": dict(adabn=False, score_norm=False, stack=True), "stack_z": dict(adabn=False, score_norm=True, stack=True),
            "stack_z+adabn": dict(adabn=True, score_norm=True, stack=True)}


def job(a):
    ds, fold_no, ddir = a
    import torch, healthlib as H, run_health as R
    torch.set_num_threads(2)
    D = R.data(ds, ddir); shapes = {m: D.X[m].shape[1:] for m in D.mods}
    fold, tr, va, te = list(R.folds(ds, D, 100))[fold_no]
    itr, iva = R.idx_of(D, tr), R.idx_of(D, va, True)
    NZ = H.Normaliser(D, itr, True)
    xtr, xva = H.to_t(NZ.transform(itr)), H.to_t(NZ.transform(iva))
    m = H.TinyPAD(shapes); m, _ = H.train_deep(m, xtr, D.y[itr], xva, D.y[iva], 100)
    out = {}
    for name, v in VARIANTS.items():
        lg, dv = R.personal_features(H, D, m, NZ, iva, v)
        y, s_of = D.y[iva], D.subj[iva]
        if not v["stack"]:
            score = lg
        else:   # cross-fit: score each validation person with a stacker fitted on the other validation people
            score = np.zeros(len(y))
            for s in np.unique(s_of):
                trm, tem = s_of != s, s_of == s
                if len(np.unique(y[trm])) < 2:
                    score[tem] = lg[tem]; continue
                lr = H.LogisticRegression(class_weight="balanced").fit(np.c_[lg[trm], dv[trm]], y[trm])
                score[tem] = lr.decision_function(np.c_[lg[tem], dv[tem]])
        per = [H.binary_metrics(y[s_of == s], score[s_of == s]) for s in np.unique(s_of) if 0 < y[s_of == s].sum() < (s_of == s).sum()]
        out[name] = {k: float(np.nanmean([p[k] for p in per])) for k in ["auc", "f1", "bacc", "sens", "spec"]}
        out[name]["pooled"] = {k: float(v2) for k, v2 in H.binary_metrics(y, score).items() if k in ("auc", "f1", "bacc")}
    return ds, fold_no, out


if __name__ == "__main__":
    ddir = sys.argv[1]; outp = sys.argv[2]
    jobs = [("wesad", f, ddir) for f in range(15)] + [("mitbih", 0, ddir)]
    res = {}
    with get_context("spawn").Pool(int(sys.argv[3]) if len(sys.argv) > 3 else 5) as p:
        for ds, f, o in p.imap_unordered(job, jobs):
            res.setdefault(ds, {})[f] = o; print(ds, f, {k: round(v["f1"], 3) for k, v in o.items()}, flush=True)
    summ = {ds: {v: {k: float(np.mean([r[f][v][k] for f in r])) for k in ["auc", "f1", "bacc"]} for v in VARIANTS} for ds, r in res.items()}
    json.dump(dict(per_fold=res, summary=summ), open(outp, "w"), indent=1)
    print(json.dumps(summ, indent=1))
