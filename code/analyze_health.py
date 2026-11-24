"""Figures 2-9 and all tables of the healthcare paper from the RunPod run records.

  python analyze_health.py --results ../results --data ../data --out ../paper_assets
All numbers in summary.json are computed here from the run JSONs (pooled / per_subject metrics, scores, cascade, missing,
calib_len, int8) and latency.json / dev_select.json.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc as sk_auc

M = ["RF-HCF", "CAE", "EF-CNN", "LF-CNN", "CNN-LSTM", "TinyTransformer", "TinyPAD"]
COL = {"RF-HCF": "#8c564b", "CAE": "#9e9e9e", "EF-CNN": "#1f77b4", "LF-CNN": "#17becf", "CNN-LSTM": "#9467bd",
       "TinyTransformer": "#ff7f0e", "TinyPAD": "#d62728", "LF-CNN+PN": "#2ca02c"}
DSN = {"wesad": "WESAD", "mitbih": "MIT-BIH"}
MODN = {"chest_acc": "chest ACC", "chest_ecg": "chest ECG", "chest_eda_temp": "chest EDA+TEMP", "chest_resp": "chest RESP",
        "wrist_acc": "wrist ACC", "wrist_bvp": "wrist BVP", "wrist_eda_temp": "wrist EDA+TEMP", "lead1": "ECG lead 1",
        "lead2": "ECG lead 2", "rhythm": "RR rhythm"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300,
                     "legend.fontsize": 7.5, "axes.titlesize": 10})
KEYS = ["acc", "f1", "macro_f1", "sens", "spec", "ppv", "auc", "bacc"]


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(f))
        if "error" in r:
            print("ERROR in", f); continue
        out[os.path.basename(f)[:-5]] = r
    return out


def runs_of(R, ds, name):
    return [r for k, r in R.items() if k.startswith(f"{ds}_{name}_s")]


def subj_metric(runs, key):
    """WESAD: subject x seed matrix of a per-subject metric."""
    subs = sorted({s for r in runs for f in r["folds"] for s in f["per_subject"]}, key=int)
    A = np.array([[np.mean([f["per_subject"][s][key] for f in r["folds"] if s in f["per_subject"]]) for s in subs] for r in runs])
    return subs, A


def summarise(R, ds, name):
    runs = runs_of(R, ds, name)
    if not runs:
        return None
    out = {}
    for k in KEYS:
        if ds == "wesad":
            _, A = subj_metric(runs, k)              # seeds x subjects
            per_seed = np.nanmean(A, 1)
            out[k], out[k + "_sd"], out[k + "_subj"] = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0, np.nanmean(A, 0).tolist()
        else:
            v = np.array([r["folds"][0]["pooled"][k] for r in runs])
            out[k], out[k + "_sd"], out[k + "_seeds"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, v.tolist()
    if ds == "mitbih":
        for cname, c in [("S", 1), ("V", 2), ("F", 3)]:
            v = []
            for r in runs:
                f = r["folds"][0]; yc = np.array(f["ycls"]); sc = np.array(f["score"])
                v.append(float(((sc > 0) & (yc == c)).sum() / max(1, (yc == c).sum())))
            out[f"se_{cname}"], out[f"se_{cname}_sd"] = float(np.mean(v)), float(np.std(v, ddof=1))
    f0 = runs[0]["folds"][0]
    out["params"], out["macs"] = f0.get("params"), f0.get("macs")
    return out


def significance(S, ds, ref="TinyPAD", key="f1"):
    out = {}
    for m in M:
        if m == ref or S[ds].get(m) is None:
            continue
        a = np.array(S[ds][ref][key + ("_subj" if ds == "wesad" else "_seeds")]); b = np.array(S[ds][m][key + ("_subj" if ds == "wesad" else "_seeds")])
        if ds == "wesad":
            out[m] = float(stats.wilcoxon(a, b).pvalue) if not np.allclose(a, b) else 1.0
            out[m + "_wins"] = int((a > b).sum())
        else:
            out[m] = float(stats.ttest_rel(a, b).pvalue)
    return out


# ------------------------------------------------------------------------------------------------- figures
def fig2(data_dir, out):
    fig = plt.figure(figsize=(14, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1, 1.6])
    mw = json.load(open(os.path.join(data_dir, "wesad_meta.json"))); mm = json.load(open(os.path.join(data_dir, "mitbih_meta.json")))
    ax = fig.add_subplot(gs[:, 0])
    subs = list(mw["per_subject"]); n = np.array([mw["per_subject"][s]["n"] for s in subs]); st = np.array([mw["per_subject"][s]["stress"] for s in subs])
    ca = np.array([mw["per_subject"][s]["calib"] for s in subs])
    ax.barh(range(len(subs)), n - st, color="#9ecae1", label="non-stress (baseline + amusement)")
    ax.barh(range(len(subs)), st, left=n - st, color="#de2d26", label="stress")
    ax.barh(range(len(subs)), ca, color="#3182bd", hatch="//", alpha=0.6, label="5-min rest calibration (part of baseline)")
    ax.set_yticks(range(len(subs))); ax.set_yticklabels([f"S{s}" for s in subs]); ax.set_xlabel("30-s windows (5-s stride)")
    ax.set_title("WESAD: windows per subject"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=6.5, ncol=1)
    ax = fig.add_subplot(gs[:, 1])
    cls = ["N", "S", "V", "F"]; x = np.arange(4)
    for i, sp in enumerate(["DS1", "DS2"]):
        ax.bar(x + (i - 0.5) * 0.38, [mm["class_counts"][sp][c] for c in cls], 0.38, label=f"{sp} ({'train/val' if sp == 'DS1' else 'test'})")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["N (normal)", "S (SVEB)", "V (VEB)", "F (fusion)"]); ax.set_ylabel("beats (log)")
    ax.set_title("MIT-BIH: AAMI beat classes"); ax.legend()
    z = np.load(os.path.join(data_dir, "wesad.npz"))
    y, sj = z["y"], z["subj"]
    i0 = np.where((sj == sj[0]) & (y == 0))[0][60]; i1 = np.where((sj == sj[0]) & (y == 1))[0][30]
    for r, (idx, lab) in enumerate([(i0, "baseline (normal)"), (i1, "stress")]):
        ax = fig.add_subplot(gs[r, 2])
        for k, (m, c) in enumerate([("chest_ecg", "#e41a1c"), ("wrist_bvp", "#377eb8"), ("chest_resp", "#4daf4a"), ("wrist_eda_temp", "#984ea3"), ("wrist_acc", "#ff7f00")]):
            sig = z["M_" + m][idx, 0]; t = np.linspace(0, 30, len(sig))
            sig = (sig - sig.mean()) / (sig.std() + 1e-9)
            ax.plot(t, sig * 0.35 - k, color=c, lw=0.6)
            ax.text(-0.5, -k, MODN[m], ha="right", va="center", fontsize=7)
        ax.set_yticks([]); ax.set_xlim(0, 30); ax.set_title(f"WESAD subject S{sj[0]}: 30-s multimodal window, {lab}", fontsize=8.5)
        ax.set_xlabel("time (s)") if r == 1 else None
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig2_data.png")); plt.close(fig)


def fig3(S, out):
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]
    subs = sorted({s for m in M for s in []} or [])
    data = [S["wesad"][m]["f1_subj"] for m in M]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    for p, m in zip(bp["boxes"], M):
        p.set_facecolor(COL[m]); p.set_alpha(0.35)
    for i, (d, m) in enumerate(zip(data, M), 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=12, color=COL[m], zorder=3)
    ax.set_xticks(range(1, len(M) + 1)); ax.set_xticklabels(M, rotation=15); ax.set_ylabel("stress F1 per left-out subject")
    ax.set_title("WESAD leave-one-subject-out: F1 of each unseen subject (mean of 3 seeds)")
    ax = axs[1]
    x = np.arange(len(M))
    for j, (k, lab) in enumerate([("sens", "Se"), ("ppv", "+P"), ("f1", "F1")]):
        ax.bar(x + (j - 1) * 0.27, [S["mitbih"][m][k] for m in M], 0.27, label=lab, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(M, rotation=35, ha="right", fontsize=7.5); ax.set_ylim(0, 1)
    ax.set_title("MIT-BIH DS2 (inter-patient): abnormal-beat detection"); ax.legend(ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig3_per_subject.png")); plt.close(fig)


def fig4(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, ds in zip(axs, ["wesad", "mitbih"]):
        for m in M:
            ys, ss = [], []
            for r in runs_of(R, ds, m):
                for f in r["folds"]:
                    sc = np.array(f["score"]); yy = np.array(f["y"])
                    if ds == "wesad":   # pool subjects after per-subject rank-free standardisation is not applied: raw scores
                        pass
                    ys += yy.tolist(); ss += sc.tolist()
            if not ys:
                continue
            fpr, tpr, _ = roc_curve(ys, ss)
            ax.plot(fpr, tpr, color=COL[m], lw=2.2 if m == "TinyPAD" else 1.1, label=f"{m} (AUC={sk_auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8); ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title(f"{DSN[ds]}: pooled ROC (all folds and seeds)"); ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig4_roc.png")); plt.close(fig)


def fig5(S, LAT, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ds in zip(axs, ["wesad", "mitbih"]):
        for m in M:
            if m in ("RF-HCF",) or m not in LAT[ds]:
                continue
            L = LAT[ds][m]
            ax.scatter(L["macs"], S[ds][m]["f1"], s=30 + L["fp32_kb"] * 2.0, color=COL[m], alpha=0.75, edgecolor="k", lw=0.5)
            ax.annotate(f"{m}\n{L['params']/1e3:.1f} k params, {L['ms']:.2f} ms", (L["macs"], S[ds][m]["f1"]), fontsize=6.5, xytext=(5, 3), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("MACs per window (log)"); ax.set_ylabel("F1 (anomaly class)")
        ax.set_title(f"{DSN[ds]}: accuracy vs. computation (marker area ∝ model size)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig5_efficiency.png")); plt.close(fig)


def cascade_table(R, ds):
    runs = runs_of(R, ds, "TinyPAD")
    targets = [x["target_recall"] for x in runs[0]["folds"][0]["cascade"]["sweep"]]
    rows = []
    for ti, t in enumerate(targets):
        vals = {k: [] for k in ["stage2_rate", "sensor_on_frac", "rel_compute", "f1", "sens", "spec", "bacc"]}
        for r in runs:
            for f in r["folds"]:
                sw = f["cascade"]["sweep"][ti]
                vals["stage2_rate"].append(sw["stage2_rate"]); vals["sensor_on_frac"].append(sw["sensor_on_frac"]); vals["rel_compute"].append(sw["rel_compute"])
                for k in ["f1", "sens", "spec", "bacc"]:
                    vals[k].append(sw["metrics"][k])
        rows.append(dict(target=t, **{k: float(np.nanmean(v)) for k, v in vals.items()}))
    st1 = {k: float(np.nanmean([f["cascade"]["stage1_alone"][k] for r in runs for f in r["folds"]])) for k in ["f1", "sens", "spec", "bacc"]}
    macs1 = runs[0]["folds"][0]["cascade"]["macs_stage1"]; macs2 = runs[0]["folds"][0]["cascade"]["macs_stage2"]
    full = {k: float(np.nanmean([f["pooled"][k] if ds == "mitbih" else np.mean([v[k] for v in f["per_subject"].values()]) for r in runs for f in r["folds"]])) for k in ["f1", "sens", "spec", "bacc"]}
    return dict(rows=rows, stage1=st1, full=full, macs1=macs1, macs2=macs2)


def fig6(C, out):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ds in zip(axs, ["wesad", "mitbih"]):
        c = C[ds]; rows = c["rows"]
        x = [r["sensor_on_frac"] if ds == "wesad" else r["rel_compute"] for r in rows]
        ax.plot(x, [r["f1"] for r in rows], "o-", color="#d62728", label="cascade F1")
        ax.plot(x, [r["sens"] for r in rows], "s--", color="#ff7f0e", label="cascade sensitivity")
        ax.plot(x, [r["spec"] for r in rows], "^:", color="#2ca02c", label="cascade specificity")
        for r, xx in zip(rows, x):
            ax.annotate(f"ρ={r['target']:.2f}", (xx, r["f1"]), fontsize=6.5, xytext=(3, -10), textcoords="offset points")
        ax.axhline(c["full"]["f1"], color="#d62728", lw=0.8, ls=":", label="always-on stage 2 F1")
        ax.axhline(c["stage1"]["f1"], color="#7f7f7f", lw=0.8, ls="-.", label="stage 1 alone F1")
        ax.set_xlabel("duty-cycled sensor on-time fraction" if ds == "wesad" else "relative computation (MACs vs always-on stage 2)")
        ax.set_ylabel("metric"); ax.set_title(f"{DSN[ds]}: energy–accuracy trade-off of the causal cascade"); ax.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_cascade.png")); plt.close(fig)


def missing_table(R, S, ds):
    out = {}
    for m in ["TinyPAD", "LF-CNN", "EF-CNN"]:
        runs = runs_of(R, ds, m)
        mods = list(runs[0]["folds"][0]["missing"].keys())
        out[m] = {}
        for mod in mods:
            if ds == "wesad":
                v = [np.mean([f["missing"][mod]["f1"] for f in r["folds"]]) for r in runs]     # pooled per fold (subject)
            else:
                v = [r["folds"][0]["missing"][mod]["f1"] for r in runs]
            out[m][mod] = float(np.mean(v))
        out[m]["none"] = float(np.mean([np.mean([f["pooled"]["f1"] for f in r["folds"]]) for r in runs]))
    return out


def fig7(MT, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for ax, ds in zip(axs, ["wesad", "mitbih"]):
        mods = ["none"] + [k for k in MT[ds]["TinyPAD"] if k != "none"]
        x = np.arange(len(mods))
        for j, m in enumerate(["EF-CNN", "LF-CNN", "TinyPAD"]):
            ax.bar(x + (j - 1) * 0.27, [MT[ds][m][k] for k in mods], 0.27, color=COL[m], label=m, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(["all sensors"] + [f"w/o {MODN[k]}" for k in mods[1:]], rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("F1 (anomaly)"); ax.set_title(f"{DSN[ds]}: one sensor missing at test time"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_missing.png")); plt.close(fig)


def ablation_summary(A, R, ds):
    names = [("full", "TinyPAD (final)"), ("no_personal_norm", "w/o personal normalisation"), ("no_attention", "mean fusion instead of attention"),
             ("no_moddrop", "w/o modality dropout"), ("concat_fusion", "concatenation fusion (LF-CNN + personal norm.)"),
             ("plus_adabn", "+ AdaBN (explored, rejected)"), ("plus_deviation_stack", "+ deviation-score stacking (explored, rejected)")]
    out = []
    for key, lab in names:
        runs = [r for r in runs_of(R, ds, "TinyPAD") if r["seed"] in (0, 1, 2)] if key == "full" else runs_of(A, ds, key)
        if not runs:
            continue
        vals = {}
        for k in ["f1", "auc", "sens", "spec", "bacc"]:
            if ds == "wesad":
                vals[k] = float(np.mean([np.mean([np.mean([v[k] for v in f["per_subject"].values()]) for f in r["folds"]]) for r in runs]))
            else:
                vals[k] = float(np.mean([r["folds"][0]["pooled"][k] for r in runs]))
        out.append(dict(key=key, label=lab, **vals))
    return out


def fig8(AB, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, ds in zip(axs, ["wesad", "mitbih"]):
        rows = AB[ds]; y = np.arange(len(rows))
        ax.barh(y, [r["f1"] for r in rows], color=["#d62728"] + ["#7f7f7f"] * (len(rows) - 1), alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5); ax.invert_yaxis()
        lo = min(r["f1"] for r in rows); ax.set_xlim(max(0, lo - 0.1), 1.0 if ds == "wesad" else min(1, max(r["f1"] for r in rows) + 0.1))
        for yi, r in zip(y, rows):
            ax.text(r["f1"] + 0.004, yi, f"{r['f1']:.3f}", va="center", fontsize=7)
        ax.set_xlabel("F1 (mean over seeds 0–2)"); ax.set_title(f"{DSN[ds]}: ablation of TinyPAD")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig8_ablation.png")); plt.close(fig)


def calib_int8(R, ds):
    runs = runs_of(R, ds, "TinyPAD")
    keys = list(runs[0]["folds"][0]["calib_len"].keys())
    cal = {}
    for k in keys:
        if ds == "wesad":
            cal[k] = float(np.mean([np.mean([f["calib_len"][k]["f1"] for f in r["folds"]]) for r in runs]))
        else:
            cal[k] = float(np.mean([r["folds"][0]["calib_len"][k]["f1"] for r in runs]))
    i8 = float(np.mean([np.mean([f["int8"]["metrics"]["f1"] for f in r["folds"]]) for r in runs]))
    fp = float(np.mean([np.mean([f["pooled"]["f1"] for f in r["folds"]]) for r in runs]))
    f0 = runs[0]["folds"][0]["int8"]
    return dict(calib=cal, int8_f1=i8, fp32_f1=fp, fp32_kb=f0["fp32_kb"], int8_kb=f0["int8_kb"])


def fig9(CI, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axs[0]
    for ds, c in [("wesad", "#d62728"), ("mitbih", "#1f77b4")]:
        ks = list(CI[ds]["calib"].keys())
        ax.plot([float(k) for k in ks], [CI[ds]["calib"][k] for k in ks], "o-", color=c, label=DSN[ds])
    ax.set_xlabel("calibration length used for personal normalisation (min)"); ax.set_ylabel("F1 (pooled per fold)")
    ax.set_title("Sensitivity to the rest-calibration length"); ax.legend()
    ax = axs[1]
    x = np.arange(2)
    ax.bar(x - 0.18, [CI[d]["fp32_f1"] for d in ["wesad", "mitbih"]], 0.36, label="fp32 weights", color="#9ecae1")
    ax.bar(x + 0.18, [CI[d]["int8_f1"] for d in ["wesad", "mitbih"]], 0.36, label="int8 weights", color="#3182bd")
    for i, d in enumerate(["wesad", "mitbih"]):
        ax.text(i, max(CI[d]["fp32_f1"], CI[d]["int8_f1"]) + 0.02, f"{CI[d]['fp32_kb']:.1f} kB → {CI[d]['int8_kb']:.1f} kB", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(["WESAD", "MIT-BIH"]); ax.set_ylim(0, 1.05); ax.set_ylabel("F1 (pooled per fold)")
    ax.set_title("Post-training int8 weight quantisation"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig9_calib_int8.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(here, "..", "results"))
    ap.add_argument("--data", default=os.path.join(here, "..", "data"))
    ap.add_argument("--out", default=os.path.join(here, "..", "paper_assets"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = load(os.path.join(a.results, "main")); A = load(os.path.join(a.results, "ablation"))
    LAT = json.load(open(os.path.join(a.results, "latency.json")))
    DEV = json.load(open(os.path.join(a.results, "dev_select.json")))
    S = {ds: {m: summarise(R, ds, m) for m in M} for ds in ["wesad", "mitbih"]}
    SIG = {ds: {k: significance(S, ds, key=k) for k in ["f1", "auc", "bacc"]} for ds in S}
    C = {ds: cascade_table(R, ds) for ds in S}
    MT = {ds: missing_table(R, S, ds) for ds in S}
    AB = {ds: ablation_summary(A, R, ds) for ds in S}
    CI = {ds: calib_int8(R, ds) for ds in S}
    fig2(a.data, a.out); fig3(S, a.out); fig4(R, a.out); fig5(S, LAT, a.out); fig6(C, a.out); fig7(MT, a.out); fig8(AB, a.out); fig9(CI, a.out)
    json.dump(dict(summary=S, significance=SIG, cascade=C, missing=MT, ablation=AB, calib_int8=CI, latency=LAT, dev_select=DEV["summary"],
                   n_runs=dict(main=len(R), ablation=len(A))), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for ds in S:
        print("\n", ds)
        for m in M:
            s = S[ds][m]
            print(f"  {m:16s} acc={s['acc']:.4f} F1={s['f1']:.4f}±{s['f1_sd']:.4f} sens={s['sens']:.4f} spec={s['spec']:.4f} AUC={s['auc']:.4f} bAcc={s['bacc']:.4f}"
                  + (f" SeS={s['se_S']:.3f} SeV={s['se_V']:.3f}" if ds == "mitbih" else "") + f"  p_f1={SIG[ds]['f1'].get(m, float('nan')):.2e}")
        print("  cascade", [(r["target"], round(r["stage2_rate"], 3), round(r["sensor_on_frac"], 3), round(r["rel_compute"], 3), round(r["f1"], 3)) for r in C[ds]["rows"]], "full", C[ds]["full"], "st1", C[ds]["stage1"])
        print("  ablation", [(r["key"], round(r["f1"], 4)) for r in AB[ds]])
        print("  calib", CI[ds])
