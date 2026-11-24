"""Compute every in-text value from summary.json / run records and build the Word manuscript."""
import glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_health as B

S = B.S
RES = os.path.join(B.ROOT, "runpod_results", "results", "main")


def pooled_auc(ds, m):
    ys, ss = [], []
    for f in glob.glob(os.path.join(RES, f"{ds}_{m}_s*.json")):
        r = json.load(open(f))
        for fo in r["folds"]:
            ys += fo["y"]; ss += fo["score"]
    return roc_auc_score(ys, ss)


def text_values(V):
    T = {}
    sm = S["summary"]
    f4 = lambda x: f"{x:.4f}"; f3 = lambda x: f"{x:.3f}"
    # WESAD extras
    T["wesad_rf_spec"] = f4(sm["wesad"]["RF-HCF"]["spec"]); T["wesad_rf_auc"] = f4(sm["wesad"]["RF-HCF"]["auc"])
    T["wesad_lf_auc"] = f4(sm["wesad"]["LF-CNN"]["auc"]); T["wesad_lf_sens"] = f4(sm["wesad"]["LF-CNN"]["sens"]); T["wesad_lf_spec"] = f4(sm["wesad"]["LF-CNN"]["spec"])
    sig = S["significance"]["wesad"]["f1"]
    T["wesad_p_rf"] = B.pfmt(sig["RF-HCF"]); T["wesad_wins_rf"] = str(S["significance"]["wesad"]["f1"]["RF-HCF_wins"])
    def holm(ds):
        sg = S["significance"][ds]["f1"]; ms = [m for m in sg if not m.endswith("_wins")]
        p = np.array([sg[m] for m in ms]); o = np.argsort(p); adj = {}; run = 0
        for r, i in enumerate(o):
            run = max(run, min(1.0, (len(p) - r) * p[i])); adj[ms[i]] = run
        return adj
    HW, HM = holm("wesad"), holm("mitbih")
    T["wesad_p_rf_holm"] = B.pfmt(HW["RF-HCF"]); T["wesad_p_holm_max4"] = B.pfmt(max(HW[m] for m in ["CAE", "EF-CNN", "CNN-LSTM", "TinyTransformer"]))
    T["mitbih_p_lf"] = B.pfmt(S["significance"]["mitbih"]["f1"]["LF-CNN"]); T["mitbih_p_lf_holm"] = B.pfmt(HM["LF-CNN"])
    T["wesad_p_max_sig"] = B.pfmt(max(sig[m] for m in ["CAE", "EF-CNN", "CNN-LSTM", "TinyTransformer"]))
    T["wesad_pool_rf"] = f3(pooled_auc("wesad", "RF-HCF")); T["wesad_pool_t"] = f3(pooled_auc("wesad", "TinyPAD")); T["wesad_pool_lf"] = f3(pooled_auc("wesad", "LF-CNN"))
    T["wesad_cae_f1"] = f4(sm["wesad"]["CAE"]["f1"]); T["mitbih_cae_f1"] = f4(sm["mitbih"]["CAE"]["f1"])
    # MIT-BIH extras
    others = [m for m in B.M if m != "TinyPAD"]
    T["mitbih_t_seS"] = f3(sm["mitbih"]["TinyPAD"]["se_S"])
    T["mitbih_min_seS"] = f3(min(sm["mitbih"][m]["se_S"] for m in others if m != "CAE"))
    T["mitbih_max_seS_other"] = f3(max(sm["mitbih"][m]["se_S"] for m in others))
    # missing
    ms = S["missing"]
    for ds in ["wesad", "mitbih"]:
        mods = [k for k in ms[ds]["TinyPAD"] if k != "none"]
        for m, tag in [("TinyPAD", "t"), ("LF-CNN", "lf"), ("EF-CNN", "ef")]:
            T[f"{ds}_miss_{tag}_mean"] = f3(np.mean([ms[ds][m][k] for k in mods])); T[f"{ds}_miss_{tag}_worst"] = f3(min(ms[ds][m][k] for k in mods))
    T["mitbih_miss_t_rhythm"] = f3(ms["mitbih"]["TinyPAD"]["rhythm"]); T["mitbih_miss_t_lead1"] = f3(ms["mitbih"]["TinyPAD"]["lead1"])
    T["mitbih_miss_t_lead2"] = f3(ms["mitbih"]["TinyPAD"]["lead2"]); T["mitbih_miss_ef_lead2"] = f3(ms["mitbih"]["EF-CNN"]["lead2"])
    # cascade
    C = S["cascade"]
    row = lambda ds, t: [r for r in C[ds]["rows"] if abs(r["target"] - t) < 1e-9][0]
    r98 = row("wesad", 0.98)
    T["wesad_c98_on"] = f"{100 * r98['sensor_on_frac']:.1f} %"; T["wesad_c98_rate"] = f"{100 * r98['stage2_rate']:.1f} %"
    T["wesad_c98_comp"] = f"{100 * r98['rel_compute']:.1f} %"; T["wesad_c98_f1"] = f4(r98["f1"])
    T["wesad_full_f1"] = f4(C["wesad"]["full"]["f1"]); T["wesad_st1_f1"] = f4(C["wesad"]["stage1"]["f1"])
    r90 = row("mitbih", 0.90)
    T["mitbih_c90_rate"] = f"{100 * r90['stage2_rate']:.1f} %"; T["mitbih_c90_comp"] = f"{100 * r90['rel_compute']:.1f} %"
    T["mitbih_c90_f1"] = f4(r90["f1"]); T["mitbih_c90_sens"] = f4(r90["sens"]); T["mitbih_full_f1"] = f4(C["mitbih"]["full"]["f1"])
    # dev selection + ablation
    dv = S["dev_select"]
    T["dev_w_logit"] = f3(dv["wesad"]["logit"]["f1"]); T["dev_m_logit"] = f3(dv["mitbih"]["logit"]["f1"])
    T["dev_w_adabn"] = f3(dv["wesad"]["logit+adabn"]["f1"]); T["dev_w_stack"] = f3(dv["wesad"]["stack_raw"]["f1"])
    ab = {ds: {r["key"]: r for r in S["ablation"][ds]} for ds in ["wesad", "mitbih"]}
    for ds, t in [("wesad", "w"), ("mitbih", "m")]:
        for k, n in [("full", "full"), ("no_personal_norm", "nopn"), ("no_moddrop", "nomd"), ("no_attention", "noatt"), ("plus_adabn", "adabn"), ("plus_deviation_stack", "dev")]:
            T[f"abl_{t}_{n}"] = f3(ab[ds][k]["f1"])
    ci = S["calib_int8"]
    T["cal_w_05"] = f3(ci["wesad"]["calib"]["0.5"]); T["cal_w_2"] = f3(ci["wesad"]["calib"]["2"]); T["cal_w_5"] = f3(ci["wesad"]["calib"]["5"])
    T["cal_m_05"] = f3(ci["mitbih"]["calib"]["0.5"]); T["cal_m_5"] = f3(ci["mitbih"]["calib"]["5"])
    T["q_w_fp"] = f3(ci["wesad"]["fp32_f1"]); T["q_w_i8"] = f3(ci["wesad"]["int8_f1"]); T["q_m_fp"] = f3(ci["mitbih"]["fp32_f1"]); T["q_m_i8"] = f3(ci["mitbih"]["int8_f1"])
    T["q_w_kb"] = f"{ci['wesad']['int8_kb']:.1f}"; T["q_m_kb"] = f"{ci['mitbih']['int8_kb']:.1f}"
    # conclusions
    T["CONCL_OBJ1"] = (f"Personal baseline normalisation from five minutes of unlabelled rest data gave TinyPAD the highest stress F1 ({V['wesad_t_f1']}), accuracy, "
                       f"sensitivity and balanced accuracy on unseen WESAD subjects, and the highest sensitivity ({V['mitbih_t_sens']}), balanced accuracy and AUC on unseen MIT-BIH patients, "
                       f"with {T['mitbih_t_seS']} of supraventricular beats detected; removing personalisation lowered F1 to {T['abl_w_nopn']} and {T['abl_m_nopn']}, respectively.")
    T["CONCL_OBJ2"] = (f"The detector needs {V['mitbih_params']}–{V['wesad_params']} parameters ({V['mitbih_kb8']}–{V['wesad_kb8']} kB with int8 weights, with less than one point of F1 change) "
                       f"and a few milliseconds per window on one CPU thread, and it keeps an average F1 of {T['wesad_miss_t_mean']} on WESAD when any single sensor is missing, compared with {T['wesad_miss_lf_mean']} for a late-fusion CNN.")
    T["CONCL_OBJ3"] = (f"The causal cascade reduced the on-time of the duty-cycled sensors to {T['wesad_c98_on']} of the monitored time on WESAD (F1 {T['wesad_c98_f1']} versus {T['wesad_full_f1']} always-on), "
                       f"and on MIT-BIH reduced computation to {T['mitbih_c90_comp']} while raising F1 to {T['mitbih_c90_f1']}; the recall target ρ provides a single control for the energy–sensitivity trade-off.")
    return T


if __name__ == "__main__":
    B.build(text_values)
