# Executed inside build_health.py (shares its namespace: S, M, LABEL, DSN, ROOT, pipe_table, mt, np, json, os).

def make_tables():
    T = {}
    mw = json.load(open(os.path.join(ROOT, "data", "wesad_meta.json"))); mm = json.load(open(os.path.join(ROOT, "data", "mitbih_meta.json")))
    names = {"chest_ecg": "Chest ECG", "chest_resp": "Chest respiration", "chest_eda_temp": "Chest EDA + skin temperature", "chest_acc": "Chest 3-axis accelerometer",
             "wrist_bvp": "Wrist BVP (PPG)", "wrist_eda_temp": "Wrist EDA + skin temperature", "wrist_acc": "Wrist 3-axis accelerometer",
             "lead1": "ECG lead 1 (MLII) beat", "lead2": "ECG second-lead beat", "rhythm": "RR-rhythm vector"}
    cheap = {"wrist_acc", "wrist_eda_temp", "rhythm"}
    rows = []
    for m, v in mw["modalities"].items():
        rows.append(["WESAD", names[m], v["channels"], v["hz"], v["length"], "always-on (stage 1)" if m in cheap else "duty-cycled (stage 2)"])
    for m, v in mm["modalities"].items():
        rows.append(["MIT-BIH", names[m], v["channels"], v["hz"] or "–", v["length"], "always-on (stage 1)" if m in cheap else "stage 2"])
    rows.append(["WESAD", f"{mw['n_windows']:,} windows, {len(mw['subjects'])} subjects, {100 * mw['anomaly_rate']:.1f} % stress", "", "", "", "LOSO"])
    c1, c2 = mm["class_counts"]["DS1"], mm["class_counts"]["DS2"]
    rows.append(["MIT-BIH", f"DS1 N/S/V/F = {c1['N']:,}/{c1['S']:,}/{c1['V']:,}/{c1['F']:,}; DS2 = {c2['N']:,}/{c2['S']:,}/{c2['V']:,}/{c2['F']:,}", "", "", "", "inter-patient"])
    T["TABLE_DATA"] = pipe_table(["Dataset", "Modality / content", "Channels", "Rate (Hz)", "Samples per window", "Role"], rows,
                                 "Table 1. Sensor modalities, sampling rates and composition of the two datasets.",
                                 widths=[10, 34, 8, 8, 10, 16], align=["l", "l", "c", "c", "c", "l"])
    T["TABLE_WESAD"] = mt("wesad", [("acc", True, "Accuracy"), ("f1", True, "F1 (stress)"), ("sens", True, "Sensitivity"), ("spec", True, "Specificity"),
                                    ("bacc", True, "Balanced acc."), ("auc", True, "AUC")],
                          "Table 3. WESAD leave-one-subject-out results (metrics averaged over the 15 unseen subjects; mean ± s.d. over 3 seeds). Best value per column in bold. p = Wilcoxon signed-rank test of per-subject F1 against TinyPAD; [wins] = subjects on which TinyPAD's F1 is higher.")
    T["TABLE_MIT"] = mt("mitbih", [("acc", True, "Accuracy"), ("f1", True, "F1 (abnormal)"), ("sens", True, "Se"), ("ppv", True, "+P"), ("spec", True, "Sp"),
                                   ("bacc", True, "Balanced acc."), ("auc", True, "AUC"), ("se_S", True, "Se (S beats)"), ("se_V", True, "Se (V beats)")],
                        "Table 4. MIT-BIH inter-patient results on DS2 (mean ± s.d. over 5 seeds). Best value per column in bold. p = paired t-test of F1 against TinyPAD over the 5 seeds.")
    rows = []
    for ds in ["wesad", "mitbih"]:
        L = S["latency"][ds]; sm = S["summary"][ds]
        for m in M:
            if m == "RF-HCF":
                rows.append([DSN[ds], m, "300 trees", "–", "–", "–", "–", f"{sm[m]['f1']:.4f}"]); continue
            l = L[m]
            rows.append([DSN[ds], LABEL[m], f"{l['params']:,}", f"{l['macs'] / 1e6:.3f}", f"{l['fp32_kb']:.1f}", f"{l['int8_kb']:.1f}", f"{l['ms']:.2f}", f"{sm[m]['f1']:.4f}"])
        l = L["TinyPAD stage-1"]
        rows.append([DSN[ds], "TinyPAD stage 1 (always-on)", f"{l['params']:,}", f"{l['macs'] / 1e6:.4f}", f"{l['fp32_kb']:.1f}", f"{l['int8_kb']:.1f}", f"{l['ms']:.2f}", f"{S['cascade'][ds]['stage1']['f1']:.4f}"])
    T["TABLE_EFF"] = pipe_table(["Dataset", "Model", "Parameters", "MMACs / window", "fp32 size (kB)", "int8 size (kB)", "CPU latency (ms)", "F1"], rows,
                                "Table 5. Model complexity and single-thread CPU latency per window (batch size 1) compared with detection F1.",
                                widths=[10, 22, 10, 10, 9, 9, 9, 9])
    rows = []
    for ds in ["wesad", "mitbih"]:
        mtab = S["missing"][ds]; mods = [k for k in mtab["TinyPAD"] if k != "none"]
        for mod in ["none"] + mods:
            vals = [mtab[m][mod] for m in ["EF-CNN", "LF-CNN", "TinyPAD"]]; b = int(np.argmax(vals))
            rows.append([DSN[ds], "all sensors available" if mod == "none" else "without " + names[mod]] + [f"**{v:.4f}**" if i == b else f"{v:.4f}" for i, v in enumerate(vals)])
        means = [np.mean([mtab[m][k] for k in mods]) for m in ["EF-CNN", "LF-CNN", "TinyPAD"]]
        worst = [min(mtab[m][k] for k in mods) for m in ["EF-CNN", "LF-CNN", "TinyPAD"]]
        rows.append([DSN[ds], "mean over single-sensor dropouts"] + [f"**{v:.4f}**" if i == int(np.argmax(means)) else f"{v:.4f}" for i, v in enumerate(means)])
        rows.append([DSN[ds], "worst single-sensor dropout"] + [f"**{v:.4f}**" if i == int(np.argmax(worst)) else f"{v:.4f}" for i, v in enumerate(worst)])
    T["TABLE_MISS"] = pipe_table(["Dataset", "Test condition", "EF-CNN", "LF-CNN", "**TinyPAD (proposed)**"], rows,
                                 "Table 6. F1 when one sensor modality is unavailable at test time (missing channels set to zero for EF-/LF-CNN; masked out of the attention for TinyPAD). Mean over seeds; best per row in bold.",
                                 widths=[10, 36, 12, 12, 16], align=["l", "l", "c", "c", "c"])
    rows = []
    for ds in ["wesad", "mitbih"]:
        c = S["cascade"][ds]
        rows.append([DSN[ds], "Always-on stage 2 (full TinyPAD)", "1.000", "1.000", "1.000", f"{c['full']['f1']:.4f}", f"{c['full']['sens']:.4f}", f"{c['full']['spec']:.4f}", f"{c['full']['bacc']:.4f}"])
        rows.append([DSN[ds], "Stage 1 only (always-on sensors)", "0.000", "0.000", f"{c['macs1'] / c['macs2']:.3f}", f"{c['stage1']['f1']:.4f}", f"{c['stage1']['sens']:.4f}", f"{c['stage1']['spec']:.4f}", f"{c['stage1']['bacc']:.4f}"])
        for r in c["rows"]:
            on = f"{r['sensor_on_frac']:.3f}" if ds == "wesad" else "1.000 (ECG)"
            rows.append([DSN[ds], f"Cascade, ρ = {r['target']:.2f}", f"{r['stage2_rate']:.3f}", on, f"{r['rel_compute']:.3f}", f"{r['f1']:.4f}", f"{r['sens']:.4f}", f"{r['spec']:.4f}", f"{r['bacc']:.4f}"])
    T["TABLE_CASC"] = pipe_table(["Dataset", "Operating mode", "Stage-2 rate", "Duty-cycled sensor on-time", "Relative computation", "F1", "Sensitivity", "Specificity", "Balanced acc."], rows,
                                 "Table 7. Energy-aware operating modes of TinyPAD (WESAD: mean over 45 subject folds; MIT-BIH: mean over 5 seeds). Stage-2 rate = fraction of windows analysed by the full model; sensor on-time = fraction of the monitored time with the duty-cycled sensors powered; relative computation = (MAC₁ + stage-2 rate × MAC₂)/MAC₂.",
                                 widths=[9, 26, 9, 10, 10, 8, 9, 9, 9], align=["l", "l"] + ["c"] * 7)
    rows = []
    for ds in ["wesad", "mitbih"]:
        for r in S["ablation"][ds]:
            rows.append([DSN[ds], r["label"], f"{r['f1']:.4f}", f"{r['auc']:.4f}", f"{r['bacc']:.4f}", f"{r['sens']:.4f}", f"{r['spec']:.4f}"])
    T["TABLE_ABL"] = pipe_table(["Dataset", "Variant (evaluation seeds 0–2)", "F1", "AUC", "Balanced acc.", "Sensitivity", "Specificity"], rows,
                                "Table 9. Ablation of TinyPAD on the evaluation seeds 0–2. The final design was fixed beforehand on validation people only (Table 8); these numbers were not used for selection.",
                                widths=[10, 40, 9, 9, 10, 10, 10], align=["l", "l"] + ["c"] * 5)
    dv = S["dev_select"]; d2 = json.load(open(os.path.join(ROOT, "runpod_results", "results", "dev_select2.json")))["summary"]
    lab = {"logit": "personal normalisation only (selected)", "logit+adabn": "+ AdaBN", "stack_raw": "+ deviation-score stacking",
           "stack_z": "+ person-standardised stacking", "stack_z+adabn": "+ standardised stacking + AdaBN"}
    rows = []
    for ds in ["wesad", "mitbih"]:
        for k in lab:
            rows.append([DSN[ds], "Personalisation", lab[k], f"{dv[ds][k]['f1']:.4f}", f"{dv[ds][k]['auc']:.4f}", f"{dv[ds][k]['bacc']:.4f}"])
        best = max(d2[ds], key=lambda k: d2[ds][k]["f1"])
        for k, v in sorted(d2[ds].items(), key=lambda kv: -kv[1]["f1"]):
            nm = k.replace("_md", " fusion, p_md = ") + (" (selected)" if k == best else "")
            rows.append([DSN[ds], "Fusion / modality dropout", nm, f"{v['f1']:.4f}", f"{v['auc']:.4f}", f"{v['bacc']:.4f}"])
    T["TABLE_DEV"] = pipe_table(["Dataset", "Decision", "Variant", "Validation F1", "Validation AUC", "Validation bal. acc."], rows,
                                "Table 8. Design selection performed exclusively on validation people with development seeds (100 for personalisation; 100–102 for fusion and modality dropout); test people were never used.",
                                widths=[10, 18, 34, 11, 11, 12], align=["l", "l", "l", "c", "c", "c"])
    rows = []
    for ds in ["wesad", "mitbih"]:
        ci = S["calib_int8"][ds]
        rows.append([DSN[ds]] + [f"{ci['calib'][k]:.4f}" for k in ["0.5", "1", "2", "3", "5"]] + [f"{ci['fp32_f1']:.4f}", f"{ci['int8_f1']:.4f}", f"{ci['fp32_kb']:.1f} → {ci['int8_kb']:.1f}"])
    T["TABLE_CAL"] = pipe_table(["Dataset", "0.5 min", "1 min", "2 min", "3 min", "5 min", "F1 fp32", "F1 int8", "Size (kB)"], rows,
                                "Table 10. F1 of TinyPAD as a function of the rest-calibration length used for personal normalisation, and with int8 post-training weight quantisation (mean over seeds).",
                                widths=[10, 9, 9, 9, 9, 9, 9, 9, 12])
    return T
