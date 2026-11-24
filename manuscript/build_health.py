"""Assemble the healthcare manuscript (.docx). All numbers come from paper_assets/summary.json (computed by
analyze_health.py from the RunPod run records); references from refs/refs_verified.json (Crossref-verified)."""
import json, os, re
import numpy as np
import pypandoc
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ASSETS = os.path.join(ROOT, "paper_assets")
S = json.load(open(os.path.join(ASSETS, "summary.json")))
REFV = json.load(open(os.path.join(ROOT, "refs", "refs_verified.json"), encoding="utf-8"))
M = ["RF-HCF", "CAE", "EF-CNN", "LF-CNN", "CNN-LSTM", "TinyTransformer", "TinyPAD"]
LABEL = {m: m for m in M}; LABEL["TinyPAD"] = "**TinyPAD (proposed)**"
DSN = {"wesad": "WESAD", "mitbih": "MIT-BIH"}
KEY2DOI = {
    "abusamah2025": "10.3390/electronics14040687", "ponnada2024": "10.1109/access.2024.3496791", "kim2023tinyml": "10.32604/cmc.2023.031663",
    "gedam2021": "10.1109/access.2021.3085502", "chen2021": "10.3390/s21041030", "dalmeida2021": "10.3390/s21082873",
    "taskasaplidis2024": "10.1109/access.2024.3373010", "pinge2024": "10.3389/fcomp.2024.1478851", "bolpagni2024": "10.3390/s24103221",
    "rashid2023": "10.1109/jiot.2023.3265768", "choi2025": "10.3390/electronics14040723", "almadhor2025": "10.3389/fbioe.2025.1659002",
    "an2024": "10.3390/s24247896", "kim2022": "10.1109/embc48229.2022.9871156", "li2022": "10.1016/j.cmpb.2021.106582",
    "zakaria2024": "10.1109/access.2024.3469640", "prabha2025": "10.1038/s41598-025-30150-x", "choi2025ecg": "10.1109/tim.2025.3547110",
    "zhang2025cop": "10.1109/tbcas.2025.3530790", "liu2025tce": "10.1109/tce.2025.3569456", "liu2025jssc": "10.1109/jssc.2025.3555512",
    "zhang2024bdscnn": "10.3390/s24227207", "fatimah2024": "10.1016/j.medengphy.2024.104102", "nazari2025": "10.1109/iscas56072.2025.11043898",
    "zhou2023": "10.3389/fphys.2023.1247587", "xiang2025": "10.3389/fphys.2025.1584299", "li2023pilot": "10.1109/jsen.2023.3247341",
    "zhang2024ecgp": "10.1109/tbcas.2024.3385993", "yoo2025": "10.1109/jbhi.2025.3589142", "tang2021": "10.1155/2021/6217601",
    "schmidt2018": "10.1145/3242969.3242985", "moody2001": "10.1109/51.932724", "dechazal2004": "10.1109/tbme.2004.827359",
    "li2018adabn": "10.1016/j.patcog.2018.03.005",
}
MANUAL = {}
NAME_OVERRIDE = {}
exec(open(os.path.join(HERE, "_helpers.py"), encoding="utf-8").read())


def fmt(m, s, d=4):
    return f"{m:.{d}f} ± {s:.{d}f}"


def mt(ds, cols, caption, extra=None):
    rows = [S["summary"][ds][m] for m in M]
    best = {k: best_idx([r[k] for r in rows], hi) for k, hi, _ in cols}
    sig = S["significance"][ds]["f1"]
    body = []
    for i, (m, r) in enumerate(zip(M, rows)):
        cells = [LABEL[m]]
        for k, hi, _ in cols:
            c = fmt(r[k], r[k + "_sd"]) if r.get(k + "_sd") is not None else f"{r[k]:.4f}"
            cells.append(f"**{c}**" if best[k] == i else c)
        p = sig.get(m)
        cells.append("–" if m == "TinyPAD" else pfmt(p) + (f" ({sig.get(m + '_wins')}/15)" if ds == "wesad" else ""))
        body.append(cells)
    hdr = ["Method"] + [c[2] for c in cols] + ["p (F1)" + (" [wins]" if ds == "wesad" else "")]
    return pipe_table(hdr, body, caption, widths=[18] + [15] * len(cols) + [12])


def compute_values():
    V = {}
    mw = json.load(open(os.path.join(ROOT, "data", "wesad_meta.json"))); mm = json.load(open(os.path.join(ROOT, "data", "mitbih_meta.json")))
    V["wesad_n_windows"] = f"{mw['n_windows']:,}"; V["wesad_stress_pct"] = f"{100 * mw['anomaly_rate']:.1f}"
    V["wesad_calib"] = str(list(mw["per_subject"].values())[0]["calib"])
    V["n_main_runs"] = str(S["n_runs"]["main"]); V["n_abl_runs"] = str(S["n_runs"]["ablation"])
    V["n_main_models"] = f"{7 * (3 * 15 + 5):,}"
    for ds in ["wesad", "mitbih"]:
        sm = S["summary"][ds]
        bb = max((m for m in M if m != "TinyPAD"), key=lambda m: sm[m]["f1"])
        V[f"{ds}_bb"] = bb
        for k in ["acc", "f1", "sens", "spec", "auc", "bacc", "ppv", "macro_f1"]:
            V[f"{ds}_t_{k}"] = f"{sm['TinyPAD'][k]:.4f}"; V[f"{ds}_b_{k}"] = f"{sm[bb][k]:.4f}"
        V[f"{ds}_gain_f1"] = f"{100 * (sm['TinyPAD']['f1'] - sm[bb]['f1']):.2f}"
        V[f"{ds}_p_bb"] = pfmt(S["significance"][ds]["f1"][bb])
        if ds == "wesad":
            V["wesad_wins_bb"] = str(S["significance"]["wesad"]["f1"][bb + "_wins"])
        L = S["latency"][ds]
        V[f"{ds}_params"] = f"{L['TinyPAD']['params']:,}"; V[f"{ds}_macs"] = f"{L['TinyPAD']['macs']:,}"
        V[f"{ds}_ms"] = f"{L['TinyPAD']['ms']:.2f}"; V[f"{ds}_kb8"] = f"{L['TinyPAD']['int8_kb']:.1f}"
        V[f"{ds}_kb32"] = f"{L['TinyPAD']['fp32_kb']:.1f}"
        V[f"{ds}_s1_macs"] = f"{L['TinyPAD stage-1']['macs']:,}"
    return V


exec(open(os.path.join(HERE, "tables_health.py"), encoding="utf-8").read())


def build(text_values):
    V = compute_values(); V.update(text_values(V))
    parts = [open(os.path.join(HERE, f), encoding="utf-8").read() for f in
             ["part0_front.md", "part1_intro_related_method.md", "part2_results.md", "part2b_results.md", "part3_end.md"]]
    text = "\n\n".join(parts)
    for k, v in make_tables().items():
        text = text.replace("{{" + k + "}}", v)
    text = fill(text, V)
    text = text.replace("p ≤ <", "p < ").replace("p = <", "p < ")
    text = re.sub(r"\\tag\{(\d+)\}", r"\\qquad\\qquad (\1)", text)
    text, refs, order = number_citations(text)
    text += "\n\n# References\n\n" + "\n\n".join(refs) + "\n"
    md = os.path.join(HERE, "manuscript_full.md"); open(md, "w", encoding="utf-8").write(text)
    out = os.path.join(HERE, "TinyPAD_manuscript.docx")
    pypandoc.convert_file(md, "docx", outputfile=out, extra_args=["--resource-path=" + ASSETS + os.pathsep + HERE, "--reference-doc=" + os.path.join(HERE, "reference.docx")])
    postprocess(out)
    recent = sum(1 for k in order if REFV[KEY2DOI[k]]["year"] >= 2021)
    print("written", out, "| references:", len(refs), "| 2021+ DOI refs:", recent)
    json.dump(dict(values=V, citation_order=order), open(os.path.join(HERE, "build_log.json"), "w"), indent=1)
