"""Prepare WESAD and MIT-BIH into a common multimodal window format.

Output data/<name>.npz with
  M_<modality>  float32 (N, C, L)   one array per sensor modality
  y             int64   (N,)        1 = health anomaly (stress / abnormal beat), 0 = normal
  ycls          int64   (N,)        fine-grained class (WESAD: 1 baseline, 2 stress, 3 amusement; MIT-BIH AAMI N,S,V,F = 0..3)
  subj          int64   (N,)        subject / patient id
  calib         bool    (N,)        window belongs to the person's label-free calibration segment
  t             float32 (N,)        window start time in seconds
plus data/<name>_meta.json.

WESAD (Schmidt et al., 2018): chest RespiBAN (700 Hz) + wrist Empatica E4. Windows of 30 s, stride 5 s; a window is kept
if >= 90 % of its label samples share one of the classes baseline/stress/amusement. Anomaly = stress. Calibration = the
first 5 min of each subject's baseline (resting) condition, i.e. data that a user would record at rest without labels.

MIT-BIH Arrhythmia (PhysioNet): 48 two-lead Holter records at 360 Hz. AAMI beat classes; paced records 102,104,107,217
excluded; inter-patient split DS1/DS2 (de Chazal et al.). Anomaly = S, V or F beat (Q beats excluded). Calibration = beats
in the first 5 min of each record.
"""
import json, os, pickle, sys, zipfile, io
import numpy as np
from scipy.signal import resample_poly

OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw")
os.makedirs(OUT, exist_ok=True)


def rs(x, fs_in, fs_out):
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=0)


# ----------------------------------------------------------------------------------------------------- WESAD
WESAD_MOD = {   # name: (source, keys, target Hz)
    "chest_ecg": ("chest", ["ECG"], 100),
    "chest_resp": ("chest", ["Resp"], 10),
    "chest_eda_temp": ("chest", ["EDA", "Temp"], 4),
    "chest_acc": ("chest", ["ACC"], 32),
    "wrist_bvp": ("wrist", ["BVP"], 32),
    "wrist_eda_temp": ("wrist", ["EDA", "TEMP"], 4),
    "wrist_acc": ("wrist", ["ACC"], 32),
}
WRIST_FS = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
WIN, STRIDE, CAL_MIN = 30, 5, 5


def prep_wesad():
    zf = zipfile.ZipFile(os.path.join(RAW, "WESAD.zip"))
    pkls = sorted(n for n in zf.namelist() if n.endswith(".pkl"))
    store = {m: [] for m in WESAD_MOD}
    Y, YC, S, C, T = [], [], [], [], []
    for pk in pkls:
        d = pickle.load(io.BytesIO(zf.read(pk)), encoding="latin1")
        sid = int(str(d["subject"]).strip("S"))
        lab = np.asarray(d["label"]).astype(int)              # 700 Hz
        n700 = len(lab)
        sig = {}
        for m, (src, keys, fs) in WESAD_MOD.items():
            chans = []
            for k in keys:
                x = np.asarray(d["signal"][src][k], dtype=np.float64)
                x = x.reshape(len(x), -1)
                fin = 700 if src == "chest" else WRIST_FS[k]
                chans.append(rs(x, fin, fs))
            L = min(len(c) for c in chans)
            sig[m] = np.concatenate([c[:L] for c in chans], axis=1).astype(np.float32)   # (time, C)
        dur = n700 / 700.0
        base_idx = np.where(lab == 1)[0]
        base_start = base_idx[0] / 700.0 if len(base_idx) else 0.0
        t0 = 0.0
        while t0 + WIN <= dur:
            seg = lab[int(t0 * 700):int((t0 + WIN) * 700)]
            vals, cnt = np.unique(seg, return_counts=True)
            c = vals[np.argmax(cnt)]
            if c in (1, 2, 3) and cnt.max() >= 0.9 * len(seg):
                ok = True
                cuts = {}
                for m, (src, keys, fs) in WESAD_MOD.items():
                    a, b = int(round(t0 * fs)), int(round(t0 * fs)) + WIN * fs
                    if b > len(sig[m]):
                        ok = False; break
                    cuts[m] = sig[m][a:b].T
                if ok:
                    for m in WESAD_MOD:
                        store[m].append(cuts[m])
                    Y.append(int(c == 2)); YC.append(int(c)); S.append(sid); T.append(t0)
                    C.append(bool(c == 1 and base_start <= t0 and t0 + WIN <= base_start + CAL_MIN * 60))
            t0 += STRIDE
        print("WESAD", pk, "windows so far", len(Y), flush=True)
    arrs = {f"M_{m}": np.stack(v).astype(np.float32) for m, v in store.items()}
    np.savez_compressed(os.path.join(OUT, "wesad.npz"), y=np.array(Y), ycls=np.array(YC), subj=np.array(S), calib=np.array(C),
                        t=np.array(T, dtype=np.float32), **arrs)
    meta = dict(name="WESAD", modalities={m: dict(channels=arrs[f"M_{m}"].shape[1], length=arrs[f"M_{m}"].shape[2], hz=WESAD_MOD[m][2],
                                                  source=WESAD_MOD[m][0], signals=WESAD_MOD[m][1]) for m in WESAD_MOD},
                n_windows=len(Y), subjects=sorted(set(S)), anomaly_rate=float(np.mean(Y)),
                per_subject={int(s): dict(n=int(np.sum(np.array(S) == s)), stress=int(np.sum((np.array(S) == s) & (np.array(Y) == 1))),
                                          calib=int(np.sum((np.array(S) == s) & np.array(C)))) for s in sorted(set(S))},
                window_s=WIN, stride_s=STRIDE, calib_min=CAL_MIN)
    json.dump(meta, open(os.path.join(OUT, "wesad_meta.json"), "w"), indent=1)
    print("WESAD done", len(Y), "windows; stress rate", np.mean(Y))


# ----------------------------------------------------------------------------------------------------- MIT-BIH
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
AAMI = {**{s: 0 for s in "NLRej"}, **{s: 1 for s in "aJAS"}, **{s: 2 for s in "VE"}, "F": 3}
FS, FS2 = 360, 180


def prep_mitbih():
    import wfdb
    beats_a, beats_b, rr, Y, YC, S, C, T, split = [], [], [], [], [], [], [], [], []
    for rec in DS1 + DS2:
        r = wfdb.rdrecord(os.path.join(RAW, "mitdb", str(rec)))
        a = wfdb.rdann(os.path.join(RAW, "mitdb", str(rec)), "atr")
        sig = rs(r.p_signal.astype(np.float64), FS, FS2)       # (n, 2) at 180 Hz
        idx = [(s, sym) for s, sym in zip(a.sample, a.symbol) if sym in AAMI or sym in "/fQ"]
        rpos = np.array([s for s, _ in idx]); syms = [sym for _, sym in idx]
        rr_all = np.diff(rpos) / FS
        for i in range(1, len(rpos) - 1):
            sym = syms[i]
            if sym not in AAMI:
                continue
            c = AAMI[sym]
            p = int(round(rpos[i] * FS2 / FS))
            a0, b0 = p - 45, p + 83                               # 0.25 s before, 0.46 s after -> 128 samples
            if a0 < 0 or b0 > len(sig):
                continue
            pre, post = rr_all[i - 1], rr_all[i]
            loc = rr_all[max(0, i - 10):i].mean()
            glob = rr_all[:i].mean()
            beats_a.append(sig[a0:b0, 0][None]); beats_b.append(sig[a0:b0, 1][None])
            rr.append(np.array([pre, post, loc, glob, post / max(pre, 1e-3)], dtype=np.float32)[None])
            Y.append(int(c > 0)); YC.append(c); S.append(rec); T.append(rpos[i] / FS)
            C.append(bool(rpos[i] / FS <= CAL_MIN * 60)); split.append(1 if rec in DS1 else 2)
        print("MIT-BIH", rec, len(Y), flush=True)
    arrs = dict(M_lead1=np.stack(beats_a).astype(np.float32), M_lead2=np.stack(beats_b).astype(np.float32),
                M_rhythm=np.stack(rr).astype(np.float32))
    np.savez_compressed(os.path.join(OUT, "mitbih.npz"), y=np.array(Y), ycls=np.array(YC), subj=np.array(S), calib=np.array(C),
                        t=np.array(T, dtype=np.float32), split=np.array(split), **arrs)
    Yn, YCn, Sn, spn = map(np.array, (Y, YC, S, split))
    meta = dict(name="MIT-BIH", modalities=dict(lead1=dict(channels=1, length=128, hz=FS2, signals=["MLII (lead 1)"]),
                                               lead2=dict(channels=1, length=128, hz=FS2, signals=["second lead (V1/V2/V4/V5)"]),
                                               rhythm=dict(channels=1, length=5, hz=None, signals=["pre-RR", "post-RR", "local RR", "global RR", "RR ratio"])),
                n_beats=len(Y), DS1=DS1, DS2=DS2,
                class_counts={sp: {k: int(((YCn == v) & (spn == i)).sum()) for k, v in dict(N=0, S=1, V=2, F=3).items()} for sp, i in [("DS1", 1), ("DS2", 2)]},
                calib_min=CAL_MIN)
    json.dump(meta, open(os.path.join(OUT, "mitbih_meta.json"), "w"), indent=1)
    print("MIT-BIH done", len(Y), "beats; abnormal rate", np.mean(Y), meta["class_counts"])


if __name__ == "__main__":
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    if which in ("all", "mitbih"):
        prep_mitbih()
    if which in ("all", "wesad"):
        prep_wesad()
