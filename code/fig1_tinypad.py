"""Figure 1: working model of TinyPAD (drawn programmatically)."""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_assets")
os.makedirs(OUT, exist_ok=True)
fig, ax = plt.subplots(figsize=(15, 8.4)); ax.set_xlim(0, 150); ax.set_ylim(0, 84); ax.axis("off")


def box(x, y, w, h, t, fc, ec="#333", fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arr(x1, y1, x2, y2, c="#333", ls="-", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color=c, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# sensors
ax.text(12, 81, "Wearable IoT sensor network", ha="center", fontsize=9.5, weight="bold", color="#08519c")
sens = [("Wrist: accelerometer", "#c6dbef", True), ("Wrist: EDA + skin temp.", "#c6dbef", True), ("Wrist: BVP / PPG", "#fdd0a2", False),
        ("Chest: ECG", "#fdd0a2", False), ("Chest: respiration", "#fdd0a2", False), ("Chest: EDA + temp.", "#fdd0a2", False),
        ("Chest: accelerometer", "#fdd0a2", False)]
for i, (t, c, cheap) in enumerate(sens):
    box(2, 72 - i * 9.6, 20, 7, t + ("\n(always on, low power)" if cheap else "\n(duty-cycled)"), c, fs=7.2)
ax.text(12, 3.5, "MIT-BIH: RR-interval features (always on)\n+ lead-1 / lead-2 ECG beats", ha="center", fontsize=7.2, style="italic")

# stage 1
ax.add_patch(FancyBboxPatch((27, 56), 30, 22, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#f7fbff", ec="#2171b5", lw=1.6))
ax.text(42, 75.5, "① Stage-1 (always-on) TinyPAD", ha="center", fontsize=8.6, weight="bold", color="#08519c")
ax.text(42, 65.5, "cheap modalities only\npersonalised score $p_1$\n\nflag if $p_1\\geq\\tau$  (τ set for ≥98 %\nrecall on validation people)", ha="center", va="center", fontsize=7.6)
for i in range(2):
    arr(22, 75.5 - i * 9.6, 27, 68)
ax.add_patch(FancyBboxPatch((27, 41), 30, 11, boxstyle="round,pad=0.3,rounding_size=1.5", fc="#fff5eb", ec="#d94801", lw=1.4))
ax.text(42, 46.5, "② Sensor gating: wake duty-cycled sensors\nfor a 60-s hold after a flag (causal)", ha="center", va="center", fontsize=7.6)
arr(42, 56, 42, 52.3, c="#d94801")
for i in range(2, 7):
    arr(27, 46.5, 22.3, 75.5 - i * 9.6, c="#d94801", ls="--", lw=1.0)

# stage 2 TinyPAD
ax.add_patch(FancyBboxPatch((62, 6), 56, 72, boxstyle="round,pad=0.3,rounding_size=2", fc="#fcfbfd", ec="#54278f", lw=1.8))
ax.text(90, 75.6, "③ Stage-2 TinyPAD (full multimodal, personalised)", ha="center", fontsize=9.2, weight="bold", color="#3f007d")
box(65, 62, 50, 9, "Personal input normalisation: per-person z-score from the label-free\n5-min rest calibration; RR features divided by the person's mean RR", "#e5f5e0", fs=7.4)
for k in range(5):
    box(65 + k * 10.2, 47, 9.2, 11, f"DS-CNN\nencoder\n$E_{k+1}$", "#dadaeb", fs=7.2)
ax.text(90, 45, "one tiny depthwise-separable 1-D CNN (or MLP for RR vectors) per modality → 32-d embedding $e_m$", ha="center", fontsize=7)
box(65, 31, 50, 11, "Masked modality-attention fusion\n$\\alpha_m=\\mathrm{softmax}_m(w^\\top\\tanh(We_m))$ over available sensors,  $z=\\sum_m\\alpha_m e_m$\n(trained with random modality dropout → robust to missing sensors)", "#fde0dd", fs=7.2)
box(65, 18, 50, 10, "Classifier head  $h$:  logit margin  $\\ell(x)=h_1(z)-h_0(z)$\ntrained with class-weighted cross-entropy on training people", "#e5f5e0", fs=7.4)
box(65, 8, 50, 7, "Training only: random modality dropout ($p_{md}=0.25$) + early stopping on validation people", "#fff7bc", fs=7.2)
arr(90, 31, 90, 28.3)

# fusion/output
box(123, 46, 25, 16, "④ Decision\n$\\hat y=\\mathbb{1}[\\ell(x)>0]$\n(stage 2 if sensors on,\nelse stage-1 output)", "#fde0dd", fs=7.6)
arr(115, 23, 123, 49)
box(123, 22, 25, 16, "⑤ Real-time alert\nstress episode /\nabnormal heartbeat\n(int8 model, ≈9–24 k params)", "#fee391", fs=7.6, bold=False)
arr(135.5, 46, 135.5, 38.3)
box(123, 66, 25, 10, "Output of stage 1 used\nwhen stage 2 is not run", "#f0f0f0", fs=7.2)
arr(57, 70, 123, 71, c="#2171b5", ls="--")
arr(135.5, 66, 135.5, 62.3, c="#2171b5", ls="--")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_architecture.png"), dpi=300)
print("saved")
