"""J_ij from one set of couplings against another, one panel per p.

Sets are the suffixes of the couplings folders:
    train   correlated_weights(couplings)_sampling/  (1M training shots)
    test    correlated_weights(couplings)_testing/   (1M testing shots)
    exact   correlated_weights(couplings)_exact/     (exact, fault propagation)

All sets describe the same chip (same per-CNOT rates), so any spread about
y = x is sampling noise. Pairs expected to have fewer than R joint (1,1)
events in 1M shots (exact P11 * shots) are drawn in grey: that is where the
noise lives.

Writes   {BASE}/plots/J_{X}_vs_{Y}_s_{SAMPLE}.png / .pdf

Usage:  BASE=<folder> SAMPLE=1 SIGMA_MULT=10 X=train Y=test python plot_J_train_test.py
"""
import os

import matplotlib.pyplot as plt
import numpy as np

BASE = os.environ.get("BASE", "72q_BB_cycles_1_sigma_sweep/sigma_10p")
SAMPLE = int(os.environ.get("SAMPLE", 1))
SIGMA_MULT = float(os.environ.get("SIGMA_MULT", 10))
R = int(os.environ.get("R", 100))
X = os.environ.get("X", "train")
Y = os.environ.get("Y", "test")
SETS = {"train": ("_sampling", "training, 1M shots"),
        "test": ("_testing", "testing, 1M shots"),
        "exact": ("_exact", "exact, fault propagation")}
shots = 1000000
PS = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009, 0.0011, 0.0013, 0.0015]

# chart chrome (light surface)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
WELL, POOR = "#2a78d6", "#b9b8b1"


def pairs(path):
    return {l.split(":")[0]: float(l.split(":")[1]) for l in open(path) if l.startswith("(")}


plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
})

data = []
for p in PS:
    name = f"correlated_weights_p_{p}_sig_{round(SIGMA_MULT * p, 10)}_s_{SAMPLE}.txt"
    tr = pairs(f"{BASE}/correlated_weights(couplings){SETS[X][0]}/{name}")
    te = pairs(f"{BASE}/correlated_weights(couplings){SETS[Y][0]}/{name}")
    n11 = pairs(f"{BASE}/correlated_weights_exact/{name}")
    keys = list(tr)
    data.append((p,
                 np.array([tr[k] for k in keys]),
                 np.array([te[k] for k in keys]),
                 np.array([n11[k] * shots for k in keys])))

lo = min(min(a.min(), b.min()) for _, a, b, _ in data)
hi = max(max(a.max(), b.max()) for _, a, b, _ in data)
pad = 0.05 * (hi - lo)
lim = (lo - pad, hi + pad)

fig, axes = plt.subplots(2, 4, figsize=(11, 6), sharex=True, sharey=True)
for ax, (p, jtr, jte, cnt) in zip(axes.flat, data):
    ok = cnt >= R
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=1)
    ax.scatter(jtr[~ok], jte[~ok], s=9, color=POOR, lw=0, zorder=2,
               label=f"< {R} expected joint events")
    ax.scatter(jtr[ok], jte[ok], s=9, color=WELL, lw=0, zorder=3,
               label=f"≥ {R} expected joint events")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    d = np.abs(jtr - jte)
    ax.set_title(f"p = {p}", color=INK, fontsize=10, loc="left")
    ax.text(0.97, 0.04,
            f"median |ΔJ| = {np.median(d):.3f}\n"
            f"≥{R} events: {np.median(d[ok]) if ok.any() else float('nan'):.3f}\n"
            f"pairs < {R}: {(~ok).sum()}/{len(ok)}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)

for ax in axes[1]:
    ax.set_xlabel(f"J  ({SETS[X][1]})")
for ax in axes[:, 0]:
    ax.set_ylabel(f"J  ({SETS[Y][1]})")

handles, labels = axes.flat[0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK2, lw=1, ls="--"))
labels.append("y = x")
fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False,
           bbox_to_anchor=(0.005, 0.965), markerscale=1.6)
fig.suptitle(f"Couplings J_ij, {X} vs {Y}  ·  σ = {SIGMA_MULT:g}p, sample {SAMPLE}, "
             f"540 Z-check clique pairs", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.5)

out = f"{BASE}/plots"
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    path = f"{out}/J_{X}_vs_{Y}_s_{SAMPLE}.{ext}"
    fig.savefig(path, dpi=200)
    print("saved", path)
