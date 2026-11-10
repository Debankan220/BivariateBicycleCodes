"""Plot the distribution of two-qubit couplings J_ij over clique edges.

Reads the coupling files written by J_ij_couplings.py, which contain
single-qubit lines "q:P_q" followed by pair lines "(i,j):J_ij", and writes one
histogram per file showing how much J_ij varies across edges.

Each plot is saved as <same folder>/plots/<same filename>.png.

Usage:
    python plot_J_ij_histogram.py                      # every file in the default folder
    python plot_J_ij_histogram.py path/to/file.txt     # one or more explicit files
    python plot_J_ij_histogram.py --bins 60
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, norm

DEFAULT_DIR = "72q_BB_cycles_1_spread_comparison/correlated_weights(couplings)_sampling"

SURFACE = "#fcfcfb"
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
GRID = "#e3e3e0"


def read_couplings(path):
    """Return (J_ij array, edge list) from one couplings file."""

    values = []
    edges = []

    for line in open(path):
        line = line.strip()

        if not line or not line.startswith("("):
            continue                      # single-qubit marginal, not a coupling

        label, val = line.rsplit(":", 1)
        i, j = map(int, label.strip("()").split(","))

        edges.append((i, j))
        values.append(float(val))

    return np.array(values), edges


def describe(path):
    """Pull p / sigma / sample out of the filename for the subtitle."""

    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    fields = {}

    for key in ("p", "sig", "s"):
        if key in parts:
            fields[key] = parts[parts.index(key) + 1]

    bits = []
    if "p" in fields:
        bits.append(f"p = {fields['p']}")
    if "sig" in fields:
        bits.append(f"$\\sigma$ = {fields['sig']}")
    if "s" in fields:
        bits.append(f"sample {fields['s']}")

    return "   ".join(bits)


def plot_one(path, out_dir=None):
    values, edges = read_couplings(path)

    if values.size == 0:
        print(f"SKIP {path}: no coupling lines found")
        return None

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(path), "plots")

    os.makedirs(out_dir, exist_ok=True)

    mean = values.mean()
    median = np.median(values)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # One point per edge, in the order the edges appear in the file, joined by a
    # line. The line carries the shape; the markers show where each edge sits.
    x = np.arange(len(values))

    ax.plot(x, values, color=SERIES, linewidth=0.8, alpha=0.55, zorder=1)
    ax.scatter(x, values, s=9, color=SERIES, linewidths=0, zorder=2)

    ax.axhline(0, color=INK, linewidth=0.9, alpha=0.6)
    ax.axhline(median, color=INK, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.annotate(
        f"median {median:.2f}",
        xy=(1.0, median), xycoords=("axes fraction", "data"),
        xytext=(-3, 3), textcoords="offset points",
        fontsize=8, color=INK_MUTED, ha="right", va="bottom",
    )

    ax.set_xlim(-4, len(values) + 3)
    ax.set_xlabel("clique edge (file order)", fontsize=11, color=INK)
    ax.set_ylabel("$J_{ij}$", fontsize=11, color=INK)

    ax.set_title(
        "Two-qubit coupling per clique edge",
        fontsize=13, color=INK, loc="left", pad=18,
    )
    ax.annotate(
        f"{len(values)} edges   {describe(path)}",
        xy=(0, 1.0), xycoords="axes fraction",
        xytext=(0, 8), textcoords="offset points",
        fontsize=9, color=INK_MUTED, ha="left", va="bottom",
    )

    stats = (f"min {values.min():.2f}   max {values.max():.2f}   "
             f"mean {mean:.2f}   std {values.std():.2f}")
    ax.annotate(
        stats,
        xy=(1.0, 1.0), xycoords="axes fraction",
        xytext=(0, 8), textcoords="offset points",
        fontsize=9, color=INK_MUTED, ha="right", va="bottom",
    )

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    out_path = os.path.join(out_dir, os.path.basename(path).replace(".txt", "") + ".png")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)

    print(f"{os.path.basename(path)}: n={len(values)} "
          f"min={values.min():.3f} max={values.max():.3f} "
          f"mean={mean:.3f} std={values.std():.3f}")
    print("  saved", out_path)

    return out_path


def plot_distribution(path, bins, out_dir=None):
    """Density view: histogram + smoothed KDE, against a normal of the same
    mean and standard deviation. The normal is a reference, not a fit claim --
    it shows at a glance how far the couplings are from bell shaped."""

    values, _ = read_couplings(path)

    if values.size == 0:
        return None

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(path), "plots")

    os.makedirs(out_dir, exist_ok=True)

    mean, std = values.mean(), values.std(ddof=1)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.hist(values, bins=bins, density=True, color=SERIES, alpha=0.30,
            edgecolor=SURFACE, linewidth=0.4)

    grid = np.linspace(values.min() - 0.5, values.max() + 0.5, 600)

    kde = gaussian_kde(values)
    ax.plot(grid, kde(grid), color=SERIES, linewidth=2, label="observed (KDE)")

    ax.plot(grid, norm.pdf(grid, mean, std), color=SERIES_2, linewidth=2,
            linestyle="--", label=f"normal ($\\mu$={mean:.2f}, $\\sigma$={std:.2f})")

    ax.set_xlabel("$J_{ij}$", fontsize=11, color=INK)
    ax.set_ylabel("density", fontsize=11, color=INK)

    ax.set_title("Distribution of two-qubit couplings",
                 fontsize=13, color=INK, loc="left", pad=18)
    ax.annotate(
        f"{len(values)} edges   {describe(path)}",
        xy=(0, 1.0), xycoords="axes fraction",
        xytext=(0, 8), textcoords="offset points",
        fontsize=9, color=INK_MUTED, ha="left", va="bottom",
    )

    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for text in leg.get_texts():
        text.set_color(INK)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    out_path = os.path.join(
        out_dir, os.path.basename(path).replace(".txt", "") + "_distribution.png")

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)

    print("  saved", out_path)

    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*",
                    help=f"Coupling files (default: every .txt in {DEFAULT_DIR})")
    ap.add_argument("--bins", type=int, default=40,
                    help="Number of bins in the distribution plot")
    ap.add_argument("--out", default=None,
                    help="Output folder (default: a 'plots' folder beside each input)")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.txt")))

    if not files:
        raise SystemExit(f"No coupling files found in {DEFAULT_DIR}")

    for path in files:
        plot_one(path, args.out)
        plot_distribution(path, args.bins, args.out)


if __name__ == "__main__":
    main()
