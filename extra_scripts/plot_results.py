"""Plots of the couplings J_ij (exact vs sampled), and of the triples.

Reads couplings_<tag>.txt, written by J_ij_couplings.py (single-qubit block, then one
row per pair: J_ij exact, J_ij sampling, HZ flag, HX flag). Pairs are colour-coded by
which checks they share:

    blue    HZ only      (pair shares a Z check only)
    orange  HX only      (pair shares an X check only)
    green   Both         (pair shares an X and a Z check)

Figures written as PDF:

    J_ij_distribution_<tag>      histogram of the exact J_ij (the two-body correlation),
                                 one row per group, with the J = 0 (independent) line
    J_ij_exact_vs_sampled_<tag>  sampled J_ij against exact J_ij, with the y = x line

and, only if P_ijk_exact_<tag>.txt (from `J_ij_couplings.py triples`) is in the same folder:

    J_abc_distribution_<tag>     histogram of the exact three-body coupling J_abc of every
                                 triple, one row per kind of triple
    P_ijk_mechanisms_<tag>       how the exact P(111) of each triple comes about: one
                                 fault flipping all three, a pair fault plus a single
                                 fault, three single faults, other

Usage:  python extra_scripts/plot_results.py [couplings_file.txt] [--out-dir DIR]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_FILE = "data/72q_BB_cycles_1/correlated_weights/couplings_p_0.0007_sig_0.007_s_1.txt"
DEFAULT_OUT = "data/72q_BB_cycles_1/plots"

# Colours (validated categorical slots 1-3) and plot chrome.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = [  # label, HZ flag, HX flag, colour
    ("HZ only", 1, 0, "#2a78d6"),
    ("HX only", 0, 1, "#eb6834"),
    ("Both HZ and HX", 1, 1, "#1baf7a"),
]
X_LABEL = "P11 = probability that both qubits of the pair have an error"


def read_table(path):
    """Read a results table: single-qubit block and pair block.

    Returns ({i: (exact, sampling)}, {(i,j): (exact, sampling, in_HZ, in_HX)})."""
    singles, pairs = {}, {}
    for line in open(path):
        if line.startswith("(") :
            label, exact, sampling, hz, hx = line.split()
            i, j = (int(x) for x in label.strip("()").split(","))
            pairs[(i, j)] = (float(exact), float(sampling), int(hz), int(hx))
        elif line[:1] == " " and line.strip():
            i, exact, sampling = line.split()
            singles[int(i)] = (float(exact), float(sampling))
    return singles, pairs


def style(ax):
    """Light surface, faint grid, no top/right frame."""
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(path, facecolor=SURFACE)
    print("wrote", path)
    plt.close(fig)


def plot_J_distribution(J_exact, masks, tag, out_dir):
    """Histogram of the exact J_ij, one row per group (shared x-axis and bins).

    J_ij = 0: the two qubits fail independently. J_ij > 0: they fail together
    more than independence predicts (correlated). J_ij < 0: less (anticorrelated).
    Each row has its own y-axis, because the HZ-only pairs all sit at J = 0."""
    bins = np.linspace(min(J_exact.min(), 0) - 0.05, J_exact.max() + 0.05, 41)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, (name, _, _, color), mask in zip(axes, GROUPS, masks):
        style(ax)
        counts, *_ = ax.hist(J_exact[mask], bins=bins, color=color, edgecolor=SURFACE,
                             linewidth=1.5)
        median = np.median(J_exact[mask])
        ax.set_ylim(0, counts.max() * 1.3)
        ax.axvline(0, color=INK2, linewidth=1.5)
        ax.axvline(median, color=INK, linestyle="--", linewidth=1.5)
        ax.text(0.985, 0.92, f"{name}  ({mask.sum()} pairs)", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="right")
        ax.text(0.985, 0.72, f"median J = {median:.3f}", transform=ax.transAxes,
                fontsize=12, va="top", ha="right")
        ax.set_ylabel("Number of pairs")
    axes[0].text(0.12, axes[0].get_ylim()[1] * 0.9, "solid line: J = 0 (independent)",
                 fontsize=12, va="top", color=INK2)
    axes[1].text(median * 1.02, axes[1].get_ylim()[1] * 0.92, "dashed line: median", fontsize=11,
                 va="top", color=INK)
    axes[-1].set_xlabel("Exact J_ij   (> 0: fail together more than independent qubits would;"
                        "  < 0: less)")
    fig.suptitle(f"Two-body correlation J_ij of every pair sharing a check   ({tag})",
                 fontsize=15)
    fig.tight_layout()
    save(fig, out_dir, f"J_ij_distribution_{tag}")


def plot_J_exact_vs_sampled(J_exact, J_samp, masks, tag, out_dir):
    """Sampled J_ij against exact J_ij, with the y = x line."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)

    # J can be negative or zero, so the axes are linear.
    lo = min(J_exact.min(), J_samp.min()) - 0.03
    hi = max(J_exact.max(), J_samp.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK2, linewidth=1.5, label="sampled = exact")
    for (name, _, _, color), mask in zip(GROUPS, masks):
        ax.scatter(J_exact[mask], J_samp[mask], s=36, color=color, edgecolor=SURFACE,
                   linewidth=0.8, label=f"{name}  ({mask.sum()} pairs)")
    ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set_aspect("equal")
    ax.set_xlabel("Exact J_ij")
    ax.set_ylabel("Sampled J_ij")
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    ax.set_title(f"Sampled vs exact J_ij   ({tag})", fontsize=14, pad=12)
    fig.tight_layout()
    save(fig, out_dir, f"J_ij_exact_vs_sampled_{tag}")


# Colours for the mechanisms (categorical slots 7, 5, 4; neutral grey for the rest).
MECHANISMS = [  # column in P_ijk_exact, label, colour
    (2, "one fault flips all three", "#6250d6"),
    (3, "pair fault + single fault", "#e87ba4"),
    (4, "three single faults", "#eda100"),
    (5, "other", "#9a9893"),
]


def read_exact_triples(path):
    """Read P_ijk_exact_<tag>.txt -> array of [P111, one, pair+single, three, other],
    the in_HZ flags, and J_abc."""
    values, hz, J_abc = [], [], []
    for line in open(path):
        if line.startswith("("):
            t = line.split()
            values.append([float(x) for x in t[1:6]])
            hz.append(int(t[6]))
            J_abc.append(float(t[8]))
    return np.array(values), np.array(hz), np.array(J_abc)


def plot_Jabc_distribution(J_abc, in_HZ, tag, out_dir):
    """Histogram of the exact J_abc, one row per kind of triple (shared x-axis, bins).

    J_abc = (a-b coupling when c has failed) - (a-b coupling when c has not failed).
    J_abc = 0: single-qubit and pair information explain the three qubits.
    J_abc > 0: the a-b link is stronger when c has failed. J_abc < 0: weaker."""
    kinds = [("HZ triples (qubits sharing a Z check)", in_HZ == 1, GROUPS[0][3]),
             ("HX triples (qubits sharing an X check)", in_HZ == 0, GROUPS[1][3])]
    bins = np.linspace(J_abc.min() - 0.1, J_abc.max() + 0.1, 46)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, dpi=150)
    fig.patch.set_facecolor(SURFACE)

    for ax, (name, mask, color) in zip(axes, kinds):
        style(ax)
        counts, *_ = ax.hist(J_abc[mask], bins=bins, color=color, edgecolor=SURFACE,
                             linewidth=1.5)
        median = np.median(J_abc[mask])
        ax.set_ylim(0, counts.max() * 1.3)
        ax.axvline(0, color=INK2, linewidth=1.5)
        ax.axvline(median, color=INK, linestyle="--", linewidth=1.5)
        ax.set_title(f"{name}  ({mask.sum()} triples),  median J_abc = {median:.2f}",
                     fontsize=13, loc="left", pad=8)
        ax.set_ylabel("Number of triples")
    axes[-1].set_xlabel("Exact J_abc   (> 0: the link between two qubits is stronger when the "
                        "third has failed;  < 0: weaker)")
    fig.suptitle(f"Three-body coupling J_abc of every triple sharing a check   ({tag})",
                 fontsize=15)
    fig.text(0.5, 0.005, "solid line: J_abc = 0 (single and pair information explain the triple)"
             "      dashed line: median", ha="center", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out_dir, f"J_abc_distribution_{tag}")


def plot_triple_mechanisms(values, in_HZ, tag, out_dir):
    """Overall split of the exact P(111) by mechanism: one bar per kind of triple.

    Each bar is 100% of the (1,1,1) events of that kind of triple (summed over
    all its triples); the colours say how those events come about."""
    kinds = [("HX triples\n(qubits sharing an X check)", in_HZ == 0),
             ("HZ triples\n(qubits sharing a Z check)", in_HZ == 1)]
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)

    for row, (name, mask) in enumerate(kinds):
        share = values[mask][:, 1:].sum(axis=0) / values[mask][:, 0].sum()
        left = 0.0
        for k, (_, _, color) in enumerate(MECHANISMS):
            ax.barh(row, share[k], left=left, height=0.55, color=color,
                    edgecolor=SURFACE, linewidth=2)
            if share[k] >= 0.08:                            # percentage inside the segment
                ax.text(left + share[k] / 2, row, f"{share[k]:.0%}", ha="center", va="center",
                        fontsize=16, fontweight="bold",
                        color="white" if color == "#6250d6" else "#0b0b0b")
            elif share[k] >= 0.01:                          # narrow segment: label above the bar
                ax.text(left + share[k] / 2, row + 0.31, f"{share[k]:.0%}", ha="center",
                        va="bottom", fontsize=13, fontweight="bold", color="#0b0b0b")
            left += share[k]

    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels([name for name, _ in kinds], fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(kinds) - 0.5)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.set_xlabel("Share of all events where all three qubits of the triple have an error")
    ax.set_title("How do all three qubits of a triple end up with an error?", fontsize=15, pad=14)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in MECHANISMS]
    labels = ["one fault flips all three qubits",
              "one fault flips two, a different fault flips the third",
              "three different faults, one qubit each",
              "other"]
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, out_dir, f"P_ijk_mechanisms_{tag}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", default=DEFAULT_FILE, help="couplings_<tag>.txt")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    tag = os.path.basename(args.file)[len("couplings_"):-len(".txt")]
    _, J_pairs = read_table(args.file)

    pairs = sorted(J_pairs)
    J_exact = np.array([J_pairs[p][0] for p in pairs])
    J_samp = np.array([J_pairs[p][1] for p in pairs])
    hz = np.array([J_pairs[p][2] for p in pairs])
    hx = np.array([J_pairs[p][3] for p in pairs])
    masks = [(hz == a) & (hx == b) for _, a, b, _ in GROUPS]

    plt.rcParams.update({"font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    plot_J_distribution(J_exact, masks, tag, args.out_dir)
    plot_J_exact_vs_sampled(J_exact, J_samp, masks, tag, args.out_dir)

    # Triple plots, only if the exact triple file is next to the couplings file.
    exact_file = os.path.join(os.path.dirname(args.file), f"P_ijk_exact_{tag}.txt")
    if os.path.exists(exact_file):
        values, in_HZ, J_abc = read_exact_triples(exact_file)
        plot_Jabc_distribution(J_abc, in_HZ, tag, args.out_dir)
        plot_triple_mechanisms(values, in_HZ, tag, args.out_dir)


if __name__ == "__main__":
    main()
