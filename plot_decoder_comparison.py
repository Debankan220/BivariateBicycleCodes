"""
Compare decoder logical error rates: BP+OSD vs plain BP.

Reads per-p result text files from a <base>/results folder:
  * BP+OSD decoder : ..._BP+OSD_failure_rates_OSD_E_order_2.txt
  * plain BP       : ..._BP_failure_rates.txt

The set of physical error rates is discovered from the BP+OSD result files
present; any p missing a BP or BP+OSD file is skipped.

Figure: logical error rate vs physical error rate p, for BP and BP+OSD.

NOTE: the Neural-BP (CSV) series and the performance-gain panel are commented
out for now (no NN CSV yet for the 72q code). Re-enable them once the CSV exists.
"""

import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# import csv  # only needed for the (disabled) NN CSV reader

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

base_directory = "90q_BB_cycles_1"
code_label = "90q_BB"          # prefix used in the per-p .txt filenames
num_cycles = 1

results_dir = os.path.join(base_directory, "results")
#nn_csv = os.path.join(results_dir, "72q_neural_fix.csv")
output_png = os.path.join(results_dir, "decoder_comparison_90q.png")

# Okabe-Ito colourblind-safe palette (fixed order, one hue per entity)
COL_OSD = "#0072B2"   # blue
COL_BP = "#D55E00"    # vermillion
COL_NN = "#009E73"    # green


# ------------------------------------------------------------
# Readers
# ------------------------------------------------------------

def read_txt_result(path):
    """Read a per-p result file: p sample failures trials LER std."""
    row = np.loadtxt(path, comments="#")
    _, _, failures, trials, ler, std = row
    return {
        "failures": int(failures),
        "trials": int(trials),
        "ler": float(ler),
        "std": float(std),
    }


# DISABLED (no NN CSV yet for 72q): parse the neural-network results.
# def read_nn_csv(path):
#     """Return {p_string: {...}} parsed from the neural-network CSV."""
#     import csv
#     nn = {}
#     with open(path, newline="") as f:
#         for r in csv.DictReader(f):
#             m = re.search(r"test_p_([0-9.]+)_s", r["error_model_parameters_description"])
#             if not m:
#                 continue
#             p_str = m.group(1)
#             trials = int(r["num_samples_per_error_rate"])
#             failures = int(r["num_failures"])
#             nn[p_str] = {
#                 "failures": failures,
#                 "trials": trials,
#                 "ler": float(r["average_logical_error_rate"]),
#                 "std": float(r["std_logical_error_rate"]),
#             }
#     return nn


def binom_se(res):
    """Binomial standard error of the mean: sqrt(p_L (1 - p_L) / N)."""
    ler, N = res["ler"], res["trials"]
    return np.sqrt(ler * (1.0 - ler) / N)


def discover_p_strings():
    """Physical error rates (as strings) that have a BP+OSD result file."""
    pattern = os.path.join(
        results_dir,
        f"{code_label}_p_*_cycles_{num_cycles}_BP+OSD_failure_rates_OSD_E_order_2.txt",
    )
    p_strings = set()
    for path in glob.glob(pattern):
        m = re.search(r"_p_([0-9.]+)_cycles_", os.path.basename(path))
        if m:
            p_strings.add(m.group(1))
    return sorted(p_strings, key=float)


# ------------------------------------------------------------
# Assemble the comparison, driven by the available BP+OSD result files
# ------------------------------------------------------------

def collect():
    ps = []
    osd_ler, osd_std = [], []
    bp_ler, bp_std = [], []

    for p_str in discover_p_strings():
        osd_path = os.path.join(
            results_dir,
            f"{code_label}_p_{p_str}_cycles_{num_cycles}_"
            f"BP+OSD_failure_rates_OSD_E_order_2.txt",
        )
        bp_path = os.path.join(
            results_dir,
            f"{code_label}_p_{p_str}_cycles_{num_cycles}_BP_failure_rates.txt",
        )

        if not (os.path.exists(osd_path) and os.path.exists(bp_path)):
            missing = [q for q, e in (("BP+OSD", osd_path), ("BP", bp_path))
                       if not os.path.exists(e)]
            print(f"SKIP p={p_str}: missing {', '.join(missing)} result(s).")
            continue

        osd = read_txt_result(osd_path)
        bp = read_txt_result(bp_path)

        ps.append(float(p_str))
        osd_ler.append(osd["ler"]); osd_std.append(binom_se(osd))
        bp_ler.append(bp["ler"]); bp_std.append(binom_se(bp))

    return {
        "p": np.array(ps),
        "osd_ler": np.array(osd_ler), "osd_std": np.array(osd_std),
        "bp_ler": np.array(bp_ler), "bp_std": np.array(bp_std),
    }


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

def main():
    d = collect()
    if d["p"].size == 0:
        raise SystemExit("No p values with both BP and BP+OSD results found; nothing to plot.")

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Logical error rate vs p
    common = dict(capsize=2, markersize=6, linewidth=1.5, elinewidth=1)
    ax1.errorbar(d["p"], d["bp_ler"], yerr=d["bp_std"], marker="^",
                 color=COL_BP, label="BP", **common)
    ax1.errorbar(d["p"], d["osd_ler"], yerr=d["osd_std"], marker="s",
                 color=COL_OSD, label="BP+OSD", **common)
    # DISABLED (no NN CSV yet): Neural-BP series.
    # ax1.errorbar(d["p"], d["nn_ler"], yerr=d["nn_std"], marker="o",
    #              color=COL_NN, label="Neural BP", **common)
    ax1.set_yscale("log")
    ax1.set_xlabel("Physical error rate  p")
    ax1.set_ylabel("Logical error rate")
    ax1.set_title(f"Decoder comparison — {code_label}, {num_cycles} cycle(s)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(frameon=False)

    # DISABLED (no NN CSV yet): performance-gain panel.
    # ax2.axhline(1.0, color="0.6", linewidth=1, linestyle="--")
    # ax2.plot(d["p"], d["gain_osd"], marker="s", color=COL_OSD,
    #          linewidth=1.5, markersize=6, label="NN / BP+OSD")
    # ax2.plot(d["p"], d["gain_bp"], marker="^", color=COL_BP,
    #          linewidth=1.5, markersize=6, label="NN / BP")
    # ax2.set_xscale("log"); ax2.set_xlabel("Physical error rate  p")
    # ax2.set_ylabel("Performance gain\n(success ratio)")
    # ax2.grid(True, which="both", alpha=0.25); ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    print(f"Plotted {d['p'].size} error rate(s). Saved figure to {output_png}")


if __name__ == "__main__":
    main()
