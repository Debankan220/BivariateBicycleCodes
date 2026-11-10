"""Couplings J_ij from the testing shots, in the same format as
correlated_weights(couplings)_sampling/ (which is built from the training shots).

Marginals are counted over all shots exactly as cer_tools.py does:

    P_i = mean over shots of E[:, i]          P11 = mean of E[:, i] * E[:, j]

and J_ij follows J_ij_couplings.py (J_ij_sampling) (same formula, same eps):

    P10 = P_i - P11,  P01 = P_j - P11,  P00 = 1 - P10 - P01 - P11
    J_ij = log( (P11 * P00) / (P10 * P01) )

The pair list is taken from the matching raw_probabilities_sampling/ file, so the
lines come out in the same order as the training couplings.

Reads    {BASE}/{SPLIT}_data/{prefix}_*.txt   (SPLIT=testing -> test_*.txt)
Writes   {BASE}/correlated_weights(couplings)_{SPLIT}/   (or OUT, if set)

Usage:  BASE=<folder> python J_ij_testing.py
"""
import glob
import math
import os

import numpy as np

shots = 1000000
eps = 1/shots
BASE = os.environ.get("BASE", "72q_BB_cycles_1_sigma_sweep/sigma_10p")
SPLIT = os.environ.get("SPLIT", "testing")
prefix = {"testing": "test", "training": "train"}[SPLIT]

out_dir = os.environ.get("OUT", f"{BASE}/correlated_weights(couplings)_{SPLIT}")
os.makedirs(out_dir, exist_ok=True)


def load_errors(path):
    """{prefix}_*.txt is n rows x shots columns of '0'/'1' separated by spaces.
    Returns a (shots, n) uint8 array."""
    raw = np.fromfile(path, dtype=np.uint8)
    rows = raw.reshape(-1, raw.size // raw[raw == ord("\n")].size)
    return (rows[:, 0:-1:2] == ord("1")).T.astype(np.uint8)


for data in sorted(glob.glob(f"{BASE}/{SPLIT}_data/{prefix}_*.txt")):
    tag = os.path.basename(data)[len(prefix) + 1:]
    name = f"raw_probabilities_{tag}"
    E = load_errors(data)

    pairs = [tuple(int(x) for x in l.split(":")[0].strip("()").split(","))
             for l in open(f"{BASE}/raw_probabilities_sampling/{name}") if l.startswith("(")]

    one_body = E.mean(axis=0)
    single = {q + 1: p_i for q, p_i in enumerate(one_body)}

    out = [f"{i}:{p_i}" for i, p_i in single.items()]
    for i, j in pairs:
        p11 = np.mean(E[:, i - 1] * E[:, j - 1])
        p10 = single[i] - p11
        p01 = single[j] - p11
        p00 = 1.0 - p10 - p01 - p11

        j_ij = math.log(((p11+eps) * (p00+eps)) /((p10+eps) * (p01 + eps)))
        out.append(f"({i},{j}):{j_ij}")

    with open(os.path.join(out_dir, name), "w") as f:
        f.write("\n".join(out) + "\n")

    print(f"{len(pairs)} edges -> {os.path.join(out_dir, name)}")
