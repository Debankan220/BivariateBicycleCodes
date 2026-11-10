"""Minimum-count filtered marginals and couplings J_ij.

A qubit or pair is kept only if it has at least R events in the training
shots; its rate is then the full-count estimate over all of them:

    P_i  = (shots where qubit i fails) / shots
    P11  = (shots where i and j fail together) / shots

Qubits or pairs with fewer than R events are dropped, and so is any pair
whose J_ij cannot be formed (a dropped qubit, or a non-positive
P10 / P01 / P00).

J_ij uses the same formula and eps as J_ij_couplings.py (J_ij_sampling):

    J_ij = log( (P11 * P00) / (P10 * P01) )

Reads    {BASE}/training_data/train_*.txt  (pairs from raw_probabilities_sampling/)
Writes   {BASE}/correlated_weights_threshold/
         {BASE}/correlated_weights(couplings)_threshold/
         {BASE}/threshold_summary.txt

Usage:  BASE=<folder> python J_ij_threshold.py
"""
import glob
import math
import os

import numpy as np

R = int(os.environ.get("R", 100))
shots = 1000000
eps = 1/shots
BASE = os.environ.get("BASE", "72q_BB_cycles_1_sigma_sweep/sigma_10p")

w_dir = f"{BASE}/correlated_weights_threshold"
j_dir = f"{BASE}/correlated_weights(couplings)_threshold"
os.makedirs(w_dir, exist_ok=True)
os.makedirs(j_dir, exist_ok=True)


def load_errors(path):
    """train_*.txt is n rows x shots columns of '0'/'1' separated by spaces.
    Returns a (shots, n) bool array."""
    raw = np.fromfile(path, dtype=np.uint8)
    rows = raw.reshape(-1, raw.size // raw[raw == ord("\n")].size)
    return (rows[:, 0:-1:2] == ord("1")).T


def rate_at_R(hits):
    """Full-count rate, or None if there are fewer than R events."""
    count = np.count_nonzero(hits)
    return count / hits.size if count >= R else None


summary = [f"# R = {R}; columns: file, qubits kept/total, pairs kept/total, "
           f"couplings kept/total"]

for train in sorted(glob.glob(f"{BASE}/training_data/train_*.txt")):
    tag = os.path.basename(train)[len("train_"):]
    name = f"raw_probabilities_{tag}"
    E = load_errors(train)
    n = E.shape[1]

    # pair list (1-indexed) taken from the full-count weights file
    pairs = [tuple(int(x) for x in l.split(":")[0].strip("()").split(","))
             for l in open(f"{BASE}/raw_probabilities_sampling/{name}") if l.startswith("(")]

    single = {i + 1: rate_at_R(E[:, i]) for i in range(n)}
    p11 = {(i, j): rate_at_R(E[:, i - 1] & E[:, j - 1]) for i, j in pairs}

    w_out = [f"{i}:{p}" for i, p in single.items() if p is not None]
    w_out += [f"({i},{j}):{p}" for (i, j), p in p11.items() if p is not None]

    j_out = [f"{i}:{p}" for i, p in single.items() if p is not None]
    for (i, j), q11 in p11.items():
        if q11 is None or single[i] is None or single[j] is None:
            continue
        q10 = single[i] - q11
        q01 = single[j] - q11
        q00 = 1.0 - q10 - q01 - q11
        if min(q10, q01, q00) <= 0:
            continue
        j_ij = math.log(((q11+eps) * (q00+eps)) / ((q10+eps) * (q01+eps)))
        j_out.append(f"({i},{j}):{j_ij}")

    with open(os.path.join(w_dir, name), "w") as f:
        f.write("\n".join(w_out) + "\n")
    with open(os.path.join(j_dir, name), "w") as f:
        f.write("\n".join(j_out) + "\n")

    n_q = sum(p is not None for p in single.values())
    n_p = sum(p is not None for p in p11.values())
    n_j = len(j_out) - n_q
    summary.append(f"{name} {n_q}/{n} {n_p}/{len(pairs)} {n_j}/{len(pairs)}")
    print(f"{name}: qubits {n_q}/{n}  pairs {n_p}/{len(pairs)}  couplings {n_j}/{len(pairs)}")

with open(f"{BASE}/threshold_summary.txt", "w") as f:
    f.write("\n".join(summary) + "\n")
