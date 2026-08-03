"""
Compute smoothed two-qubit couplings J_ij from the correlated-weights marginals.

For each clique edge (i, j):

    J_ij = log( (P_ij + eps) / (P_i * P_j + eps) ),   eps = c / N

This is add-c count smoothing written in probability space (multiplying the
numerator and denominator by N recovers (n_ij + c) / (N P_i P_j + c)). With
eps > 0 it is always finite: under-sampled pairs (P_ij ~ 0) are pulled toward
J = 0, while well-measured pairs are barely affected. Natural logarithm.

Input : <base>/correlated_weights/correlated_weights_p_{p}_s_{s}.txt
        (single-qubit lines "q:P_q", then pair lines "(i,j):P_ij")
Output: <base>/correlated_weights(couplings)/<same filename>
        single-qubit lines unchanged; each P_ij replaced by J_ij.
"""

import argparse
import glob
import math
import os


def process_file(in_path, out_dir, eps):
    lines = open(in_path).read().splitlines()

    # single-qubit marginals: qubit -> P_q
    singles = {int(line.split(":")[0]): float(line.split(":")[1])
               for line in lines if line and not line.startswith("(")}

    out = []
    for line in lines:
        if not line.strip():
            continue
        if not line.startswith("("):
            out.append(line)                      # single-qubit marginal, unchanged
            continue
        label, p_ij = line.split(":")
        i, j = map(int, label.strip("()").split(","))
        j_ij = math.log((float(p_ij) + eps) / (singles[i] * singles[j] + eps))
        out.append(f"({i},{j}):{j_ij}")

    out_path = os.path.join(out_dir, os.path.basename(in_path))
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="72q_BB_cycles_3",
                    help="Base dir containing the correlated_weights folder")
    ap.add_argument("--shots", type=int, default=1_000_000,
                    help="N: number of shots the marginals came from")
    ap.add_argument("--smoothing", type=float, default=1.0,
                    help="count smoothing c; eps = c / shots (default 1 -> eps = 1/N)")
    args = ap.parse_args()

    in_dir = os.path.join(args.base, "correlated_weights")
    out_dir = os.path.join(args.base, "correlated_weights(couplings)")
    os.makedirs(out_dir, exist_ok=True)
    eps = args.smoothing / args.shots

    files = sorted(glob.glob(os.path.join(in_dir, "correlated_weights_p_*.txt")))
    if not files:
        raise SystemExit(f"No correlated_weights files in {in_dir}")

    for f in files:
        process_file(f, out_dir, eps)
        print(f"{os.path.basename(f)} -> couplings")

    print(f"\nWrote {len(files)} file(s) to '{out_dir}' (eps = {eps:g}).")


if __name__ == "__main__":
    main()
