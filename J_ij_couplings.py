"""Log-odds-ratio couplings J_ij for every edge (i,j) of the clique.

For a pair of data qubits (i, j) let P11 = P(both fail), P10 = P(only i fails),
P01 = P(only j fails), P00 = P(neither fails). Then

    J_ij = log( (P11 * P00) / (P10 * P01) )

Two ways of getting the marginals that feed this formula:

    J_ij_sampling  marginals estimated by Monte Carlo (cer_tools.py already wrote
                   them as "raw_probabilities_*.txt"); we only convert them to J_ij.
    J_ij_exact     marginals computed exactly by propagating every single fault
                   through the circuit (no sampling noise).

Plus J_ij_old, the earlier definition log(P_ij / (P_i P_j)), kept for comparison.

File format (one qubit or pair per line):

    1:0.0123            single-qubit marginal  P_i   (qubits numbered from 1)
    (1,5):0.0004        pair line: P_ij  in the input, J_ij in the J_ij output

Command line:

    python J_ij_couplings.py couplings --base <folder> [--shots 1000000] [--tmp-dir TMP]
    python J_ij_couplings.py old       --base <folder> [--smoothing 1.0]
    python J_ij_couplings.py triples   --base <folder> [--tmp-dir TMP]

Or from Python (each returns {tag: (P, J)}: P = {i: P_i}, J = {(i,j): J_ij}):

    from J_ij_couplings import J_ij_sampling, J_ij_exact
    sampled = J_ij_sampling("my_data_folder")
    exact = J_ij_exact("my_data_folder")

Folder layout under <base>:

    raw_probabilities_sampling/  input, written by cer_tools.py: sampled marginals
                                  (and the pair list used by the exact method)
    code/                         HX.txt and HZ.txt, for the HZ / HX flags of each pair
    correlated_weights/           output: couplings_<tag>.txt, one file per data file,
                                  holding P_i (exact and sampling) and then J_ij (exact
                                  and sampling) with the HZ / HX flags of every pair;
                                  also the output of `triples`
    correlated_weights(couplings)_old/   output of J_ij_old
"""
import argparse
import glob
import math
import os
import pickle

import numpy as np


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

def coupling(p11, p10, p01, p00, eps):
    """J_ij from the four joint probabilities.

    eps is added to every probability so that an exact zero (e.g. an event never
    seen in the samples) does not give log(0) or a division by zero. Use
    eps = 1/shots: about one count's worth of smoothing."""
    return math.log(((p11 + eps) * (p00 + eps)) / ((p10 + eps) * (p01 + eps)))


def joint_probs(p_i, p_j, p11):
    """(P10, P01, P00) from the two single-qubit marginals and P11."""
    p10 = p_i - p11                    # i fails, j does not
    p01 = p_j - p11                    # j fails, i does not
    p00 = 1.0 - p10 - p01 - p11        # all four probabilities sum to 1
    return p10, p01, p00


def parse_pair(line):
    """'(i,j):value' -> (i, j, value)."""
    key, val = line.rsplit(":", 1)
    i, j = (int(x) for x in key.strip("()").split(","))
    return i, j, float(val)


def write_lines(path, lines):
    """Write a list of strings to a file, one per line."""
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def tag_of(path):
    """Tag of a data file: raw_probabilities_p_0.001_sig_0.01_s_3.txt -> p_0.001_sig_0.01_s_3."""
    return os.path.basename(path)[len("raw_probabilities_"):-len(".txt")]


def read_file(path):
    """Read a marginals file: ({i: value}, {(i,j): value})."""
    singles, pairs = {}, {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("("):
            i, j, value = parse_pair(line)
            pairs[(i, j)] = value
        else:
            i, value = line.split(":")
            singles[int(i)] = float(value)
    return singles, pairs


# --------------------------------------------------------------------------
# Method 1: couplings from Monte Carlo (sampled) marginals
# --------------------------------------------------------------------------

def J_ij_sampling(base, shots=1_000_000, paths=None):
    """Couplings J_ij from the sampled marginals. Writes no files.

    base   folder holding raw_probabilities_sampling/raw_probabilities_*.txt
    shots  number of Monte Carlo shots used to make those files (sets eps = 1/shots)
    paths  optional explicit list of input files (default: every file in
           {base}/raw_probabilities_sampling/)

    Returns {tag: (P, J)} with P = {i: P_i} and J = {(i,j): J_ij}."""
    eps = 1 / shots

    if paths is None:
        paths = sorted(glob.glob(f"{base}/raw_probabilities_sampling/raw_probabilities_*.txt"))

    results = {}
    for path in paths:
        P, P11 = read_file(path)              # single-qubit and pair marginals
        J = {}
        for (i, j), p11 in P11.items():
            p10, p01, p00 = joint_probs(P[i], P[j], p11)
            J[(i, j)] = coupling(p11, p10, p01, p00, eps)
        results[tag_of(path)] = (P, J)
    return results


# --------------------------------------------------------------------------
# Old definition: J_ij = log(P_ij / (P_i P_j)), kept for comparison
# --------------------------------------------------------------------------

def J_ij_old(base, shots=1_000_000, smoothing=1.0):
    """Older, simpler coupling (the former J_ij_calculation.py):

        J_ij = log( (P_ij + eps) / (P_i * P_j + eps) ),   eps = smoothing / shots

    This compares P_ij with what independent qubits would give (P_i * P_j).
    It is not the log-odds ratio used by J_ij_sampling / J_ij_exact.

    smoothing is add-c count smoothing in probability space: eps > 0 keeps J_ij
    finite for pairs that were never seen, pulling them toward J = 0.

    Reads  {base}/raw_probabilities_sampling/raw_probabilities_p_*.txt
    Writes {base}/correlated_weights(couplings)_old/  (a separate folder, so it
           never overwrites the J_ij_sampling output)"""
    eps = smoothing / shots

    in_dir = f"{base}/raw_probabilities_sampling"
    out_dir = f"{base}/correlated_weights(couplings)_old"
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(f"{in_dir}/raw_probabilities_p_*.txt"))
    if not paths:
        raise SystemExit(f"No correlated_weights files in {in_dir}")

    for path in paths:
        lines = [l.strip() for l in open(path) if l.strip()]

        # Single-qubit marginals P_i, keyed by qubit number.
        single = {int(l.split(":")[0]): float(l.split(":")[1])
                  for l in lines if not l.startswith("(")}

        out = []
        for line in lines:
            if not line.startswith("("):
                out.append(line)                    # single-qubit line: unchanged
                continue
            i, j, p_ij = parse_pair(line)
            j_ij = math.log((p_ij + eps) / (single[i] * single[j] + eps))
            out.append(f"({i},{j}):{j_ij}")

        write_lines(os.path.join(out_dir, os.path.basename(path)), out)
        print(f"{os.path.basename(path)} -> {out_dir}")


# --------------------------------------------------------------------------
# Method 2: exact couplings from fault propagation (no sampling)
# --------------------------------------------------------------------------
#
# Idea: put ONE fault at a time into the circuit, propagate it, and record
# which data qubits end up with an X error. Different fault locations are
# independent and the outcomes of one location are mutually exclusive, so the
# marginals follow in closed form. With a_g(i) = P(location g flips qubit i):
#
#     P_i        = (1 - prod_g (1 - 2 a_g(i))) / 2
#     P(i xor j) = (1 - prod_g (1 - 2 b_g(ij))) / 2,   b_g = P(g flips exactly one of i, j)
#     P11        = (P_i + P_j - P(i xor j)) / 2
#
# Fault locations (only the X part of a Pauli matters for X errors on data):
#
#     MeasZ  X before the measurement          p_meas
#     PrepZ  X after the preparation           p_init
#     IDLE   X or Y (2 of 3 Paulis)            2/3 p_idle
#     CNOT   15 Paulis, grouped by X action:   4/15 p_cnot each group
#              control only, target only, both

def fault_patterns(cycle, lin_order, data_qubits, simulate_circuitX):
    """List every single-fault outcome of the circuit.

    Returns one entry per outcome: (position, cnot key, kind, x_pattern)
      position   gate index in the circuit (each gate use is its own location)
      cnot key   (control, target) for CNOTs, else None (to look up its own rate)
      kind       'meas', 'init', 'idle' or 'cnot'
      x_pattern  boolean array over data qubits: which ones end with an X error
    """
    data_pos = [lin_order[q] for q in data_qubits]
    out = []
    for t, gate in enumerate(cycle):
        before, after = cycle[:t], cycle[t + 1:]
        name = gate[0]

        # Build the list of circuits, each with one fault inserted at gate t.
        if name == "MeasZ":
            faults = [(None, "meas", before + [("X", gate[1]), gate] + after)]
        elif name == "PrepZ":
            faults = [(None, "init", before + [gate, ("X", gate[1])] + after)]
        elif name == "IDLE":
            faults = [(None, "idle", before + [("X", gate[1])] + after)]
        elif name == "CNOT":
            key = (gate[1], gate[2])
            faults = [(key, "cnot", before + [gate, e] + after)
                      for e in (("X", gate[1]),            # control only
                                ("X", gate[2]),            # target only
                                ("XX", gate[1], gate[2]))]  # both
        else:
            continue

        # Propagate each faulty circuit and keep the resulting data-qubit X errors.
        for key, kind, circuit in faults:
            state = simulate_circuitX(circuit, lin_order)
            out.append((t, key, kind, state[data_pos].astype(bool)))
    return out


def outcome_prob(key, kind, rates, cnot_rates):
    """Probability of one fault outcome at a location."""
    if kind == "meas":
        return rates["meas"]
    if kind == "init":
        return rates["init"]
    if kind == "idle":
        return rates["idle"] * 2 / 3
    # CNOT: use this gate's own rate if it has one, else the global rate.
    return cnot_rates.get(key, rates["cnot"]) * 4 / 15


def group_by_location(patterns, rates, cnot_rates):
    """{gate position: [(probability, x_pattern), ...]} for one circuit.

    Outcomes within a location are mutually exclusive; different locations
    are independent."""
    by_location = {}
    for t, key, kind, x in patterns:
        q = outcome_prob(key, kind, rates, cnot_rates)
        by_location.setdefault(t, []).append((q, x))
    return by_location


def load_pickle(tmp_dir, tag):
    """The mydata pickle that matches a file tag such as p_0.001_sig_0.01_s_3."""
    p_sig, s = tag.rsplit("_s_", 1)
    (pkl,) = glob.glob(f"{tmp_dir}/mydata_*_{p_sig}_cycles_*_s_{s}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def J_ij_exact(base, shots=1_000_000, tmp_dir="TMP"):
    """Exact couplings J_ij, no Monte Carlo sampling noise. Writes no files.

    base      folder holding raw_probabilities_sampling/raw_probabilities_*.txt
              (used only for the list of pairs and the file names)
    shots     only sets eps = 1/shots so J_ij matches J_ij_sampling
    tmp_dir   folder with the pickled circuit/error-rate data
              (mydata_*_p_{p}_sig_{sig}_cycles_*_s_{s})

    Returns {tag: (P, J)} with P = {i: P_i} and J = {(i,j): J_ij}."""
    eps = 1 / shots

    from cer_tools import simulate_circuitX

    patterns_cache = {}      # fault patterns depend only on the circuit: reuse them
    results = {}

    for wpath in sorted(glob.glob(f"{base}/raw_probabilities_sampling/raw_probabilities_*.txt")):
        tag = tag_of(wpath)                                      # p_.._sig_.._s_N
        mydata = load_pickle(tmp_dir, tag)

        # Single-fault outcomes of this circuit (cached across files).
        circuit = mydata["num_cycles"] * mydata["cycle"]
        ckey = repr(circuit)
        if ckey not in patterns_cache:
            patterns_cache[ckey] = fault_patterns(
                circuit, mydata["lin_order"], mydata["data_qubits"], simulate_circuitX)
        patterns = patterns_cache[ckey]

        rates = mydata["error_rates"]
        cnot_rates = mydata.get("cnot_rates") or {}
        n = len(mydata["data_qubits"])

        # Group outcomes by location: exclusive within a location, independent across.
        by_location = group_by_location(patterns, rates, cnot_rates)

        # Pairs to compute, in the same order as the input file.
        pairs = list(read_file(wpath)[1])
        I = np.array([i - 1 for i, _ in pairs])        # 0-based qubit indices
        J = np.array([j - 1 for _, j in pairs])

        # Products over locations of (1 - 2a) and (1 - 2b) from the formulas above.
        prod_single = np.ones(n)
        prod_xor = np.ones(len(pairs))
        for outcomes in by_location.values():
            a = np.zeros(n)                            # P(location flips qubit i)
            b = np.zeros(len(pairs))                   # P(location flips exactly one of i, j)
            for q, x in outcomes:
                a += q * x
                b += q * (x[I] ^ x[J])
            prod_single *= 1 - 2 * a
            prod_xor *= 1 - 2 * b

        P = (1 - prod_single) / 2                      # single-qubit marginals P_i
        P_xor = (1 - prod_xor) / 2                     # P(exactly one of i, j fails)
        P11 = (P[I] + P[J] - P_xor) / 2                # both fail

        J_ij = {}
        for (i, j), p11 in zip(pairs, P11):
            p10, p01, p00 = joint_probs(P[i - 1], P[j - 1], p11)
            J_ij[(i, j)] = coupling(p11, p10, p01, p00, eps)
        results[tag] = ({i + 1: P[i] for i in range(n)}, J_ij)

    return results


# --------------------------------------------------------------------------
# Exact P(111) of triples, split by how the three errors come about
# --------------------------------------------------------------------------
#
# Look at one triple of qubits S = {a, b, c}. Every fault location g flips some
# subset of S with probability pi_g(r) (r = which of a, b, c it flips, written
# as 3 bits; r = 0 means "does not touch S"). The final error on S is the XOR
# of what every location does, so P(111) is exact. We split it by mechanism:
#
#   one fault        a single fault flips all three qubits (r = 111)
#   pair + single    one fault flips two of them, a different fault flips the third
#   three singles    three different faults each flip one of the three
#   other            everything else (e.g. faults that cancel each other, 4+ faults)
#
# With Z = prod_g pi_g(0) and w_g(r) = pi_g(r) / pi_g(0):
#   one fault      = Z * sum_g w_g(111)
#   pair + single  = Z * sum over distinct g != h of w_g(pair) w_h(single)
#   three singles  = Z * sum over distinct g, h, l of w_g(a) w_h(b) w_l(c)
#   total          = (1/8) sum_s (-1)^|s| prod_g (pi_g(0) + sum_r pi_g(r) (-1)^(r.s))

def P_ijk_exact(base, results_dir, tmp_dir="TMP"):
    """Exact P(111) of every triple, split by mechanism, for each data file.

    Triples are all sets of three qubits sharing an HZ or an HX check (from
    {base}/code/HX.txt and HZ.txt). Writes {results_dir}/P_ijk_exact_<tag>.txt
    with columns: (i,j,k)  P111  one-fault  pair+single  three-singles  other  HZ  HX
    J_abc  kappa."""
    from cer_tools import get_clique_triples, simulate_circuitX

    HX = np.loadtxt(f"{base}/code/HX.txt", dtype=int)
    HZ = np.loadtxt(f"{base}/code/HZ.txt", dtype=int)
    in_HZ = set(get_clique_triples(HZ))
    in_HX = set(get_clique_triples(HX))
    triples = sorted(in_HZ | in_HX)
    a, b, c = (np.array([t[m] for t in triples]) for m in range(3))      # 0-based qubits
    T = len(triples)

    os.makedirs(results_dir, exist_ok=True)
    patterns_cache = {}

    for wpath in sorted(glob.glob(f"{base}/raw_probabilities_sampling/raw_probabilities_*.txt")):
        tag = os.path.basename(wpath)[len("raw_probabilities_"):-len(".txt")]
        mydata = load_pickle(tmp_dir, tag)

        circuit = mydata["num_cycles"] * mydata["cycle"]
        if repr(circuit) not in patterns_cache:
            patterns_cache[repr(circuit)] = fault_patterns(
                circuit, mydata["lin_order"], mydata["data_qubits"], simulate_circuitX)
        by_location = group_by_location(
            patterns_cache[repr(circuit)], mydata["error_rates"],
            mydata.get("cnot_rates") or {})

        # Running sums over locations, for all triples at once.
        Z = np.ones(T)                       # prod_g pi_g(0)
        total_chars = np.ones((T, 8))        # prod_g F_g(s), one column per character s
        S = np.zeros((T, 8))                 # sum_g w_g(r)
        pair_terms = {(r1, r2): np.zeros(T)  # sum_g w_g(r1) w_g(r2): same-location overlaps
                      for r1, r2 in [(3, 4), (5, 2), (6, 1), (1, 2), (1, 4), (2, 4)]}
        T124 = np.zeros(T)

        signs = np.array([[(-1) ** bin(r & s_).count("1") for s_ in range(8)] for r in range(8)])
        for outcomes in by_location.values():
            pi = np.zeros((T, 8))
            for q, x in outcomes:            # restricted pattern of this outcome on each triple
                r = x[a].astype(int) + 2 * x[b].astype(int) + 4 * x[c].astype(int)
                pi[np.arange(T), r] += q
            pi[:, 0] = 1 - pi[:, 1:].sum(axis=1)          # nothing flipped on the triple
            Z *= pi[:, 0]
            total_chars *= pi @ signs                     # F_g(s) = sum_r pi_g(r) (-1)^(r.s)
            w = pi / pi[:, [0]]
            S += w
            for (r1, r2) in pair_terms:
                pair_terms[(r1, r2)] += w[:, r1] * w[:, r2]
            T124 += w[:, 1] * w[:, 2] * w[:, 4]

        parity = np.array([(-1) ** bin(s_).count("1") for s_ in range(8)])
        total = (total_chars * parity).sum(axis=1) / 8                      # exact P(111)
        one_fault = Z * S[:, 7]
        pair_single = Z * ((S[:, 3] * S[:, 4] - pair_terms[(3, 4)])         # {a,b} + {c}
                           + (S[:, 5] * S[:, 2] - pair_terms[(5, 2)])       # {a,c} + {b}
                           + (S[:, 6] * S[:, 1] - pair_terms[(6, 1)]))      # {b,c} + {a}
        three_singles = Z * (S[:, 1] * S[:, 2] * S[:, 4]
                             - pair_terms[(1, 2)] * S[:, 4] - pair_terms[(1, 4)] * S[:, 2]
                             - pair_terms[(2, 4)] * S[:, 1] + 2 * T124)
        other = total - one_fault - pair_single - three_singles

        # Full exact distribution of the three bits: P[:, r], r = a + 2b + 4c as bits
        # (the same transform as above, for every outcome r instead of only 111).
        P = total_chars @ signs.T / 8
        J_abc = np.log(P[:, 7] * P[:, 1] * P[:, 2] * P[:, 4]
                       / (P[:, 3] * P[:, 5] * P[:, 6] * P[:, 0]))
        # Connected three-point function: P111 minus every product of lower-order marginals.
        Pa, Pb, Pc = (P[:, [1, 3, 5, 7]].sum(axis=1), P[:, [2, 3, 6, 7]].sum(axis=1),
                      P[:, [4, 5, 6, 7]].sum(axis=1))
        Pab, Pac, Pbc = P[:, 3] + P[:, 7], P[:, 5] + P[:, 7], P[:, 6] + P[:, 7]
        kappa = P[:, 7] - Pa * Pbc - Pb * Pac - Pc * Pab + 2 * Pa * Pb * Pc

        lines = [f"# {tag}",
                 "# exact P(111) of triples sharing a check, split by mechanism",
                 "# one-fault: one fault flips all three | pair+single: one fault flips two, another the third",
                 "# three-singles: three faults flip one qubit each | other: the rest (cancellations, 4+ faults)",
                 "# J_abc: three-body log-odds coupling, log(P111 P100 P010 P001 / (P110 P101 P011 P000))",
                 "# kappa: P111 - Pa Pbc - Pb Pac - Pc Pab + 2 Pa Pb Pc  (connected three-point function)",
                 f"# {'(i,j,k)':<13} {'P111':>13} {'one-fault':>13} {'pair+single':>13} "
                 f"{'three-singles':>13} {'other':>13} {'HZ':>3} {'HX':>3} {'J_abc':>10} {'kappa':>13}"]
        for m, t in enumerate(triples):
            label = f"({t[0] + 1},{t[1] + 1},{t[2] + 1})"
            lines.append(f"{label:<15} {total[m]:>13.6e} {one_fault[m]:>13.6e} "
                         f"{pair_single[m]:>13.6e} {three_singles[m]:>13.6e} {other[m]:>13.6e} "
                         f"{int(t in in_HZ):>3} {int(t in in_HX):>3} {J_abc[m]:>10.5f} {kappa[m]:>13.6e}")
        out_path = os.path.join(results_dir, f"P_ijk_exact_{tag}.txt")
        write_lines(out_path, lines)
        print(f"{T} triples -> {out_path}")


# --------------------------------------------------------------------------
# The couplings file: exact and sampling side by side
# --------------------------------------------------------------------------

def write_table(path, tag, single_title, pair_title, singles, pairs, pair_fmt, in_HZ, in_HX):
    """Write one side-by-side table (exact next to sampling).

    singles  (name_a, values_a, name_b, values_b) for the single-qubit block,
             values are {i: value}
    pairs    (name_a, values_a, name_b, values_b) for the pair block,
             values are {(i,j): value}
    pair_fmt format of a pair value, e.g. ".8f" or ".8e"
    in_HZ / in_HX  sets of pairs sharing a Z / an X check (the last two columns)"""
    name_a, single_a, name_b, single_b = singles
    lines = [f"# {tag}",
             f"# {single_title}",
             f"# {'i':>3} {name_a:>16} {name_b:>16}"]
    for i in sorted(single_a):
        lines.append(f"{i:>5} {single_a[i]:>16.8e} {single_b[i]:>16.8e}")

    name_a, pair_a, name_b, pair_b = pairs
    lines += ["",
              f"# {pair_title}",
              f"# {'(i,j)':<9} {name_a:>14} {name_b:>14} {'HZ':>3} {'HX':>3}"]
    for pair in sorted(pair_a):
        label = f"({pair[0]},{pair[1]})"
        lines.append(f"{label:<11} {pair_a[pair]:>14{pair_fmt}} {pair_b[pair]:>14{pair_fmt}} "
                     f"{int(pair in in_HZ):>3} {int(pair in in_HX):>3}")
    write_lines(path, lines)


def write_couplings(base, out_dir=None, shots=1_000_000, tmp_dir="TMP"):
    """Compute the exact and the sampled couplings and write one file per data file.

    Needs {base}/code/HX.txt and HZ.txt (written by cer_tools.py), which give
    the pairs that share a Z check (HZ) or an X check (HX). Written to
    {out_dir} (default {base}/correlated_weights), for each data file <tag>:

      couplings_<tag>.txt
          single-qubit block:  i   P_i exact   P_i sampling
          pair block:          (i,j)   J_ij exact   J_ij sampling   in HZ   in HX

    Pairs in both HZ and HX have 1 in both of the last two columns."""
    from cer_tools import get_clique_edges           # light import: no side effects

    out_dir = out_dir or f"{base}/correlated_weights"
    os.makedirs(out_dir, exist_ok=True)

    sampling = J_ij_sampling(base, shots)
    exact = J_ij_exact(base, shots, tmp_dir)

    # Pairs sharing a check, 1-based like the data files.
    HX = np.loadtxt(f"{base}/code/HX.txt", dtype=int)
    HZ = np.loadtxt(f"{base}/code/HZ.txt", dtype=int)
    in_HZ = {(a + 1, b + 1) for a, b in get_clique_edges(HZ)}
    in_HX = {(a + 1, b + 1) for a, b in get_clique_edges(HX)}

    for tag in sorted(sampling):
        P_samp, J_samp = sampling[tag]
        P_exact, J_exact = exact[tag]
        assert J_samp.keys() == J_exact.keys(), "sampling and exact pair lists differ"

        write_table(os.path.join(out_dir, f"couplings_{tag}.txt"), tag,
                    "single-qubit error rates P_i",
                    "pairs: J_ij exact, J_ij sampling, belongs to HZ (1/0), belongs to HX (1/0)",
                    ("P_i exact", P_exact, "P_i sampling", P_samp),
                    ("J_exact", J_exact, "J_sampling", J_samp),
                    ".8f", in_HZ, in_HX)

        print(f"{len(J_exact)} pairs -> {out_dir}/couplings_{tag}.txt")


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute J_ij couplings.")
    parser.add_argument("method", choices=["couplings", "old", "triples"])
    parser.add_argument("--base", default=os.environ.get("BASE"),
                        help="data folder (default: $BASE)")
    parser.add_argument("--shots", type=int, default=1_000_000,
                        help="Monte Carlo shots used for the marginals (sets eps=1/shots)")
    parser.add_argument("--smoothing", type=float, default=1.0,
                        help="old only: count smoothing c, eps = c / shots")
    parser.add_argument("--tmp-dir", default="TMP",
                        help="folder with the mydata_* pickles (couplings, triples)")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the files (default: <base>/correlated_weights)")
    args = parser.parse_args()

    if not args.base:
        parser.error("give --base <folder> or set the BASE environment variable")
    out_dir = args.out_dir or f"{args.base}/correlated_weights"

    if args.method == "couplings":
        write_couplings(args.base, out_dir, args.shots, args.tmp_dir)
    elif args.method == "old":
        J_ij_old(args.base, args.shots, args.smoothing)
    else:
        P_ijk_exact(args.base, out_dir, args.tmp_dir)
