"""Generate noisy-circuit shots and the sampled error marginals for a BB code.

Starting from a pickle written by decoder_setup.py (circuit, error rates, code
matrices), this script

  1. computes the logical operators of the code,
  2. simulates many noisy syndrome-extraction circuits and records which data
     qubits end with an X error (training shots and testing shots),
  3. estimates the single-qubit marginals P_i and the two-qubit marginals P_ij
     (for every pair sharing an HZ or HX check) from the training shots.

Output folders, all under --out (default: $BB_OUT):

    code/                  HX, HZ, LX, LZ, logical checks, clique edges (HZ and HX pairs),
                           clique pairs (HZ / HX / common)
    training_data/         train_<tag>.txt   (n x shots, one column per shot)
    testing_data/          test_<tag>.txt
    raw_probabilities_sampling/   raw_probabilities_<tag>.txt  (P_i and P_ij)
                                   P_ijk_results_<tag>.txt       (P(111) of triples)
    cnot_error_rates/      the per-CNOT rates used for this disorder realisation

<tag> is p_<error rate>_sig_<sigma>_s_<sample>.

Usage:
    python cer_tools.py --error-rates 0.001 --sample 1
    python cer_tools.py --only-code --error-rates 0.001   # shared code files only
    python cer_tools.py --error-rates 0.001 --skip-code   # data only (parallel jobs)
"""
import argparse
import glob
import os
import pickle
from itertools import combinations

import numpy as np
from tqdm import tqdm

# ============================================================
# Defaults (all can be overridden on the command line)
# ============================================================

# Physical error rates to process. Each needs a pickle from decoder_setup.py in TMP/.
# Full sweep: 0.0001, 0.0003, ..., 0.002, 0.0024, ..., 0.005
DEFAULT_ERROR_RATES = [0.0007]

# Pickle written by decoder_setup.py. {p} = error rate, {s} = sample (disorder
# realisation) index. The '*' absorbs the sigma field, which scales with p.
DEFAULT_MYDATA_TEMPLATE = "TMP/mydata_72_12_p_{p}_sig_*_cycles_1_s_{s}"

DEFAULT_OUT = os.environ.get("BB_OUT", "72q_BB_cycles_1_spread_comparison")
DEFAULT_SHOTS = 1_000_000          # for both training and testing
DEFAULT_SAMPLE = 3                 # disorder realisation: a fixed draw of per-CNOT rates
DEFAULT_SEED = 1234


def output_folders(out):
    """Create (if needed) and return the output sub-folders under `out`."""
    names = {
        "code": "code",
        "train": "training_data",
        "test": "testing_data",
        "weights": "raw_probabilities_sampling",
        "cnot_rates": "cnot_error_rates",
    }
    folders = {key: os.path.join(out, name) for key, name in names.items()}
    for path in folders.values():
        os.makedirs(path, exist_ok=True)
    return folders


# ============================================================
# Clique graph from HZ and HX
# ============================================================

def get_clique_edges(H):
    """Edges (q1, q2), q1 < q2 (0-based), between data qubits sharing a check of H."""
    edges = set()
    for check in H:
        support = np.where(check == 1)[0]
        for q1, q2 in combinations(support, 2):
            edges.add((int(q1), int(q2)))       # combinations keeps q1 < q2
    return sorted(edges)


def get_clique_triples(H):
    """Triples (q1, q2, q3), q1 < q2 < q3 (0-based), of data qubits that all
    share one check of H."""
    triples = set()
    for check in H:
        support = np.where(check == 1)[0]
        for triple in combinations(support, 3):
            triples.add(tuple(int(q) for q in triple))
    return sorted(triples)


def get_all_clique_edges(HX, HZ):
    """Edges from the Z checks and from the X checks together.

    A pair that is in both (a common pair) appears only once. These are the
    pairs for which P_ij and J_ij are computed."""
    return sorted(set(get_clique_edges(HZ)) | set(get_clique_edges(HX)))


# ============================================================
# GF(2) linear algebra
# ============================================================

def as_gf2(A):
    """Copy of A as a 0/1 uint8 array."""
    return np.array(A, dtype=np.uint8) % 2


def gf2_row_reduce(A, ncols=None):
    """Reduce A in place to reduced row echelon form over GF(2).

    Only the first `ncols` columns are searched for pivots (default: all), which
    lets an augmented matrix [A | I] be reduced by A's columns alone.
    Returns the list of pivot columns."""
    ncols = A.shape[1] if ncols is None else ncols
    pivots = []
    row = 0                                      # next row that will hold a pivot

    for col in range(ncols):
        # Find a row at or below `row` with a 1 in this column.
        candidates = np.nonzero(A[row:, col])[0]
        if len(candidates) == 0:
            continue
        pivot = row + candidates[0]
        A[[row, pivot]] = A[[pivot, row]]

        # Clear this column in every other row.
        others = np.nonzero(A[:, col])[0]
        others = others[others != row]
        A[others] ^= A[row]

        pivots.append(col)
        row += 1
        if row == A.shape[0]:
            break
    return pivots


def gf2_rank(A):
    """Rank of a binary matrix over GF(2)."""
    A = as_gf2(A)
    if A.ndim != 2:
        raise ValueError("gf2_rank expects a 2D matrix")
    return len(gf2_row_reduce(A))


def gf2_kernel(H):
    """Basis (as rows v) of the kernel of H over GF(2): H @ v = 0 mod 2."""
    H = as_gf2(H)
    if H.ndim != 2:
        raise ValueError("gf2_kernel expects a 2D matrix")
    n = H.shape[1]

    pivots = gf2_row_reduce(H)
    free_cols = [c for c in range(n) if c not in pivots]

    # One basis vector per free column: set it to 1, solve for the pivot columns.
    basis = np.zeros((len(free_cols), n), dtype=np.uint8)
    for b, free in enumerate(free_cols):
        basis[b, free] = 1
        basis[b, pivots] = H[:len(pivots), free]
    return basis


def gf2_inverse(A):
    """Inverse of a square binary matrix over GF(2)."""
    A = as_gf2(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("gf2_inverse expects a square matrix")
    n = A.shape[0]

    # Reduce [A | I]; the right half becomes the inverse.
    augmented = np.hstack([A, np.eye(n, dtype=np.uint8)])
    if len(gf2_row_reduce(augmented, ncols=n)) < n:
        raise ValueError("Matrix is not invertible over GF(2)")
    return augmented[:, n:]


def quotient_logicals(kernel_basis, stabilizer_matrix):
    """Keep the kernel vectors that are independent modulo the stabilizers.

    For a CSS code:  LX = kernel(HZ) modulo rowspace(HX)
                     LZ = kernel(HX) modulo rowspace(HZ)"""
    stabilizers = as_gf2(stabilizer_matrix)
    n = stabilizers.shape[1]

    current = stabilizers
    current_rank = gf2_rank(current)
    logicals = []

    for v in as_gf2(kernel_basis):
        trial = np.vstack([current, v])
        trial_rank = gf2_rank(trial)
        if trial_rank > current_rank:            # v adds something new
            logicals.append(v)
            current, current_rank = trial, trial_rank

    return np.array(logicals, dtype=np.uint8).reshape(-1, n)


def compute_css_logicals(HX, HZ):
    """Paired CSS logical operators LX and LZ with LX @ LZ.T = identity mod 2.

    HX rows are X stabilizers, HZ rows are Z stabilizers.
    Returns LX, LZ, k, rank(HX), rank(HZ), LX @ LZ.T."""
    HX, HZ = as_gf2(HX), as_gf2(HZ)
    if HX.ndim != 2 or HZ.ndim != 2:
        raise ValueError("HX and HZ must be 2D matrices")
    if HX.shape[1] != HZ.shape[1]:
        raise ValueError("HX and HZ must have the same number of columns")

    n = HX.shape[1]
    rank_HX, rank_HZ = gf2_rank(HX), gf2_rank(HZ)
    k = n - rank_HX - rank_HZ
    print(f"rank(HX) = {rank_HX}, rank(HZ) = {rank_HZ}, k = {k}")
    if k < 0:
        raise ValueError("Computed k is negative. Check HX and HZ.")

    assert np.all((HX @ HZ.T) % 2 == 0), "ERROR: HX and HZ do not commute"

    # X logicals commute with the Z stabilizers, Z logicals with the X stabilizers.
    LX = quotient_logicals(gf2_kernel(HZ), HX)
    LZ = quotient_logicals(gf2_kernel(HX), HZ)

    assert LX.shape[0] == k, "ERROR: number of LX operators does not match k"
    assert LZ.shape[0] == k, "ERROR: number of LZ operators does not match k"
    assert gf2_rank(np.vstack([HX, LX])) - rank_HX == k, "ERROR: LX not independent modulo HX"
    assert gf2_rank(np.vstack([HZ, LZ])) - rank_HZ == k, "ERROR: LZ not independent modulo HZ"
    assert np.all((LX @ HZ.T) % 2 == 0), "ERROR: LX does not commute with HZ"
    assert np.all((LZ @ HX.T) % 2 == 0), "ERROR: LZ does not commute with HX"

    # The raw pairing matrix must be invertible for the logicals to be valid.
    logical_comm = (LX @ LZ.T) % 2
    print("Raw LX @ LZ.T mod 2 =")
    print(logical_comm)
    assert gf2_rank(logical_comm) == k, "ERROR: LX and LZ do not form a full-rank pairing"

    # Change the LX basis (keeping LZ fixed) so that LX @ LZ.T = identity.
    LX = (gf2_inverse(logical_comm) @ LX) % 2
    logical_comm = (LX @ LZ.T) % 2
    print("Paired LX @ LZ.T mod 2 =")
    print(logical_comm)
    assert np.array_equal(logical_comm, np.eye(k, dtype=np.uint8)), "ERROR: failed to pair logicals"

    return LX, LZ, k, rank_HX, rank_HZ, logical_comm


# ============================================================
# Writing files
# ============================================================

def save_matrix(path, M):
    """Save a binary matrix as text, one row per line."""
    np.savetxt(path, M.astype(int), fmt="%d")


def write_code_files(code_dir, HX, HZ, LX, LZ, k, rank_HX, rank_HZ, logical_comm, clique_edges):
    """Write the files that are the same for every error rate.

    Kept separate so a cluster run can write them once (--only-code) and the
    per-rate data jobs can skip them (--skip-code), avoiding concurrent writes."""
    n = HX.shape[1]

    for name, M in [("HX", HX), ("HZ", HZ), ("LX", LX), ("LZ", LZ)]:
        save_matrix(os.path.join(code_dir, f"{name}.txt"), M)

    with open(os.path.join(code_dir, "logical_checks.txt"), "w") as f:
        f.write("Logical operator checks\n=======================\n\n")
        f.write(f"n = {n}\n")
        f.write(f"rank(HX) = {rank_HX}\n")
        f.write(f"rank(HZ) = {rank_HZ}\n")
        f.write(f"k = {k}\n")
        f.write(f"LX shape = {LX.shape}\n")
        f.write(f"LZ shape = {LZ.shape}\n\n")
        f.write("Checks:\n")
        f.write(f"HX @ HZ.T == 0: {np.all((HX @ HZ.T) % 2 == 0)}\n")
        f.write(f"LX @ HZ.T == 0: {np.all((LX @ HZ.T) % 2 == 0)}\n")
        f.write(f"LZ @ HX.T == 0: {np.all((LZ @ HX.T) % 2 == 0)}\n")
        f.write(f"LX @ LZ.T == I: {np.array_equal(logical_comm, np.eye(k, dtype=np.uint8))}\n\n")
        f.write("LX @ LZ.T mod 2:\n")
        f.write(np.array2string(logical_comm))
        f.write("\n")

    # Clique edges are written 1-based.
    with open(os.path.join(code_dir, "clique_edges.txt"), "w") as f:
        for q1, q2 in clique_edges:
            f.write(f"({q1 + 1},{q2 + 1})\n")

    write_clique_pairs(os.path.join(code_dir, "clique_pairs.txt"), HX, HZ)


def write_clique_pairs(path, HX, HZ):
    """Table of the pairs of qubits that share a check.

    Columns: pairs from HZ, pairs from HX, and the pairs common to both.
    The count of each column is at the bottom.
    Qubits are numbered from 1."""
    pairs_HZ = {(q1 + 1, q2 + 1) for q1, q2 in get_clique_edges(HZ)}
    pairs_HX = {(q1 + 1, q2 + 1) for q1, q2 in get_clique_edges(HX)}
    columns = [("HZ pairs", sorted(pairs_HZ)),
               ("HX pairs", sorted(pairs_HX)),
               ("common pairs", sorted(pairs_HZ & pairs_HX))]

    width = 14
    rule = "-" * (width * len(columns))
    lines = ["".join(f"{name:<{width}}" for name, _ in columns).rstrip(), rule]
    for r in range(max(len(pairs) for _, pairs in columns)):
        cells = [f"({p[r][0]},{p[r][1]})" if r < len(p) else "" for _, p in columns]
        lines.append("".join(f"{c:<{width}}" for c in cells).rstrip())
    lines.append(rule)
    lines.append("".join(f"{'count = ' + str(len(p)):<{width}}" for _, p in columns).rstrip())

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_cnot_rates(path, cnot_rates, mydata, error_rate, sample):
    """Save the per-CNOT error rates of this disorder realisation.

    They are drawn once in decoder_setup.py and held fixed for every shot, so
    one file fully describes the "chip" the data was generated on."""
    with open(path, "w") as f:
        f.write("# per-CNOT two-qubit error rates (fixed for all shots)\n")
        f.write(f"# p={error_rate} sigma={mydata.get('cnot_sigma')} "
                f"sample={sample} seed={mydata.get('cnot_seed')}\n")
        f.write(f"# gates={len(cnot_rates)} mean={np.mean(list(cnot_rates.values())):.8e}\n")
        f.write("# control_type control_index target_type target_index rate\n")
        for (ctrl, tgt), rate in cnot_rates.items():
            f.write(f"{ctrl[0]} {ctrl[1]} {tgt[0]} {tgt[1]} {rate:.10e}\n")


def write_triple_marginals(path, E, HX, HZ, tag=""):
    """P(111) for every triple of qubits sharing an HZ or an HX check.

    E is the shots x n matrix of data-qubit X errors (the training shots).
    P(111) = fraction of shots where all three qubits have an error.
    A triple belongs to HZ (HX) if one Z (X) check contains all three qubits.
    Columns: (i,j,k)  P111  events  in HZ  in HX  (qubits numbered from 1)."""
    in_HZ = set(get_clique_triples(HZ))
    in_HX = set(get_clique_triples(HX))
    triples = sorted(in_HZ | in_HX)

    shots = E.shape[0]
    Et = np.ascontiguousarray(E.T)               # one contiguous row per qubit
    lines = [f"# {tag}",
             "# triples of qubits sharing a check: P(111) from the sampled shots,",
             "# number of shots with all three failing, belongs to HZ (1/0), belongs to HX (1/0)",
             f"# {'(i,j,k)':<13} {'P111':>14} {'events':>8} {'HZ':>3} {'HX':>3}"]
    for t in triples:
        events = int(np.count_nonzero(Et[t[0]] & Et[t[1]] & Et[t[2]]))
        label = f"({t[0] + 1},{t[1] + 1},{t[2] + 1})"
        lines.append(f"{label:<15} {events / shots:>14.8e} {events:>8d} "
                     f"{int(t in in_HZ):>3} {int(t in in_HX):>3}")

    counts = (f"count = {len(triples)}   HZ = {len(in_HZ)}   HX = {len(in_HX)}   "
              f"both = {len(in_HZ & in_HX)}")
    lines += ["-" * 50, counts]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_marginals(path, one_body, two_body):
    """Write P_i lines "i:P_i" then P_ij lines "(i,j):P_ij" (1-based)."""
    with open(path, "w") as f:
        for q, p_i in enumerate(one_body):
            f.write(f"{q + 1}:{p_i}\n")
        for (q1, q2), p_ij in two_body.items():
            f.write(f"({q1 + 1},{q2 + 1}):{p_ij}\n")


def resolve_mydata_path(template, p, s):
    """Fill {p} and {s} into `template` and find the matching pickle.

    The template may contain glob wildcards (the default does, to absorb the
    sigma field). Returns None if nothing matches. If several files match (other
    disorder realisations, or a stale pickle) it warns and uses the newest."""
    pattern = template.format(p=p, s=s)

    if not any(c in pattern for c in "*?["):
        return pattern if os.path.exists(pattern) else None

    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"WARNING: {len(matches)} pickles match p={p}; using newest {matches[0]}")
        for m in matches[1:]:
            print(f"         ignoring {m}")
    return matches[0]


def load_code_matrices(mydata):
    """Code-level HX, HZ from the pickle (not the circuit-level overwritten ones)."""
    return as_gf2(mydata["hx_code"]), as_gf2(mydata["hz_code"])


def write_code_only(mydata_path, code_dir):
    """Write just the shared code files from one pickle (the code does not
    depend on the error rate). Used by --only-code."""
    print("Loading (code only):", mydata_path)
    with open(mydata_path, "rb") as f:
        mydata = pickle.load(f)

    HX, HZ = load_code_matrices(mydata)
    LX, LZ, k, rank_HX, rank_HZ, logical_comm = compute_css_logicals(HX, HZ)
    write_code_files(code_dir, HX, HZ, LX, LZ, k, rank_HX, rank_HZ, logical_comm,
                     get_all_clique_edges(HX, HZ))
    print(f"Wrote shared code files to '{code_dir}' (n={HX.shape[1]}, k={k}).")


# ============================================================
# Noisy circuit
# ============================================================

def generate_noisy_circuit(cycle_repeated, rates, rng, cnot_rates=None):
    """One random noisy copy of the circuit (circuit-level depolarizing noise).

    Same gate set and fault model as the single-fault circuits built in
    decoder_setup.py.

    `rates` is the per-location rate dict saved by decoder_setup.py,
        {'init': ..., 'idle': ..., 'meas': ..., 'cnot': ...}
    so faults are injected at exactly the rates the decoder has priors for.
    `cnot_rates` optionally maps (control, target) -> rate, overriding
    rates['cnot'] for that physical gate.
    """
    p_init = rates["init"]
    p_idle = rates["idle"]
    p_meas = rates["meas"]
    p_cnot_default = rates["cnot"]

    circ = []
    for gate in cycle_repeated:
        assert(gate[0] in ['CNOT', 'IDLE', 'PrepX', 'PrepZ', 'MeasX', 'MeasZ'])

        # measurement error: flip the measured basis just before the measurement
        if gate[0] == 'MeasX':
            assert(len(gate) == 2)
            if rng.random() <= p_meas:
                circ.append(('Z', gate[1]))
            circ.append(gate)
            continue
        if gate[0] == 'MeasZ':
            assert(len(gate) == 2)
            if rng.random() <= p_meas:
                circ.append(('X', gate[1]))
            circ.append(gate)
            continue

        # preparation error: flip the fresh qubit just after the preparation
        if gate[0] == 'PrepX':
            assert(len(gate) == 2)
            circ.append(gate)
            if rng.random() <= p_init:
                circ.append(('Z', gate[1]))
            continue
        if gate[0] == 'PrepZ':
            assert(len(gate) == 2)
            circ.append(gate)
            if rng.random() <= p_init:
                circ.append(('X', gate[1]))
            continue

        # idle error: X, Y or Z with equal probability (the IDLE gate itself is dropped)
        if gate[0] == 'IDLE':
            assert(len(gate) == 2)
            if rng.random() <= p_idle:
                circ.append((['X', 'Y', 'Z'][rng.integers(3)], gate[1]))
            continue

        # CNOT error: one of the 15 non-identity two-qubit Paulis
        if gate[0] == 'CNOT':
            assert(len(gate) == 3)
            control = gate[1]
            target = gate[2]
            circ.append(gate)
            p_cnot = p_cnot_default
            if cnot_rates is not None:
                p_cnot = cnot_rates.get((control, target), p_cnot_default)
            if rng.random() <= p_cnot:
                paulis = []
                # error on the control qubit only
                paulis += [('X', control), ('Y', control), ('Z', control)]
                # error on the target qubit only
                paulis += [('X', target), ('Y', target), ('Z', target)]
                # errors on both the control and the target qubits
                for pauli in ['XX', 'YY', 'ZZ', 'XY', 'YX', 'YZ', 'ZY', 'XZ', 'ZX']:
                    paulis.append((pauli, control, target))
                circ.append(paulis[rng.integers(15)])
            continue
    return circ


# we only look at the action of the circuit on X errors; 0 means no error, 1 means error
# (same as simulate_circuitX in decoder_setup.py, but without the syndrome bookkeeping;
#  here we only need the final error state)
def simulate_circuitX(C, lin_order):
    state = np.zeros(len(lin_order), dtype=np.uint8)
    for gate in C:
        if gate[0] == 'CNOT':
            assert(len(gate) == 3)
            control = lin_order[gate[1]]
            target = lin_order[gate[2]]
            state[target] = (state[target] + state[control]) % 2
            continue
        if gate[0] == 'PrepZ':
            assert(len(gate) == 2)
            q = lin_order[gate[1]]
            state[q] = 0
            continue
        if gate[0] in ['X', 'Y']:
            assert(len(gate) == 2)
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['XZ', 'YZ']:
            assert(len(gate) == 3)
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['ZX', 'ZY']:
            assert(len(gate) == 3)
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['XX', 'YY', 'XY', 'YX']:
            assert(len(gate) == 3)
            q1 = lin_order[gate[1]]
            q2 = lin_order[gate[2]]
            state[q1] = (state[q1] + 1) % 2
            state[q2] = (state[q2] + 1) % 2
            continue
        # all other gates (PrepX, MeasX, MeasZ, IDLE, Z, ZZ) do not change X errors
    return state


def generate_error_matrix(num_shots, cycle_repeated, rates, rng, lin_order,
                          data_qubits, label, cnot_rates=None):
    """Final X errors on the data qubits: E[shot, q] = 1 if qubit q has an X error."""
    data_pos = [lin_order[q] for q in data_qubits]
    E = np.zeros((num_shots, len(data_qubits)), dtype=np.uint8)

    for shot in tqdm(range(num_shots), desc=label, leave=False, unit="shot"):
        noisy_circuit = generate_noisy_circuit(cycle_repeated, rates, rng, cnot_rates)
        E[shot] = simulate_circuitX(noisy_circuit, lin_order)[data_pos]

    return E


# ============================================================
# Processing one (error rate, sample)
# ============================================================

def get_fault_rates(mydata):
    """Per-location rates the decoder priors were built with."""
    if "error_rates" in mydata:
        return mydata["error_rates"]
    # Pickles from before that field existed used one uniform rate.
    p = mydata["error_rate"]
    print("WARNING: pickle predates 'error_rates'; falling back to uniform p "
          "for all fault locations (this does NOT match the decoder priors).")
    return {"init": p, "idle": p, "meas": p, "cnot": p}


def process_one(mydata_path, folders, train_shots, test_shots, seed,
                skip_code=False, sample=1):
    """Generate shots and write all outputs for one pickle."""
    print("Loading:", mydata_path)
    with open(mydata_path, "rb") as f:
        mydata = pickle.load(f)

    # Take the sample index from the pickle so output names always agree with
    # the disorder realisation the data came from.
    sample = mydata.get("sample", sample)

    error_rate = mydata["error_rate"]
    rates = get_fault_rates(mydata)
    cnot_rates = mydata.get("cnot_rates")
    # Disorder spread (0.0 for pickles from before per-CNOT sampling). It is in
    # every filename so runs that differ only in sigma never overwrite each other.
    cnot_sigma = mydata.get("cnot_sigma", 0.0)
    tag = f"p_{error_rate}_sig_{cnot_sigma}_s_{sample}"

    lin_order = mydata["lin_order"]
    data_qubits = mydata["data_qubits"]
    n = len(data_qubits)
    cycle_repeated = mydata["num_cycles"] * mydata["cycle"]

    HX, HZ = load_code_matrices(mydata)
    assert HX.shape[1] == n, "ERROR: HX columns do not match number of data qubits"
    assert HZ.shape[1] == n, "ERROR: HZ columns do not match number of data qubits"

    LX, LZ, k, rank_HX, rank_HZ, logical_comm = compute_css_logicals(HX, HZ)

    # Summary of this run.
    print(f"n = {n}, k = {k}, num_cycles = {mydata['num_cycles']}")
    print(f"error_rate = {error_rate}, sample = {sample}, cnot_sigma = {cnot_sigma}")
    print("fault rates =", rates)
    if cnot_rates:
        v = list(cnot_rates.values())
        print("per-CNOT rates: n=%d mean=%.4e min=%.4e max=%.4e"
              % (len(v), np.mean(v), min(v), max(v)))
    print(f"gates in one cycle = {len(mydata['cycle'])}")

    # Independent shot streams per sample, so realisations do not share noise.
    rng_train = np.random.default_rng([seed, sample, 0])
    rng_test = np.random.default_rng([seed, sample, 1])

    # Training shots feed the one- and two-qubit marginals (and Neural BP training).
    print("\nGenerating training errors...")
    E_train = generate_error_matrix(train_shots, cycle_repeated, rates, rng_train,
                                    lin_order, data_qubits, "TRAIN", cnot_rates)
    # Testing shots are only used for testing.
    print("\nGenerating testing errors...")
    E_test = generate_error_matrix(test_shots, cycle_repeated, rates, rng_test,
                                   lin_order, data_qubits, "TEST", cnot_rates)

    # Marginals from the training shots: P_i, and P_ij on every clique edge.
    clique_edges = get_all_clique_edges(HX, HZ)
    one_body = E_train.mean(axis=0)
    two_body = {(q1, q2): np.mean(E_train[:, q1] * E_train[:, q2])
                for q1, q2 in clique_edges}

    # Shared code files: skipped on cluster data jobs to avoid concurrent writes.
    if skip_code:
        print("Skipping shared code files (--skip-code).")
    else:
        write_code_files(folders["code"], HX, HZ, LX, LZ, k, rank_HX, rank_HZ,
                         logical_comm, clique_edges)

    # Error patterns, shape n x shots (one column per shot).
    save_matrix(os.path.join(folders["train"], f"train_{tag}.txt"), E_train.T)
    save_matrix(os.path.join(folders["test"], f"test_{tag}.txt"), E_test.T)

    if cnot_rates:
        cnot_path = os.path.join(folders["cnot_rates"], f"cnot_error_rates_{tag}.txt")
        write_cnot_rates(cnot_path, cnot_rates, mydata, error_rate, sample)
        print("Saved per-CNOT error rates to:", cnot_path)

    write_marginals(os.path.join(folders["weights"], f"raw_probabilities_{tag}.txt"),
                    one_body, two_body)
    write_triple_marginals(os.path.join(folders["weights"], f"P_ijk_results_{tag}.txt"),
                           E_train, HX, HZ, tag)

    print(f"\nSaved {tag}: {train_shots} training shots, "
          f"{len(clique_edges)} clique edges, k = {k}")


# ============================================================
# Command line
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--error-rates", type=float, nargs="+", default=DEFAULT_ERROR_RATES,
                        help="Physical error rates to process (default: list in this file)")
    parser.add_argument("--mydata-template", default=DEFAULT_MYDATA_TEMPLATE,
                        help="Path template for the decoder_setup.py pickle; {p}, {s} are filled in")
    parser.add_argument("--train-shots", type=int, default=DEFAULT_SHOTS,
                        help="Noisy-circuit shots for training / marginals")
    parser.add_argument("--test-shots", type=int, default=DEFAULT_SHOTS,
                        help="Noisy-circuit shots for testing")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base random seed (combined with --sample)")
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="Disorder realisation index; selects the _s_<n> pickle")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="Output folder (default: $BB_OUT)")
    parser.add_argument("--skip-code", action="store_true",
                        help="Do not write the shared code/*.txt files (parallel cluster data jobs)")
    parser.add_argument("--only-code", action="store_true",
                        help="Only write the shared code/*.txt files, then exit")
    args = parser.parse_args()

    folders = output_folders(args.out)
    resolved = {p: resolve_mydata_path(args.mydata_template, p, args.sample)
                for p in args.error_rates}

    # --only-code: write the rate-independent files once, from any existing pickle.
    if args.only_code:
        for path in resolved.values():
            if path is not None:
                write_code_only(path, folders["code"])
                return
        raise SystemExit("ERROR: --only-code found no existing pickle among the given error rates.")

    missing = [p for p, path in resolved.items() if path is None]
    if missing:
        print("WARNING: no pickle found for these error rates, skipping them:")
        for p in missing:
            print(f"  p={p}  ->  {args.mydata_template.format(p=p, s=args.sample)}")
        print("Generate them first with decoder_setup.py.\n")

    rates_to_run = [p for p in args.error_rates if resolved[p] is not None]
    for p in tqdm(rates_to_run, desc="error rates", unit="rate"):
        process_one(resolved[p], folders, args.train_shots, args.test_shots, args.seed,
                    skip_code=args.skip_code, sample=args.sample)

    print(f"\nAll done. Processed {len(rates_to_run)} error rate(s) into '{args.out}'.")


if __name__ == "__main__":
    main()
