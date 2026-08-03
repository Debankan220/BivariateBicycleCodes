

import argparse
import glob
import os
import pickle
import numpy as np
from itertools import combinations
from tqdm import tqdm

# List of physical error rates to process in one run.
# Each rate must have a matching pickle produced by decoder_setup.py in TMP/.
# Full sweep (commented out; using the smaller test list below):
# error_rates = [
#     0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
#     0.0011, 0.0013, 0.0015, 0.0017, 0.0019, 0.002,
#     0.0024, 0.0028, 0.0032, 0.0036,
#     0.004, 0.0044, 0.0048, 0.005,
# ]

# checking with a smaller list of error rates
error_rates = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]
# Template for the pickle file produced by decoder_setup.py.
# {p} is the error rate, {s} the sample (disorder realisation) index. The '*'
# absorbs the sigma field, which scales with p and so cannot be a fixed string.
mydata_template = "TMP/mydata_72_12_p_{p}_sig_*_cycles_3_s_{s}"

output_folder = "72q_BB_cycles_1_spread_comparison"
code_details_folder = os.path.join(output_folder, "code")
training_data_folder = os.path.join(output_folder, "training_data")
testing_data_folder = os.path.join(output_folder, "testing_data")
correlated_weights_folder = os.path.join(output_folder, "correlated_weights")
cnot_error_rates_folder = os.path.join(output_folder, "cnot_error_rates")

os.makedirs(code_details_folder, exist_ok=True)
os.makedirs(training_data_folder, exist_ok=True)
os.makedirs(testing_data_folder, exist_ok=True)
os.makedirs(correlated_weights_folder, exist_ok=True)
os.makedirs(cnot_error_rates_folder, exist_ok=True)
training_shots = 1000000
testing_shots = 1000000
# Default disorder realisation index; override with --sample. Each sample is a
# different draw of the per-CNOT rates, held fixed across all shots of that run.
sample = 1



# ============================================================
# Clique graph from HZ
# ============================================================

def get_clique_edges(HZ):
    """
    Clique graph from HZ.

    If two data qubits are connected to the same Z check,
    put an edge between them.
    """

    clique_edges = set()

    for a in range(HZ.shape[0]):
        support = np.where(HZ[a] == 1)[0]

        for q1, q2 in combinations(support, 2):
            q1 = int(q1)
            q2 = int(q2)

            if q1 > q2:
                q1, q2 = q2, q1

            clique_edges.add((q1, q2))

    return sorted(clique_edges)


# ============================================================
# GF(2) linear algebra
# ============================================================

def gf2_rank(A):
    """
    Rank over GF(2).
    """

    A = np.array(A, dtype=np.uint8, copy=True) % 2

    if A.ndim != 2:
        raise ValueError("gf2_rank expects a 2D matrix")

    m, n = A.shape
    rank = 0

    for col in range(n):
        pivot = None

        for r in range(rank, m):
            if A[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        A[[rank, pivot]] = A[[pivot, rank]]

        for r in range(m):
            if r != rank and A[r, col] == 1:
                A[r] ^= A[rank]

        rank += 1

    return rank


def gf2_kernel(H):
    """
    Return a basis for kernel(H) over GF(2).

    Output rows v satisfy:

        H @ v = 0 mod 2
    """

    H = np.array(H, dtype=np.uint8, copy=True) % 2

    if H.ndim != 2:
        raise ValueError("gf2_kernel expects a 2D matrix")

    m, n = H.shape

    A = H.copy()
    pivots = []
    row = 0

    for col in range(n):
        pivot = None

        for r in range(row, m):
            if A[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            continue

        A[[row, pivot]] = A[[pivot, row]]

        for r in range(m):
            if r != row and A[r, col] == 1:
                A[r] ^= A[row]

        pivots.append(col)
        row += 1

    free_cols = [c for c in range(n) if c not in pivots]

    if len(free_cols) == 0:
        return np.zeros((0, n), dtype=np.uint8)

    basis = []

    for free in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[free] = 1

        for i, p in enumerate(pivots):
            v[p] = A[i, free]

        basis.append(v)

    return np.array(basis, dtype=np.uint8)


def gf2_inverse(A):
    """
    Inverse of a square binary matrix over GF(2).
    """

    A = np.array(A, dtype=np.uint8, copy=True) % 2

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("gf2_inverse expects a square matrix")

    n = A.shape[0]

    if n == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    I = np.eye(n, dtype=np.uint8)
    aug = np.hstack([A, I])

    row = 0

    for col in range(n):
        pivot = None

        for r in range(row, n):
            if aug[r, col] == 1:
                pivot = r
                break

        if pivot is None:
            raise ValueError("Matrix is not invertible over GF(2)")

        aug[[row, pivot]] = aug[[pivot, row]]

        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] ^= aug[row]

        row += 1

    return aug[:, n:]


def quotient_logicals(kernel_basis, stabilizer_matrix):
    """
    Keep only kernel vectors that are independent modulo stabilizers.

    For CSS codes:

        LX = kernel(HZ) modulo rowspace(HX)
        LZ = kernel(HX) modulo rowspace(HZ)
    """

    kernel_basis = np.array(kernel_basis, dtype=np.uint8) % 2
    stabilizer_matrix = np.array(stabilizer_matrix, dtype=np.uint8) % 2

    n = stabilizer_matrix.shape[1]

    logicals = []

    current = stabilizer_matrix.copy()
    current_rank = gf2_rank(current)

    for v in kernel_basis:
        trial = np.vstack([current, v.reshape(1, n)])
        trial_rank = gf2_rank(trial)

        if trial_rank > current_rank:
            logicals.append(v.copy())
            current = trial
            current_rank = trial_rank

    if len(logicals) == 0:
        return np.zeros((0, n), dtype=np.uint8)

    return np.array(logicals, dtype=np.uint8)


def compute_css_logicals(HX, HZ):
    """
    Compute paired CSS logical operators LX and LZ.

    Convention:
        HX rows are X stabilizers.
        HZ rows are Z stabilizers.

    Therefore:
        LX = kernel(HZ) modulo rowspace(HX)
        LZ = kernel(HX) modulo rowspace(HZ)

    The final output is paired so that:

        LX @ LZ.T = identity mod 2
    """

    HX = np.array(HX, dtype=np.uint8) % 2
    HZ = np.array(HZ, dtype=np.uint8) % 2

    if HX.ndim != 2 or HZ.ndim != 2:
        raise ValueError("HX and HZ must be 2D matrices")

    if HX.shape[1] != HZ.shape[1]:
        raise ValueError("HX and HZ must have the same number of columns")

    n = HX.shape[1]

    rank_HX = gf2_rank(HX)
    rank_HZ = gf2_rank(HZ)

    k = n - rank_HX - rank_HZ

    print("rank(HX) =", rank_HX)
    print("rank(HZ) =", rank_HZ)
    print("k =", k)

    if k < 0:
        raise ValueError("Computed k is negative. Check HX and HZ.")

    # Stabilizers must commute.
    assert np.all((HX @ HZ.T) % 2 == 0), "ERROR: HX and HZ do not commute"

    # X logicals commute with Z stabilizers.
    LX = quotient_logicals(gf2_kernel(HZ), HX)

    # Z logicals commute with X stabilizers.
    LZ = quotient_logicals(gf2_kernel(HX), HZ)

    assert LX.shape[0] == k, "ERROR: number of LX operators does not match k"
    assert LZ.shape[0] == k, "ERROR: number of LZ operators does not match k"
    assert gf2_rank(np.vstack([HX, LX])) - gf2_rank(HX) == k, (
        "ERROR: LX operators are not independent modulo HX stabilizers"
    )

    assert gf2_rank(np.vstack([HZ, LZ])) - gf2_rank(HZ) == k, (
        "ERROR: LZ operators are not independent modulo HZ stabilizers"
    )

    assert np.all((LX @ HZ.T) % 2 == 0), "ERROR: LX does not commute with HZ"
    assert np.all((LZ @ HX.T) % 2 == 0), "ERROR: LZ does not commute with HX"

    # This is the raw logical commutation matrix.
    logical_comm = (LX @ LZ.T) % 2

    print("Raw LX @ LZ.T mod 2 =")
    print(logical_comm)

    # It must be full rank. If yes, the logicals are valid.
    assert gf2_rank(logical_comm) == k, (
        "ERROR: LX and LZ do not form a full-rank logical pairing"
    )

    # Pair logicals so that LX @ LZ.T = identity.
    # Keep LZ fixed and change the LX basis.
    logical_comm_inv = gf2_inverse(logical_comm)
    LX = (logical_comm_inv @ LX) % 2

    # Check final paired commutation matrix.
    logical_comm = (LX @ LZ.T) % 2

    print("Paired LX @ LZ.T mod 2 =")
    print(logical_comm)

    assert np.array_equal(logical_comm, np.eye(k, dtype=np.uint8)), (
        "ERROR: failed to pair logical operators"
    )

    return LX, LZ, k, rank_HX, rank_HZ, logical_comm


def save_matrix(path, M):
    """
    Save binary matrix.
    """

    np.savetxt(path, M.astype(int), fmt="%d")


def write_code_files(HX, HZ, LX, LZ, k, rank_HX, rank_HZ, logical_comm, n, clique_edges):
    """
    Write the code-level files that are identical for every error rate:
    HX.txt, HZ.txt, LX.txt, LZ.txt, logical_checks.txt, clique_edges.txt.

    These are separated out so a cluster run can generate them once
    (--only-code) and the per-rate data jobs can skip them (--skip-code),
    avoiding concurrent writes to the same shared files.
    """

    save_matrix(os.path.join(code_details_folder, "HX.txt"), HX)
    save_matrix(os.path.join(code_details_folder, "HZ.txt"), HZ)
    save_matrix(os.path.join(code_details_folder, "LX.txt"), LX)
    save_matrix(os.path.join(code_details_folder, "LZ.txt"), LZ)

    with open(os.path.join(code_details_folder, "logical_checks.txt"), "w") as f:
        f.write("Logical operator checks\n")
        f.write("=======================\n\n")

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

    with open(os.path.join(code_details_folder, "clique_edges.txt"), "w") as f:
        for q1, q2 in clique_edges:
            f.write(f"({q1 + 1},{q2 + 1})\n")


def write_code_only(mydata_path):
    """
    Load one pickle (any error rate; the code is rate-independent) and
    write just the shared code files, then return. Used by --only-code.
    """

    print("Loading (code only):", mydata_path)

    with open(mydata_path, "rb") as f:
        mydata = pickle.load(f)

    HX = np.array(mydata["hx_code"], dtype=np.uint8) % 2
    HZ = np.array(mydata["hz_code"], dtype=np.uint8) % 2
    n = HX.shape[1]

    LX, LZ, k, rank_HX, rank_HZ, logical_comm = compute_css_logicals(HX, HZ)
    clique_edges = get_clique_edges(HZ)

    write_code_files(HX, HZ, LX, LZ, k, rank_HX, rank_HZ, logical_comm, n, clique_edges)

    print(f"Wrote shared code files to '{code_details_folder}' (n={n}, k={k}).")


# ============================================================
# Noisy circuit
# ============================================================

def resolve_mydata_path(template, p, s):
    """
    Fill {p} and {s} into `template` and resolve it to an existing pickle.

    The template may contain glob wildcards (the default one does, to absorb the
    sigma field in the filename). Returns None if nothing matches.
    Multiple matches means several disorder realisations (or a stale pickle from
    an older filename convention) exist for the same p. That is ambiguous, so
    warn and take the most recently written one -- normally the one you just
    generated. Narrow the template if you want a specific realisation.
    """

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


def generate_noisy_circuit(cycle_repeated, rates, rng, cnot_rates=None):
    """
    Bravyi-style circuit-level depolarizing noise.

    `rates` is the per-location rate dict saved by decoder_setup.py:
        {'init': ..., 'idle': ..., 'meas': ..., 'cnot': ...}
    so that faults are injected at exactly the rates the decoder was given
    priors for. `cnot_rates` optionally maps (control, target) -> rate,
    overriding rates['cnot'] per physical gate; missing pairs fall back to
    rates['cnot'].

    For CNOT:
        with probability p_gate, add one of the 15 non-identity
        two-qubit Pauli errors.
    """

    p_init = rates["init"]
    p_idle = rates["idle"]
    p_meas = rates["meas"]
    p_cnot_default = rates["cnot"]

    circ = []

    for gate in cycle_repeated:
        name = gate[0]

        assert name in ["CNOT", "IDLE", "PrepX", "PrepZ", "MeasX", "MeasZ"]

        if name == "MeasX":
            if rng.random() <= p_meas:
                circ.append(("Z", gate[1]))

            circ.append(gate)
            continue

        if name == "MeasZ":
            if rng.random() <= p_meas:
                circ.append(("X", gate[1]))

            circ.append(gate)
            continue

        if name == "PrepX":
            circ.append(gate)

            if rng.random() <= p_init:
                circ.append(("Z", gate[1]))

            continue

        if name == "PrepZ":
            circ.append(gate)

            if rng.random() <= p_init:
                circ.append(("X", gate[1]))

            continue

        if name == "IDLE":
            if rng.random() <= p_idle:
                ptype = rng.integers(3)

                if ptype == 0:
                    circ.append(("X", gate[1]))
                elif ptype == 1:
                    circ.append(("Y", gate[1]))
                else:
                    circ.append(("Z", gate[1]))

            continue

        if name == "CNOT":
            circ.append(gate)

            control = gate[1]
            target = gate[2]

            if cnot_rates is None:
                p_gate = p_cnot_default
            else:
                p_gate = cnot_rates.get((control, target), p_cnot_default)

            if rng.random() <= p_gate:

                pauli_errors = [
                    ("X", control),
                    ("Y", control),
                    ("Z", control),

                    ("X", target),
                    ("Y", target),
                    ("Z", target),

                    ("XX", control, target),
                    ("YY", control, target),
                    ("ZZ", control, target),

                    ("XY", control, target),
                    ("YX", control, target),

                    ("YZ", control, target),
                    ("ZY", control, target),

                    ("XZ", control, target),
                    ("ZX", control, target),
                ]

                circ.append(pauli_errors[rng.integers(15)])

            continue

    return circ


def simulate_circuitX(C, lin_order):
    """
    Track only X-type errors through the circuit.

    This is relevant for Z stabilizer checks,
    because Z checks detect X errors.

    Binary state:
        state[q] = 0 means no X error on qubit q
        state[q] = 1 means X error on qubit q

    CNOT X propagation:
        X on control propagates to target:

            state[target] ^= state[control]
    """

    state = np.zeros(len(lin_order), dtype=np.uint8)

    for gate in C:
        name = gate[0]

        if name == "CNOT":
            control = lin_order[gate[1]]
            target = lin_order[gate[2]]

            state[target] ^= state[control]
            continue

        if name == "PrepZ":
            q = lin_order[gate[1]]
            state[q] = 0
            continue

        # Single-qubit errors with X component.
        if name in ["X", "Y"]:
            q = lin_order[gate[1]]
            state[q] ^= 1
            continue

        # Two-qubit errors where first qubit has X or Y.
        if name in ["XZ", "YZ"]:
            q = lin_order[gate[1]]
            state[q] ^= 1
            continue

        # Two-qubit errors where second qubit has X or Y.
        if name in ["ZX", "ZY"]:
            q = lin_order[gate[2]]
            state[q] ^= 1
            continue

        # Two-qubit errors where both qubits have X/Y component.
        if name in ["XX", "YY", "XY", "YX"]:
            q1 = lin_order[gate[1]]
            q2 = lin_order[gate[2]]

            state[q1] ^= 1
            state[q2] ^= 1
            continue

        # These do not affect X-error tracking:
        # PrepX, MeasX, MeasZ, IDLE, Z, ZZ.
        continue

    return state


def generate_error_matrix(
    num_shots,
    cycle_repeated,
    rates,
    rng,
    lin_order,
    data_qubits,
    label,
    cnot_rates=None,
):
    """
    Generate final X-error patterns on data qubits.

    Output:
        E[shot, q] = final X error on data qubit q
    """

    n = len(data_qubits)
    E = np.zeros((num_shots, n), dtype=np.uint8)

    for shot in tqdm(range(num_shots), desc=label, leave=False, unit="shot"):
        noisy_circuit = generate_noisy_circuit(
            cycle_repeated,
            rates,
            rng,
            cnot_rates,
        )

        state = simulate_circuitX(
            noisy_circuit,
            lin_order,
        )

        data_error = np.array(
            [state[lin_order[q]] for q in data_qubits],
            dtype=np.uint8,
        )

        E[shot] = data_error

    return E


# ============================================================
# Main
# ============================================================

def process_one(mydata_path, train_shots, test_shots, seed, out, skip_code=False,
                sample=1):
    print("Loading:", mydata_path)

    with open(mydata_path, "rb") as f:
        mydata = pickle.load(f)

    # Take the sample index from the pickle itself so the output filenames can
    # never disagree with the disorder realisation they were generated from.
    sample = mydata.get("sample", sample)

    # Independent shot streams per sample, so realisations do not share noise.
    rng_train = np.random.default_rng([seed, sample, 0])
    rng_test = np.random.default_rng([seed, sample, 1])

    cycle = mydata["cycle"]
    num_cycles = mydata["num_cycles"]
    error_rate = mydata["error_rate"]
    lin_order = mydata["lin_order"]
    data_qubits = mydata["data_qubits"]

    # Per-location rates as used by decoder_setup.py when it built the priors.
    # Pickles written before that field existed fall back to the old uniform
    # model, which is what those runs actually used.
    if "error_rates" in mydata:
        rates = mydata["error_rates"]
    else:
        print("WARNING: pickle predates 'error_rates'; falling back to uniform p "
              "for all fault locations (this does NOT match the decoder priors).")
        rates = {
            "init": error_rate,
            "idle": error_rate,
            "meas": error_rate,
            "cnot": error_rate,
        }

    cnot_rates = mydata.get("cnot_rates")

    n = len(data_qubits)

    # IMPORTANT:
    # Use code-level matrices, not circuit-level overwritten matrices.
    HX = np.array(mydata["hx_code"], dtype=np.uint8) % 2
    HZ = np.array(mydata["hz_code"], dtype=np.uint8) % 2

    assert HX.shape[1] == n, "ERROR: HX columns do not match number of data qubits"
    assert HZ.shape[1] == n, "ERROR: HZ columns do not match number of data qubits"

    # ------------------------------------------------------------
    # Compute logical operators
    # ------------------------------------------------------------

    LX, LZ, k, rank_HX, rank_HZ, logical_comm = compute_css_logicals(HX, HZ)

    print("n =", n)
    print("k =", k)
    print("num_cycles =", num_cycles)
    print("error_rate =", error_rate)
    print("sample =", sample)
    print("fault rates =", rates)
    if cnot_rates:
        _v = list(cnot_rates.values())
        print("per-CNOT rates: n=%d mean=%.4e min=%.4e max=%.4e"
              % (len(cnot_rates), np.mean(_v), min(_v), max(_v)))
    print("number of gates in one cycle =", len(cycle))
    print("number of data qubits =", n)
    print("HX shape =", HX.shape)
    print("HZ shape =", HZ.shape)
    print("HX row weights =", HX.sum(axis=1))
    print("HZ row weights =", HZ.sum(axis=1))
    print("LX shape =", LX.shape)
    print("LZ shape =", LZ.shape)

    # Save the p-dependent files directly in training_data,
    # testing_data, and correlated_weights.

    cycle_repeated = num_cycles * cycle

    # ------------------------------------------------------------
    # Generate training errors
    # These are used for:
    #   1. training Neural BP
    #   2. one-qubit marginals
    #   3. two-qubit clique marginals
    # ------------------------------------------------------------

    print("\nGenerating training errors...")

    E_train = generate_error_matrix(
        num_shots=train_shots,
        cycle_repeated=cycle_repeated,
        rates=rates,
        rng=rng_train,
        lin_order=lin_order,
        data_qubits=data_qubits,
        label="TRAIN",
        cnot_rates=cnot_rates,
    )

    # ------------------------------------------------------------
    # Generate testing errors
    # These are used only for testing.
    # ------------------------------------------------------------

    print("\nGenerating testing errors...")

    E_test = generate_error_matrix(
        num_shots=test_shots,
        cycle_repeated=cycle_repeated,
        rates=rates,
        rng=rng_test,
        lin_order=lin_order,
        data_qubits=data_qubits,
        label="TEST",
        cnot_rates=cnot_rates,
    )

    # ------------------------------------------------------------
    # Clique edges from HZ (needed for the shared code files below)
    # ------------------------------------------------------------

    clique_edges = get_clique_edges(HZ)

    # ------------------------------------------------------------
    # One- and two-qubit marginals from training errors.
    # ------------------------------------------------------------

    one_body = E_train.mean(axis=0)

    two_body_clique = {}

    for q1, q2 in clique_edges:
        p_ij = np.mean(E_train[:, q1] * E_train[:, q2])
        two_body_clique[(q1 + 1, q2 + 1)] = p_ij

    # ------------------------------------------------------------
    # Save PCMs, logical operators, logical checks and clique edges.
    # These are rate-independent shared files; skip them on cluster
    # data jobs (--skip-code) to avoid concurrent writes.
    # ------------------------------------------------------------

    if skip_code:
        print("Skipping shared code files (--skip-code).")
    else:
        write_code_files(
            HX, HZ, LX, LZ, k, rank_HX, rank_HZ, logical_comm, n, clique_edges
        )

    # ------------------------------------------------------------
    # Save training and testing error patterns
    # Shape: n x shots
    # Useful if Neural BP expects one error pattern per column.
    # ------------------------------------------------------------

    save_matrix(
        os.path.join(training_data_folder, f"train_p_{error_rate}_s_{sample}.txt"),
        E_train.T,
    )

    save_matrix(
        os.path.join(testing_data_folder, f"test_p_{error_rate}_s_{sample}.txt"),
        E_test.T,
    )

    # ------------------------------------------------------------
    # Save one-qubit and two-qubit marginals
    # 1-indexed format:
    # 1: value
    # (1,3): value
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Save the per-CNOT error rates for this disorder realisation.
    # These are drawn once in decoder_setup.py and held fixed across every
    # shot of this run, so one file fully describes the "chip" the data
    # was generated on.
    # ------------------------------------------------------------

    if cnot_rates:
        cnot_path = os.path.join(
            cnot_error_rates_folder,
            f"cnot_error_rates_p_{error_rate}_sig_{mydata.get('cnot_sigma')}_s_{sample}.txt",
        )

        with open(cnot_path, "w") as f:
            f.write(f"# per-CNOT two-qubit error rates (fixed for all shots)\n")
            f.write(f"# p={error_rate} sigma={mydata.get('cnot_sigma')} "
                    f"sample={sample} seed={mydata.get('cnot_seed')}\n")
            f.write(f"# gates={len(cnot_rates)} "
                    f"mean={np.mean(list(cnot_rates.values())):.8e}\n")
            f.write("# control_type control_index target_type target_index rate\n")

            for (ctrl, tgt), rate in cnot_rates.items():
                f.write(f"{ctrl[0]} {ctrl[1]} {tgt[0]} {tgt[1]} {rate:.10e}\n")

        print("Saved per-CNOT error rates to:", cnot_path)

    with open(os.path.join(correlated_weights_folder, f"correlated_weights_p_{error_rate}_s_{sample}.txt"), "w") as f:
        for q, p_i in enumerate(one_body):
            f.write(f"{q + 1}:{p_i}\n")

        for q1, q2 in clique_edges:
            p_ij = two_body_clique[(q1 + 1, q2 + 1)]
            f.write(f"({q1 + 1},{q2 + 1}):{p_ij}\n")

    print("\nSaved output to:", out)
    print("Training shots:", train_shots)
    print("Number of clique edges:", len(clique_edges))
    print("Number of logical qubits k:", k)
    if not skip_code:
        print("Saved HX.txt")
        print("Saved HZ.txt")
        print("Saved LX.txt")
        print("Saved LZ.txt")
        print("Saved logical_checks.txt")

    print("\nOne-qubit marginals from training data:")
    for q, p_i in enumerate(one_body):
        print(f"{q + 1}: {p_i}")

    print("\nFirst few clique-edge two-qubit marginals from training data:")
    for q1, q2 in clique_edges[:10]:
        print(f"({q1 + 1},{q2 + 1}): {two_body_clique[(q1 + 1, q2 + 1)]}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--error-rates",
        type=float,
        nargs="+",
        default=error_rates,
        help="List of physical error rates to process (defaults to the list in this file)",
    )

    parser.add_argument(
        "--mydata-template",
        type=str,
        default=mydata_template,
        help="Template path for the pickle from decoder_setup.py; {p} is filled with each error rate",
    )

    parser.add_argument(
        "--train-shots",
        type=int,
        default=training_shots,
        help="Number of noisy circuit shots for training",
    )

    parser.add_argument(
        "--test-shots",
        type=int,
        default=testing_shots,
        help="Number of noisy circuit shots for testing",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base random seed for the shot sampling (combined with --sample)",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=sample,
        help="Disorder realisation index; selects the _s_<n> pickle and labels the output",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=output_folder,
        help="Output folder",
    )

    parser.add_argument(
        "--skip-code",
        action="store_true",
        help="Do not (re)write the shared code/*.txt files; for parallel cluster data jobs",
    )

    parser.add_argument(
        "--only-code",
        action="store_true",
        help="Only write the shared code/*.txt files (no shot generation) and exit",
    )

    args = parser.parse_args()

    # --only-code: write the rate-independent code files once, then stop.
    if args.only_code:
        for p in args.error_rates:
            mydata_path = resolve_mydata_path(args.mydata_template, p, args.sample)
            if mydata_path is not None:
                write_code_only(mydata_path)
                return
        raise SystemExit(
            "ERROR: --only-code found no existing pickle among the given error rates."
        )

    resolved = {p: resolve_mydata_path(args.mydata_template, p, args.sample)
                for p in args.error_rates}

    missing = [(p, args.mydata_template.format(p=p, s=args.sample))
               for p in args.error_rates if resolved[p] is None]

    if missing:
        print("WARNING: the following pickles are missing and will be skipped:")
        for p, path in missing:
            print(f"  p={p}  ->  {path}")
        print("Generate them first with decoder_setup.py / run_error_rates.py.\n")

    rates_to_run = [p for p in args.error_rates if resolved[p] is not None]

    for p in tqdm(rates_to_run, desc="error rates", unit="rate"):
        process_one(
            resolved[p],
            args.train_shots,
            args.test_shots,
            args.seed,
            args.out,
            skip_code=args.skip_code,
            sample=args.sample,
        )

    print(f"\nAll done. Processed {len(rates_to_run)} error rate(s) into '{args.out}'.")


if __name__ == "__main__":
    main()