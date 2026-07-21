

import argparse
import os
import pickle
import numpy as np
from itertools import combinations

input_file_name = "TMP/mydata_18_8_p_0.005_cycles_1"
output_folder = "18q_BB_cycles_1"
code_details_folder = os.path.join(output_folder, "code")
training_data_folder = os.path.join(output_folder, "training_data")
testing_data_folder = os.path.join(output_folder, "testing_data")
correlated_weights_folder = os.path.join(output_folder, "correlated_weights")

os.makedirs(code_details_folder, exist_ok=True)
os.makedirs(training_data_folder, exist_ok=True)
os.makedirs(testing_data_folder, exist_ok=True)
os.makedirs(correlated_weights_folder, exist_ok=True)
training_shots = 1000000
testing_shots = 100000
sample=1



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


# ============================================================
# Noisy circuit
# ============================================================

def generate_noisy_circuit(cycle_repeated, p, rng):
    """
    Bravyi-style circuit-level depolarizing noise.

    Each location fails with probability p.

    For CNOT:
        with probability p, add one of the 15 non-identity
        two-qubit Pauli errors.
    """

    circ = []

    for gate in cycle_repeated:
        name = gate[0]

        assert name in ["CNOT", "IDLE", "PrepX", "PrepZ", "MeasX", "MeasZ"]

        if name == "MeasX":
            if rng.random() <= p:
                circ.append(("Z", gate[1]))

            circ.append(gate)
            continue

        if name == "MeasZ":
            if rng.random() <= p:
                circ.append(("X", gate[1]))

            circ.append(gate)
            continue

        if name == "PrepX":
            circ.append(gate)

            if rng.random() <= p:
                circ.append(("Z", gate[1]))

            continue

        if name == "PrepZ":
            circ.append(gate)

            if rng.random() <= p:
                circ.append(("X", gate[1]))

            continue

        if name == "IDLE":
            if rng.random() <= p:
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

            if rng.random() <= p:
                control = gate[1]
                target = gate[2]

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
    error_rate,
    rng,
    lin_order,
    data_qubits,
    label,
):
    """
    Generate final X-error patterns on data qubits.

    Output:
        E[shot, q] = final X error on data qubit q
    """

    n = len(data_qubits)
    E = np.zeros((num_shots, n), dtype=np.uint8)

    for shot in range(num_shots):
        noisy_circuit = generate_noisy_circuit(
            cycle_repeated,
            error_rate,
            rng,
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

        if (shot + 1) % max(1, num_shots // 10) == 0:
            print(f"{label}: Completed {shot + 1}/{num_shots} shots")

    return E


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mydata",
        type=str,
        default=input_file_name,
        help="Pickle file produced by decoder_setup.py",
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
        help="Random seed",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=output_folder,
        help="Output folder",
    )

    args = parser.parse_args()

    rng_train = np.random.default_rng(args.seed)
    rng_test = np.random.default_rng(args.seed + 1)

    print("Loading:", args.mydata)

    with open(args.mydata, "rb") as f:
        mydata = pickle.load(f)

    cycle = mydata["cycle"]
    num_cycles = mydata["num_cycles"]
    error_rate = mydata["error_rate"]
    lin_order = mydata["lin_order"]
    data_qubits = mydata["data_qubits"]

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
        num_shots=args.train_shots,
        cycle_repeated=cycle_repeated,
        error_rate=error_rate,
        rng=rng_train,
        lin_order=lin_order,
        data_qubits=data_qubits,
        label="TRAIN",
    )

    # ------------------------------------------------------------
    # Generate testing errors
    # These are used only for testing.
    # No marginals are computed from E_test.
    # ------------------------------------------------------------

    print("\nGenerating testing errors...")

    E_test = generate_error_matrix(
        num_shots=args.test_shots,
        cycle_repeated=cycle_repeated,
        error_rate=error_rate,
        rng=rng_test,
        lin_order=lin_order,
        data_qubits=data_qubits,
        label="TEST",
    )

    # ------------------------------------------------------------
    # One-qubit marginals from training errors only
    # ------------------------------------------------------------

    one_body = E_train.mean(axis=0)

    # ------------------------------------------------------------
    # Clique edges from HZ
    # ------------------------------------------------------------

    clique_edges = get_clique_edges(HZ)

    # ------------------------------------------------------------
    # Two-qubit marginals only on clique edges
    # Computed from training errors only
    # ------------------------------------------------------------

    two_body_clique = {}

    for q1, q2 in clique_edges:
        p_ij = np.mean(E_train[:, q1] * E_train[:, q2])
        two_body_clique[(q1 + 1, q2 + 1)] = p_ij

    # ------------------------------------------------------------
    # Save PCMs and logical operators
    # ------------------------------------------------------------

    save_matrix(
        os.path.join(code_details_folder, "HX.txt"),
        HX,
    )

    save_matrix(
        os.path.join(code_details_folder, "HZ.txt"),
        HZ,
    )

    save_matrix(
        os.path.join(code_details_folder, "LX.txt"),
        LX,
    )

    save_matrix(
        os.path.join(code_details_folder, "LZ.txt"),
        LZ,
    )

    # ------------------------------------------------------------
    # Save logical checks
    # ------------------------------------------------------------

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
    # Save clique edges
    # 1-indexed format:
    # (1,3)
    # (1,6)
    # ...
    # ------------------------------------------------------------

    with open(os.path.join(code_details_folder, "clique_edges.txt"), "w") as f:
        for q1, q2 in clique_edges:
            f.write(f"({q1 + 1},{q2 + 1})\n")

    # ------------------------------------------------------------
    # Save one-qubit and two-qubit marginals
    # 1-indexed format:
    # 1: value
    # (1,3): value
    # ------------------------------------------------------------

    with open(os.path.join(correlated_weights_folder, f"correlated_weights_p_{error_rate}_s_{sample}.txt"), "w") as f:
        for q, p_i in enumerate(one_body):
            f.write(f"{q + 1}:{p_i}\n")

        for q1, q2 in clique_edges:
            p_ij = two_body_clique[(q1 + 1, q2 + 1)]
            f.write(f"({q1 + 1},{q2 + 1}):{p_ij}\n")

    print("\nSaved output to:", args.out)
    print("Training shots:", args.train_shots)
    print("Testing shots:", args.test_shots)
    print("Number of clique edges:", len(clique_edges))
    print("Number of logical qubits k:", k)
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


if __name__ == "__main__":
    main()