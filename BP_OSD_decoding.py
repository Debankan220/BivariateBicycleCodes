import argparse
import os
import numpy as np
from tqdm import tqdm
from ldpc import BpOsdDecoder


# ============================================================
# Parameters
# ============================================================

# List of physical error rates to decode in one run.
# Each rate must have a matching testing file test_p_{p}_s_{sample}.txt.
error_rates = [
    0.0001, 0.0003, 0.0005, 0.0007, 0.0009,
    0.0011, 0.0013, 0.0015, 0.0017, 0.0019, 0.002,
    0.0024, 0.0028, 0.0032, 0.0036,
    0.004, 0.0044, 0.0048, 0.005,
]

sample = 1
num_cycles = 1

base_directory = "90q_BB_cycles_1"
code_directory = os.path.join(base_directory, "code")
testing_data_directory = os.path.join(base_directory, "testing_data")
results_directory = os.path.join(base_directory, "results")

# Create the results directory if it does not already exist
os.makedirs(results_directory, exist_ok=True)

# BP-OSD decoder settings (identical for every error rate)
decoder_settings = dict(
    error_rate=0.1,
    max_iter=500,
    bp_method="product_sum",
    schedule="parallel",
    osd_method="OSD_E",
    osd_order=2,
)


# ============================================================
# Decode a single error rate
# ============================================================

def decode_one(p_value, HZ, H_perp, decoder_x, code_label):
    """
    Decode the testing errors for one physical error rate and
    save the logical-error-rate result. Returns a summary tuple
    or None if the testing file is missing / empty.
    """

    test_file = os.path.join(
        testing_data_directory,
        f"test_p_{p_value}_s_{sample}.txt",
    )

    if not os.path.exists(test_file):
        print(f"SKIP p={p_value}: missing {test_file}")
        return None

    # The file has shape: number_of_qubits x number_of_errors.
    # After transposing: number_of_errors x number_of_qubits.
    sample_errors_X = np.loadtxt(test_file, dtype=int).T

    # Guard against a file that holds a single error (loadtxt returns 1D).
    if sample_errors_X.ndim == 1:
        sample_errors_X = sample_errors_X.reshape(1, -1)

    n_errors = sample_errors_X.shape[0]

    if n_errors == 0:
        print(f"SKIP p={p_value}: testing file contains no error samples.")
        return None

    successes = 0

    for error in tqdm(
        sample_errors_X,
        desc=f"decode p={p_value}",
        leave=False,
        unit="err",
    ):
        # Calculate the syndrome
        syndrome = (HZ @ error) % 2

        # Decode the syndrome
        correction = decoder_x.decode(syndrome)

        # Residual error after applying the correction
        residual_error = (error + correction) % 2

        # Successful decoding: zero syndrome and no logical error
        if np.all((H_perp @ residual_error) % 2 == 0):
            successes += 1

    failures = n_errors - successes
    average_logical_error_rate = failures / n_errors

    # Binomial standard error: sigma = sqrt[p_L (1 - p_L) / N]
    standard_error = np.sqrt(
        average_logical_error_rate
        * (1.0 - average_logical_error_rate)
        / n_errors
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    result_file = os.path.join(
        results_directory,
        (
            f"{code_label}_p_{p_value}_cycles_{num_cycles}_"
            f"BP+OSD_failure_rates_OSD_E_order_2.txt"
        ),
    )

    # Six columns:
    # 1. p
    # 2. sample
    # 3. number of failures
    # 4. total number of trials
    # 5. average logical error rate
    # 6. binomial standard error
    results = np.array(
        [
            [
                p_value,
                sample,
                failures,
                n_errors,
                average_logical_error_rate,
                standard_error,
            ]
        ]
    )

    np.savetxt(
        result_file,
        results,
        fmt=[
            "%.10g",   # p
            "%d",      # sample
            "%d",      # number of failures
            "%d",      # total number of trials
            "%.10e",   # average logical error rate
            "%.10e",   # standard error
        ],
        header=(
            "p sample number_of_failures total_number_of_trials "
            "average_logical_error_rate standard_error"
        ),
    )

    print(
        f"p={p_value}: failures={failures}/{n_errors} "
        f"p_L={average_logical_error_rate:.3e} +/- {standard_error:.3e} "
        f"-> {result_file}"
    )

    return (p_value, failures, n_errors, average_logical_error_rate, standard_error)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--error-rates",
        type=float,
        nargs="+",
        default=error_rates,
        help="List of physical error rates to decode (defaults to the list in this file)",
    )
    args = parser.parse_args()
    rates = args.error_rates

    # --------------------------------------------------------
    # Load code matrices once (identical for every error rate)
    # --------------------------------------------------------

    HZ = np.loadtxt(os.path.join(code_directory, "HZ.txt"), dtype=int)
    LZ = np.loadtxt(os.path.join(code_directory, "LZ.txt"), dtype=int)

    # Prefix derived from the number of data qubits (HZ columns),
    # e.g. an 18-column HZ -> "18q_BB".
    code_label = f"{HZ.shape[1]}q_BB"

    # A residual error is successful when it commutes with both the
    # Z stabilizers and the Z logical operators.
    H_perp = np.vstack([HZ, LZ])

    # Build the decoder once and reuse it for every rate.
    decoder_x = BpOsdDecoder(pcm=HZ, **decoder_settings)

    print("Code label:", code_label)
    print("HZ shape:", HZ.shape)
    print("LZ shape:", LZ.shape)
    print("Decoding error rates:", rates)

    summaries = []

    for p_value in tqdm(rates, desc="error rates", unit="rate"):
        summary = decode_one(p_value, HZ, H_perp, decoder_x, code_label)
        if summary is not None:
            summaries.append(summary)

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\nDecoding completed.")
    print(f"Processed {len(summaries)} / {len(rates)} error rate(s).")
    print("Results saved in:", results_directory)


if __name__ == "__main__":
    main()
