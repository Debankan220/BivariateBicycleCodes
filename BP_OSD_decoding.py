import os
import numpy as np
from tqdm import tqdm
from ldpc import BpOsdDecoder


# ============================================================
# Parameters
# ============================================================

p_value = 0.005
sample = 1
num_cycles = 1

base_directory = "18q_BB_cycles_1"
code_directory = os.path.join(base_directory, "code")
testing_data_directory = os.path.join(base_directory, "testing_data")
results_directory = os.path.join(base_directory, "results")

# Create the results directory if it does not already exist
os.makedirs(results_directory, exist_ok=True)


# ============================================================
# Load code matrices
# ============================================================

HZ = np.loadtxt(
    os.path.join(code_directory, "HZ.txt"),
    dtype=int,
)

LZ = np.loadtxt(
    os.path.join(code_directory, "LZ.txt"),
    dtype=int,
)


# ============================================================
# Load testing errors
# ============================================================

test_file = os.path.join(
    testing_data_directory,
    f"test_p_{p_value}_s_{sample}.txt",
)

# The file has shape:
#     number_of_qubits x number_of_errors
#
# After transposing:
#     number_of_errors x number_of_qubits
sample_errors_X = np.loadtxt(
    test_file,
    dtype=int,
).T

n_errors = sample_errors_X.shape[0]

if n_errors == 0:
    raise ValueError("The testing-error file contains no error samples.")

print("Test file:", test_file)
print("Number of testing errors:", n_errors)
print("Number of qubits:", sample_errors_X.shape[1])


# ============================================================
# Initialize BP-OSD decoder
# ============================================================

decoder_x = BpOsdDecoder(
    pcm=HZ,
    error_rate=0.1,
    max_iter=500,
    bp_method="product_sum",
    schedule="parallel",
    osd_method="OSD_E",
    osd_order=2,
)


# ============================================================
# Stabilizer and logical checks
# ============================================================

# A residual error is successful when it commutes with both:
#     1. Z stabilizers
#     2. Z logical operators
H_perp = np.vstack([HZ, LZ])


# ============================================================
# Run decoding
# ============================================================

successes = 0

for error in tqdm(sample_errors_X, desc="BP-OSD decoding"):

    # Calculate the syndrome
    syndrome = (HZ @ error) % 2

    # Decode the syndrome
    correction = decoder_x.decode(syndrome)

    # Residual error after applying the correction
    residual_error = (error + correction) % 2

    # Successful decoding means that the residual error has:
    #     zero syndrome
    #     no logical error
    if np.all((H_perp @ residual_error) % 2 == 0):
        successes += 1


# ============================================================
# Calculate logical error statistics
# ============================================================

failures = n_errors - successes

average_logical_error_rate = failures / n_errors

# Binomial standard error:
#
# sigma = sqrt[p_L (1 - p_L) / N]
standard_error = np.sqrt(
    average_logical_error_rate
    * (1.0 - average_logical_error_rate)
    / n_errors
)


# ============================================================
# Save results
# ============================================================

result_file = os.path.join(
    results_directory,
    (
        f"72q_BB_p_{p_value}_cycles_{num_cycles}_"
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


# ============================================================
# Print summary
# ============================================================

print("\nDecoding completed.")
print("p value:", p_value)
print("Sample:", sample)
print("Number of successes:", successes)
print("Number of failures:", failures)
print("Total number of trials:", n_errors)
print("Average logical error rate:", average_logical_error_rate)
print("Binomial standard error:", standard_error)
print("Results saved to:", result_file)