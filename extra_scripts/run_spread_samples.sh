#!/usr/bin/env bash
# Generate the spread-comparison data for several disorder realisations.
#
#   ./run_spread_samples.sh 2 3        # samples 2 and 3 at the current sigma
#
# decoder_setup.py's cnot_sigma decides which arm you are building; set it to
# 0 * two_qubit_error_rate for the homogeneous arm, 1.0 * ... for sigma = p.
# The shared code/*.txt files already exist, so --skip-code avoids rewriting
# them on every sample.
set -euo pipefail

P=0.0005
SAMPLES=("$@")
[ ${#SAMPLES[@]} -eq 0 ] && SAMPLES=(1 2 3)

for s in "${SAMPLES[@]}"; do
    echo "=== sample $s : decoder_setup ==="
    python decoder_setup.py "$P" "$s"

    echo "=== sample $s : shots ==="
    python cer_tools.py --error-rates "$P" --sample "$s" --skip-code
done

echo "=== couplings + two-qubit marginals (all samples) ==="
python J_ij_couplings.py sampling --base "${BASE:-72q_BB_cycles_1_spread_comparison}"

echo "DONE"
