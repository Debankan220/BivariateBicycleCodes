#!/bin/bash
# Submit the full pipeline as four dependent SLURM stages, one rate per task.
#
#   setup (array) -> code (single) -> data (array) -> decode (array)
#
# Run from anywhere:  bash cluster/submit_all.sh
set -euo pipefail

# Move to the project root (parent of this script's directory).
cd "$(dirname "$0")/.."

mkdir -p cluster/logs

# Number of rates -> array range 0..N-1 (ignores blank lines).
N=$(grep -c '[^[:space:]]' cluster/rates.txt)
LAST=$((N - 1))
echo "Submitting pipeline for $N error rates (array 0-$LAST)."

jid_setup=$(sbatch --parsable --array=0-"$LAST" cluster/00_setup.sbatch)
echo "  setup   job $jid_setup"

jid_code=$(sbatch --parsable --dependency=afterok:"$jid_setup" cluster/01_code.sbatch)
echo "  code    job $jid_code  (after setup)"

jid_data=$(sbatch --parsable --dependency=afterok:"$jid_code" --array=0-"$LAST" cluster/02_data.sbatch)
echo "  data    job $jid_data  (after code)"

jid_decode=$(sbatch --parsable --dependency=afterok:"$jid_data" --array=0-"$LAST" cluster/03_decode.sbatch)
echo "  decode  job $jid_decode  (after data)"

echo
echo "Track with:  squeue -u \$USER"
echo "Logs in:     cluster/logs/"
