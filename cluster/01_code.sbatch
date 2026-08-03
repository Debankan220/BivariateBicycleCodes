#!/bin/bash
# Stage 2: write the shared, rate-independent code/*.txt files ONCE.
# Runs as a single job so the shared files are never written concurrently.
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=bb_code
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=cluster/logs/code_%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
source cluster/env.sh

# The code is identical for every rate; use the first rate's pickle.
FIRST=$(sed -n "1p" cluster/rates.txt)
echo "cer_tools.py --only-code using p=$FIRST"

python cer_tools.py --only-code --error-rates "$FIRST"
