#!/bin/bash
# Stage 4: BP-OSD decode the testing data for one error rate (job array).
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=bb_decode
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19
#SBATCH --output=cluster/logs/decode_%A_%a.out

cd "$SLURM_SUBMIT_DIR" || exit 1
source cluster/env.sh

RATE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" cluster/rates.txt)
echo "Task $SLURM_ARRAY_TASK_ID -> BP_OSD_decoding.py p=$RATE"

python BP_OSD_decoding.py --error-rates "$RATE"
