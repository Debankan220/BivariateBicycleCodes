#!/bin/bash
# Stage 1: generate one decoder pickle per error rate (job array).
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=bb_setup
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19
#SBATCH --output=cluster/logs/setup_%A_%a.out

cd "$SLURM_SUBMIT_DIR" || exit 1
source cluster/env.sh

RATE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" cluster/rates.txt)
echo "Task $SLURM_ARRAY_TASK_ID -> decoder_setup.py p=$RATE"

python decoder_setup.py "$RATE"
