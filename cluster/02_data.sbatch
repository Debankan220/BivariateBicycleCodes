#!/bin/bash
# Stage 3: generate training + testing data for one error rate (job array).
# --skip-code so tasks never touch the shared code/*.txt files.
#SBATCH --account=def-CHANGEME
#SBATCH --job-name=bb_data
#SBATCH --time=00:40:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19
#SBATCH --output=cluster/logs/data_%A_%a.out

cd "$SLURM_SUBMIT_DIR" || exit 1
source cluster/env.sh

RATE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" cluster/rates.txt)
echo "Task $SLURM_ARRAY_TASK_ID -> cer_tools.py p=$RATE (--skip-code)"

python cer_tools.py --error-rates "$RATE" --skip-code
