#!/bin/bash
# ------------------------------------------------------------
# Environment setup for Alliance (Compute Canada) jobs.
# This file is `source`d at the start of every job.
#
# One-time setup on a login node before submitting:
#
#   module load StdEnv/2023 python/3.11
#   virtualenv --no-download "$HOME/BB_env"
#   source "$HOME/BB_env/bin/activate"
#   pip install --no-index --upgrade pip
#   pip install --no-index numpy scipy        # Alliance wheelhouse
#   pip install ldpc bposd tqdm               # from PyPI (login node has internet)
#
# Then edit the two lines below if your module/venv names differ.
# ------------------------------------------------------------

module load StdEnv/2023 python/3.11
source "$HOME/BB_env/bin/activate"
