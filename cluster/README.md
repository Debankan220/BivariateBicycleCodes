# Running the BB pipeline on Alliance (Compute Canada)

The 20 error rates are independent, so we run them as **SLURM job arrays** —
one rate per array task — instead of a Python loop. Each stage waits for the
previous one via job dependencies.

```
setup (array 0-19)  ->  code (single)  ->  data (array 0-19)  ->  decode (array 0-19)
decoder_setup.py         cer_tools.py         cer_tools.py            BP_OSD_decoding.py
  p per task             --only-code          p per task, --skip-code  p per task
```

The `code` stage is separate and single because the `code/*.txt` files are the
same for every rate; writing them once avoids 20 tasks clobbering the same
files at once.

## One-time setup (on a login node)

```bash
module load StdEnv/2023 python/3.11
virtualenv --no-download "$HOME/BB_env"
source "$HOME/BB_env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scipy      # from the Alliance wheelhouse
pip install ldpc bposd tqdm             # from PyPI (login nodes have internet)
```

Then edit these before submitting:

- **`cluster/env.sh`** — module + venv names, if different from above.
- **`--account=def-CHANGEME`** in all four `*.sbatch` files — your allocation
  (e.g. `def-yourpi`). Set it once: `sed -i 's/def-CHANGEME/def-yourpi/' cluster/*.sbatch`
- **`cluster/rates.txt`** — the error rates, one per line (array size adapts).
- The code being run is set **inside the Python scripts** (`decoder_setup.py`
  code params, and the `output_folder` / `base_directory` / `mydata_template`
  in `cer_tools.py` and `BP_OSD_decoding.py`). Make sure all three agree
  (currently the 72q code -> `72q_BB_cycles_1`).

## Submit everything

```bash
bash cluster/submit_all.sh
```

Prints the four job IDs. Monitor with `squeue -u $USER`; logs land in
`cluster/logs/`.

## Run a single stage manually

```bash
sbatch --array=0-19 cluster/00_setup.sbatch     # pickles
sbatch cluster/01_code.sbatch                   # shared code files
sbatch --array=0-19 cluster/02_data.sbatch      # train/test data
sbatch --array=0-19 cluster/03_decode.sbatch    # results
```

If you already generated all pickles (they exist in `TMP/`), skip the setup
stage and start from `01_code` / `02_data`.

## Resource requests

Defaults are set from a local benchmark of the 72q code (~11 min/rate data,
~5 min/rate decode). If you move to a larger code (288q, 784q), raise `--time`
and `--mem` in the relevant `*.sbatch` file.

## Rate <-> array-index mapping

Array task `i` uses line `i+1` of `cluster/rates.txt`. To run a subset, submit
with e.g. `--array=0,5,18` or `--array=0-4`.
