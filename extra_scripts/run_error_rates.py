"""Run decoder_setup.py for a list of physical error rates.

Each (rate, sample) pair is run as a separate process, producing one pickle
under ./TMP/. The output filename encodes both the error rate and the sample
index, so nothing is overwritten.

The sample index selects the disorder realisation: each sample draws a fresh
set of per-CNOT error rates, held fixed for that whole run.

Usage:
    python run_error_rates.py                      # default rates, sample 1
    python run_error_rates.py 0.001 0.002          # rates on the command line
    python run_error_rates.py --samples 1 2 3      # three disorder realisations
    python run_error_rates.py 0.001 --samples 1-20 # ranges work too
"""
import argparse
import subprocess
import sys

# default sweep; override by passing rates as positional arguments
error_rates = [0.0005]


def parse_samples(tokens):
    """Expand tokens like ['1', '3-5'] into [1, 3, 4, 5]."""
    out = []
    for tok in tokens:
        if "-" in tok:
            lo, hi = tok.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    return out


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("rates", nargs="*", type=float, default=None,
                    help="Physical error rates (defaults to the list in this file)")
parser.add_argument("--samples", nargs="+", default=["1"],
                    help="Disorder realisation indices, e.g. '1 2 3' or '1-20'")
args = parser.parse_args()

rates = args.rates if args.rates else error_rates
samples = parse_samples(args.samples)

for s in samples:
    for p in rates:
        print(f"\n=== Running decoder_setup.py for error_rate={p} sample={s} ===")
        subprocess.run([sys.executable, "decoder_setup.py", str(p), str(s)],
                       check=True)
