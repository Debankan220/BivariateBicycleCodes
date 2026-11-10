#!/usr/bin/env bash
# Full (p, sigma, sample) sweep for the BB spread study.
#
#   ./run_sigma_sweep.sh generate   # decoder_setup + cer_tools for every cell
#   ./run_sigma_sweep.sh couplings  # J_ij for every raw_probabilities_sampling file
#   ./run_sigma_sweep.sh decode     # BP and BP-OSD for every cell
#   ./run_sigma_sweep.sh all
#
# sigma is set per arm as a multiple of p, passed to decoder_setup.py as argv[3]
# and to the decoders via BB_SIGMA_MULT so both build the same filenames.
set -uo pipefail

export BB_OUT="${BB_OUT:-72q_BB_cycles_1_sigma_sweep/sigma_10p}"
JOBS="${JOBS:-8}"
PS=(0.0001 0.0003 0.0005 0.0007 0.0009 0.0011 0.0013 0.0015)

# arm:  "<sigma multiplier> <number of samples>"
#ARMS=("0 3" "1 5" "2 5")
ARMS=("10 5")

LOGS="$BB_OUT/logs"; mkdir -p "$LOGS"

cells() {                                   # emit "p mult sample" per line
    for p in "${PS[@]}"; do
        for arm in "${ARMS[@]}"; do
            set -- $arm; mult=$1; nsamp=$2
            for s in $(seq 1 "$nsamp"); do echo "$p $mult $s"; done
        done
    done
}

gen_one() {
    p=$1; mult=$2; s=$3
    tag="p${p}_m${mult}_s${s}"
    python decoder_setup.py "$p" "$s" "$mult" > "$LOGS/setup_$tag.log" 2>&1 || {
        echo "FAIL setup $tag"; return 1; }
    pick=$(grep -a "saving data to" "$LOGS/setup_$tag.log" | awk '{print $NF}')
    [ -f "$pick" ] || { echo "FAIL no pickle $tag"; return 1; }
    python cer_tools.py --error-rates "$p" --sample "$s" --skip-code \
        --mydata-template "$pick" > "$LOGS/cer_$tag.log" 2>&1 || {
        echo "FAIL cer $tag"; return 1; }
    echo "ok $tag"
}
export -f gen_one; export LOGS

decode_one() {
    p=$1; mult=$2; s=$3
    tag="p${p}_m${mult}_s${s}"
    BB_SAMPLE=$s BB_SIGMA_MULT=$mult python BP_decoding.py --error-rates "$p" \
        > "$LOGS/bp_$tag.log" 2>&1
    BB_SAMPLE=$s BB_SIGMA_MULT=$mult python BP_OSD_decoding.py --error-rates "$p" \
        > "$LOGS/bposd_$tag.log" 2>&1
    echo "ok $tag"
}
export -f decode_one

case "${1:-all}" in
  generate)  cells | xargs -P "$JOBS" -n 3 bash -c 'gen_one "$@"' _ ;;
  couplings) python J_ij_couplings.py sampling --base "$BB_OUT" ;;
  decode)    cells | xargs -P "$JOBS" -n 3 bash -c 'decode_one "$@"' _ ;;
  all)       "$0" generate && "$0" couplings && "$0" decode ;;
esac
echo "STAGE ${1:-all} COMPLETE"
