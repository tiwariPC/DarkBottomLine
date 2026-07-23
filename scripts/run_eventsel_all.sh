#!/usr/bin/env bash
#
# Run event-selection over every .root in an input dir, one job per file.
# Data files (name matches DATA_PATTERN) get the --data flag; everything else
# is treated as MC. Output name = input basename with the "___<uuid>" NanoAOD
# hash suffix stripped, plus _EVENTSELECTION.root.
#
# Usage:
#   scripts/run_eventsel_all.sh [INPUT_DIR] [OUTPUT_DIR] [CONFIG] [--dry-run]
#
# --dry-run prints the planned commands without executing them.
#
# Defaults target the NanoAODv12_2022 testing samples.

set -u

DRY_RUN=0
args=()
for a in "$@"; do
    if [ "$a" = "--dry-run" ]; then
        DRY_RUN=1
    else
        args+=("$a")
    fi
done

INPUT_DIR="${args[0]:-../TestingSamples/NanoAODv15_2024}"
OUTPUT_DIR="${args[1]:-outputs/eventsel}"
CONFIG="${args[2]:-configs/2024.yaml}"

# Filename substring pattern that identifies collision data (gets --data)
DATA_PATTERN="JetMET-Run|JetMET0-Run|JetMET1-Run|MET-Run|EGamma-Run|EGamma0-Run|EGamma1-Run"

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
files=("$INPUT_DIR"/*.root)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "No .root files found in $INPUT_DIR" >&2
    exit 1
fi

echo "Config     : $CONFIG"
echo "Input dir  : $INPUT_DIR"
echo "Output dir : $OUTPUT_DIR"
echo "Files      : ${#files[@]}"
[ $DRY_RUN -eq 1 ] && echo "Mode       : DRY RUN (no jobs executed)"
echo

n=0
fail=0
for f in "${files[@]}"; do
    n=$((n + 1))
    base="$(basename "$f" .root)"
    # strip NanoAOD "___<uuid>" suffix -> clean sample name
    sample="${base%%___*}"
    out="$OUTPUT_DIR/${sample}_EVENTSELECTION.root"

    data_flag=""
    tag="MC"
    if [[ "$base" =~ $DATA_PATTERN ]]; then
        data_flag="--data"
        tag="DATA"
    fi

    echo "[$n/${#files[@]}] ($tag) $sample"

    if [ $DRY_RUN -eq 1 ]; then
        echo "  darkbottomline analyze --mode event-selection --config $CONFIG --input $f --event-selection-output $out $data_flag"
        echo
        continue
    fi

    darkbottomline analyze \
        --mode event-selection \
        --config "$CONFIG" \
        --input "$f" \
        --event-selection-output "$out" \
        $data_flag
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "  FAILED (rc=$rc): $f" >&2
        fail=$((fail + 1))
    fi
    echo
done

if [ $DRY_RUN -eq 1 ]; then
    echo "Dry run complete. $n file(s) would be processed."
    exit 0
fi

echo "Done. $((n - fail))/$n succeeded, $fail failed."
[ $fail -eq 0 ]
