#!/usr/bin/env bash
#
# Run event-selection over every .root in an input dir, one job per file.
# Data files (name matches DATA_PATTERN) get the --data flag; everything else
# is treated as MC. Output name = input basename with the "___<uuid>" NanoAOD
# hash suffix stripped, plus _EVENTSELECTION.root.
#
# Usage:
#   scripts/run_eventsel_all.sh [INPUT_DIR] [OUTPUT_DIR] [CONFIG]
#
# Defaults target the NanoAODv12_2022 testing samples.

set -u

INPUT_DIR="${1:-../TestingSamples/NanoAODv15_2024}"
OUTPUT_DIR="${2:-outputs/eventsel}"
CONFIG="${3:-configs/2024.yaml}"

# Filename substring that identifies collision data (gets --data)
DATA_PATTERN="JetMET-Run"

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
    if [[ "$base" == *"$DATA_PATTERN"* ]]; then
        data_flag="--data"
        tag="DATA"
    fi

    echo "[$n/${#files[@]}] ($tag) $sample"
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

echo "Done. $((n - fail))/$n succeeded, $fail failed."
[ $fail -eq 0 ]
