#!/bin/bash
# ---------------------------------------------------------------------------
# A/B benchmark: old vs new (redundant-copy cleanup) DNN training code.
#
# The point of this script is NOT just "how much memory does the new code
# use": it has to show that the new code produces the SAME model as the old
# code while using less memory.  So it does three things:
#
#   1. starts both code versions at the same time, on the same data, with the
#      same seed and configuration;
#   2. records each process' own peak RSS (/usr/bin/time -v) and wall clock;
#   3. compares the two trained models field by field (architecture, split
#      sizes, AUCs, scaler, network weights) and writes a PASS/FAIL verdict.
#
# Everything lands in a single summary file, refreshed after each run:
#   reports/memory_ab/summary_<stamp>.md
#
# How the two versions run at once
# --------------------------------
# The old code is materialised from git (OLD_REF, default HEAD) into a
# temporary tree (only the code paths, ~3 MB).  The editable install
# registers a meta-path finder, but it is *appended* to sys.meta_path, so a
# directory listed on PYTHONPATH is still resolved first; that is what lets
# the old tree be imported in a second process without touching either the
# working tree or the installed environment.  Nothing is copied over the
# working tree, unlike a swap-in-place approach.
#
# Usage:
#   bash scripts/dnn/bench_memory_ab.sh
#   BKG_FRACTION=0.5 MASS_POINT=MH3_600_MH4_10_Mchi_1 bash scripts/dnn/bench_memory_ab.sh
#
# Environment knobs:
#   OLD_REF       git ref holding the pre-cleanup code (default: HEAD)
#   MASS_POINT    mass point label (default: MH3_1500_MH4_1000_Mchi_1)
#   BKG_FRACTION  background fraction (default: 0.25)
#   INPUT         event-selection directory (default: /home/zzq/eventsel-merged)
#   OUTROOT       output root for logs + summary (default: <repo>/reports/memory_ab)
#   PY            python interpreter
# ---------------------------------------------------------------------------
set -u

REPO="${REPO:-/home/zzq/DarkBottomLine}"
PY="${PY:-/home/zzq/miniconda3/envs/darkbottomline/bin/python}"
export LD_LIBRARY_PATH="/home/zzq/miniconda3/envs/darkbottomline/lib:${LD_LIBRARY_PATH:-}"

INPUT="${INPUT:-/home/zzq/eventsel-merged}"
DNN_CFG="${DNN_CFG:-configs/dnn.yaml}"
MASS_POINT="${MASS_POINT:-MH3_1500_MH4_1000_Mchi_1}"
BKG_FRACTION="${BKG_FRACTION:-0.25}"
OLD_REF="${OLD_REF:-HEAD}"
OUTROOT="${OUTROOT:-$REPO/reports/memory_ab}"

# Code paths that differ between the two versions (only these are extracted
# from git for the old tree).
CODE_PATHS=(darkbottomline dnn utils scripts configs data/cross-section)

STAMP="$(date '+%Y%m%d-%H%M%S')"
WORK="/tmp/dbl_memab_${STAMP}"
OLD_TREE="$WORK/old_tree"
SUMMARY="$OUTROOT/summary_${STAMP}.md"
OLD_LOG="$OUTROOT/run_old_${STAMP}.log"
NEW_LOG="$OUTROOT/run_new_${STAMP}.log"
OLD_TIME="$OUTROOT/time_old_${STAMP}.txt"
NEW_TIME="$OUTROOT/time_new_${STAMP}.txt"
OLD_MODEL="${OLD_MODEL:-data/memab_old/${MASS_POINT,,}}"
NEW_MODEL="${NEW_MODEL:-data/memab_new/${MASS_POINT,,}}"

mkdir -p "$WORK" "$OUTROOT" "$OLD_TREE"
cd "$REPO"

# ---------------------------------------------------------------------------
# Build the old code tree from git
# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] extracting old code (${OLD_REF}) into ${OLD_TREE}"
git archive "$OLD_REF" -- "${CODE_PATHS[@]}" | tar -x -C "$OLD_TREE" || exit 1
if grep -q "reuse them instead of allocating a second full copy" \
        "$OLD_TREE/darkbottomline/dnn_trainer.py"; then
    echo "ERROR: old tree already contains the new code - is ${OLD_REF} the new commit?" >&2
    exit 1
fi
echo "[$(date '+%F %T')] old tree: $(grep -c '' "$OLD_TREE/darkbottomline/dnn_trainer.py") lines; \
working tree: $(grep -c '' "$REPO/darkbottomline/dnn_trainer.py") lines"

# A leftover model from an earlier attempt would make the training script
# skip this mass point ("already trained"), which would silently turn the
# comparison into a no-op.  The two dirs are scratch output of this script.
for d in "$REPO/data/memab_old" "$REPO/data/memab_new"; do
    if [ -e "$d" ]; then
        echo "[$(date '+%F %T')] removing stale scratch output $d"
        rm -rf "$d"
    fi
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
peak_rss_gb() {  # $1 = /usr/bin/time -v output
    [ -f "$1" ] || { echo "n/a"; return; }
    awk '/Maximum resident set size \(kbytes\)/{v=$NF} END{
             if (v!="") printf "%.2f", v/1048576; else printf "n/a"}' "$1"
}
wall_clock() {  # $1 = /usr/bin/time -v output
    [ -f "$1" ] || { echo "n/a"; return; }
    awk -F': ' '/Elapsed \(wall clock\) time/{print $NF}' "$1" | tail -1
}
json_field() {  # $1 = model dir, $2 = key
    "$PY" - "$1/train_metrics.json" "$2" <<'PY' 2>/dev/null || echo "n/a"
import json, sys
try:
    m = json.load(open(sys.argv[1]))
    print(repr(m.get(sys.argv[2], "n/a")))
except Exception:
    print("n/a")
PY
}

run_one() {  # $1 = tag, $2 = tree root, $3 = script, $4 = log, $5 = time file
    local tag="$1" tree="$2" script="$3" log="$4" timef="$5"
    cd "$REPO"
    PYTHONPATH="$tree" /usr/bin/time -v -o "$timef" \
        "$PY" "$script" \
            --dnn-config "$DNN_CFG" \
            --input "$INPUT" \
            --weight-branch full_event_weight \
            --xsection-signal-json data/cross-section/xsection_signal.json \
            --xsection-json data/cross-section/xsection_background_run3.json \
            --outdir-base "data/memab_${tag}" \
            --plot-dir-base "outputs/memab_${tag}" \
            --bkg-fraction "$BKG_FRACTION" \
            --masspoints "$MASS_POINT" \
        > "$log" 2>&1
    echo $? > "$OUTROOT/exit_${tag}_${STAMP}.txt"
}

exit_code() {  # $1 = tag
    local f="$OUTROOT/exit_${1}_${STAMP}.txt"
    [ -f "$f" ] && cat "$f" || echo "?"
}

# ---------------------------------------------------------------------------
# Result-equivalence check: compare the two trained models numerically
# ---------------------------------------------------------------------------
compare_models() {
    "$PY" - "$OLD_MODEL" "$NEW_MODEL" <<'PY'
import json, sys, math
from pathlib import Path

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
notes = []


def flat_numbers(obj, prefix=""):
    """Flatten nested dict/list structures into {path: float}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flat_numbers(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flat_numbers(v, f"{prefix}[{i}]"))
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def compare_numbers(a, b):
    """Return (max abs diff, worst key, n compared, n missing) for two dicts."""
    keys = [k for k in a if k in b]
    missing = len(set(a) ^ set(b))
    worst, where = 0.0, "-"
    for k in keys:
        va, vb = a[k], b[k]
        if math.isnan(va) and math.isnan(vb):
            d = 0.0
        elif math.isnan(va) or math.isnan(vb):
            d = float("inf")
        else:
            d = abs(va - vb)
        if d > worst:
            worst, where = d, k
    return worst, where, len(keys), missing


if not (old_dir / "train_metrics.json").exists():
    print("VERDICT: INCOMPLETE - old run produced no train_metrics.json")
    raise SystemExit(0)
if not (new_dir / "train_metrics.json").exists():
    print("VERDICT: INCOMPLETE - new run produced no train_metrics.json")
    raise SystemExit(0)

mo = json.load(open(old_dir / "train_metrics.json"))
mn = json.load(open(new_dir / "train_metrics.json"))

def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return {}


def tensors(obj, prefix=""):
    """Flatten anything that ends up in the checkpoint into {path: tensor}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(tensors(v, f"{prefix}[{i}]"))
    else:
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                out[prefix] = obj.detach().to("cpu").to(torch.float64).flatten()
        except Exception:
            pass
    return out


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.12g}"
    return str(v)


# --- quantities that describe the experiment itself ------------------------
rows = []
for k in ("n_train", "n_val", "n_test"):
    rows.append((k, mo.get(k), mn.get(k), abs((mo.get(k) or 0) - (mn.get(k) or 0))))

same_features = mo.get("features") == mn.get("features")
rows.append(("features (names + order)",
             f"{len(mo.get('features') or [])} names",
             f"{len(mn.get('features') or [])} names",
             "0" if same_features else "DIFFERENT"))

spec_old = flat_numbers(mo.get("model_spec") or {}, "model_spec")
spec_new = flat_numbers(mn.get("model_spec") or {}, "model_spec")
spec_diff, spec_where, spec_n, spec_missing = compare_numbers(spec_old, spec_new)
spec_same = (mo.get("model_spec") or {}) == (mn.get("model_spec") or {})
rows.append(("model_spec (n_inputs/hidden_layers/dropout/parametric)",
             "identical" if spec_same else "differs",
             "identical" if spec_same else "differs",
             "0" if spec_same else f"{spec_diff:.3e}"))

# --- training outcome -----------------------------------------------------
for k in ("auc_train", "auc_val", "auc_test"):
    a, b = mo.get(k), mn.get(k)
    d = abs(a - b) if isinstance(a, float) and isinstance(b, float) else float("nan")
    rows.append((k, a, b, f"{d:.3e}"))

# --- the full numeric dumps ----------------------------------------------
detail_keys = ("class_balance_factors", "score_diagnostics",
               "weight_diagnostics", "negative_weight_handling")
detail_old = flat_numbers({k: mo.get(k) for k in detail_keys})
detail_new = flat_numbers({k: mn.get(k) for k in detail_keys})
detail_diff, detail_where, detail_n, detail_missing = compare_numbers(detail_old, detail_new)

# --- feature scaler -------------------------------------------------------
scaler_diff, scaler_where, scaler_n, scaler_missing = compare_numbers(
    flat_numbers(load_json(old_dir / "scaler.json")),
    flat_numbers(load_json(new_dir / "scaler.json")),
)

# --- the trained weights --------------------------------------------------
weight_diff, weight_where, weight_n = float("nan"), "-", 0
try:
    import torch
    co = torch.load(old_dir / "dnn_model.pt", map_location="cpu", weights_only=False)
    cn = torch.load(new_dir / "dnn_model.pt", map_location="cpu", weights_only=False)
    to, tn = tensors(co), tensors(cn)
    common = [k for k in to if k in tn]
    weight_n = len(common)
    weight_diff = 0.0
    for k in common:
        d = float((to[k] - tn[k]).abs().max())
        if d > weight_diff:
            weight_diff, weight_where = d, k
    if len(common) != len(set(to) | set(tn)):
        notes.append("checkpoint tensor sets differ")
except Exception as exc:  # pragma: no cover
    notes.append(f"checkpoint comparison failed: {exc}")

rows.append(("scaler.json (feature mean/scale)",
             f"{scaler_n} values", f"{scaler_n} values", f"{scaler_diff:.3e}"))
rows.append(("checkpoint dnn_model.pt",
             f"{weight_n} tensors", f"{weight_n} tensors", f"{weight_diff:.3e}"))
rows.append((f"{len(detail_keys)} diagnostics blocks "
             f"(scores / weights / class balance)",
             f"{detail_n} fields", f"{detail_n} fields", f"{detail_diff:.3e}"))

everything_equal = (same_features and spec_same
                    and detail_diff <= 1e-9 and scaler_diff <= 1e-9
                    and weight_diff <= 1e-9 and detail_missing == 0)
close_enough = (same_features and spec_same
                and detail_diff <= 1e-6 and scaler_diff <= 1e-6
                and weight_diff <= 1e-6)

if everything_equal:
    verdict = "PASS - identical to the last bit"
elif close_enough:
    verdict = "PASS within float32 rounding (<= 1e-6) - not bit-identical"
else:
    verdict = "FAIL - the two runs disagree"

print(f"VERDICT: {verdict}")
print()
print("| quantity | old code | new code | max abs diff |")
print("|---|---|---|---|")
for name, a, b, d in rows:
    d_str = d if isinstance(d, str) else f"{d:.3e}"
    print(f"| `{name}` | {fmt(a)} | {fmt(b)} | {d_str} |")
print()
if notes:
    for n in notes:
        print(f"- {n}")
    print()
print(f"- diagnostics compared: {detail_n} numeric fields, {detail_missing} missing"
      f" (worst: `{detail_where}`)")
print(f"- scaler compared: {scaler_n} numeric fields, {scaler_missing} missing"
      f" (worst: `{scaler_where}`)")
print(f"- checkpoint compared: {weight_n} tensors (worst: `{weight_where}`)")
print(f"- architecture compared: {spec_n} numeric fields (worst: `{spec_where}`)")
PY
}

# ---------------------------------------------------------------------------
# Summary file
# ---------------------------------------------------------------------------
write_summary() {
    local o_exit n_exit o_rss n_rss o_wall n_wall o_aucv n_aucv o_auct n_auct
    o_rss=$(peak_rss_gb "$OLD_TIME"); n_rss=$(peak_rss_gb "$NEW_TIME")
    o_wall=$(wall_clock "$OLD_TIME"); n_wall=$(wall_clock "$NEW_TIME")
    o_aucv=$(json_field "$OLD_MODEL" auc_val); n_aucv=$(json_field "$NEW_MODEL" auc_val)
    o_auct=$(json_field "$OLD_MODEL" auc_test); n_auct=$(json_field "$NEW_MODEL" auc_test)
    o_exit=$(exit_code old); n_exit=$(exit_code new)

    local cmp cmp_verdict
    cmp=$(compare_models)
    cmp_verdict=$(echo "$cmp" | sed -n 's/^VERDICT: //p' | head -1)
    [ -n "$cmp_verdict" ] || cmp_verdict="not available yet"

    {
        echo "# DNN training A/B: old code vs memory-cleanup code"
        echo
        echo "- started: \`${STAMP}\`"
        echo "- data: \`${INPUT}\`"
        echo "- config: \`${DNN_CFG}\`  |  mass point: \`${MASS_POINT}\`  |  background fraction: \`${BKG_FRACTION}\`"
        echo "- old code: git \`${OLD_REF}\` (\`${OLD_TREE}\`)"
        echo "- new code: working tree \`${REPO}\` (redundant-copy cleanup, uncommitted)"
        echo "- both runs launched simultaneously, same seed (from \`${DNN_CFG}\`)"
        echo
        echo "## 1. Results (must be identical)"
        echo
        echo "**${cmp_verdict}**"
        echo
        echo "$cmp" | grep -E '^\||^- ' || echo "_no comparison available yet_"
        echo
        echo "## 2. Resource usage"
        echo
        echo "| run | exit code | peak RSS (GB) | wall clock | AUC(val) | AUC(test) |"
        echo "|---|---|---|---|---|---|"
        echo "| old code | ${o_exit} | ${o_rss} | ${o_wall} | ${o_aucv} | ${o_auct} |"
        echo "| new code | ${n_exit} | ${n_rss} | ${n_wall} | ${n_aucv} | ${n_auct} |"
        if [ "$o_rss" != "n/a" ] && [ "$n_rss" != "n/a" ]; then
            echo
            echo "Peak RSS saved: \`$(awk -v o="$o_rss" -v n="$n_rss" 'BEGIN{printf "%.2f GB (%.1f%%)", o-n, 100*(o-n)/o}')\`"
        fi
        echo
        echo "## 3. Artifacts"
        echo
        echo "- old model: \`${OLD_MODEL}\`"
        echo "- new model: \`${NEW_MODEL}\`"
        echo "- training logs: \`${OLD_LOG}\`, \`${NEW_LOG}\`"
        echo "- \`/usr/bin/time -v\` reports: \`${OLD_TIME}\`, \`${NEW_TIME}\`"
    } > "$SUMMARY"
}

# ---------------------------------------------------------------------------
# Run both versions at the same time
# ---------------------------------------------------------------------------
# COMPARE_ONLY=1 re-runs just the comparison against models that already
# exist (useful for debugging, or to re-check an earlier run's verdict).
if [ "${COMPARE_ONLY:-0}" = "1" ]; then
    echo "[$(date '+%F %T')] COMPARE_ONLY - not starting any training"
    write_summary
    cat "$SUMMARY"
    exit 0
fi

echo "[$(date '+%F %T')] A/B started: old=${OLD_REF} new=working tree, both at once"
echo "[$(date '+%F %T')] mass point ${MASS_POINT}, bkg fraction ${BKG_FRACTION}"

run_one old "$OLD_TREE" "$OLD_TREE/scripts/dnn/train_per_masspoint.py" "$OLD_LOG" "$OLD_TIME" &
OLD_PID=$!
run_one new "$REPO" "$REPO/scripts/dnn/train_per_masspoint.py" "$NEW_LOG" "$NEW_TIME" &
NEW_PID=$!

echo "[$(date '+%F %T')] old pid ${OLD_PID}, new pid ${NEW_PID}"

wait "$OLD_PID"; echo "[$(date '+%F %T')] old finished (exit $(exit_code old), peak RSS $(peak_rss_gb "$OLD_TIME") GB)"
write_summary
echo "[$(date '+%F %T')] summary refreshed (old done)"

wait "$NEW_PID"; echo "[$(date '+%F %T')] new finished (exit $(exit_code new), peak RSS $(peak_rss_gb "$NEW_TIME") GB)"
write_summary

echo
echo "[$(date '+%F %T')] summary written to ${SUMMARY}"
echo "-------------------------------------------------------------------"
cat "$SUMMARY"
echo "-------------------------------------------------------------------"
