#!/usr/bin/env python3
"""Train one independent binary DNN per signal mass point.

The BBDM signal EVENTSELECTION ROOT files contain 29 boolean
``GenModel_MH3_<a>_MH4_<b>_Mchi_<c>`` branches.  This script treats each such
branch as a separate signal class: for every mass point it keeps only the
signal events whose branch is true, combines them with the (shared) background
events, and trains a normal non-parametric binary classifier.

It deliberately does NOT use the parametric ``(MH3, MH4)`` input path: the goal
is 29 independent models, not one model conditioned on mass.

Signal identification
---------------------
Only the production signal sample (``DEFAULT_SIGNAL_SAMPLE``) is treated as
signal.  The input directory also contains ``BBDM-2HDMa_500_EVENTSELECTION.root``
— a small test file carrying the same ``GenModel_*`` branches — which is
skipped outright, as signal *or* background.  ``--signal-pattern`` /
``--signal-prefix`` override the default when a different sample is wanted.

Example
-------
python scripts/dnn/train_per_masspoint.py \
    --dnn-config configs/dnn.yaml \
    --input /home/zzq/eventsel-merged \
    --weight-branch full_event_weight \
    --xsection-signal-json data/cross-section/xsection_signal.json \
    --xsection-json data/cross-section/xsection_background_run3.json \
    --outdir-base data/dnn_masspoint \
    --plot-dir-base outputs/dnn_masspoint \
    --bkg-fraction 0.25 \
    --masspoints MH3_600_MH4_10_Mchi_1   # omit to train all 29
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml

from dnn.common import sanitize_feature_frame
from dnn.data import read_branch_as_array, read_tree_branches_as_arrays
from dnn.feature_engineering import build_feature_frame_from_tree
from dnn.make_trees import (
    _is_data,
    _is_signal_heuristic,
    _is_test_sample,
    _sample_name,
)
from darkbottomline.dnn_trainer import DNNTrainer
from darkbottomline.plotting import PlotManager, _find_xsec


log = logging.getLogger("train_per_masspoint")

# The production signal sample for this analysis.  Unless the caller passes
# --signal-pattern / --signal-prefix, only files whose name starts with this
# are taken as signal.  Do not fall back to the loose "BBDM" keyword: the
# directory holds other BBDM files that are not production signal.
DEFAULT_SIGNAL_SAMPLE = "BBDM-2HDMa-5f"

# MH3/MH4 can also be used as the grouping variable instead of training one
# model per mass point (see --ratio-bins).
_MASS_RE = re.compile(r"MH3_(\d+)_MH4_(\d+)", re.IGNORECASE)


def _mass_ratio(label: str) -> float | None:
    """$M_{H_3}/M_{H_4}$ for a GenModel label, or None if unparseable."""
    m = _MASS_RE.search(label)
    if not m:
        return None
    mh3, mh4 = int(m.group(1)), int(m.group(2))
    return mh3 / mh4 if mh4 else None


def _fmt_edge(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _make_ratio_bins(
    labels: list[str],
    n_bins: int,
    edges: list[float] | None,
) -> "dict[str, list[str]]":
    """Group mass points into contiguous bins of $M_{H_3}/M_{H_4}$.

    Without explicit *edges* the bins are terciles by number of mass points, so
    each bin holds a comparable share of the parameter space.  The returned
    mapping is ordered from low to high ratio and its keys are used as model
    names (and therefore as output directory names).
    """
    pairs = []
    for lab in labels:
        r = _mass_ratio(lab)
        if r is None:
            raise SystemExit(f"cannot parse MH3/MH4 from mass point label {lab!r}")
        pairs.append((r, lab))
    pairs.sort()

    if n_bins < 2:
        raise SystemExit(f"--ratio-bins must be at least 2, got {n_bins}")

    if edges:
        cuts = sorted(float(e) for e in edges)
        if len(cuts) != n_bins - 1:
            raise SystemExit(
                f"--ratio-bin-edges needs {n_bins - 1} values for "
                f"{n_bins} bins, got {len(cuts)}"
            )
    else:
        n = len(pairs)
        if n_bins > n:
            raise SystemExit(f"cannot split {n} mass points into {n_bins} bins")
        # upper edge of bin k = ratio of the last mass point in that bin
        cuts = [pairs[round((k + 1) * n / n_bins) - 1][0] for k in range(n_bins - 1)]
        cuts = sorted(set(cuts))
        if len(cuts) != n_bins - 1:
            raise SystemExit(
                "equal-count bins collapse with this many mass points - "
                "pass --ratio-bin-edges explicitly"
            )

    bins: dict[str, list[str]] = {}
    prev: float | None = None
    for k in range(n_bins):
        hi = cuts[k] if k < len(cuts) else None
        if k == 0:
            name = f"ratio_le_{_fmt_edge(hi)}"
            members = [lab for r, lab in pairs if r <= hi]
        elif hi is None:
            name = f"ratio_gt_{_fmt_edge(prev)}"
            members = [lab for r, lab in pairs if r > prev]
        else:
            name = f"ratio_{_fmt_edge(prev)}_to_{_fmt_edge(hi)}"
            members = [lab for r, lab in pairs if prev < r <= hi]
        if not members:
            raise SystemExit(f"ratio bin {name!r} would be empty")
        bins[name] = members
        prev = hi
    return bins


def _discover_files(inputs: list[str]) -> list[str]:
    """Expand directories and .txt lists into a sorted list of .root files."""
    files: list[str] = []
    for entry in inputs:
        p = Path(entry)
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.iterdir() if f.suffix == ".root"))
        elif p.suffix == ".txt":
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        files.append(line)
        else:
            files.append(entry)
    return files


def _read_wte(f: uproot.ReadOnlyFile) -> float:
    for key in ("weighted_total_events", "weighted_total_events;1"):
        if key in f:
            try:
                return float(f[key].values()[0])
            except Exception:
                pass
    return 0.0


def _read_wte_per_genmodel(f: uproot.ReadOnlyFile) -> dict[str, float]:
    """{"GenModel_<point>": generated weighted events} from weighted_total_events_GenModel_* TH1s."""
    out: dict[str, float] = {}
    prefix = "weighted_total_events_GenModel_"
    for key in f.keys():
        name = str(key).split(";")[0]
        if name.startswith(prefix):
            try:
                out["GenModel_" + name[len(prefix):]] = float(f[key].values()[0])
            except Exception:
                pass
    return out


def _load_signal_cross_sections(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    flat: dict[str, float] = {}
    for _model, entries in raw.items():
        if not isinstance(entries, dict):
            continue
        for key, value in entries.items():
            if key.startswith("_") or not isinstance(value, (int, float)):
                continue
            flat[str(key)] = float(value)
    return flat


def _load_background_cross_sections(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    return PlotManager._normalize_cross_sections(raw)


def _load_file(
    path: str,
    features: list[str],
    weight_branch: str,
    max_events: int | None,
    fraction: float | None,
    rng_seed,
    sig_flag: bool,
) -> dict | None:
    """Read one EVENTSELECTION file into a feature frame + weight + mass flags.

    The caller has already classified the file (see :func:`_classify`) and only
    passes files that take part in training; this function just reads them.
    """
    sample = _sample_name(path)

    with uproot.open(path) as f:
        if "Events" not in f:
            return None
        tree = f["Events"]

        # If a random fraction is requested, subsample ~fraction of this file's
        # events in place (memory-efficient, via entry_mask) instead of reading
        # the first max_events entries.
        if fraction is not None and 0.0 < fraction < 1.0:
            n_total = int(tree.num_entries)
            keep = np.random.default_rng(rng_seed).random(n_total) < fraction
            read_max = None
        else:
            keep = None
            read_max = max_events

        df, _src, _used = build_feature_frame_from_tree(
            tree, features, max_events=read_max, entry_mask=keep
        )
        df = sanitize_feature_frame(df).astype("float32", copy=False)
        n = len(df)

        avail = set(tree.keys())
        if weight_branch in avail:
            w = read_branch_as_array(
                tree, weight_branch, max_events=read_max, entry_mask=keep
            ).astype("f8")
            w = np.where(np.isfinite(w), w, 0.0)
        else:
            w = np.ones(n, dtype="f8")

        n = min(n, len(w))
        df = df.iloc[:n].reset_index(drop=True)
        w = w[:n]

        flags: dict[str, np.ndarray] = {}
        if sig_flag:
            gm_cols = sorted(k for k in avail if str(k).startswith("GenModel_"))
            if gm_cols:
                gm = read_tree_branches_as_arrays(
                    tree, gm_cols, max_events=read_max, entry_mask=keep
                )
                for col in gm_cols:
                    arr = np.asarray(gm[col][:n], dtype=bool)
                    flags[col[len("GenModel_"):]] = arr
                if flags:
                    common = min(len(v) for v in flags.values())
                    df = df.iloc[:common].reset_index(drop=True)
                    w = w[:common]
                    flags = {k: v[:common] for k, v in flags.items()}

        wte = _read_wte(f)
        wte_gm = _read_wte_per_genmodel(f)

    return {
        "path": path,
        "sample": sample,
        "sig_flag": bool(sig_flag),
        "df": df,
        "w": w,
        "flags": flags,
        "wte": wte,
        "wte_gm": wte_gm,
    }


def _classify(
    path: str,
    label_map: dict[str, int] | None,
    signal_patterns: tuple[str, ...],
    signal_prefix: str | None,
) -> tuple[bool, bool] | None:
    """Return (is_signal, is_data), or None if the path should be skipped.

    This is the only place that decides what a file is; the reader downstream
    just consumes the verdict, so the two can never disagree.
    """
    # Test/placeholder samples carry the same GenModel_* branches as the
    # production signal, so they must be kept out of training entirely — as
    # signal *and* as background.
    if _is_test_sample(path):
        log.info("Skipping test sample (never used for training): %s", path)
        return None
    if label_map is not None:
        key = path if path in label_map else Path(path).name
        if key not in label_map:
            log.warning("Path %s not in label CSV — skipping", path)
            return None
        return bool(label_map[key] == 1), False
    if _is_data(path):
        return False, True
    if signal_patterns or signal_prefix:
        return _is_signal_heuristic(path, signal_patterns, signal_prefix), False
    # Default: only the production signal sample counts as signal.
    return Path(path).name.lower().startswith(
        DEFAULT_SIGNAL_SAMPLE.lower()
    ), False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train one independent DNN per GenModel mass point."
    )
    ap.add_argument("--dnn-config", required=True)
    ap.add_argument("--input", nargs="+", required=True,
                    help="Directory, list of files, or a .txt file.")
    ap.add_argument("--weight-branch", default="full_event_weight")
    ap.add_argument("--xsection-signal-json",
                    default="data/cross-section/xsection_signal.json")
    ap.add_argument("--xsection-json",
                    default="data/cross-section/xsection_background_run3.json")
    ap.add_argument("--outdir-base", default="data/dnn_masspoint")
    ap.add_argument("--plot-dir-base", default="outputs/dnn_masspoint")
    ap.add_argument("--max-events-signal", type=int, default=None,
                    help="Cap events per signal file (default: read all).")
    ap.add_argument("--bkg-fraction", type=float, default=0.25,
                    help="Randomly keep this fraction of each background file "
                         "(default 0.25, i.e. about 1/4 of total background).")
    ap.add_argument("--masspoints", default=None,
                    help="Comma-separated GenModel labels (e.g. "
                         "MH3_600_MH4_10_Mchi_1). Default: all found.")
    ap.add_argument("--ratio-bins", type=int, default=0,
                    help="Instead of one model per mass point, group the mass "
                         "points into this many bins of M_H3/M_H4 and train "
                         "one model per bin (0 = off).")
    ap.add_argument("--ratio-bin-edges", default=None,
                    help="Comma-separated bin edges for --ratio-bins, e.g. "
                         "'2,5'. Default: equal-count bins.")
    ap.add_argument("--signal-pattern", action="append", default=None,
                    dest="signal_pattern",
                    help="Regex(es) selecting signal files. Overrides the "
                         f"default (name starts with {DEFAULT_SIGNAL_SAMPLE}).")
    ap.add_argument("--signal-prefix", default=None,
                    help="Signal files start with this string. Overrides the "
                         f"default ({DEFAULT_SIGNAL_SAMPLE}).")
    ap.add_argument("--label-csv", default=None)
    ap.add_argument("--lumi", type=float, default=1.0,
                    help="Luminosity for lumi*xsec*1000/wte weighting "
                         "(default 1.0, same as train-dnn).")
    ap.add_argument("--config-year", default=None,
                    help="Optional year config; overrides --lumi with its lumi value.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.dnn_config) as fh:
        dnn_cfg = yaml.safe_load(fh)
    features = list(dnn_cfg.get("features") or [])
    if not features:
        log.error("dnn config has no features list")
        return 1
    seed = int((dnn_cfg.get("training") or {}).get("seed", 7))

    lumi = args.lumi
    if args.config_year:
        with open(args.config_year) as fh:
            year_cfg = yaml.safe_load(fh)
        lumi = float(year_cfg.get("lumi", year_cfg.get("luminosity", lumi)))
    log.info("Using lumi = %s", lumi)

    label_map: dict[str, int] | None = None
    if args.label_csv:
        label_map = {}
        with open(args.label_csv) as fh:
            import csv
            reader = csv.DictReader(fh)
            for row in reader:
                label_map[str(row["path"]).strip()] = int(row["label"])

    signal_xsec = _load_signal_cross_sections(args.xsection_signal_json)
    bkg_xsec = _load_background_cross_sections(args.xsection_json)

    sig_patterns = tuple(args.signal_pattern) if args.signal_pattern else ()

    files = _discover_files(args.input)
    log.info("Found %d input file(s)", len(files))

    bkg_parts: list[dict] = []
    sig_parts: list[dict] = []
    for idx, path in enumerate(files):
        cls = _classify(path, label_map, sig_patterns, args.signal_prefix)
        if cls is None:
            continue
        is_sig, is_data = cls
        if is_data:
            continue
        if is_sig:
            cap, fraction, rng_seed = args.max_events_signal, None, None
        else:
            cap, fraction, rng_seed = None, args.bkg_fraction, (seed, idx)
        item = _load_file(
            path,
            features,
            args.weight_branch,
            cap,
            fraction,
            rng_seed,
            is_sig,
        )
        if item is None:
            continue
        if item["sig_flag"]:
            sig_parts.append(item)
        else:
            bkg_parts.append(item)

    if not bkg_parts:
        log.error("No background files loaded")
        return 1
    if not sig_parts:
        log.error(
            "No signal files loaded — expected file names starting with %s "
            "(override with --signal-prefix / --signal-pattern)",
            DEFAULT_SIGNAL_SAMPLE,
        )
        return 1

    # Assemble shared background once.
    X_bkg = pd.concat([p["df"] for p in bkg_parts], ignore_index=True).astype("float32")
    # The per-file feature frames are no longer needed once concatenated; drop
    # them (keeping only the small per-file metadata) so the background does
    # not sit in memory twice.
    for p in bkg_parts:
        p["n"] = int(len(p["df"]))
        p["df"] = None
    w_bkg_list: list[np.ndarray] = []
    sid_bkg_list: list[np.ndarray] = []
    for p in bkg_parts:
        w = p["w"].copy()
        xsec = _find_xsec(p["sample"], bkg_xsec)
        if xsec is not None and p["wte"] > 0:
            w = w * ((lumi * xsec * 1000.0) / p["wte"])
        w_bkg_list.append(w)
        sid_bkg_list.append(np.full(len(w), p["path"], dtype=object))
    w_bkg = np.concatenate(w_bkg_list)
    sid_bkg = np.concatenate(sid_bkg_list)
    log.info("Background: %d events", len(w_bkg))

    # Assemble signal events with their per-event mass-point flags.
    X_sig = pd.concat([p["df"] for p in sig_parts], ignore_index=True).astype("float32")
    for p in sig_parts:
        p["n"] = int(len(p["df"]))
        p["df"] = None
    w_sig_raw = np.concatenate([p["w"] for p in sig_parts])
    sid_sig = np.concatenate([
        np.full(len(p["w"]), p["path"], dtype=object) for p in sig_parts
    ])
    # Precompute per-event signal weights scaled by each event's mass-point
    # cross section and its file's weighted_total_events (wte).
    sig_scale = np.ones(len(X_sig), dtype="f8")
    offset = 0
    for p in sig_parts:
        n = p["n"]
        wte = p["wte"]
        idx = np.arange(offset, offset + n)
        for label, flag in p["flags"].items():
            xsec = signal_xsec.get(label)
            if xsec is None:
                continue
            # per mass point: a multi-point file's wte covers every point
            if not flag[:n].any():
                continue   # flag present for the grid, no events in this file
            if "GenModel_" + label not in p["wte_gm"]:
                raise ValueError(f"{p['path']}: no weighted_total_events_GenModel_{label}; "
                                 "add it with scripts/tools/add_genmodel_counts.py")
            mp_wte = p["wte_gm"]["GenModel_" + label]
            if mp_wte <= 0:
                continue   # no generated events for this point: its flag selects nothing
            s = (lumi * xsec * 1000.0) / mp_wte
            m = flag[:n]
            sig_scale[idx[m]] = s
        offset += n
    w_sig_scaled = w_sig_raw * sig_scale

    all_labels: list[str] = []
    for p in sig_parts:
        for label in p["flags"]:
            if label not in all_labels:
                all_labels.append(label)
    all_labels.sort()

    if args.masspoints:
        requested = [s.strip() for s in args.masspoints.split(",") if s.strip()]
        labels = [lab for lab in all_labels if lab in requested]
        missing = [r for r in requested if r not in all_labels]
        if missing:
            log.warning("Requested mass points not found: %s", missing)
    else:
        labels = all_labels

    if not labels:
        log.error("No mass points to train")
        return 1

    # Per-masspoint event mask aligned with X_sig / w_sig_raw / sid_sig.
    signal_masks: dict[str, np.ndarray] = {}
    offset = 0
    for p in sig_parts:
        n = p["n"]
        for label in labels:
            if label not in p["flags"]:
                continue
            mask = signal_masks.setdefault(label, np.zeros(len(X_sig), dtype=bool))
            mask[offset:offset + n] |= p["flags"][label]
        offset += n

    log.info("Will train %d mass point(s): %s", len(labels), labels)

    # Optional: merge the mass points into bins of M_H3/M_H4 and train one
    # model per bin.  The per-event weights are untouched by this — every
    # signal event keeps the scaling of its own mass point (computed above) —
    # only which events form the signal class changes.
    if args.ratio_bins:
        edges = None
        if args.ratio_bin_edges:
            edges = [float(x) for x in args.ratio_bin_edges.split(",") if x.strip()]
        groups = _make_ratio_bins(labels, args.ratio_bins, edges)
        merged: dict[str, np.ndarray] = {}
        for gname, members in groups.items():
            arr = None
            for m in members:
                mm = signal_masks.get(m)
                if mm is None:
                    continue
                arr = mm.copy() if arr is None else (arr | mm)
            if arr is None:
                raise SystemExit(f"ratio bin {gname!r} has no signal events")
            merged[gname] = arr

        log.info("Grouped %d mass point(s) into %d M_H3/M_H4 ratio bin(s):",
                 len(labels), len(groups))
        composition: dict[str, dict] = {}
        for gname, members in groups.items():
            ratios = sorted(_mass_ratio(m) for m in members)
            n_ev = int(merged[gname].sum())
            log.info("  %-24s M_H3/M_H4 %.3f-%.3f  %2d point(s)  %8d signal events",
                     gname, ratios[0], ratios[-1], len(members), n_ev)
            composition[gname] = {
                "ratio_min": ratios[0], "ratio_max": ratios[-1],
                "n_mass_points": len(members), "n_signal_events": n_ev,
                "mass_points": members,
            }
        signal_masks = merged
        labels = list(groups.keys())
        out_base = Path(args.outdir_base)
        out_base.mkdir(parents=True, exist_ok=True)
        (out_base / "ratio_bins.json").write_text(
            json.dumps({"n_bins": args.ratio_bins, "bins": composition},
                       indent=2) + "\n"
        )

    for label in labels:
        sel = signal_masks.get(label)
        n_sig = int(sel.sum())
        if n_sig == 0:
            log.warning("Mass point %s has no signal events — skipping", label)
            continue

        # Drop the previous mass point's combined dataset before building the
        # next one, so two full copies never coexist.
        X = y = w = sid = trainer = None
        gc.collect()

        X = pd.concat([X_sig[sel].reset_index(drop=True), X_bkg], ignore_index=True)
        y = np.concatenate([np.ones(n_sig, dtype="i4"), np.zeros(len(w_bkg), dtype="i4")])
        w = np.concatenate([w_sig_scaled[sel], w_bkg])
        sid = np.concatenate([sid_sig[sel], sid_bkg])

        outdir = str(Path(args.outdir_base) / label.lower())
        plotdir = str(Path(args.plot_dir_base) / label.lower())
        if (Path(outdir) / "dnn_model.pt").exists():
            log.info("Skipping %s (already trained): %s", label, outdir)
            continue
        log.info("Training %s: signal=%d background=%d -> %s", label, n_sig, len(w_bkg), outdir)

        trainer = DNNTrainer(args.dnn_config)
        trainer.train_from_arrays(
            X,
            y,
            w,
            sample_ids=sid,
            outdir=outdir,
            plot_dir=plotdir,
        )

    log.info("Done. Trained %d mass point model(s).", len(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
