#!/usr/bin/env python3
"""Train one independent binary DNN per signal mass point.

The BBDM signal EVENTSELECTION ROOT files contain 29 boolean
``GenModel_MH3_<a>_MH4_<b>_Mchi_<c>`` branches.  This script treats each such
branch as a separate signal class: for every mass point it keeps only the
signal events whose branch is true, combines them with the (shared) background
events, and trains a normal non-parametric binary classifier.

It deliberately does NOT use the parametric ``(MH3, MH4)`` input path: the goal
is 29 independent models, not one model conditioned on mass.

Example
-------
python scripts/train_per_masspoint.py \
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
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import uproot
import yaml

from dnn.common import sanitize_feature_frame
from dnn.data import read_branch_as_array, read_tree_branches_as_arrays
from dnn.feature_engineering import build_feature_frame_from_tree
from dnn.make_trees import _is_data, _is_signal_heuristic, _sample_name
from darkbottomline.dnn_trainer import DNNTrainer
from darkbottomline.plotting import PlotManager, _find_xsec


log = logging.getLogger("train_per_masspoint")


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
    label_map: dict[str, int] | None,
    signal_patterns: tuple[str, ...],
    signal_prefix: str | None,
) -> dict | None:
    """Read one EVENTSELECTION file into a feature frame + weight + mass flags."""
    sample = _sample_name(path)

    if label_map is not None:
        key = path if path in label_map else Path(path).name
        if key not in label_map:
            log.warning("Path %s not in label CSV — skipping", path)
            return None
        sig_flag = bool(label_map[key] == 1)
        data_flag = False
    else:
        data_flag = _is_data(path)
        sig_flag = False if data_flag else _is_signal_heuristic(
            path, signal_patterns, signal_prefix
        )

    if data_flag:
        return None

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
        df = sanitize_feature_frame(df)
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

    return {
        "path": path,
        "sample": sample,
        "sig_flag": sig_flag,
        "df": df,
        "w": w,
        "flags": flags,
        "wte": wte,
    }


def _classify(
    path: str,
    label_map: dict[str, int] | None,
    signal_patterns: tuple[str, ...],
    signal_prefix: str | None,
) -> tuple[bool, bool] | None:
    """Return (is_signal, is_data), or None if the path should be skipped."""
    if label_map is not None:
        key = path if path in label_map else Path(path).name
        if key not in label_map:
            return None
        return bool(label_map[key] == 1), False
    if _is_data(path):
        return False, True
    return _is_signal_heuristic(path, signal_patterns, signal_prefix), False


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
    ap.add_argument("--signal-pattern", action="append", default=None,
                    dest="signal_pattern")
    ap.add_argument("--signal-prefix", default=None)
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
        if cls is None or cls[1]:
            continue
        is_sig = cls[0]
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
            label_map,
            sig_patterns,
            args.signal_prefix,
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
        log.error("No signal files loaded")
        return 1

    # Assemble shared background once.
    X_bkg = pd.concat([p["df"] for p in bkg_parts], ignore_index=True).astype("float32")
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
    w_sig_raw = np.concatenate([p["w"] for p in sig_parts])
    sid_sig = np.concatenate([
        np.full(len(p["w"]), p["path"], dtype=object) for p in sig_parts
    ])
    # Precompute per-event signal weights scaled by each event's mass-point
    # cross section and its file's weighted_total_events (wte).
    sig_scale = np.ones(len(X_sig), dtype="f8")
    offset = 0
    for p in sig_parts:
        n = len(p["df"])
        wte = p["wte"]
        idx = np.arange(offset, offset + n)
        for label, flag in p["flags"].items():
            xsec = signal_xsec.get(label)
            if xsec is None:
                continue
            s = (lumi * xsec * 1000.0) / wte if wte > 0 else 1.0
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
        n = len(p["df"])
        for label in labels:
            if label not in p["flags"]:
                continue
            mask = signal_masks.setdefault(label, np.zeros(len(X_sig), dtype=bool))
            mask[offset:offset + n] |= p["flags"][label]
        offset += n

    log.info("Will train %d mass point(s): %s", len(labels), labels)

    for label in labels:
        sel = signal_masks.get(label)
        n_sig = int(sel.sum())
        if n_sig == 0:
            log.warning("Mass point %s has no signal events — skipping", label)
            continue

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
