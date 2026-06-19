#!/usr/bin/env python3
"""Run analyzer + feature plots + DNN training from EOS EVENTSELECTION outputs.

Workflow:
1) Auto-select latest date folder under EOS outputs that has a pure-year directory.
2) Optionally require both new*/run* subdirectories in the selected year folder.
3) Run analyzer to produce ppbbchichi trees.
4) Plot feature comparisons and train DNN with strict run* exclusion.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from dnn.data import find_latest_year_dir, summarize_year_subdirs
except ModuleNotFoundError:
    from data import find_latest_year_dir, summarize_year_subdirs


DEFAULT_BASE = "/eos/user/x/xdu/dbl_praveen/DarkBottomLine/outputs"


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_")


def _resolve_signal_prefix(signal_prefix: str | None, signal_category: str | None) -> str | None:
    if signal_prefix:
        return str(signal_prefix).strip().lower()
    if signal_category:
        cat = str(signal_category).strip().lower()
        if not cat:
            return None
        return cat if cat.startswith("new") else f"new{cat}"
    return None


def _pick_explicit(
    base_dir: Path,
    date_dir: str,
    target_year: str | None,
    require_new_run: bool,
    signal_prefix: str | None,
) -> dict:
    ddir = (base_dir / date_dir).resolve()
    if not ddir.exists() or not ddir.is_dir():
        raise FileNotFoundError(f"Date directory not found: {ddir}")

    year_dirs = sorted([x for x in ddir.iterdir() if x.is_dir() and re.fullmatch(r"\d{4}", x.name)])
    if target_year is not None:
        year_dirs = [x for x in year_dirs if x.name == target_year]
    if not year_dirs:
        raise FileNotFoundError(f"No pure-year folder found under: {ddir}")

    selected = year_dirs[-1]
    summary = summarize_year_subdirs(selected)
    if require_new_run and (len(summary["new"]) == 0 or len(summary["run"]) == 0):
        raise FileNotFoundError(
            f"Explicit selection {selected} does not contain both new*/run* subdirectories"
        )
    if signal_prefix and not any(s.lower().startswith(signal_prefix) for s in summary["all"]):
        raise FileNotFoundError(
            f"Explicit selection {selected} has no sample matching signal prefix: {signal_prefix}"
        )

    return {
        "date": date_dir,
        "date_dir": str(ddir),
        "year": selected.name,
        "year_dir": str(selected),
        "summary": summary,
    }


def _run(cmd: list[str], dry_run: bool = False) -> None:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto workflow for EOS EVENTSELECTION -> analyzer -> DNN")
    ap.add_argument("--base-dir", default=DEFAULT_BASE, help="EOS outputs root")
    ap.add_argument("--year", default="2022", help="Target year folder name, e.g. 2022")
    ap.add_argument("--date-dir", default=None, help="Optional explicit date folder under base-dir (e.g. 4-2)")
    ap.add_argument(
        "--allow-missing-run",
        action="store_true",
        help="Allow selected year folder without run* subdirs (default: require both new* and run*).",
    )
    ap.add_argument("--signal-prefix", default=None, help="Signal sample prefix, e.g. newdiboson")
    ap.add_argument(
        "--signal-category",
        default="diboson",
        help="Signal category shorthand; diboson -> newdiboson",
    )
    ap.add_argument("--exclude-prefixes", default="run", help="Comma-separated sample prefixes excluded from ML")

    ap.add_argument("--era", default="All", help="Analyzer era argument")
    ap.add_argument("--region", default="preselection", help="Region for plotting/training")
    ap.add_argument("--tree", default=None, help="Optional input ROOT tree name for analyzer")
    ap.add_argument("--pid-chi", type=int, default=52, help="Analyzer --pid-chi value")
    ap.add_argument("--with-hists", action="store_true", help="Enable histogram output in analyzer")
    ap.add_argument("--force-analyzer", action="store_true", help="Force rerun analyzer even if tree exists")

    ap.add_argument("--features", default=None, help="Optional comma-separated feature list")
    ap.add_argument("--max-events-per-sample", type=int, default=200000)
    ap.add_argument("--test-size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden", default="128,128")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-clip", type=float, default=100.0)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--top-k-significance", type=int, default=5)
    ap.add_argument("--single-feature-epochs", type=int, default=20)
    ap.add_argument("--topology-decorrelation-weight", type=float, default=0.0)
    ap.add_argument("--topology-decorrelation-features", default="M_Jet1Jet2,dRJet12")
    ap.add_argument("--topology-decorrelation-min-signal", type=int, default=16)
    ap.add_argument("--drop-constant-features", action="store_true")
    ap.add_argument(
        "--balance-classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reweight signal/background to contribute equally to the training loss.",
    )
    ap.add_argument(
        "--balance-strength",
        type=float,
        default=1.0,
        help="Class-balance strength in [0,1], passed through to the trainer.",
    )

    ap.add_argument("--tag", default=None, help="Analyzer output tag (default auto)")
    ap.add_argument("--outdir-root", default="outputs_dnn", help="Prefix for training output directory")
    ap.add_argument("--plot-root", default="plot", help="Root plot directory")
    ap.add_argument("--clean-output", action="store_true", help="Remove existing output dirs for this tag")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    analyzer_script = repo_root / "ppbbchichi_analyzer_par.py"
    plot_script = repo_root / "dnn" / "plot_feature_comparison.py"
    train_script = repo_root / "dnn" / "train_classifier.py"

    signal_prefix = _resolve_signal_prefix(args.signal_prefix, args.signal_category)
    if not signal_prefix:
        raise ValueError("Please provide --signal-prefix or --signal-category")

    require_new_run = not bool(args.allow_missing_run)
    base_dir = Path(args.base_dir).resolve()

    if args.date_dir:
        picked = _pick_explicit(
            base_dir,
            args.date_dir,
            str(args.year),
            require_new_run=require_new_run,
            signal_prefix=signal_prefix,
        )
    else:
        picked = find_latest_year_dir(
            base_dir,
            target_year=str(args.year),
            require_new_and_run=require_new_run,
            required_signal_prefix=signal_prefix,
        )

    summary = picked["summary"]
    if len(summary["new"]) == 0:
        raise RuntimeError("Selected year folder has no new* sample subdirectories")

    date_name = Path(picked["date_dir"]).name
    auto_tag = f"eos_{_slug(date_name)}_{picked['year']}_{_slug(signal_prefix)}"
    tag = args.tag if args.tag else auto_tag

    merged_dir = repo_root / "outputfiles" / "merged" / tag
    tree_root = merged_dir / f"{picked['year']}_ppbbchichi-trees.root"

    train_outdir = repo_root / f"{args.outdir_root}_{tag}"
    plot_dir = repo_root / args.plot_root / tag
    train_plot_dir = plot_dir / "training"

    if args.clean_output:
        for p in [merged_dir, train_outdir, plot_dir]:
            if p.exists():
                print(f"[CLEAN] Removing {p}")
                if not args.dry_run:
                    shutil.rmtree(p)

    print("[INFO] Selected dataset:")
    print(json.dumps(picked, indent=2))

    if args.force_analyzer or not tree_root.exists():
        cmd = [
            sys.executable,
            str(analyzer_script),
            "--year",
            str(picked["year"]),
            "--era",
            str(args.era),
            "-i",
            str(picked["year_dir"]),
            "--tag",
            str(tag),
            "--output-prefix",
            str(picked["year"]),
            "--pid-chi",
            str(args.pid_chi),
            "--skip-bad-files",
        ]
        if args.tree:
            cmd.extend(["--tree", str(args.tree)])
        if not args.with_hists:
            cmd.append("--skip-hists")
        _run(cmd, dry_run=args.dry_run)
    else:
        print(f"[INFO] Reusing existing analyzer tree: {tree_root}")

    if not args.dry_run and not tree_root.exists():
        raise FileNotFoundError(f"Analyzer output tree not found: {tree_root}")

    plot_cmd = [
        sys.executable,
        str(plot_script),
        "--root",
        str(tree_root),
        "--region",
        str(args.region),
        "--signal-prefix",
        str(signal_prefix),
        "--exclude-prefixes",
        str(args.exclude_prefixes),
        "--max-events-per-sample",
        str(args.max_events_per_sample),
        "--outdir",
        str(plot_dir),
    ]
    if args.features:
        plot_cmd.extend(["--features", str(args.features)])
    _run(plot_cmd, dry_run=args.dry_run)

    train_cmd = [
        sys.executable,
        str(train_script),
        "--root",
        str(tree_root),
        "--region",
        str(args.region),
        "--signal-prefix",
        str(signal_prefix),
        "--exclude-prefixes",
        str(args.exclude_prefixes),
        "--max-events-per-sample",
        str(args.max_events_per_sample),
        "--test-size",
        str(args.test_size),
        "--seed",
        str(args.seed),
        "--outdir",
        str(train_outdir),
        "--plot-dir",
        str(train_plot_dir),
        "--hidden",
        str(args.hidden),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-clip",
        str(args.weight_clip),
        "--patience",
        str(args.patience),
        "--device",
        str(args.device),
        "--top-k-significance",
        str(args.top_k_significance),
        "--single-feature-epochs",
        str(args.single_feature_epochs),
        "--topology-decorrelation-weight",
        str(args.topology_decorrelation_weight),
        "--topology-decorrelation-features",
        str(args.topology_decorrelation_features),
        "--topology-decorrelation-min-signal",
        str(args.topology_decorrelation_min_signal),
    ]
    if args.balance_classes:
        train_cmd.append("--balance-classes")
    else:
        train_cmd.append("--no-balance-classes")
    train_cmd.extend(["--balance-strength", str(args.balance_strength)])
    if args.features:
        train_cmd.extend(["--features", str(args.features)])
    if args.drop_constant_features:
        train_cmd.append("--drop-constant-features")
    _run(train_cmd, dry_run=args.dry_run)

    print("[OK] Workflow finished")
    print(f"[OK] Analyzer tree: {tree_root}")
    print(f"[OK] Feature plots/significance: {plot_dir}")
    print(f"[OK] Training outputs (incl. roc): {train_outdir}")
    print(f"[OK] Loss/AUC curves: {train_plot_dir}")


if __name__ == "__main__":
    main()
