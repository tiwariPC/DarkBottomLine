#!/usr/bin/env python3
"""Apply a trained classifier to ppbbchichi trees and write scores back to ROOT.

Purpose:
- Load `features.json` and a trained model produced by the training script.
- Iterate the selected region trees in `ppbbchichi-trees.root`.
- Compute per-event classifier score and write it as a new branch (default: `ml_score`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

try:
    # When executed as a module: `python -m dnn.apply_classifier ...`
    from dnn.model import load_checkpoint
    from dnn.scaler import StandardScaler
    from dnn.data import list_sample_region_trees
except ModuleNotFoundError:
    # When executed as a script: `python dnn/apply_classifier.py ...`
    from model import load_checkpoint
    from scaler import StandardScaler
    from data import list_sample_region_trees


def load_features(features_path: str) -> list[str]:
    return list(json.loads(Path(features_path).read_text()))

def load_scaler(path: str) -> StandardScaler:
    return StandardScaler.from_jsonable(json.loads(Path(path).read_text()))


def ensure_dir(upfile, treedir: str):
    curr = upfile
    for part in [p for p in treedir.split("/") if p]:
        curr = curr.mkdir(part) if part not in curr.keys() else curr[part]
    return curr


def main():
    ap = argparse.ArgumentParser(description="Apply a trained ML model and write ml_score into a new ROOT file.")
    ap.add_argument("--root", required=True, help="Input ppbbchichi-trees.root")
    ap.add_argument("--region", default="preselection", help="Which region tree to score")
    ap.add_argument(
        "--features",
        default="outputs_dnn/features.json",
        help="Path to features.json produced by training",
    )
    ap.add_argument("--model", default="outputs_dnn/dnn_model.pt", help="Path to trained DNN checkpoint")
    ap.add_argument("--scaler", default="outputs_dnn/scaler.json", help="Path to scaler.json produced by training")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out", default=None, help="Output ROOT path (default: <input> with -scored suffix)")
    ap.add_argument("--score-branch", default="ml_score", help="Name of the score branch to add")
    ap.add_argument("--step", type=int, default=200000, help="Chunk size")
    args = ap.parse_args()

    features = load_features(args.features)
    scaler = load_scaler(args.scaler)

    import torch

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _spec = load_checkpoint(args.model, map_location=str(device))
    model = model.to(device)
    model.eval()

    in_path = Path(args.root)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "-scored" + in_path.suffix)

    region = args.region
    score_branch = args.score_branch

    with uproot.open(str(in_path)) as fin:
        trees = list_sample_region_trees(fin, region)
        if not trees:
            raise FileNotFoundError(f"No trees found for region '{region}' in {in_path}")

        with uproot.recreate(str(out_path)) as fout:
            for sample, tpath in trees:
                tree = fin[tpath]
                all_branches = list(map(str, tree.keys()))

                required = list(features)
                missing = [b for b in required if b not in set(all_branches)]
                if missing:
                    raise KeyError(f"Tree '{tpath}' missing required feature branches: {missing}")

                # We'll copy ALL branches and add score branch.
                copy_branches = list(all_branches)

                # Create output tree schema from first chunk
                first = True
                out_tree = None

                for arrays in tree.iterate(copy_branches, step_size=int(args.step), library="np"):
                    n = len(next(iter(arrays.values()))) if arrays else 0
                    if n == 0:
                        continue

                    X = np.column_stack([np.asarray(arrays[f], dtype="f8") for f in features])
                    X = np.where(np.isfinite(X), X, -9999.0)

                    Xs = scaler.transform(X).astype("float32")
                    with torch.no_grad():
                        xb = torch.from_numpy(Xs).to(device)
                        logits = model(xb).squeeze(1)
                        score = torch.sigmoid(logits).cpu().numpy().astype("float32")

                    out_arrays = {k: arrays[k] for k in copy_branches}
                    out_arrays[score_branch] = score

                    # ensure directory and tree
                    if first:
                        d = ensure_dir(fout, sample)
                        # infer branch types from numpy dtypes
                        types = {}
                        for k, v in out_arrays.items():
                            a = np.asarray(v)
                            if a.dtype == np.bool_:
                                types[k] = "int8"
                                out_arrays[k] = a.astype("int8")
                            elif a.dtype.kind in ("i", "u"):
                                types[k] = "int64"
                                out_arrays[k] = a.astype("int64")
                            elif a.dtype.kind == "f":
                                types[k] = "float64"
                                out_arrays[k] = a.astype("float64")
                            else:
                                types[k] = "float64"
                                out_arrays[k] = a.astype("float64")
                        out_tree = d.mktree(region, types)
                        first = False

                    # Cast consistently each chunk
                    for k, v in list(out_arrays.items()):
                        a = np.asarray(v)
                        if a.dtype == np.bool_:
                            out_arrays[k] = a.astype("int8")
                        elif a.dtype.kind in ("i", "u"):
                            out_arrays[k] = a.astype("int64")
                        elif a.dtype.kind == "f":
                            out_arrays[k] = a.astype("float64")
                        else:
                            out_arrays[k] = a.astype("float64")

                    assert out_tree is not None
                    out_tree.extend(out_arrays)

    print(f"[OK] Wrote scored ROOT: {out_path}")
    print(f"[OK] Added branch '{score_branch}' in trees '<sample>/{region}'")


if __name__ == "__main__":
    main()
