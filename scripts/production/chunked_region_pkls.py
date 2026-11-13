#!/usr/bin/env python3
"""Memory-bounded stand-in for the per-sample half of
`darkbottomline analyze --mode region-analysis`.

Same per-file logic as cli._run_analyzer_from_eventselection (xsec lookup,
weighted_total_events, DNN scoring, analyzer.process_from_eventselection),
but reads Events in chunks and combines the chunk results, and does not
build merged.pkl. Writes <out>/per_sample/<stem>.pkl for --region-results
replay. Plotting reads only region_histograms (raw per-event arrays,
concatenated across chunks), region_cutflow_steps (summed) and
metadata.weighted_total_events (the file's value, set once).
"""
import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _sum_numeric(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _sum_numeric(dst.setdefault(k, {}), v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            dst[k] = dst.get(k, 0.0) + float(v)


def process_file(file_path, a):
    sys.path.insert(0, a.repo)
    import uproot
    from darkbottomline.analyzer import DarkBottomLineAnalyzer
    from darkbottomline.cli import load_config, _resolve_mass_scan
    from darkbottomline.plotting import PlotManager, _find_xsec

    config = load_config(a.config)
    is_data = config.get("data", {}).get("is_data", False)
    with open(a.xsection_json) as f:
        cross_sections = PlotManager._normalize_cross_sections(json.load(f))
    dnn_inference = dnn_mass_scan = None
    if a.dnn_model:
        from darkbottomline.dnn_inference import DNNInference
        dnn_inference = DNNInference(a.dnn_model, config_path=a.dnn_config)
        dnn_mass_scan = _resolve_mass_scan(None, dnn_inference)
    analyzer = DarkBottomLineAnalyzer(config, a.regions_config)

    stem = Path(file_path).stem
    xsec = _find_xsec(stem, cross_sections)
    if xsec is None:
        xsec = _find_xsec(stem.replace("_EVENTSELECTION", ""), cross_sections)

    t0 = time.time()
    with uproot.open(file_path) as f:
        wte = 0.0
        for key in ("weighted_total_events", "weighted_total_events;1"):
            if key in f:
                try:
                    wte = float(f[key].values()[0])
                    break
                except Exception:
                    pass
        from darkbottomline.cli import _read_weighted_total_events_per_genmodel
        wte_gm = _read_weighted_total_events_per_genmodel(f)
        if "Events" not in f:
            return stem, "no Events tree", 0, 0.0
        tree = f["Events"]
        n_total = tree.num_entries
        effective_wte = wte if wte > 0 else 1.0

        out = None
        hist_parts = {}  # region -> {"weight": [arrs], "values": {field: [arrs]}}
        for chunk in tree.iterate(library="np", step_size=a.step_size):
            res = analyzer.process_from_eventselection(
                branches=dict(chunk), weighted_total_events=effective_wte, is_data=is_data,
                dnn_inference=dnn_inference, dnn_mass_scan=dnn_mass_scan,
            )
            rh = res.get("region_histograms", {})
            if out is None:
                # analyzer.accumulator is reused across calls: copy everything
                # except the big per-event arrays, which are collected below.
                out = {k: copy.deepcopy(v) for k, v in res.items() if k != "region_histograms"}
                out["region_cutflow_steps"] = {}
                out["region_cutflow"] = {}
            else:
                out["metadata"]["n_events_processed"] += res["metadata"]["n_events_processed"]
            _sum_numeric(out["region_cutflow_steps"], res.get("region_cutflow_steps", {}))
            _sum_numeric(out["region_cutflow"], res.get("region_cutflow", {}))
            for region, payload in rh.items():
                d = hist_parts.setdefault(region, {"weight": [], "values": {}})
                if payload.get("weight") is not None:
                    d["weight"].append(np.asarray(payload["weight"]))
                for field, arr in payload.get("values", {}).items():
                    d["values"].setdefault(field, []).append(np.asarray(arr))

    if out is None:
        return stem, "empty", 0, time.time() - t0
    out["region_histograms"] = {
        r: {"weight": np.concatenate(d["weight"]) if d["weight"] else np.zeros(0),
            "values": {k: np.concatenate(v) for k, v in d["values"].items()}}
        for r, d in hist_parts.items()
    }
    out["metadata"]["weighted_total_events"] = float(effective_wte)
    if wte_gm:
        out["metadata"]["weighted_total_events_per_genmodel"] = wte_gm
    out["metadata"]["xsec"] = xsec
    out["metadata"]["sample"] = stem

    pkl_dir = os.path.join(a.out, "per_sample")
    os.makedirs(pkl_dir, exist_ok=True)
    tmp = os.path.join(pkl_dir, stem + ".pkl.tmp")
    with open(tmp, "wb") as pf:
        pickle.dump(out, pf)
    os.replace(tmp, os.path.join(pkl_dir, stem + ".pkl"))
    return stem, "ok", n_total, time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--regions-config", required=True)
    p.add_argument("--xsection-json", required=True)
    p.add_argument("--dnn-model")
    p.add_argument("--dnn-config")
    p.add_argument("--repo", required=True)
    p.add_argument("--step-size", type=int, default=300_000)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()

    files = sorted(str(f) for f in Path(a.input).iterdir() if f.suffix == ".root")
    pkl_dir = os.path.join(a.out, "per_sample")
    done = ({f.removesuffix(".pkl") for f in os.listdir(pkl_dir) if f.endswith(".pkl")}
            if os.path.isdir(pkl_dir) else set())
    todo = [f for f in files if Path(f).stem not in done]
    # Largest first so the long ones don't straggle at the end.
    todo.sort(key=lambda f: -os.path.getsize(f))
    logging.info("%d files, %d already done, %d to process", len(files), len(files) - len(todo), len(todo))

    failed = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(process_file, f, a): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                stem, status, n, secs = fut.result()
                logging.info("[%d/%d] %s: %s (%d events, %.0fs)", i, len(todo), stem, status, n, secs)
                if status != "ok":
                    failed.append(stem)
            except Exception as e:
                logging.exception("[%d/%d] FAILED %s: %s", i, len(todo), futs[fut], e)
                failed.append(Path(futs[fut]).stem)
    logging.info("PKLS_DONE failed=%d %s", len(failed), failed)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
