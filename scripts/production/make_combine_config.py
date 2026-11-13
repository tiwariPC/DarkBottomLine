#!/usr/bin/env python
"""Write a Combine config for a production directory from configs/combine.yaml.

  python scripts/production/make_combine_config.py --base BASE --out-base BASE/limits/combine --config-out BASE/limits/configs/run3.yaml \
      [--years 2024 2025 2026] [--points MH3_600_MH4_150_Mchi_1 ...] [--shape-cleaning]

BASE must contain <year>/region_incl/root (symlink to the region_lumi*_incl output) and 2024/region_lumi*_incl/config_used/regions_used.yaml.
--shape-cleaning enables the low-statistics shape-variation cleaning (datacard.shape_cleaning; needed for the --one-mc-for-all
production where 2026 has only ~10% of the MC); thresholds are in the config (stat_sigma, max_rel_shift, min_rel_err)."""
import argparse, copy, glob, json
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--base", required=True); ap.add_argument("--out-base", required=True); ap.add_argument("--config-out", required=True)
ap.add_argument("--years", nargs="+", default=["2024", "2025", "2026"])
ap.add_argument("--points", nargs="+", default=None, help="signal points (default: the whole grid of xsection_signal.json)")
ap.add_argument("--exclude-points", nargs="*", default=["MH3_1500_MH4_1450_Mchi_1"])
ap.add_argument("--shape-cleaning", action="store_true")
ap.add_argument("--stat-sigma", type=float, default=1.0); ap.add_argument("--max-rel-shift", type=float, default=0.5)
ap.add_argument("--min-rel-err", type=float, default=0.5)
a = ap.parse_args()

c = copy.deepcopy(yaml.safe_load(open(REPO / "configs/combine.yaml")))
c["eras"] = [{"year": int(y), "year_config": f"configs/{y}.yaml", "active": True} for y in a.years]
xs = json.load(open(REPO / "data/cross-section/xsection_signal.json"))["2HDMa"]
c["signal_grid"]["points"] = a.points or [k for k in xs if not k.startswith("_") and k not in a.exclude_points]
c["regions_config"] = glob.glob(f"{a.base}/{a.years[0]}/region_lumi*_incl/config_used/regions_used.yaml")[0]
c["input"]["region_root_dir_template"] = a.base + "/{year}/region_incl/root"
c["output"]["base_dir"] = a.out_base
if a.shape_cleaning:
    c["datacard"]["shape_cleaning"] = {"enabled": True, "stat_sigma": a.stat_sigma, "max_rel_shift": a.max_rel_shift, "min_rel_err": a.min_rel_err}
Path(a.config_out).parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(c, open(a.config_out, "w"), sort_keys=False)
print("wrote", a.config_out, "-", len(c["signal_grid"]["points"]), "points,", len(c["eras"]), "eras")
