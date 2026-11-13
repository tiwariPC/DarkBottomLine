#!/usr/bin/env python3
"""Build a process-group -> dataset -> xsec table through the REAL code path.

Uses darkbottomline.plotting._clean_sample_name / _find_xsec and the plotting.yaml
process-group patterns, so the table proves the cross-year mapping actually works.

For every dataset in every year's samplelist we:
  1. canonicalize the dataset name (_clean_sample_name),
  2. find which plotting.yaml process group its pattern matches (same substring-on-
     canonical logic as _resolve_group_files),
  3. look up the cross-section (_find_xsec against the normalized xsec JSON).
"""
import sys, json, re, yaml
from pathlib import Path

REPO = Path("/Users/ptiwari/Development/bbdmRun3/DarkBottomLine")
sys.path.insert(0, str(REPO))
from darkbottomline.plotting import _clean_sample_name, _find_xsec, PlotManager

# --- load groups + xsec (same objects the plotter uses) ---
pcfg = yaml.safe_load((REPO / "configs/plotting.yaml").read_text())
groups = pcfg["process_groups"]
raw_xsec = json.loads((REPO / "data/cross-section/xsection_background.json").read_text())
flat_xsec = PlotManager._normalize_cross_sections(raw_xsec)


def group_for(canon_name: str):
    """Return (group_label, matched_pattern) using the same canonical-substring
    rule as _resolve_group_files. First group/pattern that matches wins.
    Includes data + signal groups so data files are classified too."""
    for label, spec in groups.items():
        for pat in spec.get("patterns", []):
            if _clean_sample_name(pat) in canon_name:
                return label, pat
    return None, None


def dataset_dir_names(year: str):
    """Extract unique dataset-directory names from a year's samplelist .txt files
    (the middle /store/mc/<campaign>/<DATASET>/NANOAODSIM/... segment)."""
    d = REPO / "data/samplelist" / year
    if not d.exists():
        return []
    names = set()
    pat = re.compile(r"/store/mc/[^/]+/([^/]+)/NANOAODSIM")
    for f in sorted(d.glob("*.txt")):
        # Data files carry no /store/mc/ MC dataset; their EVENTSELECTION stem is
        # the samplelist filename itself (e.g. JetMET0-Run2024C-...). Use that.
        is_mc = False
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = pat.search(line)
            if m:
                names.add(m.group(1))
                is_mc = True
        if not is_mc:
            names.add(f.stem)   # data: the .txt basename is the dataset stem
    return sorted(names)


YEARS = ["2022", "2022EE", "2023", "2024"]

rows = []  # (group, year, dataset, canonical, xsec)
for year in YEARS:
    for ds in dataset_dir_names(year):
        canon = _clean_sample_name(ds)
        label, pat = group_for(canon)
        xsec = _find_xsec(ds, flat_xsec)
        rows.append((label or "UNMATCHED", year, ds, canon, xsec))

# --- print grouped table ---
def fmt_xsec(x):
    return "None" if x is None else f"{x:g}"

by_group = {}
for r in rows:
    by_group.setdefault(r[0], []).append(r)

order = list(groups.keys()) + ["UNMATCHED"]
print(f"{'GROUP':14} {'YEAR':7} {'XSEC(pb)':>10}  DATASET  ->  CANONICAL")
print("=" * 120)
for g in order:
    if g not in by_group:
        continue
    for _, year, ds, canon, xsec in sorted(by_group[g], key=lambda r: (r[2], r[1])):
        flag = "  <-- NO XSEC" if (xsec is None and g not in ("MET_Data", "EGamma_Data", "Signal_2HDMa")) else ""
        print(f"{g:14} {year:7} {fmt_xsec(xsec):>10}  {ds}")
        print(f"{'':14} {'':7} {'':>10}     -> {canon}{flag}")
    print("-" * 120)

# --- summary ---
n_total = len(rows)
n_unmatched = sum(1 for r in rows if r[0] == "UNMATCHED")
n_noxsec = sum(1 for r in rows
               if r[4] is None and r[0] not in ("MET_Data", "EGamma_Data",
                                                 "Signal_2HDMa", "UNMATCHED"))
print(f"\nSUMMARY: {n_total} datasets | UNMATCHED groups: {n_unmatched} | "
      f"MC missing xsec: {n_noxsec}")
