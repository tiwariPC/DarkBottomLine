"""
Histo-production input helpers for the Combine limit-setting pipeline.

Reads the per-region ROOT histogram artifacts already produced by
plotting.py's region-plot mode, resolves per-region systematic applicability
against configs/regions.yaml, and implements the e/mu channel merge and PDF
weight normalization stages (Agent 2 in the build plan).
"""

import glob
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import uproot

from .regions import RegionManager

ZERO_FLOOR = 1e-6


def resolve_version_dir(region_root_dir_template: str, version: Optional[str]) -> Path:
    """Resolve the version directory under a region-plots tree.

    region_root_dir_template is the raw template string containing a literal
    "{version}" placeholder (e.g. "outputs/region_plots/{version}/root").
    If version is given, substitutes it directly. If None, picks the most
    recently modified version directory under the template's parent dir.
    """
    plots_root = Path(region_root_dir_template.split("{version}")[0].rstrip("/"))

    if version is not None:
        resolved = Path(region_root_dir_template.format(version=version))
        if not resolved.exists():
            raise FileNotFoundError(f"Version directory not found: {resolved}")
        return resolved

    if not plots_root.is_dir():
        raise FileNotFoundError(f"No plots directory found at {plots_root}")

    candidates = [d for d in plots_root.iterdir() if d.is_dir() and (d / "root").is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No version directories with a root/ subdir under {plots_root}")

    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return latest / "root"


def region_hist_path(region_root_dir: str, category: str, region_dir: str,
                      variable: str, log: bool = True) -> Path:
    """Build the path to a region's histogram ROOT file.

    Matches plotting.py's _create_region_plots_single naming:
    hist_{category}_{region_dir}_{variable}[_log].root
    plotting.py always writes with use_log determined by the variable's plot
    config, and in practice every region ROOT file observed in the wild has
    carried the _log suffix — default to log=True, not False.
    """
    suffix = "_log" if log else ""
    fname = f"hist_{category}_{region_dir}_{variable}{suffix}.root"
    return Path(region_root_dir) / fname


def region_syst_hist_path(region_root_dir: str, category: str, region_dir: str,
                           variable: str, syst_suffix: str, direction: str,
                           log: bool = True) -> Path:
    """Build the path to a region's shape-systematic variant ROOT file.

    Shape systematics live in SEPARATE files from the nominal histogram
    (verified against plotting.py:2764-2855's _weight_systs/_kin_systs and
    root_stem = f"hist_{cat}_{rdir}_{svar}_{wsyst}_log" convention), each
    containing the same process-label keys as the nominal file:
    hist_{category}_{region_dir}_{variable}_{syst_suffix}{direction}[_log].root

    Args:
        syst_suffix: filename token, e.g. "weight_btag" or "JES" (NOT the
            combine.yaml systematic key — see combine.yaml's syst_suffix field)
        direction: "UP" or "DOWN" (uppercase, matching plotting.py's file names)
    """
    suffix = "_log" if log else ""
    fname = f"hist_{category}_{region_dir}_{variable}_{syst_suffix}{direction}{suffix}.root"
    return Path(region_root_dir) / fname


def region_dir_from_role(region_role: str) -> str:
    """Convert a regions.yaml region role (e.g. "CR_Wmunu") to the region_dir
    plotting.py uses for file naming (strips the "CR_" prefix, matches
    plotting.py's PlotManager._parse_region_name)."""
    if region_role.startswith("CR_"):
        return region_role.replace("CR_", "")
    return region_role


# Single source of truth for which e/mu-channel CR pairs get merged when
# combine_emu: true, and what merged region_dir each pair produces. Shared by
# merge_emu (cli.py) and make_datacard's Wlnu/Zll region_dir_override lookup —
# both must agree on this mapping or make-datacard silently reads the wrong
# (unmerged, stale, or missing) histogram directory.
EMU_PAIRS = {
    "1b": [("CR_Wmunu", "CR_Wenu", "Wlnu"), ("CR_Zmumu", "CR_Zee", "Zll")],
    "2b": [("CR_Topmunu", "CR_Topenu", "Topl"), ("CR_Zmumu", "CR_Zee", "Zll")],
}


def merged_region_dir_for_role(category: str, region_role: str) -> Optional[str]:
    """If region_role is one half of an EMU_PAIRS pair for this category,
    return the merged region_dir (e.g. "CR_Wmunu" -> "Wlnu"). Returns None if
    region_role isn't part of any e/mu pair (SR, or a category with no pairing
    defined for that role)."""
    for mu_role, e_role, merged_dir in EMU_PAIRS.get(category, []):
        if region_role in (mu_role, e_role):
            return merged_dir
    return None


def load_region_histogram(region_root_dir: str, category: str, region_dir: str,
                           variable: str, key: str) -> np.ndarray:
    """Read one histogram's bin values from a region ROOT file.

    Args:
        region_root_dir: Directory containing hist_*.root files
        category: "1b" or "2b"
        region_dir: plotting.py's region_dir (e.g. "Wmunu", "SR")
        variable: fit/discriminant variable name
        key: histogram key in the ROOT file (process group label, "TotalBkg",
             "data_obs", or "sig_<mass_point>")

    Returns:
        Bin values as a numpy array.
    """
    path = region_hist_path(region_root_dir, category, region_dir, variable)
    if not path.exists():
        raise FileNotFoundError(f"Region histogram file not found: {path}")

    with uproot.open(str(path)) as f:
        if key not in f:
            raise KeyError(f"Histogram key '{key}' not found in {path}. "
                            f"Available keys: {list(f.keys())}")
        values, _ = f[key].to_numpy()
    return values


def load_region_syst_histogram(region_root_dir: str, category: str, region_dir: str,
                                variable: str, syst_suffix: str, direction: str,
                                key: str) -> np.ndarray:
    """Read one process histogram's bin values from a shape-systematic
    variant file (separate file from the nominal, see region_syst_hist_path)."""
    path = region_syst_hist_path(region_root_dir, category, region_dir, variable,
                                  syst_suffix, direction)
    if not path.exists():
        raise FileNotFoundError(f"Systematic histogram file not found: {path}")

    with uproot.open(str(path)) as f:
        if key not in f:
            raise KeyError(f"Histogram key '{key}' not found in {path}. "
                            f"Available keys: {list(f.keys())}")
        values, _ = f[key].to_numpy()
    return values


def load_region_bin_edges(region_root_dir: str, category: str, region_dir: str,
                           variable: str) -> np.ndarray:
    """Read the bin edges of a region's TotalBkg histogram."""
    path = region_hist_path(region_root_dir, category, region_dir, variable)
    if not path.exists():
        raise FileNotFoundError(f"Region histogram file not found: {path}")

    with uproot.open(str(path)) as f:
        if "TotalBkg" not in f:
            raise KeyError(f"'TotalBkg' not found in {path}")
        _, edges = f["TotalBkg"].to_numpy()
    return edges


def systematic_applies_to_region(regions_config: str, region_key: str,
                                  gated_by_cut: Optional[List[str]]) -> bool:
    """Check whether a shape systematic applies to a region, based on whether
    any of `gated_by_cut`'s cut keys are present in that region's actual
    configs/regions.yaml cuts block.

    If gated_by_cut is None/empty, the systematic applies everywhere (default).

    Args:
        regions_config: path to configs/regions.yaml
        region_key: full region key, e.g. "1b:CR_Zmumu"
        gated_by_cut: list of regions.yaml cut keys (e.g. ["Bjet1bCond", "Bjet2bCond"])
    """
    if not gated_by_cut:
        return True

    region_manager = RegionManager(regions_config)
    region = region_manager.get_region(region_key)
    if region is None:
        raise KeyError(f"Region '{region_key}' not found in {regions_config}")

    return any(cut_key in region.cuts for cut_key in gated_by_cut)


def load_signal_grid(xsection_json: str, model_key: str,
                      points: Optional[List[str]] = None) -> Dict[str, float]:
    """Load the signal mass-point grid and cross sections.

    Args:
        xsection_json: path to data/cross-section/xsection_signal.json
        model_key: top-level key, e.g. "2HDMa"
        points: subset of mass-point labels to keep, or None for all

    Returns:
        {mass_point_label: cross_section_pb}
    """
    with open(xsection_json) as f:
        data = json.load(f)

    model_grid = data[model_key]
    grid = {k: v for k, v in model_grid.items() if not k.startswith("_")}

    if points is not None:
        missing = set(points) - set(grid.keys())
        if missing:
            raise KeyError(f"Requested signal points not in {xsection_json}[{model_key}]: {missing}")
        grid = {k: grid[k] for k in points}

    return grid


def resolve_active_eras(combine_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of era entries marked active: true in combine.yaml."""
    return [era for era in combine_config["eras"] if era.get("active", False)]


def resolve_era(combine_config: Dict[str, Any], year) -> Dict[str, Any]:
    """Look up a single era entry by year from combine.yaml's eras list."""
    year_str = str(year)
    for era in combine_config["eras"]:
        if str(era["year"]) == year_str:
            return era
    raise KeyError(f"Era '{year}' not found in combine.yaml eras list")


def _merge_two_histogram_files(mu_path: Path, e_path: Path, out_path: Path) -> None:
    """Sum bin values for every matching histogram key present in both files;
    keys present in only one file are carried through unchanged. Shared by
    merge_emu_histograms' nominal-file merge and its per-systematic-variant
    merge below — same combine logic, different source file pair."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.open(str(mu_path)) as f_mu, uproot.open(str(e_path)) as f_e:
        mu_keys = {k.split(";")[0] for k in f_mu.keys()}
        e_keys = {k.split(";")[0] for k in f_e.keys()}
        all_keys = mu_keys | e_keys

        with uproot.recreate(str(out_path)) as f_out:
            for key in all_keys:
                if key in mu_keys and key in e_keys:
                    v_mu, edges = f_mu[key].to_numpy()
                    v_e, _ = f_e[key].to_numpy()
                    merged = v_mu + v_e
                elif key in mu_keys:
                    merged, edges = f_mu[key].to_numpy()
                else:
                    merged, edges = f_e[key].to_numpy()
                f_out[key] = (merged.astype(float), edges.astype(float))


def merge_emu_histograms(region_root_dir: str, output_dir: str, category: str,
                          variable: str, mu_region_role: str, e_region_role: str,
                          merged_region_dir: str,
                          syst_suffixes: Optional[List[str]] = None) -> str:
    """Merge a muon-channel and electron-channel CR's histograms into one
    region ROOT file (Run2 combineEMU.py logic, ported to uproot/numpy).

    Also merges each shape systematic's UP/DOWN variant file (syst_suffixes,
    normally combine.yaml's datacard.systematics[*].syst_suffix values) the
    same way — otherwise the merged CR silently has NO shape systematics at
    all (write_shapes correctly reports "not applicable" when the Wlnu-named
    variant file doesn't exist, but the real gap is upstream: this function
    only ever merged the nominal file, never the systematic variants, so
    every shape systematic vanishes from combine_emu-merged CR datacards —
    reproduced against real Wmunu/Wenu region-plot output, which does have
    per-systematic variant files that were never being read here).
    """
    mu_dir = region_dir_from_role(mu_region_role)
    e_dir = region_dir_from_role(e_region_role)

    mu_path = region_hist_path(region_root_dir, category, mu_dir, variable)
    e_path = region_hist_path(region_root_dir, category, e_dir, variable)

    if not mu_path.exists() or not e_path.exists():
        raise FileNotFoundError(f"Missing input for e/mu merge: {mu_path} or {e_path}")

    out_path = region_hist_path(output_dir, category, merged_region_dir, variable)
    _merge_two_histogram_files(mu_path, e_path, out_path)

    for syst_suffix in (syst_suffixes or []):
        for direction in ("UP", "DOWN"):
            mu_syst_path = region_syst_hist_path(region_root_dir, category, mu_dir,
                                                  variable, syst_suffix, direction)
            e_syst_path = region_syst_hist_path(region_root_dir, category, e_dir,
                                                 variable, syst_suffix, direction)
            mu_exists, e_exists = mu_syst_path.exists(), e_syst_path.exists()
            if not mu_exists and not e_exists:
                continue
            out_syst_path = region_syst_hist_path(output_dir, category, merged_region_dir,
                                                    variable, syst_suffix, direction)
            if mu_exists and e_exists:
                _merge_two_histogram_files(mu_syst_path, e_syst_path, out_syst_path)
            else:
                # Systematic variant present for only one channel (e.g. a
                # channel-specific SF) — carry it through as-is, same
                # single-file passthrough the nominal merge does per-key.
                shutil.copy(mu_syst_path if mu_exists else e_syst_path, out_syst_path)

    logging.info(f"Merged e/mu histograms for {category}:{merged_region_dir} -> {out_path}")
    return str(out_path)


def passthrough_region_histogram(region_root_dir: str, output_dir: str,
                                  category: str, region_dir: str, variable: str,
                                  syst_suffixes: Optional[List[str]] = None) -> str:
    """Copy a region's histogram file unchanged (used when combine_emu/pdf_normalize
    is disabled, so downstream stages always read from a consistent output_dir).

    Shape systematics live in SEPARATE files from the nominal histogram
    (region_syst_hist_path) — syst_suffixes (combine.yaml's per-systematic
    syst_suffix values) must be passed through so those files get copied too,
    or every shape systematic silently vanishes downstream (same class of bug
    documented on merge_emu_histograms, reproduced here for the passthrough
    and normalize-pdf paths)."""
    src = region_hist_path(region_root_dir, category, region_dir, variable)
    if not src.exists():
        raise FileNotFoundError(f"Region histogram file not found: {src}")

    dst = region_hist_path(output_dir, category, region_dir, variable)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with uproot.open(str(src)) as f_in, uproot.recreate(str(dst)) as f_out:
        for key in {k.split(";")[0] for k in f_in.keys()}:
            values, edges = f_in[key].to_numpy()
            f_out[key] = (values.astype(float), edges.astype(float))

    for syst_suffix in (syst_suffixes or []):
        for direction in ("UP", "DOWN"):
            syst_src = region_syst_hist_path(region_root_dir, category, region_dir,
                                              variable, syst_suffix, direction)
            if not syst_src.exists():
                continue
            syst_dst = region_syst_hist_path(output_dir, category, region_dir,
                                              variable, syst_suffix, direction)
            syst_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(syst_src, syst_dst)

    return str(dst)


def normalize_pdf_histograms(region_root_dir: str, output_dir: str, category: str,
                              region_dir: str, variable: str,
                              syst_suffixes: Optional[List[str]] = None,
                              pdf_syst_suffix: Optional[str] = None) -> str:
    """Rescale pdf Up/Down shape-variant histograms to preserve the central
    (nominal) integral, per process (Run2 normalisePDF.py logic, ported to
    uproot/numpy).

    Shape systematics live in SEPARATE per-systematic files, not extra keys
    inside the nominal file (region_syst_hist_path convention). Every
    systematic in syst_suffixes is copied through unchanged EXCEPT
    pdf_syst_suffix's UP/DOWN files, which get rescaled per-process so each
    process's integral matches its nominal integral (the actual pdf-envelope
    normalization Run2's normalisePDF.py performs).
    """
    src = region_hist_path(region_root_dir, category, region_dir, variable)
    if not src.exists():
        raise FileNotFoundError(f"Region histogram file not found: {src}")

    dst = region_hist_path(output_dir, category, region_dir, variable)
    dst.parent.mkdir(parents=True, exist_ok=True)

    with uproot.open(str(src)) as f_in, uproot.recreate(str(dst)) as f_out:
        for key in {k.split(";")[0] for k in f_in.keys()}:
            values, edges = f_in[key].to_numpy()
            f_out[key] = (values.astype(float), edges.astype(float))
        nominal_histograms = {k: f_in[k].to_numpy() for k in {kk.split(";")[0] for kk in f_in.keys()}}

    for syst_suffix in (syst_suffixes or []):
        is_pdf = syst_suffix == pdf_syst_suffix
        for direction in ("UP", "DOWN"):
            syst_src = region_syst_hist_path(region_root_dir, category, region_dir,
                                              variable, syst_suffix, direction)
            if not syst_src.exists():
                continue
            syst_dst = region_syst_hist_path(output_dir, category, region_dir,
                                              variable, syst_suffix, direction)
            syst_dst.parent.mkdir(parents=True, exist_ok=True)

            if not is_pdf:
                shutil.copy(syst_src, syst_dst)
                continue

            with uproot.open(str(syst_src)) as f_in:
                keys = {k.split(";")[0] for k in f_in.keys()}
                histograms = {k: f_in[k].to_numpy() for k in keys}

            with uproot.recreate(str(syst_dst)) as f_out:
                for key, (values, edges) in histograms.items():
                    if key not in nominal_histograms:
                        f_out[key] = (values.astype(float), edges.astype(float))
                        continue

                    nominal_values, _ = nominal_histograms[key]
                    nominal_integral = nominal_values.sum()
                    variant_integral = values.sum()
                    if variant_integral <= 0:
                        f_out[key] = (values.astype(float), edges.astype(float))
                        continue

                    scale = nominal_integral / variant_integral
                    rescaled = np.clip(values * scale, ZERO_FLOOR, None)
                    f_out[key] = (rescaled.astype(float), edges.astype(float))

    logging.info(f"Normalized PDF variants for {category}:{region_dir} -> {dst}")
    return str(dst)
