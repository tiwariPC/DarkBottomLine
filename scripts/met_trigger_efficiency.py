#!/usr/bin/env python3
"""
MET (PFMETNoMu) trigger efficiency and data/MC scale factor by tag-and-probe.

Implements the AN-26-094 Section 5 methodology: the Run 3 MET trigger turn-on is
measured in single-muon (W->mu nu) and double-muon (Z->mu mu) control samples,
selected from the Muon primary dataset with the orthogonal single-muon reference
trigger (HLT_IsoMu24 OR HLT_IsoMu27). The efficiency is binned in the hadronic
recoil U = |PuppiMET_vec + sum(mu pT_vec)| (muons added back, since the HLT is
PFMETNoMu). The data/MC ratio is the trigger scale factor; the W->mu nu vs Z->mu mu
difference drives the systematic (sfup/sfdown).

Outputs (fixed names):
  outputs/plots/hlt_sf/HLT_PFMETNoMu_120To140_IDTight_DATA_eff.{png,pdf}
  outputs/plots/hlt_sf/HLT_PFMETNoMu_120To140_IDTight_MC_eff.{png,pdf}
  outputs/plots/hlt_sf/HLT_PFMETNoMu_120To140_IDTight_sf.{png,pdf}
  data/corrections/2024/Run3-Summer24-NanoAODv15_metHlt.json.gz  (correction: MET-HLT-SF)

Two-step mode (recommended for hundreds of remote files):

    # Step 1 - skim: preselect NanoAOD, write one small ROOT per input file as
    #   <outdir>/<txtfile stem>_<unique last string of the input ROOT filename>.root
    python scripts/met_trigger_efficiency.py skim --config configs/2024.yaml \
        --data-files data/samplelist/2024/Muon*.txt \
        --mc-files data/samplelist/2024/DYto2L-2Jets_*.txt \
                   data/samplelist/2024/WtoLNu-2Jets_*.txt \
        --outdir outputs/skims/2024

    # (optional) hadd skims per samplelist yourself, then...

    # Step 2 - analyze: read the skims, make plots + correctionlib JSON (fast, local)
    python scripts/met_trigger_efficiency.py analyze --config configs/2024.yaml \
        --data-skims "outputs/skims/2024/Muon*.root" \
        --mc-skims   "outputs/skims/2024/DYto2L-2Jets_*.root" \
                     "outputs/skims/2024/WtoLNu-2Jets_*.root"

One-shot mode (single pass, no skim files):

    python scripts/met_trigger_efficiency.py run --config configs/2024.yaml \
        --data-files data/samplelist/2024/Muon*.txt \
        --mc-files data/samplelist/2024/DYto2L-2Jets_*.txt \
                   data/samplelist/2024/WtoLNu-2Jets_*.txt \
        --channel both
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import awkward as ak
import matplotlib as mpl

mpl.use("Agg")  # headless backend; must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402
import numpy as np
import uproot
import yaml

plt.style.use(hep.style.CMS)

# Make the repo root importable when this file is run by path (python3 scripts/x.py),
# where sys.path[0] is scripts/ rather than the repo root, so `darkbottomline` (a
# repo-root package) would otherwise not be found.
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from darkbottomline.objects import (
    build_muon_collection,
    calculate_recoil,
    select_electrons,
    select_muons,
)
from darkbottomline.selections import pass_met_filters, pass_triggers

# We load NanoAOD via uproot directly rather than coffea NanoEvents: the framework's
# object builders / selection masks read flat branches (events["Muon_pt"] etc.), which
# work identically on a plain uproot record array, and this avoids a hard coffea<->uproot
# schema version coupling. Only the flat branches we actually use are read per file.
BRANCH_PREFIXES = (
    "Muon_", "Electron_", "Jet_", "PuppiMET_", "PFMET_", "MET_",
)
SCALAR_BRANCHES = ("event", "run", "luminosityBlock", "genWeight")

CHANNELS = ("wmn", "zmm")
DEFAULT_RECOIL_BINS = [
    0, 20, 40, 60, 80, 100, 110, 120, 130, 140,
    150, 160, 180, 200, 250, 300, 400, 600, 800, 1000, 1200, 1300, 1400, 1500
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a year YAML. Missing keys are surfaced loudly downstream (no defaults)."""
    with open(path) as f:
        return yaml.safe_load(f)


def _require(cfg: Dict[str, Any], *keys: str) -> Any:
    """Fetch a nested config key, raising a loud KeyError on any missing level."""
    node = cfg
    for k in keys:
        if k not in node:
            raise KeyError(f"Missing config key: {'.'.join(keys)} (at '{k}')")
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# File-list resolution
# ---------------------------------------------------------------------------

def _expand_txt(path: str) -> List[str]:
    """Expand a samplelist .txt (one ROOT path per line, '#' = comment)."""
    out: List[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def resolve_inputs(patterns: List[str]) -> List[Tuple[str, str]]:
    """
    Expand CLI globs / .txt samplelists into a flat list of (root_path, sample_stem).

    sample_stem is the samplelist basename (used for the MC xsec lookup) when the
    input is a .txt, else the ROOT file basename.
    """
    resolved: List[Tuple[str, str]] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat)) or [pat]
        for m in matches:
            if m.endswith(".txt"):
                stem = os.path.splitext(os.path.basename(m))[0]
                for root_path in _expand_txt(m):
                    resolved.append((root_path, stem))
            else:
                stem = os.path.splitext(os.path.basename(m))[0]
                resolved.append((m, stem))
    return resolved


def mc_channels(requested: List[str]) -> List[str]:
    """
    Channels an MC sample contributes to: ALL requested channels. Routing is NOT
    per-sample — every MC event is tried against each channel's preselection and
    lands wherever the muon multiplicity places it. So DY populates mostly Zmm, W
    mostly Wmn, and tt/single-top/diboson contaminate both, exactly as in data.
    """
    return list(requested)


# ---------------------------------------------------------------------------
# Cross-section / sumw bookkeeping (MC pT-bin stitching)
# ---------------------------------------------------------------------------

def load_xsections(json_path: str) -> Dict[str, float]:
    """
    Flatten an xsection JSON to {full_dataset: xsection}.

    Keyed by category -> list of {process, xsection, full_dataset}. `full_dataset`
    may be a single string OR a list of naming variants (the run3 file lists both the
    '_2J_' and 'Bin-2J' dataset names per process) — every variant is registered so a
    skim named with either convention resolves.
    """
    with open(json_path) as f:
        raw = json.load(f)
    flat: Dict[str, float] = {}
    for entries in raw.values():
        if not isinstance(entries, list):
            continue
        for e in entries:
            xs = e.get("xsection")
            ds = e.get("full_dataset")
            if xs is None or not ds:
                continue
            names = ds if isinstance(ds, list) else [ds]
            for name in names:
                if name and name not in flat:
                    flat[name] = float(xs)
    return flat


def _xsec_key(name: str) -> Tuple[str, str]:
    """
    Reduce a sample name to (family, pt_range) for robust xsec matching.

    Handles samplelist stems that carry extra tokens vs the JSON full_dataset,
    e.g. 'DYto2L-2Jets_Bin-2J-MLL-50-PTLL-100to200_...' vs
    'DYto2L-2Jets_MLL-50-PTLL-100to200_...' (both -> ('dyto2l-2jets', '100to200')).
    """
    import re
    low = name.lower().split("___")[0]
    if "dyto2l" in low:
        family = "dyto2l-2jets"
    elif "wtolnu" in low:
        family = "wtolnu-2jets"
    else:
        family = low.split("_")[0]
    pt = ""
    m_range = re.search(r"(\d+to\d+)", low)
    if m_range:
        pt = m_range.group(1)
    else:
        m_open = re.search(r"(?:ptll|ptlnu)-?(\d+)", low)
        if m_open:
            pt = m_open.group(1)
    return family, pt


def find_xsec(stem: str, xsecs: Dict[str, float]) -> Optional[float]:
    """
    Look up an xsection by sample stem. Order matters: exact match, then the
    specific (family, pt-range) key, and only then a loose substring match as a
    last resort (substring alone is ambiguous, e.g. 'WtoLNu' vs 'WtoLNu-4Jets').
    """
    if stem in xsecs:
        return xsecs[stem]
    key = _xsec_key(stem)
    if key[0] and key[1]:
        for ds, xs in xsecs.items():
            if _xsec_key(ds) == key:
                return xs
    for ds, xs in xsecs.items():
        if ds == stem or ds in stem or stem in ds:
            return xs
    return None


def read_gensumw(tfile: "uproot.ReadOnlyDirectory") -> Tuple[float, float]:
    """
    Return (genTotalSumw, genTotalCount) over ALL events in the file:
      genTotalSumw  = sum of sign(genWeight)  (effective event count; the norm
                      denominator, consistent with the sign-only per-event weight),
      genTotalCount = raw number of generated events.

    We use the SIGN of genWeight, not its magnitude: the magnitude cancels in the
    efficiency ratio anyway, and sign avoids the huge NLO weight scale. The Runs-tree
    genEventSumw is magnitude-weighted, so it cannot be used here — instead we sum
    sign(genWeight) over the full Events tree (NOT the entry_stop-limited slice).
    """
    try:
        if "Events" not in tfile:
            return 0.0, 0.0
        events_tree = tfile["Events"]
        count = float(events_tree.num_entries)
        if "genWeight" not in events_tree:
            return count, count  # no genWeight (e.g. data): treat every event as +1
        gw = events_tree["genWeight"].array(library="np")
        sumw = float(np.sign(gw).sum())
        return sumw, count
    except Exception as exc:  # noqa: BLE001 - defensive, one bad file shouldn't kill run
        print(f"WARNING: could not read genWeight sum: {exc}")
        return 0.0, 0.0


def load_events(tfile: "uproot.ReadOnlyDirectory",
                max_events: Optional[int]) -> ak.Array:
    """
    Read the flat NanoAOD branches we need from the Events tree via uproot.

    Loaded as a plain awkward record array — the framework object builders and
    selection masks read flat branches (events["Muon_pt"] etc.), so no coffea
    NanoEvents schema is required.
    """
    tree = tfile["Events"]
    available = set(tree.keys())
    wanted = [b for b in SCALAR_BRANCHES if b in available]
    wanted += [b for b in available
               if any(b.startswith(p) for p in BRANCH_PREFIXES)]
    wanted += [b for b in available if b.startswith("HLT_")]
    return tree.arrays(wanted, entry_stop=max_events)


# ---------------------------------------------------------------------------
# Event-level preselection + recoil
# ---------------------------------------------------------------------------

def _leading_jet_pt(events: ak.Array, jet_cfg: Dict[str, Any]) -> ak.Array:
    """
    Max pt among AK4 jets passing |eta| < eta_max (0 when none). Matches the
    framework's select_jets, which cuts on pt/eta only — no jet-ID bit, as the
    puId branch is absent in NanoAOD v12+ (see objects.py::select_jets).
    """
    if "Jet_pt" not in events.fields or "Jet_eta" not in events.fields:
        return ak.zeros_like(events["event"], dtype=np.float64)
    eta_ok = abs(events["Jet_eta"]) < jet_cfg["eta_max"]
    jet_pt = ak.where(eta_ok, events["Jet_pt"], 0.0)
    return ak.fill_none(ak.max(jet_pt, axis=1), 0.0)


def preselection_mask(
    events: ak.Array,
    cfg: Dict[str, Any],
    channel: str,
) -> Tuple[ak.Array, ak.Array]:
    """
    Build the preselection mask and the tight-muon collection for the recoil.

    Preselection (both channels): reference trigger, tight-muon count, leading jet
    pt > 50, loose-electron veto. Z->mu mu additionally requires a second loose muon.
    Returns (event_mask, tight_muons_zip) where the zip is masked to selected muons.
    """
    muon_cfg = _require(cfg, "objects", "muons")
    ele_cfg = _require(cfg, "objects", "electrons")
    jet_cfg = _require(cfg, "objects", "jets")
    ref_triggers = _require(cfg, "triggers", "SingleMuon")
    noise_filters = _require(cfg, "noise_filters")

    # Reference (orthogonal single-muon) trigger + MET noise filters.
    ref_mask = pass_triggers(events, ref_triggers)
    filter_mask = pass_met_filters(events, noise_filters)

    # Muon collections and per-object selection masks.
    muons = build_muon_collection(events)
    tight_muon_mask = select_muons(events, muon_cfg, wp="tight")
    loose_muon_mask = select_muons(events, muon_cfg, wp="loose")
    tight_muons = muons[tight_muon_mask]
    n_tight = ak.sum(tight_muon_mask, axis=1)
    n_loose = ak.sum(loose_muon_mask, axis=1)

    if channel == "wmn":
        muon_count_mask = n_tight == 1
    elif channel == "zmm":
        # >=1 tight muon plus a second muon passing Loose ID (>=2 loose total).
        muon_count_mask = (n_tight >= 1) & (n_loose >= 2)
    else:
        raise ValueError(f"Unknown channel: {channel}")

    # Leading AK4 jet (|eta| < eta_max) with pt > 50 GeV.
    lead_jet_mask = _leading_jet_pt(events, jet_cfg) > 50.0

    # Loose-electron veto.
    loose_ele_mask = select_electrons(events, ele_cfg, wp="loose")
    ele_veto_mask = ~ak.any(loose_ele_mask, axis=1)

    event_mask = (
        ref_mask & filter_mask & muon_count_mask & lead_jet_mask & ele_veto_mask
    )
    return event_mask, tight_muons


def compute_recoil(events: ak.Array, tight_muons: ak.Array) -> np.ndarray:
    """Hadronic recoil U = |PuppiMET_vec + sum(mu pT_vec)| for the selected muons."""
    # calculate_recoil sums pt/phi over BOTH the "muons" and "electrons" collections.
    # We want a muon-only add-back (electrons are vetoed in preselection anyway), so
    # we pass a genuinely empty electron collection. We derive it by masking the muon
    # zip to nothing (tight_muons[all-False]) rather than ak.Array([]): this preserves
    # the same record layout (pt/phi fields, correct per-event jaggedness), so
    # calculate_recoil's ak.sum(...pt*cos(phi), axis=1) sees empty sublists (-> 0),
    # not a fieldless array that would raise on `.pt`.
    empty_electrons = tight_muons[tight_muons.pt < -1.0]
    recoil_pt, _ = calculate_recoil(
        events, {"muons": tight_muons, "electrons": empty_electrons}
    )
    return ak.to_numpy(recoil_pt)


# ---------------------------------------------------------------------------
# Per-file accumulation
# ---------------------------------------------------------------------------

class BinCounts:
    """Weighted numerator/denominator per recoil bin, plus sum-of-weights-squared
    on the denominator for the effective-count (weighted) Clopper-Pearson errors."""

    def __init__(self, n_bins: int):
        self.den = np.zeros(n_bins, dtype=np.float64)
        self.num = np.zeros(n_bins, dtype=np.float64)
        self.den_w2 = np.zeros(n_bins, dtype=np.float64)

    def fill(self, recoil: np.ndarray, passed: np.ndarray,
             weights: np.ndarray, edges: np.ndarray) -> None:
        idx = np.digitize(recoil, edges) - 1
        valid = (idx >= 0) & (idx < len(self.den))
        idx = idx[valid]
        w = weights[valid]
        p = passed[valid]
        np.add.at(self.den, idx, w)
        np.add.at(self.den_w2, idx, w * w)
        np.add.at(self.num, idx, w * p)


def compute_event_rows(
    root_path: str,
    channels: List[str],
    cfg: Dict[str, Any],
    is_mc: bool,
    max_events: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    Load one ROOT file, apply preselection per channel, and return the per-event
    skim rows plus the file's total generator weight.

    Deliberately NO cross-section / luminosity / sumw normalisation here: the skim
    stores only the per-event sign(genWeight) (1.0 for data) and the file's total
    sum of sign(genWeight) (as a scalar). Step 2 (analyze) applies xsec*lumi/SUMW
    once, with SUMW = sum of file_sumw over ALL files of the sample. That makes the
    weight of an event independent of which file it landed in, so skims of one sample
    can be hadd-merged and analysed correctly.

    Returns dict {recoil, met_pass, genweight, channel, file_sumw} or None on failure.
    """
    try:
        tfile = uproot.open(root_path)
    except Exception as exc:  # noqa: BLE001 - skip bad XRootD files, keep going
        print(f"WARNING: skipping unopenable file {root_path}: {exc}")
        return None

    # Context-manage the open file so the descriptor is always released, even if
    # load_events or the per-channel processing raises (important at 1000s of files).
    with tfile:
        try:
            events = load_events(tfile, max_events)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: skipping unreadable file {root_path}: {exc}")
            return None

        met_triggers = _require(cfg, "triggers", "MET")
        met_pass = ak.to_numpy(pass_triggers(events, met_triggers))

        # Per-event weight = SIGN of genWeight (+/-1), 1.0 for data / no branch. We use
        # the sign only: the genWeight magnitude cancels in the efficiency ratio, and
        # the sumw denominator below is likewise sum of sign(genWeight).
        if is_mc and "genWeight" in events.fields:
            genweight = np.sign(ak.to_numpy(events["genWeight"])).astype(np.float64)
        else:
            genweight = np.ones(len(events), dtype=np.float64)

        # File-total sum of sign(genWeight) + raw event count over ALL events in the
        # file. 0 for data (unused there).
        file_sumw, file_count = read_gensumw(tfile) if is_mc else (0.0, 0.0)

        rec_all: List[np.ndarray] = []
        pass_all: List[np.ndarray] = []
        w_all: List[np.ndarray] = []
        ch_all: List[np.ndarray] = []
        for ci, channel in enumerate(channels):
            mask, tight_muons = preselection_mask(events, cfg, channel)
            mask_np = ak.to_numpy(mask)
            if not mask_np.any():
                continue
            rec_all.append(compute_recoil(events, tight_muons)[mask_np])
            pass_all.append(met_pass[mask_np].astype(np.float64))
            w_all.append(genweight[mask_np])
            ch_all.append(np.full(int(mask_np.sum()), ci, dtype=np.int32))

    if not rec_all:
        return {"recoil": np.array([]), "met_pass": np.array([]),
                "genweight": np.array([]), "channel": np.array([], dtype=np.int32),
                "file_sumw": file_sumw, "file_count": file_count}
    return {
        "recoil": np.concatenate(rec_all),
        "met_pass": np.concatenate(pass_all),
        "genweight": np.concatenate(w_all),
        "channel": np.concatenate(ch_all),
        "file_sumw": file_sumw,
        "file_count": file_count,
    }


def accumulate_rows(rows: Dict[str, Any], weight: np.ndarray, channels: List[str],
                    edges: np.ndarray, counts: Dict[str, BinCounts]) -> None:
    """
    Fill per-channel BinCounts from a skim-row dict, using a caller-supplied final
    per-event weight array (already = genweight * xsec * lumi / SUMW for MC, or all
    ones for data). channel index i maps to channels[i].
    """
    ch = rows["channel"]
    for ci, channel in enumerate(channels):
        sel = ch == ci
        if not sel.any():
            continue
        counts[channel].fill(
            rows["recoil"][sel], rows["met_pass"][sel], weight[sel], edges
        )


def norm_factor(sumw: float, xsec: Optional[float], lumi: float) -> float:
    """MC normalisation xsec*lumi/SUMW (1.0 when xsec/sumw unavailable, i.e. data)."""
    if xsec is not None and sumw > 0:
        return xsec * lumi / sumw
    return 1.0


def accumulate_file(
    root_path: str,
    stem: str,
    channels: List[str],
    cfg: Dict[str, Any],
    edges: np.ndarray,
    counts: Dict[str, BinCounts],
    is_mc: bool,
    xsecs: Dict[str, float],
    lumi: float,
    max_events: Optional[int],
) -> None:
    """
    Single-pass (one-shot 'run' mode): load a NanoAOD file, normalise, fill counts.

    Here one file = one sample instance, so the file's own genEventSumw IS the sample
    SUMW (single-file normalisation). For the multi-file two-step path use skim +
    analyze, which sums SUMW across a sample's files.
    """
    rows = compute_event_rows(root_path, channels, cfg, is_mc, max_events)
    if rows is None:
        return
    if is_mc:
        xsec = find_xsec(stem, xsecs)
        if xsec is None:
            print(f"WARNING: no xsection for '{stem}' - using norm 1.0")
        norm = norm_factor(rows["file_sumw"], xsec, lumi)
    else:
        norm = 1.0
    weight = rows["genweight"] * norm
    accumulate_rows(rows, weight, channels, edges, counts)


# ---------------------------------------------------------------------------
# Skim I/O (two-step mode)
# ---------------------------------------------------------------------------

SKIM_TREE = "MetTriggerSkim"
SUMW_HIST = "genTotalSumw"
COUNT_HIST = "genTotalCount"


def skim_output_name(outdir: str, txt_stem: str, root_path: str) -> str:
    """
    <outdir>/<txtfile stem>_<unique last string of the input ROOT filename>.root

    The 'unique last string' is the input ROOT basename without extension (the file
    hash/id that makes each entry in a samplelist unique).
    """
    root_id = os.path.splitext(os.path.basename(root_path))[0]
    return os.path.join(outdir, f"{txt_stem}_{root_id}.root")


def write_skim(rows: Dict[str, Any], out_path: str) -> None:
    """
    Write the per-event skim (recoil, met_pass, genweight, channel) plus the file's
    total generator weight as a 1-bin TH1 (genTotalSumw). Storing the sumw as a
    histogram means `hadd` sums it automatically across a sample's files, so a merged
    skim carries the correct sample-total SUMW for step 2.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with uproot.recreate(out_path) as f:
        f[SKIM_TREE] = {
            "recoil": rows["recoil"].astype(np.float64),
            "met_pass": rows["met_pass"].astype(np.float64),
            "genweight": rows["genweight"].astype(np.float64),
            "channel": rows["channel"].astype(np.int32),
        }
        # 1-bin TH1s: signed genEventSumw (used for norm) and raw genEventCount
        # (reference only — shows the effect of negative NLO weights). Both hadd-sum.
        f[SUMW_HIST] = (
            np.array([float(rows["file_sumw"])], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
        f[COUNT_HIST] = (
            np.array([float(rows["file_count"])], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
    print(f"Saved: {out_path}")


def read_skim(path: str) -> Optional[Dict[str, Any]]:
    """
    Read a skim ROOT into {recoil, met_pass, genweight, channel, sumw}. `sumw` is the
    integral of the genTotalSumw histogram (sample-total when the skim is a hadd of a
    sample's files). None on unreadable/missing tree.
    """
    try:
        with uproot.open(path) as f:
            if SKIM_TREE not in f:
                print(f"WARNING: no '{SKIM_TREE}' tree in {path}")
                return None
            arr = f[SKIM_TREE].arrays(library="np")
            sumw = 0.0
            if SUMW_HIST in f:
                sumw = float(np.sum(f[SUMW_HIST].values()))
            else:
                print(f"WARNING: no '{SUMW_HIST}' histogram in {path}; sumw=0")
            count = float(np.sum(f[COUNT_HIST].values())) if COUNT_HIST in f else 0.0
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: skipping unreadable skim {path}: {exc}")
        return None
    return {
        "recoil": arr["recoil"],
        "met_pass": arr["met_pass"],
        "genweight": arr["genweight"],
        "channel": arr["channel"].astype(np.int32),
        "sumw": sumw,
        "count": count,
    }


# ---------------------------------------------------------------------------
# Efficiency + errors
# ---------------------------------------------------------------------------

def clopper_pearson(k: np.ndarray, n: np.ndarray,
                    cl: float = 0.6827) -> Tuple[np.ndarray, np.ndarray]:
    """Clopper-Pearson interval (lo, hi) for k successes of n trials."""
    from scipy.stats import beta
    alpha = 1.0 - cl
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    lo = np.nan_to_num(lo, nan=0.0)
    hi = np.nan_to_num(hi, nan=1.0)
    return lo, hi


def efficiency_with_errors(
    c: BinCounts, weighted: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Efficiency and asymmetric (lo, hi) errors per bin.

    Data (weighted=False): Clopper-Pearson on raw integer counts.
    MC (weighted=True): effective-count Clopper-Pearson (n_eff = (sum w)^2 / sum w^2).
    """
    eff = np.divide(c.num, c.den, out=np.zeros_like(c.num), where=c.den > 0)

    if not weighted:
        # Data weights are all 1.0, so num/den are integer counts; round before the
        # int cast so float accumulation (e.g. 41.9999999) doesn't truncate down.
        k = np.rint(c.num).astype(np.int64)
        n = np.rint(c.den).astype(np.int64)
    else:
        n_eff = np.divide(c.den ** 2, c.den_w2,
                          out=np.zeros_like(c.den), where=c.den_w2 > 0)
        n = n_eff
        # k = eff * n_eff can slightly exceed n_eff from float rounding, which makes
        # beta.ppf(..., n-k+1) receive a non-positive shape -> NaN. Clamp k <= n.
        k = np.minimum(eff * n_eff, n)

    lo, hi = clopper_pearson(k, n)
    err_lo = np.clip(eff - lo, 0.0, None)
    err_hi = np.clip(hi - eff, 0.0, None)
    return eff, err_lo, err_hi


def scale_factor(
    eff_d: np.ndarray, ed_lo: np.ndarray, ed_hi: np.ndarray,
    eff_m: np.ndarray, em_lo: np.ndarray, em_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SF = eff_data / eff_mc with symmetrised errors propagated in quadrature."""
    # np.divide(..., where=) computes only where the denominator is > 0 (no
    # divide-by-zero even under warnings); elsewhere the preset 0.0 output stands.
    sf = np.divide(eff_d, eff_m, out=np.zeros_like(eff_d), where=eff_m > 0)
    ed = 0.5 * (ed_lo + ed_hi)
    em = 0.5 * (em_lo + em_hi)
    rel_d = np.divide(ed, eff_d, out=np.zeros_like(ed), where=eff_d > 0)
    rel_m = np.divide(em, eff_m, out=np.zeros_like(em), where=eff_m > 0)
    sf_err = np.abs(sf) * np.sqrt(rel_d ** 2 + rel_m ** 2)
    return sf, sf_err, sf_err


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _bin_centres(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ctr = 0.5 * (edges[:-1] + edges[1:])
    half = 0.5 * (edges[1:] - edges[:-1])
    return ctr, half


def _save(fig: plt.Figure, outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    pdf_path = os.path.splitext(outpath)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")
    print(f"Saved: {pdf_path}")


def plot_efficiency(
    edges: np.ndarray,
    per_channel: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    outpath: str,
    lumi: Optional[float],
    com: float,
    label: str,
    is_data: bool,
) -> None:
    """Turn-on efficiency vs recoil, one curve per channel, C-P error bars."""
    ctr, half = _bin_centres(edges)
    fig, ax = plt.subplots(figsize=(9, 8))
    styles = {"wmn": ("o", r"$W\to\mu\nu$"), "zmm": ("s", r"$Z\to\mu\mu$")}
    for ch, (eff, elo, ehi) in per_channel.items():
        marker, leg = styles.get(ch, ("o", ch))
        ax.errorbar(
            ctr, eff, yerr=[elo, ehi], xerr=half,
            fmt=marker, capsize=2, markersize=6, label=leg,
        )
    ax.set_xlabel(r"Hadronic recoil $U$ [GeV]", fontsize=18)
    ax.set_ylabel("Trigger efficiency", fontsize=18)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlim(edges[0], edges[-1])
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.legend(loc="lower right", fontsize=16)
    hep.cms.label(text=label, data=is_data, lumi=lumi, com=com, ax=ax, fontsize=18)
    _save(fig, outpath)


def plot_scale_factor(
    edges: np.ndarray,
    per_channel: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    outpath: str,
    lumi: Optional[float],
    com: float,
    label: str,
) -> None:
    """Data/MC scale factor vs recoil, one curve per channel."""
    ctr, half = _bin_centres(edges)
    fig, ax = plt.subplots(figsize=(9, 8))
    styles = {"wmn": ("o", r"$W\to\mu\nu$"), "zmm": ("s", r"$Z\to\mu\mu$")}
    for ch, (sf, slo, shi) in per_channel.items():
        marker, leg = styles.get(ch, ("o", ch))
        ax.errorbar(
            ctr, sf, yerr=[slo, shi], xerr=half,
            fmt=marker, capsize=2, markersize=6, label=leg,
        )
    ax.set_xlabel(r"Hadronic recoil $U$ [GeV]", fontsize=18)
    ax.set_ylabel("Data / MC scale factor", fontsize=18)
    # Wide enough for the turn-on region where data/MC can disagree by >20%.
    ax.set_ylim(0.5, 1.5)
    ax.set_xlim(edges[0], edges[-1])
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.legend(loc="lower right", fontsize=16)
    hep.cms.label(text=label, data=True, lumi=lumi, com=com, ax=ax, fontsize=18)
    _save(fig, outpath)


# ---------------------------------------------------------------------------
# correctionlib JSON
# ---------------------------------------------------------------------------

def build_correction_set(
    edges: np.ndarray,
    sf: np.ndarray,
    sf_up: np.ndarray,
    sf_down: np.ndarray,
) -> Dict[str, Any]:
    """
    Assemble a schema_version-2 CorrectionSet with a single 'MET-HLT-SF' correction:
    category on ValType (sf/sfup/sfdown) -> multibinning over the recoil axis.
    """
    def _edges_json(e: np.ndarray) -> List[Any]:
        # The recoil binning is finite by construction. Reject infinite edges rather
        # than emit them: out-of-range recoil is already handled by flow="clamp".
        if not np.all(np.isfinite(e)):
            raise ValueError(
                "infinite recoil bin edges are not supported "
                "(out-of-range values are clamped via flow='clamp')"
            )
        return [float(v) for v in e]

    def _leaf(content: np.ndarray) -> Dict[str, Any]:
        return {
            "nodetype": "multibinning",
            "inputs": ["recoil"],
            "edges": [_edges_json(edges)],
            "content": [float(x) for x in content],
            "flow": "clamp",
        }

    correction = {
        "name": "MET-HLT-SF",
        "description": "MET (PFMETNoMu120/130/140_IDTight OR) trigger data/MC scale "
                       "factor vs hadronic recoil U. Systematic = W->munu vs Z->mumu "
                       "difference (sfup/sfdown).",
        "version": 1,
        "inputs": [
            {"name": "ValType", "type": "string",
             "description": "sf/sfup/sfdown (sfup = sf + syst, sfdown = sf - syst)"},
            {"name": "recoil", "type": "real",
             "description": "hadronic recoil U [GeV]"},
        ],
        "output": {"name": "weight", "type": "real",
                   "description": "value of scale factor (nominal, up or down)"},
        "data": {
            "nodetype": "category",
            "input": "ValType",
            "content": [
                {"key": "sf", "value": _leaf(sf)},
                {"key": "sfup", "value": _leaf(sf_up)},
                {"key": "sfdown", "value": _leaf(sf_down)},
            ],
        },
    }
    return {
        "schema_version": 2,
        "description": "MET trigger scale factors (tag-and-probe, AN-26-094 Sec 5).",
        "corrections": [correction],
    }


def write_correction(cset: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        json.dump(cset, f)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def make_plots_and_json(
    data_counts: Dict[str, BinCounts],
    mc_counts: Dict[str, BinCounts],
    channels: List[str],
    edges: np.ndarray,
    outdir: str,
    lumi: Optional[float],
    com: float,
    label: str,
    json_out: str,
) -> None:
    """Turn accumulated counts into the 3 plots + the correctionlib JSON."""
    data_eff = {ch: efficiency_with_errors(data_counts[ch], weighted=False)
                for ch in channels}
    mc_eff = {ch: efficiency_with_errors(mc_counts[ch], weighted=True)
              for ch in channels}

    plot_efficiency(
        edges, data_eff,
        os.path.join(outdir, "HLT_PFMETNoMu_120To140_IDTight_DATA_eff.png"),
        lumi, com, label, is_data=True,
    )
    plot_efficiency(
        edges, mc_eff,
        os.path.join(outdir, "HLT_PFMETNoMu_120To140_IDTight_MC_eff.png"),
        lumi, com, label, is_data=False,
    )

    sf_per_channel: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sf_values: Dict[str, np.ndarray] = {}
    for ch in channels:
        ed, edlo, edhi = data_eff[ch]
        em, emlo, emhi = mc_eff[ch]
        sf, slo, shi = scale_factor(ed, edlo, edhi, em, emlo, emhi)
        sf_per_channel[ch] = (sf, slo, shi)
        sf_values[ch] = sf

    plot_scale_factor(
        edges, sf_per_channel,
        os.path.join(outdir, "HLT_PFMETNoMu_120To140_IDTight_sf.png"),
        lumi, com, label,
    )

    # correctionlib: nominal SF + systematic from the W->munu vs Z->mumu difference.
    nominal_ch = "wmn" if "wmn" in channels else channels[0]
    sf_nom = sf_values[nominal_ch]
    stat_err = sf_per_channel[nominal_ch][1]
    if "wmn" in sf_values and "zmm" in sf_values:
        # Channel difference as the systematic — but only where BOTH channels have a
        # valid (non-zero) SF. In sparse high-U bins one channel can be empty (SF=0),
        # which would fake a huge |wmn-zmm|; there, fall back to the statistical error.
        both_valid = (sf_values["wmn"] > 0) & (sf_values["zmm"] > 0)
        syst = np.where(
            both_valid, np.abs(sf_values["wmn"] - sf_values["zmm"]), stat_err
        )
    else:
        syst = stat_err
    write_correction(
        build_correction_set(edges, sf_nom, sf_nom + syst, sf_nom - syst),
        json_out,
    )


def _resolve_channels(channel: str) -> List[str]:
    return list(CHANNELS) if channel == "both" else [channel]


def cmd_skim(args: argparse.Namespace) -> None:
    """
    Step 1: preselect NanoAOD, write one skim ROOT per input file. The skim stores
    raw per-event genWeight + the file's total genEventSumw only (NO xsec/lumi/sumw
    normalisation — that is applied in analyze, after summing SUMW over the sample).
    """
    cfg = load_config(args.config)
    channels = _resolve_channels(args.channel)

    data_inputs = resolve_inputs(args.data_files) if args.data_files else []
    mc_inputs = resolve_inputs(args.mc_files) if args.mc_files else []
    if args.max_files is not None:
        data_inputs = data_inputs[: args.max_files]
        mc_inputs = mc_inputs[: args.max_files]

    print(f"Skim: {len(data_inputs)} data + {len(mc_inputs)} MC files -> {args.outdir}")

    for i, (root_path, stem) in enumerate(data_inputs):
        print(f"[data {i + 1}/{len(data_inputs)}] {root_path}")
        rows = compute_event_rows(root_path, channels, cfg, False, args.max_events)
        if rows is not None:
            write_skim(rows, skim_output_name(args.outdir, stem, root_path))

    for i, (root_path, stem) in enumerate(mc_inputs):
        target = mc_channels(channels)
        print(f"[mc {i + 1}/{len(mc_inputs)}] ({'+'.join(target)}) {root_path}")
        rows = compute_event_rows(root_path, target, cfg, True, args.max_events)
        if rows is not None:
            write_skim(rows, skim_output_name(args.outdir, stem, root_path))


def cmd_analyze(args: argparse.Namespace) -> None:
    """
    Step 2: read skim ROOTs, normalise, fill counts, make plots + JSON.

    Each MC skim is expected to be ONE merged file per sample (hadd of that sample's
    per-file skims): its genTotalSumw histogram then holds the sample-total SUMW, and
    the xsec is looked up from the file name. Final MC weight per event is
    genweight * xsec * lumi / SUMW. Data skims are unweighted.
    """
    cfg = load_config(args.config)
    lumi = args.lumi if args.lumi is not None else float(_require(cfg, "lumi"))
    edges = np.array(
        args.recoil_bins if args.recoil_bins else DEFAULT_RECOIL_BINS, dtype=np.float64
    )
    n_bins = len(edges) - 1
    channels = _resolve_channels(args.channel)
    xsecs = load_xsections(args.xsection_json)

    data_skims = [p for pat in args.data_skims for p in (sorted(glob.glob(pat)) or [pat])]
    mc_skims = [p for pat in args.mc_skims for p in (sorted(glob.glob(pat)) or [pat])]

    data_counts = {ch: BinCounts(n_bins) for ch in channels}
    mc_counts = {ch: BinCounts(n_bins) for ch in channels}

    print(f"Analyze: {len(data_skims)} data + {len(mc_skims)} MC skims")
    for p in data_skims:
        rows = read_skim(p)
        if rows is not None:
            accumulate_rows(rows, rows["genweight"], channels, edges, data_counts)
    for p in mc_skims:
        rows = read_skim(p)
        if rows is None:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        xsec = find_xsec(stem, xsecs)
        if xsec is None:
            print(f"WARNING: no xsection for skim '{stem}' - using norm 1.0")
        norm = norm_factor(rows["sumw"], xsec, lumi)
        print(f"[mc] {stem}: xsec={xsec} sumw={rows['sumw']:.4g} "
              f"count={rows['count']:.4g} norm={norm:.4g}")
        accumulate_rows(rows, rows["genweight"] * norm, channels, edges, mc_counts)

    make_plots_and_json(data_counts, mc_counts, channels, edges,
                        args.outdir, lumi, args.com, args.label, args.json_out)


def cmd_run(args: argparse.Namespace) -> None:
    """One-shot: NanoAOD -> plots + JSON in a single pass (no skim files)."""
    cfg = load_config(args.config)
    lumi = args.lumi if args.lumi is not None else float(_require(cfg, "lumi"))
    edges = np.array(
        args.recoil_bins if args.recoil_bins else DEFAULT_RECOIL_BINS, dtype=np.float64
    )
    n_bins = len(edges) - 1
    channels = _resolve_channels(args.channel)
    xsecs = load_xsections(args.xsection_json)

    data_inputs = resolve_inputs(args.data_files)
    mc_inputs = resolve_inputs(args.mc_files)
    if args.max_files is not None:
        data_inputs = data_inputs[: args.max_files]
        mc_inputs = mc_inputs[: args.max_files]

    data_counts = {ch: BinCounts(n_bins) for ch in channels}
    mc_counts = {ch: BinCounts(n_bins) for ch in channels}

    print(f"Run: {len(data_inputs)} data + {len(mc_inputs)} MC files")
    for i, (root_path, stem) in enumerate(data_inputs):
        print(f"[data {i + 1}/{len(data_inputs)}] {root_path}")
        accumulate_file(root_path, stem, channels, cfg, edges, data_counts,
                        is_mc=False, xsecs=xsecs, lumi=lumi,
                        max_events=args.max_events)
    for i, (root_path, stem) in enumerate(mc_inputs):
        target = mc_channels(channels)
        print(f"[mc {i + 1}/{len(mc_inputs)}] ({'+'.join(target)}) {root_path}")
        accumulate_file(root_path, stem, target, cfg, edges, mc_counts,
                        is_mc=True, xsecs=xsecs, lumi=lumi,
                        max_events=args.max_events)

    make_plots_and_json(data_counts, mc_counts, channels, edges,
                        args.outdir, lumi, args.com, args.label, args.json_out)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Year YAML, e.g. configs/2024.yaml")
    p.add_argument("--channel", choices=["wmn", "zmm", "both"], default="both")
    p.add_argument("--lumi", type=float, default=None,
                   help="Luminosity fb^-1 (default: cfg 'lumi')")
    p.add_argument("--xsection-json",
                   default="data/cross-section/xsection_background_run3.json")
    p.add_argument("--max-events", type=int, default=None,
                   help="entry_stop per file (fast test slices)")
    p.add_argument("--max-files", type=int, default=None,
                   help="cap number of input files (fast test slices)")


def _add_analysis(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recoil-bins", type=float, nargs="+", default=None,
                   help="Recoil bin edges [GeV] (default: physics-sensible 0-1200)")
    p.add_argument("--outdir", default="outputs/plots/hlt_sf")
    p.add_argument("--com", type=float, default=13.6)
    p.add_argument("--label", default="Internal")
    p.add_argument(
        "--json-out",
        default="data/corrections/2024/Run3-Summer24-NanoAODv15_metHlt.json.gz",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MET trigger efficiency and data/MC scale factor (tag-and-probe)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # skim: NanoAOD -> per-file skim ROOTs
    ps = sub.add_parser("skim", help="Step 1: preselect NanoAOD -> skim ROOTs")
    _add_common(ps)
    ps.add_argument("--data-files", nargs="+", default=None,
                    help="Muon-PD ROOT globs or samplelist .txt paths")
    ps.add_argument("--mc-files", nargs="+", default=None,
                    help="DY / W->munu ROOT globs or samplelist .txt paths")
    ps.add_argument("--outdir", required=True,
                    help="Directory for skim ROOTs "
                         "(<outdir>/<txtstem>_<rootid>.root)")
    ps.set_defaults(func=cmd_skim)

    # analyze: skim ROOTs -> plots + JSON
    pa = sub.add_parser("analyze", help="Step 2: skim ROOTs -> plots + JSON")
    _add_common(pa)
    _add_analysis(pa)
    pa.add_argument("--data-skims", nargs="+", required=True,
                    help="Skim ROOT globs for data")
    pa.add_argument("--mc-skims", nargs="+", required=True,
                    help="Skim ROOT globs for MC")
    pa.set_defaults(func=cmd_analyze)

    # run: single-pass NanoAOD -> plots + JSON
    pr = sub.add_parser("run", help="One-shot: NanoAOD -> plots + JSON (no skims)")
    _add_common(pr)
    _add_analysis(pr)
    pr.add_argument("--data-files", nargs="+", required=True,
                    help="Muon-PD ROOT globs or samplelist .txt paths")
    pr.add_argument("--mc-files", nargs="+", required=True,
                    help="DY / W->munu ROOT globs or samplelist .txt paths")
    pr.set_defaults(func=cmd_run)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
