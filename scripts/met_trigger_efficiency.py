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

    # Step 1 - skim: preselect NanoAOD, any number of input files per stem merge
    #   into one skim ROOT: <outdir>/<stem>.root, one per samplelist/file stem
    #   (data and MC alike) — the stem is also the xsec lookup key used in analyze
    #   for MC.
    python scripts/met_trigger_efficiency.py skim --config configs/2024.yaml \
        --data-files data/samplelist/2024/Muon*.txt \
        --mc-files data/samplelist/2024/DYto2L-2Jets_*.txt \
                   data/samplelist/2024/WtoLNu-2Jets_*.txt \
        --outdir outputs/skims/2024

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

# This script is SELF-CONTAINED: all object building, selection, trigger and recoil
# logic is defined below (no darkbottomline import). NanoAOD is read via uproot as a
# plain awkward record array. Only the flat branches we actually use are read per file.
BRANCH_PREFIXES = (
    "Muon_", "Electron_", "Jet_", "PuppiMET_", "PFMET_", "MET_",
)
SCALAR_BRANCHES = ("event", "run", "luminosityBlock", "genWeight")


# ---------------------------------------------------------------------------
# Object building + selection (inlined, previously from darkbottomline)
# ---------------------------------------------------------------------------

def build_muon_collection(events: ak.Array) -> ak.Array:
    """Zip the muon collection from flat branches (kinematics + ID + iso)."""
    fields = {
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
        "looseId": events["Muon_looseId"],
        "tightId": events["Muon_tightId"],
        "pfIsoId": events["Muon_pfIsoId"],
    }
    if "Muon_charge" in events.fields:
        fields["charge"] = events["Muon_charge"]
    if "Muon_mass" in events.fields:
        fields["mass"] = events["Muon_mass"]
    return ak.zip(fields)


def select_muons(events: ak.Array, config: Dict[str, Any], wp: str = "loose") -> ak.Array:
    """Per-muon boolean mask. wp='loose' uses pt_min_loose + looseId, 'tight' uses
    pt_min + tightId. Isolation: pfIsoId >= iso_wp_{loose,tight}."""
    pt_min = config["pt_min_loose"] if wp == "loose" else config["pt_min"]
    pt_mask = events["Muon_pt"] > pt_min
    eta_mask = abs(events["Muon_eta"]) < config["eta_max"]
    iso_wp = config["iso_wp_loose"] if wp == "loose" else config["iso_wp_tight"]
    iso_mask = events["Muon_pfIsoId"] >= iso_wp
    id_mask = (events["Muon_looseId"] == 1) if wp == "loose" else (events["Muon_tightId"] == 1)
    return pt_mask & eta_mask & id_mask & iso_mask


def select_electrons(events: ak.Array, config: Dict[str, Any], wp: str = "loose") -> ak.Array:
    """Per-electron boolean mask (used for the loose-electron veto). cutBased ID +
    mvaIso WP, with the ECAL barrel-endcap gap (1.4442<|eta|<1.566) excluded."""
    ele_eta = events["Electron_eta"]
    pt_min = config["pt_min_loose"] if wp == "loose" else config["pt_min"]
    pt_mask = events["Electron_pt"] > pt_min
    eta_mask = abs(ele_eta) < config["eta_max"]
    in_gap = (abs(ele_eta) > 1.4442) & (abs(ele_eta) < 1.566)
    gap_veto_mask = ~in_gap
    if wp == "loose":
        id_wp = config["id_wp_loose"]
        iso_mask = events["Electron_mvaIso_WP90"] == 1
    else:
        id_wp = config["id_wp_tight"]
        iso_mask = events["Electron_mvaIso_WP80"] == 1
    id_mask = events["Electron_cutBased"] >= id_wp
    return pt_mask & eta_mask & gap_veto_mask & id_mask & iso_mask


# ---------------------------------------------------------------------------
# Triggers + MET filters (inlined; STRICT on the analysis MET/reference paths)
# ---------------------------------------------------------------------------

def pass_triggers(events: ak.Array, trigger_paths: List[str],
                  require_present: bool = False) -> ak.Array:
    """
    OR of the given HLT paths (per-event bool). Empty list -> all True.

    require_present=True: EVERY listed path must exist in the file, else raise. This
    guards the MET / reference triggers: if a file is missing those branches, the old
    silent behaviour returned all-False, so its events entered the denominator but
    never the numerator -> the efficiency was dragged DOWN (worst in the sparse
    high-recoil tail). We now fail loudly instead of biasing the turn-on.
    """
    if not trigger_paths:
        return ak.ones_like(events["event"], dtype=bool)
    present = [t for t in trigger_paths if t in events.fields]
    if require_present:
        missing = [t for t in trigger_paths if t not in events.fields]
        if missing:
            raise KeyError(
                f"required trigger branch(es) absent from file: {missing} "
                f"(present HLT-like fields would silently bias the efficiency)"
            )
    mask = ak.zeros_like(events["event"], dtype=bool)
    for t in present:
        mask = mask | events[t]
    return mask


def pass_met_filters(events: ak.Array, filter_names: List[str]) -> ak.Array:
    """AND of the given MET noise filters (per-event bool). Empty list -> all True.
    Absent filters are skipped (a missing recommended filter is not a hard error)."""
    if not filter_names:
        return ak.ones_like(events["event"], dtype=bool)
    mask = ak.ones_like(events["event"], dtype=bool)
    for f in filter_names:
        if f in events.fields:
            mask = mask & events[f]
    return mask


def _met_pt_phi(events: ak.Array) -> Tuple[ak.Array, ak.Array]:
    """MET (pt, phi) with the PuppiMET -> PFMET -> MET fallback. Loud if none present."""
    def _get(*cands: str) -> ak.Array:
        for v in cands:
            if v in events.fields:
                return events[v]
        raise KeyError(f"No MET branch found among {cands}")
    return (_get("PuppiMET_pt", "PFMET_pt", "MET_pt"),
            _get("PuppiMET_phi", "PFMET_phi", "MET_phi"))


def calculate_recoil(events: ak.Array, muons: ak.Array
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hadronic recoil (U, phi) with U = | MET_vec + sum(mu pT_vec) | (muons added back,
    since the HLT is PFMETNoMu). MET is PuppiMET (fallback PFMET/MET), via px/py.
    """
    met_pt, met_phi = _met_pt_phi(events)
    lep_px = ak.sum(muons.pt * np.cos(muons.phi), axis=1)
    lep_py = ak.sum(muons.pt * np.sin(muons.phi), axis=1)
    ux = met_pt * np.cos(met_phi) + lep_px
    uy = met_pt * np.sin(met_phi) + lep_py
    return (ak.to_numpy(np.sqrt(ux ** 2 + uy ** 2)),
            ak.to_numpy(np.arctan2(uy, ux)))

# Skim schema: ONE row per selected event. wmu/zmu are orthogonal booleans (an event
# is a W->munu OR a Z->mumu candidate, never both). Object scalars are leading-object
# kinematics (SENTINEL when absent, e.g. lep2 in a W event). genweight is kept for MC
# normalisation in analyze. Order preserved for readability.
SENTINEL = -999.0
SKIM_BRANCHES: Dict[str, Any] = {
    "recoil": np.float64, "recoilPhi": np.float64,
    "metPt": np.float64,
    "lep1Pt": np.float64, "lep1Eta": np.float64, "lep1Phi": np.float64,
    "lep2Pt": np.float64, "lep2Eta": np.float64, "lep2Phi": np.float64,
    "Jet1Pt": np.float64, "Jet1Eta": np.float64, "Jet1Phi": np.float64,
    "muTrigPass": np.int32, "metTrigPass": np.int32,
    "wmu": np.int32, "zmu": np.int32,
    "genweight": np.float64,
}

CHANNELS = ("wmn", "zmm")
DEFAULT_RECOIL_BINS = [
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150,
    160, 170, 180, 190, 200,
    220, 240, 260, 280, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000,
    1100, 1200
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
# Golden-JSON (certified lumi) mask — data only
# ---------------------------------------------------------------------------

def load_golden_json(path: Optional[str]) -> Optional[Dict[int, List[List[int]]]]:
    """
    Load a CMS golden JSON {run: [[lo,hi], ...]} of certified lumisection ranges.
    Returns {int(run): [[lo,hi],...]} or None if path is None/missing (mask disabled).
    """
    if not path or not os.path.isfile(path):
        if path:
            print(f"WARNING: golden JSON not found: {path} — lumi mask disabled")
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(run): ranges for run, ranges in raw.items()}


def golden_mask(events: ak.Array,
                golden: Optional[Dict[int, List[List[int]]]]) -> np.ndarray:
    """
    Per-event boolean: True if (run, luminosityBlock) is in a certified range. Removes
    events from lumisections where the detector / triggers were not good — the proper
    fix for run-era data-quality dips (e.g. a MET-trigger-off period). All-True when no
    golden JSON (MC, or mask disabled).
    """
    n = len(events["event"])
    if golden is None or "run" not in events.fields or "luminosityBlock" not in events.fields:
        return np.ones(n, dtype=bool)
    run = ak.to_numpy(events["run"]).astype(np.int64)
    lumi = ak.to_numpy(events["luminosityBlock"]).astype(np.int64)
    mask = np.zeros(n, dtype=bool)
    # Group by unique run so each run's ranges are checked once.
    for r in np.unique(run):
        ranges = golden.get(int(r))
        if not ranges:
            continue
        in_run = run == r
        lr = lumi[in_run]
        good = np.zeros(lr.shape, dtype=bool)
        for lo, hi in ranges:
            good |= (lr >= lo) & (lr <= hi)
        mask[in_run] = good
    return mask


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


def _dataset_stem(root_path: str) -> str:
    """
    Derive a sample stem from a bare ROOT path (CLI-passed, not via .txt).

    CMS (xrootd or local) NanoAOD paths look like
    .../store/{data,mc}/<campaign>/<DATASET_NAME>/NANOAOD[SIM]/<version>/<numbered
    dir>/<hash>.root — DATASET_NAME is the component right before NANOAOD[SIM], the
    same token samplelist .txt stems are named after (so xsec lookup matches either
    way). Falls back to the file basename if that layout isn't found, so plain local
    files still get a usable (if per-file) stem.
    """
    parts = root_path.rstrip("/").split("/")
    for i, part in enumerate(parts):
        if part in ("NANOAOD", "NANOAODSIM") and i > 0:
            return parts[i - 1]
    return os.path.splitext(os.path.basename(root_path))[0]


def resolve_inputs(patterns: List[str]) -> List[Tuple[str, str]]:
    """
    Expand CLI globs / .txt samplelists into a flat list of (root_path, sample_stem).

    sample_stem is the samplelist basename (used for the MC xsec lookup) when the
    input is a .txt; for bare ROOT paths it's the CMS dataset name parsed out of the
    path (see _dataset_stem) so multiple files of the same sample share a stem and
    merge into one skim, instead of each bare file getting its own stem.
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
                resolved.append((m, _dataset_stem(m)))
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

def _clean_jet_mask(jet_eta: ak.Array, jet_phi: ak.Array, muons: ak.Array,
                    dr_min: float) -> ak.Array:
    """
    Per-jet bool: True if the jet is >= dr_min from every muon in `muons` (jets in
    events with zero muons pass automatically). Guards against a muon's HCAL/ECAL
    deposit being reconstructed as a fake AK4 jet and faking the leading-jet cut.
    """
    jeta, meta = ak.unzip(ak.cartesian([jet_eta, muons.eta], nested=True))
    jphi, mphi = ak.unzip(ak.cartesian([jet_phi, muons.phi], nested=True))
    dphi = np.arctan2(np.sin(jphi - mphi), np.cos(jphi - mphi))
    dr = np.sqrt((jeta - meta) ** 2 + dphi ** 2)
    min_dr = ak.fill_none(ak.min(dr, axis=-1), np.inf)
    return min_dr > dr_min


def _jet_selection_mask(events: ak.Array, jet_cfg: Dict[str, Any],
                        muons: ak.Array) -> ak.Array:
    """
    Per-jet bool: |eta| < eta_max, hadron-EF jet ID (neHEF < neHEF_max, chHEF >
    chHEF_min), and muon-jet ΔR cleaning (dr_min). Mirrors the old ntuple's
    THINjetNHadEF<0.8 / THINjetCHadEF>0.1 / DeltaR<0.5 selection; keys are optional
    (default to those old values) so an unmodified config still works. No jet-ID bit
    beyond that, as the puId branch is absent in NanoAOD v12+ (see
    objects.py::select_jets).
    """
    eta_ok = abs(events["Jet_eta"]) < jet_cfg["eta_max"]
    id_ok = ak.ones_like(events["Jet_eta"], dtype=bool)
    if "Jet_neHEF" in events.fields:
        id_ok = id_ok & (events["Jet_neHEF"] < jet_cfg.get("neHEF_max", 0.8))
    if "Jet_chHEF" in events.fields:
        id_ok = id_ok & (events["Jet_chHEF"] > jet_cfg.get("chHEF_min", 0.1))
    clean_ok = _clean_jet_mask(events["Jet_eta"], events["Jet_phi"], muons,
                              jet_cfg.get("muon_dr_min", 0.5))
    return eta_ok & id_ok & clean_ok


def _leading_jet_pt(events: ak.Array, jet_cfg: Dict[str, Any],
                    muons: ak.Array) -> ak.Array:
    """Max pt among jets passing `_jet_selection_mask` (0 when none)."""
    if "Jet_pt" not in events.fields or "Jet_eta" not in events.fields:
        return ak.zeros_like(events["event"], dtype=np.float64)
    pass_mask = _jet_selection_mask(events, jet_cfg, muons)
    jet_pt = ak.where(pass_mask, events["Jet_pt"], 0.0)
    return ak.fill_none(ak.max(jet_pt, axis=1), 0.0)


def _nth(values: ak.Array, n: int) -> np.ndarray:
    """The n-th (0-based) entry per event of a jagged array; SENTINEL when absent."""
    padded = ak.pad_none(values, n + 1, axis=1)
    return ak.to_numpy(ak.fill_none(padded[:, n], SENTINEL))


def _leading_jet_obj(events: ak.Array, jet_cfg: Dict[str, Any], muons: ak.Array
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-event leading (max-pt) jet (pt, eta, phi) passing `_jet_selection_mask`;
    SENTINEL when none pass."""
    n = len(events["event"])
    if "Jet_pt" not in events.fields or "Jet_eta" not in events.fields:
        s = np.full(n, SENTINEL)
        return s, s, s
    pass_mask = _jet_selection_mask(events, jet_cfg, muons)
    jpt = events["Jet_pt"][pass_mask]
    jeta = events["Jet_eta"][pass_mask]
    jphi = events["Jet_phi"][pass_mask] if "Jet_phi" in events.fields else jpt * 0.0
    order = ak.argsort(jpt, axis=1, ascending=False)
    jpt, jeta, jphi = jpt[order], jeta[order], jphi[order]
    return _nth(jpt, 0), _nth(jeta, 0), _nth(jphi, 0)


def build_event_rows(
    events: ak.Array,
    cfg: Dict[str, Any],
    channels: List[str],
    golden: Optional[Dict[int, List[List[int]]]] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate every event for the W->munu and Z->mumu preselections and return ONE
    row per selected event (an event selected in neither channel is dropped). wmu/zmu
    are orthogonal booleans set by the muon multiplicity, so an event is at most one of
    them. Kinematic branches follow the MetTrigSkims schema (SKIM_BRANCHES).

    The two channels require the leading (pt-highest) loose muon to be tight (the
    tag); the subleading muon (the probe, when present) only needs to be loose,
    tight or not:
      wmu: leading muon tight, no second loose muon (lead_is_tight, n_loose==1)
      zmu: leading muon tight, exactly one extra loose muon (lead_is_tight, n_loose==2)
    Both also require: certified lumi (golden JSON, data only), reference IsoMu
    trigger, MET filters, leading jet pt>50, and a loose-electron veto.
    """
    muon_cfg = _require(cfg, "objects", "muons")
    ele_cfg = _require(cfg, "objects", "electrons")
    jet_cfg = _require(cfg, "objects", "jets")
    ref_triggers = _require(cfg, "triggers", "SingleMuon")
    met_triggers = _require(cfg, "triggers", "MET")
    noise_filters = _require(cfg, "noise_filters")

    # Triggers (both REQUIRED present -> loud error if a file lacks the branches, so a
    # missing HLT never silently biases the efficiency) + MET filters.
    mu_trig = ak.to_numpy(pass_triggers(events, ref_triggers, require_present=True))
    met_trig = ak.to_numpy(pass_triggers(events, met_triggers, require_present=True))
    filter_mask = ak.to_numpy(pass_met_filters(events, noise_filters))

    # Muons, ordered by pt so lep1/lep2 are leading/subleading.
    muons = build_muon_collection(events)
    order = ak.argsort(muons.pt, axis=1, ascending=False)
    muons = muons[order]
    tight_mask = select_muons(events, muon_cfg, wp="tight")[order]
    loose_mask = select_muons(events, muon_cfg, wp="loose")[order]
    n_loose = ak.to_numpy(ak.sum(loose_mask, axis=1))
    loose_muons = muons[loose_mask]  # candidates that define lep1/lep2
    # Tightness of the leading (pt-highest) loose muon: the tag leg must be tight,
    # regardless of the subleading (probe) leg's tightness.
    lead_is_tight = ak.to_numpy(ak.fill_none(ak.firsts(tight_mask[loose_mask]), False))

    # Common event-level cuts.
    lead_jet_mask = ak.to_numpy(_leading_jet_pt(events, jet_cfg, loose_muons) > 50.0)
    loose_ele = select_electrons(events, ele_cfg, wp="loose")
    ele_veto = ak.to_numpy(~ak.any(loose_ele, axis=1))
    lumi_mask = golden_mask(events, golden)  # certified lumi (data); all-True for MC
    common = (lumi_mask & mu_trig.astype(bool) & filter_mask.astype(bool)
              & lead_jet_mask & ele_veto)

    wmu = common & lead_is_tight & (n_loose == 1)
    zmu = common & lead_is_tight & (n_loose == 2)
    sel = wmu | zmu  # one row per selected event; wmu/zmu are disjoint by n_loose
    if not sel.any():
        return {k: np.array([], dtype=dt) for k, dt in SKIM_BRANCHES.items()}

    # Recoil uses the loose muons added back (both channels: all selected muons).
    recoil, recoil_phi = calculate_recoil(events, loose_muons)
    met_pt, _ = _met_pt_phi(events)
    met_pt = ak.to_numpy(met_pt)

    lep1_pt, lep1_eta, lep1_phi = _nth(loose_muons.pt, 0), _nth(loose_muons.eta, 0), _nth(loose_muons.phi, 0)
    lep2_pt, lep2_eta, lep2_phi = _nth(loose_muons.pt, 1), _nth(loose_muons.eta, 1), _nth(loose_muons.phi, 1)
    j_pt, j_eta, j_phi = _leading_jet_obj(events, jet_cfg, loose_muons)

    if "genWeight" in events.fields:
        genweight = np.sign(ak.to_numpy(events["genWeight"])).astype(np.float64)
    else:
        genweight = np.ones(len(events["event"]), dtype=np.float64)

    full = {
        "recoil": recoil, "recoilPhi": recoil_phi,
        "metPt": met_pt,
        "lep1Pt": lep1_pt, "lep1Eta": lep1_eta, "lep1Phi": lep1_phi,
        "lep2Pt": lep2_pt, "lep2Eta": lep2_eta, "lep2Phi": lep2_phi,
        "Jet1Pt": j_pt, "Jet1Eta": j_eta, "Jet1Phi": j_phi,
        "muTrigPass": mu_trig.astype(np.int32), "metTrigPass": met_trig.astype(np.int32),
        "wmu": wmu.astype(np.int32), "zmu": zmu.astype(np.int32),
        "genweight": genweight,
    }
    return {k: np.asarray(v)[sel].astype(SKIM_BRANCHES[k]) for k, v in full.items()}


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


def _read_one_file(root_path: str, channels: List[str], cfg: Dict[str, Any],
                   is_mc: bool, max_events: Optional[int],
                   golden: Optional[Dict[int, List[List[int]]]] = None) -> Dict[str, Any]:
    """Open + process a single file (one attempt). Raises on any XRootD/read error."""
    with uproot.open(root_path) as tfile:
        events = load_events(tfile, max_events)
        rows = build_event_rows(events, cfg, channels, golden=golden)
        # File-total sum of sign(genWeight) + raw event count over ALL events. Data has
        # no genWeight -> both equal the raw NanoAOD event count (all-ones data weight).
        if is_mc:
            file_sumw, file_count = read_gensumw(tfile)
        else:
            file_count = float(tfile["Events"].num_entries)
            file_sumw = file_count
    rows = dict(rows)
    rows["file_sumw"] = file_sumw
    rows["file_count"] = file_count
    return rows


def compute_event_rows(
    root_path: str,
    channels: List[str],
    cfg: Dict[str, Any],
    is_mc: bool,
    max_events: Optional[int],
    n_retries: int = 3,
    golden: Optional[Dict[int, List[List[int]]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load one ROOT file, build the one-row-per-selected-event skim (SKIM_BRANCHES),
    and attach the file's total generator weight (file_sumw) + event count.

    XRootD reads are flaky, so each file is attempted up to n_retries times (with a
    short backoff) before being skipped — a transient "Invalid operation" / timeout on
    one attempt usually succeeds on the next. Returns the row dict, or None only after
    all attempts fail.

    NO xsec/lumi/sumw normalisation here — the skim stores raw sign(genWeight) and the
    file-total sum-of-sign(genWeight); analyze applies xsec*lumi/SUMW once.
    """
    import time
    last_exc: Optional[Exception] = None
    for attempt in range(1, n_retries + 1):
        try:
            return _read_one_file(root_path, channels, cfg, is_mc, max_events, golden)
        except Exception as exc:  # noqa: BLE001 - retry transient XRootD errors
            last_exc = exc
            if attempt < n_retries:
                print(f"WARNING: read attempt {attempt}/{n_retries} failed for "
                      f"{root_path}: {exc} — retrying")
                time.sleep(5 * attempt)  # linear backoff: 5s, 10s, ...
    print(f"WARNING: skipping file after {n_retries} attempts {root_path}: {last_exc}")
    return None


def accumulate_rows(rows: Dict[str, Any], weight: np.ndarray, channels: List[str],
                    edges: np.ndarray, counts: Dict[str, BinCounts]) -> None:
    """
    Fill per-channel BinCounts from the one-row-per-event skim. A row belongs to 'wmn'
    when wmu==1 and 'zmm' when zmu==1 (orthogonal). numerator = metTrigPass. The
    caller supplies the final per-event weight (genweight*xsec*lumi/SUMW for MC, ones
    for data).
    """
    flag = {"wmn": rows["wmu"], "zmm": rows["zmu"]}
    for channel in channels:
        sel = flag[channel].astype(bool)
        if not sel.any():
            continue
        counts[channel].fill(
            rows["recoil"][sel], rows["metTrigPass"][sel].astype(np.float64),
            weight[sel], edges,
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
    golden: Optional[Dict[int, List[List[int]]]] = None,
) -> None:
    """
    Single-pass (one-shot 'run' mode): load a NanoAOD file, normalise, fill counts.

    Here one file = one sample instance, so the file's own genEventSumw IS the sample
    SUMW (single-file normalisation). For the multi-file two-step path use skim +
    analyze, which sums SUMW across a sample's files.
    """
    rows = compute_event_rows(root_path, channels, cfg, is_mc, max_events,
                              golden=golden)
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


def merged_skim_output_name(outdir: str, stem: str) -> str:
    """<outdir>/<stem>.root — one merged skim per sample stem (or 'data')."""
    return os.path.join(outdir, f"{stem}.root")


def _merge_rows(rows_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Concatenate per-file row dicts: SKIM_BRANCHES arrays concat, file_sumw/count sum."""
    out: Dict[str, Any] = {
        k: np.concatenate([r[k] for r in rows_list]) for k in SKIM_BRANCHES
    }
    out["file_sumw"] = sum(r["file_sumw"] for r in rows_list)
    out["file_count"] = sum(r["file_count"] for r in rows_list)
    return out


def write_skim(rows: Dict[str, Any], out_path: str) -> None:
    """
    Write the one-row-per-event skim (SKIM_BRANCHES) as a classic TTree, plus 1-bin
    TH1s genTotalSumw / genTotalCount (both hadd-sum, giving the sample-total SUMW when
    a sample's per-file skims are merged).
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {k: rows[k].astype(dt) for k, dt in SKIM_BRANCHES.items()}
    with uproot.recreate(out_path) as f:
        # Force a classic TTree, NOT an RNTuple: uproot 5.x's `f[name] = dict` now
        # defaults to RNTuple, which older ROOT viewers and `hadd` cannot read (the
        # tree looks empty and merges fail). mktree + extend writes a TTree.
        f.mktree(SKIM_TREE, {k: v.dtype for k, v in payload.items()})
        if len(payload["recoil"]) > 0:
            f[SKIM_TREE].extend(payload)
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
    Read a skim ROOT into the SKIM_BRANCHES arrays plus `sumw` (integral of the
    genTotalSumw histogram = sample-total when the skim is a hadd of a sample's files)
    and `count`. None on unreadable/missing tree.
    """
    try:
        with uproot.open(path) as f:
            # Integrity guard against TRUNCATED skims (a condor job that died mid-write
            # leaves a partial ROOT: the tree metadata reads, arrays() returns garbage
            # high-recoil rows, and the plateau "dips"). A clean skim always has the
            # tree + BOTH 1-bin histos (they are written AFTER the tree), so a missing
            # histogram is a reliable truncation signal. Force a full key list too,
            # which raises on a corrupt directory record.
            keys = set(k.split(";")[0] for k in f.keys())
            missing = [n for n in (SKIM_TREE, SUMW_HIST, COUNT_HIST) if n not in keys]
            if missing:
                print(f"WARNING: skipping truncated/incomplete skim {path} "
                      f"(missing {missing})")
                return None
            arr = f[SKIM_TREE].arrays(library="np")
            sumw = float(np.sum(f[SUMW_HIST].values()))
            count = float(np.sum(f[COUNT_HIST].values()))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: skipping unreadable skim {path}: {exc}")
        return None
    out = {k: arr[k] for k in SKIM_BRANCHES if k in arr}
    out["sumw"] = sumw
    out["count"] = count
    return out


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


def _cms_label(label: str, data: bool, lumi: Optional[float], com: float,
               ax: "plt.Axes") -> None:
    """hep.cms.label across mplhep versions: newer wants text=, older (LCG_109)
    wants label=. Try text= first, fall back to label=."""
    try:
        hep.cms.label(text=label, data=data, lumi=lumi, com=com, ax=ax, fontsize=18)
    except TypeError:
        hep.cms.label(label=label, data=data, lumi=lumi, com=com, ax=ax, fontsize=18)


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
    _cms_label(label, is_data, lumi, com, ax)
    _save(fig, outpath)


def plot_efficiency_all(
    edges: np.ndarray,
    data_eff: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    mc_eff: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    outpath: str,
    lumi: Optional[float],
    com: float,
    label: str,
) -> None:
    """All four efficiency curves on one axes, each a DISTINCT colour+marker:
    data W/Z and MC W/Z."""
    ctr, half = _bin_centres(edges)
    fig, ax = plt.subplots(figsize=(9, 8))
    ch_leg = {"wmn": r"$W\to\mu\nu$", "zmm": r"$Z\to\mu\mu$"}
    # One (colour, marker) per curve = per (source, channel).
    curve_style = {
        ("data", "wmn"): ("#3f90da", "o"),   # blue circle
        ("data", "zmm"): ("#bd1f01", "s"),   # red square
        ("mc",   "wmn"): ("#e76300", "^"),   # orange triangle
        ("mc",   "zmm"): ("#832db6", "D"),   # purple diamond
    }
    for src, effs in (("data", data_eff), ("mc", mc_eff)):
        for ch, (eff, elo, ehi) in effs.items():
            col, mk = curve_style.get((src, ch), ("C0", "o"))
            leg = f"{ch_leg.get(ch, ch)} ({'data' if src == 'data' else 'MC'})"
            ax.errorbar(ctr, eff, yerr=[elo, ehi], xerr=half, fmt=mk, color=col,
                        capsize=2, markersize=6, label=leg)
    ax.set_xlabel(r"Hadronic recoil $U$ [GeV]", fontsize=18)
    ax.set_ylabel("Trigger efficiency", fontsize=18)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlim(edges[0], edges[-1])
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.legend(loc="lower right", fontsize=14, ncol=2)
    _cms_label(label, True, lumi, com, ax)
    _save(fig, outpath)


def plot_scale_factor(
    edges: np.ndarray,
    sf_nom: np.ndarray,
    stat_lo: np.ndarray,
    stat_hi: np.ndarray,
    syst: np.ndarray,
    outpath: str,
    lumi: Optional[float],
    com: float,
    label: str,
) -> None:
    """
    Data/MC scale factor vs recoil: single W->munu series.
    Inner error bar = stat only; outer error bar = stat (+) syst in quadrature,
    where syst = |SF_wmn - SF_zmm| per bin (channel-difference systematic).
    """
    ctr, half = _bin_centres(edges)
    total_lo = np.sqrt(stat_lo ** 2 + syst ** 2)
    total_hi = np.sqrt(stat_hi ** 2 + syst ** 2)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.errorbar(
        ctr, sf_nom, yerr=[total_lo, total_hi], xerr=half,
        fmt="none", ecolor="lightgray", elinewidth=6, capsize=0,
        label="stat + syst",
        zorder=1,
    )
    ax.errorbar(
        ctr, sf_nom, yerr=[stat_lo, stat_hi],
        fmt="o", color="black", capsize=2, markersize=6,
        label=r"$W\to\mu\nu$ (stat)", zorder=2,
    )
    ax.set_xlabel(r"Hadronic recoil $U$ [GeV]", fontsize=18)
    ax.set_ylabel("Data / MC scale factor", fontsize=18)
    # Wide enough for the turn-on region where data/MC can disagree by >20%.
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(250.0, edges[-1])
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.legend(loc="lower right", fontsize=16)
    _cms_label(label, True, lumi, com, ax)
    _save(fig, outpath)


def write_eff_sf_root(
    edges: np.ndarray,
    data_eff: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    mc_eff: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    sf_nom: np.ndarray,
    sf_stat_lo: np.ndarray,
    sf_stat_hi: np.ndarray,
    outpath: str,
) -> None:
    """
    Write the 5 curves behind the _all_eff and _sf plots (data-wmn, data-zmm,
    mc-wmn, mc-zmm efficiency + the nominal W->munu data/MC SF) into one ROOT
    TTree, one row per recoil bin.

    uproot 5.x cannot serialize TGraphAsymmErrors (only plain TGraph, no
    errors), so the asymmetric Clopper-Pearson errors are instead stored as
    plain val/err_lo/err_hi branches, one row per bin - equivalent information,
    trivially convertible to TGraphAsymmErrors on the ROOT/PyROOT side.
    """
    ctr, _ = _bin_centres(edges)
    d_wmn = data_eff.get("wmn", (np.zeros_like(ctr),) * 3)
    d_zmm = data_eff.get("zmm", (np.zeros_like(ctr),) * 3)
    m_wmn = mc_eff.get("wmn", (np.zeros_like(ctr),) * 3)
    m_zmm = mc_eff.get("zmm", (np.zeros_like(ctr),) * 3)
    payload = {
        "recoil_lo": edges[:-1], "recoil_hi": edges[1:], "recoil_ctr": ctr,
        "data_wmn_eff": d_wmn[0], "data_wmn_errlo": d_wmn[1], "data_wmn_errhi": d_wmn[2],
        "data_zmm_eff": d_zmm[0], "data_zmm_errlo": d_zmm[1], "data_zmm_errhi": d_zmm[2],
        "mc_wmn_eff": m_wmn[0], "mc_wmn_errlo": m_wmn[1], "mc_wmn_errhi": m_wmn[2],
        "mc_zmm_eff": m_zmm[0], "mc_zmm_errlo": m_zmm[1], "mc_zmm_errhi": m_zmm[2],
        "sf": sf_nom, "sf_errlo": sf_stat_lo, "sf_errhi": sf_stat_hi,
    }
    payload = {k: np.asarray(v, dtype=np.float64) for k, v in payload.items()}
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with uproot.recreate(outpath) as f:
        # mktree + extend forces a classic TTree (see write_skim): uproot 5.x's
        # `f[name] = dict` now defaults to RNTuple, which ROOT/PyROOT readers
        # downstream of this file may not yet support.
        f.mktree("eff_sf", {k: v.dtype for k, v in payload.items()})
        f["eff_sf"].extend(payload)
    print(f"Saved: {outpath}")


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
    # All four curves (data + MC, both channels) on one axes.
    plot_efficiency_all(
        edges, data_eff, mc_eff,
        os.path.join(outdir, "HLT_PFMETNoMu_120To140_IDTight_all_eff.png"),
        lumi, com, label,
    )

    sf_per_channel: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sf_values: Dict[str, np.ndarray] = {}
    for ch in channels:
        ed, edlo, edhi = data_eff[ch]
        em, emlo, emhi = mc_eff[ch]
        sf, slo, shi = scale_factor(ed, edlo, edhi, em, emlo, emhi)
        sf_per_channel[ch] = (sf, slo, shi)
        sf_values[ch] = sf

    # Nominal SF + systematic from the W->munu vs Z->mumu channel difference.
    nominal_ch = "wmn" if "wmn" in channels else channels[0]
    sf_nom, stat_lo, stat_hi = sf_per_channel[nominal_ch]
    if "wmn" in sf_values and "zmm" in sf_values:
        # Channel difference as the systematic — but only where BOTH channels have a
        # valid (non-zero) SF. In sparse high-U bins one channel can be empty (SF=0),
        # which would fake a huge |wmn-zmm|; there, fall back to the statistical error.
        both_valid = (sf_values["wmn"] > 0) & (sf_values["zmm"] > 0)
        syst = np.where(
            both_valid, np.abs(sf_values["wmn"] - sf_values["zmm"]), stat_lo
        )
    else:
        syst = stat_lo

    plot_scale_factor(
        edges, sf_nom, stat_lo, stat_hi, syst,
        os.path.join(outdir, "HLT_PFMETNoMu_120To140_IDTight_sf.png"),
        lumi, com, label,
    )

    # The 5 curves behind _all_eff.png + _sf.png (data-wmn/zmm eff, mc-wmn/zmm eff,
    # nominal SF), as one ROOT TTree alongside the PNG/PDF plots.
    write_eff_sf_root(
        edges, data_eff, mc_eff, sf_nom, stat_lo, stat_hi,
        os.path.join(outdir, "HLT_PFMETNoMu_120To140_IDTight_eff_sf.root"),
    )

    write_correction(
        build_correction_set(edges, sf_nom, sf_nom + syst, sf_nom - syst),
        json_out,
    )


def _resolve_channels(channel: str) -> List[str]:
    return list(CHANNELS) if channel == "both" else [channel]


def cmd_skim(args: argparse.Namespace) -> None:
    """
    Step 1: preselect NanoAOD, write one merged skim ROOT per stem: data and MC
    inputs are each grouped by their samplelist/file stem (the same stem is the
    xsec lookup key analyze uses for MC) -> <outdir>/<stem>.root. The skim stores
    raw per-event genWeight + the file's total genEventSumw only (NO xsec/lumi/sumw
    normalisation — that is applied in analyze, after summing SUMW over the sample).
    """
    cfg = load_config(args.config)
    channels = _resolve_channels(args.channel)
    # Golden JSON (certified lumi) applied to DATA only, at skim time. Path from CLI
    # override or configs/<year>.yaml data.golden_json; None disables the mask.
    golden_path = args.golden_json or cfg.get("data", {}).get("golden_json")
    golden = load_golden_json(golden_path)
    print(f"Golden JSON: {golden_path or 'DISABLED'} "
          f"({len(golden)} runs)" if golden else f"Golden JSON: DISABLED")

    data_inputs = resolve_inputs(args.data_files) if args.data_files else []
    mc_inputs = resolve_inputs(args.mc_files) if args.mc_files else []
    if args.max_files is not None:
        data_inputs = data_inputs[: args.max_files]
        mc_inputs = mc_inputs[: args.max_files]

    print(f"Skim: {len(data_inputs)} data + {len(mc_inputs)} MC files -> {args.outdir}")

    data_rows_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for i, (root_path, stem) in enumerate(data_inputs):
        print(f"[data {i + 1}/{len(data_inputs)}] {root_path}")
        rows = compute_event_rows(root_path, channels, cfg, False, args.max_events,
                                  golden=golden)
        if rows is not None:
            data_rows_by_stem.setdefault(stem, []).append(rows)
    for stem, rows_list in data_rows_by_stem.items():
        write_skim(_merge_rows(rows_list), merged_skim_output_name(args.outdir, stem))

    mc_rows_by_stem: Dict[str, List[Dict[str, Any]]] = {}
    for i, (root_path, stem) in enumerate(mc_inputs):
        target = mc_channels(channels)
        print(f"[mc {i + 1}/{len(mc_inputs)}] ({'+'.join(target)}) {root_path}")
        rows = compute_event_rows(root_path, target, cfg, True, args.max_events)
        if rows is not None:
            mc_rows_by_stem.setdefault(stem, []).append(rows)
    for stem, rows_list in mc_rows_by_stem.items():
        write_skim(_merge_rows(rows_list), merged_skim_output_name(args.outdir, stem))


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
    golden_path = args.golden_json or cfg.get("data", {}).get("golden_json")
    golden = load_golden_json(golden_path)

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
                        max_events=args.max_events, golden=golden)
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
    p.add_argument("--golden-json", default=None,
                   help="Golden JSON of certified lumisections (data only). "
                        "Default: configs/<year>.yaml data.golden_json.")
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
