#!/usr/bin/env python3
"""
B-tag (UParT, 2024) MC-truth tagging-efficiency SLIMMER.

Usage:
    python scripts/btag_efficiency_slimmer.py \
        --config configs/2024.yaml \
        --input /path/to/one_nanoaod_file.root \
        --output outputs/btageff/2024/sample_0.root \
        [--max-events 500000]

    # Then, once all per-file outputs exist, run stage 2:
    #   python scripts/btag_efficiency_maker.py \
    #       --inputs "outputs/btageff/2024/*.root" \
    #       --output outputs/btageff/2024/heavyflavor_efficiency_maps.root
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Tuple

import awkward as ak
import numpy as np
import uproot
import yaml

# Only branches under these prefixes (plus SCALAR_BRANCHES) are read from the
# file — same optimisation as scripts/met_trigger_efficiency.py::load_events.
BRANCH_PREFIXES = (
    "Muon_", "Electron_", "Jet_", "PuppiMET_", "PFMET_", "MET_",
)
SCALAR_BRANCHES = ("event",)

# (eta, pt) binning for the efficiency-map TH2Ds — matches the real Run2
# production efficiency maps exactly (bTagEffs_{2016,2017,2018}.root: 6 signed
# eta bins, 9 pt bins), not the old bEff_Slimmer.py script's own finer
# hardcoded constants (those were never actually used for the production
# output on disk). Using the same binning for both slimmer (fill) and maker
# (output) means the maker's rebin step is a trivial 1:1 pass-through, no
# risk of the coarse-source/fine-output mismatch that silently dropped bins
# when the two grids differed.
ETA_EDGES = np.array([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])  # 6 bins, signed
PT_EDGES = np.array([20., 50., 80., 120., 200., 300., 400., 500., 700., 1000.])  # 9 bins

FLAVOR_LABELS = {5: "b", 4: "c", 0: "light"}

# config key -> output-histogram-name suffix, one per b-tag working point.
WORKING_POINTS = {"score_loose": "lwp", "score": "mwp", "score_tight": "twp"}


# ---------------------------------------------------------------------------
# Object building + selection (inlined, adapted from
# scripts/met_trigger_efficiency.py; darkbottomline/objects.py conventions
# for config keys/thresholds)
# ---------------------------------------------------------------------------

def build_muon_collection(events: ak.Array) -> ak.Array:
    """Zip the muon collection from flat branches (kinematics + ID + iso)."""
    fields = {
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
    }
    return ak.zip(fields)


def build_electron_collection(events: ak.Array) -> ak.Array:
    """Zip the electron collection from flat branches (kinematics only)."""
    return ak.zip({
        "pt": events["Electron_pt"],
        "eta": events["Electron_eta"],
        "phi": events["Electron_phi"],
    })


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
    """Per-electron boolean mask. cutBased ID + mvaIso WP, with the ECAL
    barrel-endcap gap (1.4442<|eta|<1.566) excluded."""
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


def pass_triggers(events: ak.Array, trigger_paths: List[str]) -> ak.Array:
    """OR of the given HLT paths (per-event bool). Empty/absent -> all True/skip."""
    if not trigger_paths:
        return ak.ones_like(events["event"], dtype=bool)
    present = [t for t in trigger_paths if t in events.fields]
    mask = ak.zeros_like(events["event"], dtype=bool)
    for t in present:
        mask = mask | events[t]
    return mask


def pass_met_filters(events: ak.Array, filter_names: List[str]) -> ak.Array:
    """AND of the given MET noise filters (per-event bool). Absent filters skipped."""
    if not filter_names:
        return ak.ones_like(events["event"], dtype=bool)
    mask = ak.ones_like(events["event"], dtype=bool)
    for f in filter_names:
        if f in events.fields:
            mask = mask & events[f]
    return mask


def _met_pt_phi(events: ak.Array) -> Tuple[ak.Array, ak.Array]:
    """MET (pt, phi) with the PuppiMET -> PFMET -> MET fallback."""
    def _get(*cands: str) -> ak.Array:
        for v in cands:
            if v in events.fields:
                return events[v]
        raise KeyError(f"No MET branch found among {cands}")
    return (_get("PuppiMET_pt", "PFMET_pt", "MET_pt"),
            _get("PuppiMET_phi", "PFMET_phi", "MET_phi"))


def calculate_recoil(events: ak.Array, leptons: ak.Array) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hadronic recoil (pt, phi): |-(MET_vec + sum lepton pT_vec)|, matching the
    framework's recoil convention (darkbottomline/objects.py::calculate_recoil).
    `leptons` must be a single jagged collection with .pt/.phi (muons+electrons
    concatenated by the caller).
    """
    met_pt, met_phi = _met_pt_phi(events)
    lep_px = ak.sum(leptons.pt * np.cos(leptons.phi), axis=1)
    lep_py = ak.sum(leptons.pt * np.sin(leptons.phi), axis=1)
    rx = -(met_pt * np.cos(met_phi) + lep_px)
    ry = -(met_pt * np.sin(met_phi) + lep_py)
    return (ak.to_numpy(np.sqrt(rx ** 2 + ry ** 2)),
            ak.to_numpy(np.arctan2(ry, rx)))


def _clean_jet_mask(jet_eta: ak.Array, jet_phi: ak.Array, leptons: ak.Array,
                    dr_min: float) -> ak.Array:
    """Per-jet bool: True if the jet is >= dr_min from every lepton in `leptons`
    (jets in events with zero leptons pass automatically)."""
    jeta, leta = ak.unzip(ak.cartesian([jet_eta, leptons.eta], nested=True))
    jphi, lphi = ak.unzip(ak.cartesian([jet_phi, leptons.phi], nested=True))
    dphi = np.arctan2(np.sin(jphi - lphi), np.cos(jphi - lphi))
    dr = np.sqrt((jeta - leta) ** 2 + dphi ** 2)
    min_dr = ak.fill_none(ak.min(dr, axis=-1), np.inf)
    return min_dr > dr_min


def jet_selection_mask(events: ak.Array, jet_cfg: Dict[str, Any],
                       leptons: ak.Array, dr_min: float) -> ak.Array:
    """Per-jet bool: pt > pt_min, |eta| < eta_max, cleaned (dR) against
    muons+electrons. No jet-ID cut (puId branch absent in NanoAOD v12+, same
    as darkbottomline/objects.py::select_jets)."""
    pt_ok = events["Jet_pt"] > jet_cfg["pt_min"]
    eta_ok = abs(events["Jet_eta"]) < jet_cfg["eta_max"]
    clean_ok = _clean_jet_mask(events["Jet_eta"], events["Jet_phi"], leptons, dr_min)
    return pt_ok & eta_ok & clean_ok


# ---------------------------------------------------------------------------
# Config + I/O
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a year YAML. Missing keys are surfaced loudly downstream (no defaults)."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_events(tfile: "uproot.ReadOnlyDirectory", max_events) -> ak.Array:
    """Read only the flat NanoAOD branches this script needs."""
    tree = tfile["Events"]
    available = set(tree.keys())
    wanted = [b for b in SCALAR_BRANCHES if b in available]
    wanted += [b for b in available if any(b.startswith(p) for p in BRANCH_PREFIXES)]
    wanted += [b for b in available if b.startswith("HLT_")]
    wanted += [b for b in available if b.startswith("Flag_")]
    return tree.arrays(wanted, entry_stop=max_events)


# ---------------------------------------------------------------------------
# Main fill logic
# ---------------------------------------------------------------------------

def fill_slimmer_histograms(events: ak.Array, config: Dict[str, Any]
                            ) -> Dict[str, np.ndarray]:
    """Build the 21 (eta, pt) 2D COUNT histograms (per flavor b/c/light: 1
    denominator + pass/fail x 3 working points L/M/T) as plain numpy arrays,
    from one file's worth of preselected+cleaned jets.
    Additive counts only — no division into an efficiency ratio (see module
    docstring: that's a separate stage-2 step, run once after hadd-ing every
    per-file output together)."""
    muon_cfg = config["objects"]["muons"]
    ele_cfg = config["objects"]["electrons"]
    jet_cfg = config["objects"]["jets"]
    btag_cfg = config["btagging"]
    dr_jet = config["cleaning"]["dr_jet"]

    if "Jet_hadronFlavour" not in events.fields:
        raise KeyError(
            "Jet_hadronFlavour absent — this script is MC-only "
            "(efficiency maps are not meaningful on data)."
        )

    # Noise filters (event-level QC; loud is unnecessary here, just skip absent).
    filter_mask = pass_met_filters(events, config.get("noise_filters", []))

    # Loose lepton veto/cleaning collections.
    muons = build_muon_collection(events)
    loose_mu_mask = select_muons(events, muon_cfg, wp="loose")
    loose_muons = muons[loose_mu_mask]

    electrons = build_electron_collection(events)
    loose_ele_mask = select_electrons(events, ele_cfg, wp="loose")
    loose_electrons = electrons[loose_ele_mask]

    loose_leptons = ak.concatenate([loose_muons, loose_electrons], axis=1)

    # Loose "any CR/SR-like" preselection: MET>100 or recoil>200. This is a
    # denominator-enrichment gate (matches the old Run2 script's intent), not
    # the analysis's actual region cuts.
    met_pt, _ = _met_pt_phi(events)
    recoil_pt, _ = calculate_recoil(events, loose_leptons)
    preselected = filter_mask & ((ak.to_numpy(met_pt) > 100.0) | (recoil_pt > 200.0))

    # Cleaned jets passing the preselection gate.
    jet_ok = jet_selection_mask(events, jet_cfg, loose_leptons, dr_jet)
    jet_ok = jet_ok & preselected

    jet_eta = ak.flatten(events["Jet_eta"][jet_ok])
    jet_pt = ak.flatten(events["Jet_pt"][jet_ok])
    jet_flavor = ak.flatten(events["Jet_hadronFlavour"][jet_ok])
    jet_score = ak.flatten(events[btag_cfg["branch"]][jet_ok])

    jet_eta = ak.to_numpy(jet_eta)
    jet_pt = ak.to_numpy(jet_pt)
    jet_flavor = ak.to_numpy(jet_flavor)
    jet_score = ak.to_numpy(jet_score)

    hists: Dict[str, np.ndarray] = {}
    for flav_val, label in FLAVOR_LABELS.items():
        if flav_val == 0:
            in_flavor = (jet_flavor != 4) & (jet_flavor != 5)
        else:
            in_flavor = jet_flavor == flav_val

        # Denominator (all truth jets of this flavor passing preselection) is
        # shared across working points; only pass/fail differ per WP.
        den, _, _ = np.histogram2d(
            jet_eta[in_flavor], jet_pt[in_flavor], bins=[ETA_EDGES, PT_EDGES]
        )
        hists[f"hist_{label}_efficiency_denominator"] = den

        for wp_key, wp_suffix in WORKING_POINTS.items():
            is_tagged = jet_score > btag_cfg[wp_key]
            pass_, _, _ = np.histogram2d(
                jet_eta[in_flavor & is_tagged], jet_pt[in_flavor & is_tagged],
                bins=[ETA_EDGES, PT_EDGES],
            )
            fail = den - pass_

            hists[f"hist_{label}_efficiency_pass_{wp_suffix}"] = pass_
            hists[f"hist_{label}_efficiency_fail_{wp_suffix}"] = fail

    return hists


def write_histograms(output_path: str, hists: Dict[str, np.ndarray]) -> None:
    with uproot.recreate(output_path) as rf:
        for name, values in hists.items():
            rf[name] = (values, ETA_EDGES, PT_EDGES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Year YAML, e.g. configs/2024.yaml")
    parser.add_argument("--input", required=True, help="Single NanoAOD ROOT file (local or xrootd)")
    parser.add_argument("--output", required=True, help="Output ROOT file path")
    parser.add_argument("--max-events", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    with uproot.open(args.input) as tfile:
        events = load_events(tfile, args.max_events)

    print(f"Loaded {len(events)} events from {args.input}")
    hists = fill_slimmer_histograms(events, config)

    for name, values in hists.items():
        print(f"  {name}: sum={values.sum():.1f}")

    import os
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_histograms(args.output, hists)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
