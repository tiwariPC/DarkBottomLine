"""
Event-level variable computation for DarkBottomLine output trees.

All physics variables saved to the output ROOT TTree are defined here.
To add a new variable: implement a function and register it in compute_event_variables().

Return value of compute_event_variables(): flat dict[str, np.ndarray | ak.Array]
  - 1D numpy arrays  → scalar branches (one value per event)
  - 2D ak.Arrays     → jagged branches (vector<float> in ROOT)
  - SENTINEL (-9.0)  → variable undefined for this event (e.g. < N jets required)
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import awkward as ak
from typing import Dict, Any, Optional, Tuple
from darkbottomline.objects import SENTINEL
from darkbottomline.corrections import jes_sources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar(arr, default=0.0):
    """ak.Array or numpy → 1D numpy float32, filling None with default."""
    return ak.to_numpy(ak.fill_none(ak.values_astype(arr, np.float32), np.float32(default)))


def _lead(jagged, idx, default=SENTINEL):
    """Extract scalar at position idx from a jagged array; fill missing with default."""
    pad = ak.pad_none(jagged, idx + 1, clip=True)
    return ak.to_numpy(ak.fill_none(pad[:, idx], np.float32(default))).astype(np.float32)


def _dphi(phi1, phi2):
    """Delta phi wrapped to [0, pi]."""
    raw = np.abs(phi1 - phi2)
    return np.where(raw > np.pi, 2 * np.pi - raw, raw).astype(np.float32)


def _4vec(pt, eta, phi, m):
    """Build Cartesian 4-vector from (pt, eta, phi, mass).

    eta is clipped before sinh() so sentinel-filled inputs (e.g. eta=SENTINEL
    for a missing jet) can't overflow float — callers mask the result out via
    the has_Nj flags regardless, this just keeps the intermediate finite.
    """
    eta_safe = np.clip(eta, -20.0, 20.0)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta_safe)
    e  = np.sqrt(px**2 + py**2 + pz**2 + np.maximum(m, 0.0)**2)
    return px, py, pz, e


def _dijet(px1, py1, pz1, e1, px2, py2, pz2, e2):
    """Compute dijet 4-vector quantities: (mass, pt, eta, phi)."""
    px = px1 + px2; py = py1 + py2
    pz = pz1 + pz2; e  = e1  + e2
    pt  = np.sqrt(px**2 + py**2)
    m   = np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0.0))
    eta = np.where(pt > 0, np.arcsinh(pz / np.maximum(pt, 1e-6)), 0.0)
    phi = np.arctan2(py, px)
    return m, pt, eta, phi


# ---------------------------------------------------------------------------
# Variable groups
# ---------------------------------------------------------------------------

def _met_variables(events: ak.Array, objects: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """MET kinematics. The Type-1 MET recomputed in build_objects
    (objects["met_pt"/"met_phi"], same MET as Recoil) when available;
    otherwise PuppiMET, falling back to PFMET then MET."""
    def _get(*candidates, default=SENTINEL):
        for v in candidates:
            if v in events.fields:
                return ak.to_numpy(events[v])
        return np.full(len(events), default, dtype=np.float32)

    objects = objects or {}
    met_pt, met_phi = objects.get('met_pt'), objects.get('met_phi')
    return {
        'MET_pt':  np.asarray(ak.to_numpy(met_pt)) if met_pt is not None else _get('PuppiMET_pt',  'PFMET_pt',  'MET_pt'),
        'MET_phi': np.asarray(ak.to_numpy(met_phi)) if met_phi is not None else _get('PuppiMET_phi', 'PFMET_phi', 'MET_phi'),
        'MET_significance': _get('PFMET_significance', 'MET_significance'),
    }


def _recoil_variables(objects: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Recoil pT and phi (precomputed in build_objects), plus JES (11-source,
    see darkbottomline.corrections.jes_sources)/JER shifted variants. The
    year-suffixed source names (and therefore branch names) depend on
    config["year"] — must match objects.py's recoil_jes_variants label
    format exactly (see that module's _jes_label helper)."""
    recoil     = objects.get('recoil')
    recoil_phi = objects.get('recoil_phi')
    n = len(recoil) if recoil is not None else 0
    _zeros = ak.Array(np.zeros(n, dtype=np.float32))
    central = recoil if recoil is not None else _zeros
    out = {
        'Recoil':          _scalar(central),
        'RecoilPhi':       _scalar(recoil_phi if recoil_phi is not None else _zeros),
        'Recoil_JERUp':   _scalar(objects.get('recoil_JERUp',   central)),
        'Recoil_JERDown': _scalar(objects.get('recoil_JERDown', central)),
    }
    # jes_sources() already gives the real-year-suffixed name (e.g.
    # "EC2_2024") — matches objects.py's _jes_label exactly, no further
    # munging needed.
    for source in jes_sources(config.get('year', '')):
        for direction in ("Up", "Down"):
            obj_key = f"recoil_JEC{source}{direction}"
            branch_key = f"Recoil_JEC{source}{direction}"
            out[branch_key] = _scalar(objects.get(obj_key, central))
    return out


def _multiplicity_variables(objects: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Object counts per event."""
    def _num(key):
        arr = objects.get(key, ak.Array([]))
        try:
            return ak.to_numpy(ak.num(arr, axis=1)).astype(np.int32)
        except Exception:
            return np.zeros(0, dtype=np.int32)

    jets = objects.get('jets', ak.Array([]))
    try:
        lead2_flavour = ak.pad_none(jets.hadronFlavour, 2, clip=True)
        b_flavor_count = ak.to_numpy(
            ak.sum(lead2_flavour == 5, axis=1)
        ).astype(np.int32)
    except Exception:
        n = ak.to_numpy(ak.num(jets, axis=1)).size if hasattr(jets, '__len__') else 0
        b_flavor_count = np.zeros(n, dtype=np.int32)

    def _scalar_int(key):
        arr = objects.get(key)
        if arr is None:
            return None
        try:
            return ak.to_numpy(ak.fill_none(ak.values_astype(arr, np.int32), np.int32(0)))
        except Exception:
            return None

    out = {
        'njets':          _num('jets'),
        'n_bjets':        _num('bjets'),
        'n_muons':        _num('muons'),
        'n_electrons':    _num('electrons'),
        'n_tight_muons':     _num('tight_muons'),
        'n_tight_electrons': _num('tight_electrons'),
        'n_taus':         _num('taus'),
        'b_flavor_count': b_flavor_count,
    }
    # Z-candidate lepton counts and mass — required for CR_Zee/CR_Zmumu region cuts
    for key, branch in (('n_z_muons', 'n_z_muons'), ('n_z_electrons', 'n_z_electrons')):
        v = _scalar_int(key)
        if v is not None:
            out[branch] = v
    # Z dilepton mass: muon pair if NmuonsZ==2 else electron pair
    mll_mu = objects.get('mll_mu')
    mll_el = objects.get('mll_el')
    nzm = objects.get('n_z_muons')
    nze = objects.get('n_z_electrons')
    if mll_mu is not None and mll_el is not None and nzm is not None and nze is not None:
        try:
            mll = ak.where(
                ak.fill_none(ak.values_astype(nzm, np.int32), 0) == 2, mll_mu,
                ak.where(ak.fill_none(ak.values_astype(nze, np.int32), 0) == 2, mll_el,
                         np.float32(SENTINEL))
            )
            out['mll'] = ak.to_numpy(ak.fill_none(ak.values_astype(mll, np.float32), np.float32(SENTINEL)))
        except Exception:
            pass
    # Z pT: muon-candidate pT if NmuonsZ==2 else electron-candidate pT.
    # Same candidate leptons as mll (computed in build_z_candidates).
    z_pt_mu = objects.get('z_pt_mu')
    z_pt_el = objects.get('z_pt_el')
    if z_pt_mu is not None and z_pt_el is not None and nzm is not None and nze is not None:
        try:
            zpt = ak.where(
                ak.fill_none(ak.values_astype(nzm, np.int32), 0) == 2, z_pt_mu,
                ak.where(ak.fill_none(ak.values_astype(nze, np.int32), 0) == 2, z_pt_el,
                         np.float32(SENTINEL))
            )
            out['Zpt'] = ak.to_numpy(ak.fill_none(ak.values_astype(zpt, np.float32), np.float32(SENTINEL)))
        except Exception:
            pass
    # W transverse mass: muon-channel MT if a tight muon present, else electron-channel MT.
    # Orthogonal per-event pick (not config/trigger-driven) — mirrors the mll combine above.
    mt_mu = objects.get('mt_mu')
    mt_el = objects.get('mt_el')
    if mt_mu is not None and mt_el is not None:
        try:
            n_tight_mu = out['n_tight_muons']
            n_tight_el = out['n_tight_electrons']
            mt = ak.where(
                n_tight_mu >= 1, mt_mu,
                ak.where(n_tight_el >= 1, mt_el, np.float32(SENTINEL))
            )
            out['mt'] = ak.to_numpy(ak.fill_none(ak.values_astype(mt, np.float32), np.float32(SENTINEL)))
        except Exception:
            pass
    return out


def _jet_lead_variables(objects: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Leading 3 jet kinematics + btag scores.

    Output btag branch names are fixed (Jet{1,2,3}BTagScore) regardless of
    which algorithm (PNet/deepJet/UParT/...) produced the score, so
    downstream consumers don't need to track the algorithm choice per year.
    """
    jets = objects.get('jets', ak.Array([]))
    btag = jets.btagScore if hasattr(jets, 'btagScore') else ak.zeros_like(jets.pt)

    return {
        'Jet1Pt':              _lead(jets.pt,  0),
        'Jet1Eta':             _lead(jets.eta, 0),
        'Jet1Phi':             _lead(jets.phi, 0),
        'Jet1BTagScore':       _lead(btag,     0, default=SENTINEL),
        'Jet2Pt':              _lead(jets.pt,  1),
        'Jet2Eta':             _lead(jets.eta, 1),
        'Jet2Phi':             _lead(jets.phi, 1),
        'Jet2BTagScore':       _lead(btag,     1, default=SENTINEL),
        'Jet3Pt':              _lead(jets.pt,  2),
        'Jet3Eta':             _lead(jets.eta, 2),
        'Jet3Phi':             _lead(jets.phi, 2),
        'Jet3BTagScore':       _lead(btag,     2, default=SENTINEL),
        'JetHT':               _scalar(ak.sum(jets.pt, axis=1)),
    }


def _positional_btag_flags(objects: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-event positional b-tag flags + composite category conditions.

    Compares the leading/subleading jet btagScore to the configured WP so the
    b-tag requirement is positional (jet index 0/1), matching the Run2 SR/CR
    definition. Events with < 1 (< 2) jets get flag 0.

    Composite flags merge the b-count and positional requirement into a single
    region cut (one cutflow step, matching the Run2 b-tag bin):
      Bjet1bCond = (n_bjets == 1) & (leading jet b-tagged)
      Bjet2bCond = (n_bjets == 2) & (leading & subleading jets b-tagged)
    """
    jets = objects.get('jets', ak.Array([]))
    score = config["btagging"]["score"]
    btag = jets.btagScore if hasattr(jets, 'btagScore') else ak.zeros_like(jets.pt)
    # _lead fills missing (fewer jets) with SENTINEL (-9) → below WP → flag 0.
    lead_score    = _lead(btag, 0, default=SENTINEL)
    sublead_score = _lead(btag, 1, default=SENTINEL)
    is_lead    = lead_score    > score
    is_sublead = sublead_score > score

    bjets = objects.get('bjets', ak.Array([]))
    try:
        n_bjets = ak.to_numpy(ak.num(bjets, axis=1)).astype(np.int32)
    except Exception:
        n_bjets = np.zeros(len(is_lead), dtype=np.int32)

    return {
        'is_lead_bjet':    is_lead.astype(np.int32),
        'is_sublead_bjet': is_sublead.astype(np.int32),
        'Bjet1bCond': ((n_bjets == 1) & is_lead).astype(np.int32),
        'Bjet2bCond': ((n_bjets == 2) & is_lead & is_sublead).astype(np.int32),
    }


def _jet_composite_variables(
    jet_lead: Dict[str, np.ndarray],
    met_vars: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Derived dijet + jet-MET variables. Requires jet_lead and met_vars already computed."""
    j1pt  = jet_lead['Jet1Pt'];  j2pt  = jet_lead['Jet2Pt'];  j3pt  = jet_lead['Jet3Pt']
    j1eta = jet_lead['Jet1Eta']; j2eta = jet_lead['Jet2Eta']; j3eta = jet_lead['Jet3Eta']
    j1phi = jet_lead['Jet1Phi']; j2phi = jet_lead['Jet2Phi']; j3phi = jet_lead['Jet3Phi']
    met_pt  = met_vars['MET_pt'].astype(np.float32)
    met_phi = met_vars['MET_phi'].astype(np.float32)

    has_j1 = j1pt > 0
    has_2j = has_j1 & (j2pt > 0)
    has_3j = has_2j & (j3pt > 0)

    out: Dict[str, np.ndarray] = {}

    # Jet1/MET ratios
    out['ratioJet1PtMET']   = np.where(has_j1 & (met_pt > 0), j1pt / np.maximum(met_pt, 1e-6), np.float32(SENTINEL)).astype(np.float32)
    out['ratioPtJet21'] = np.where(has_2j,                 j2pt / np.maximum(j1pt,  1e-6), np.float32(SENTINEL)).astype(np.float32)

    # Dijet angular
    out['dEtaJet12'] = np.where(has_2j, np.abs(j1eta - j2eta),  np.float32(SENTINEL)).astype(np.float32)
    out['dPhiJet12'] = np.where(has_2j, _dphi(j1phi, j2phi),    np.float32(SENTINEL)).astype(np.float32)
    out['dRJet12']   = np.where(has_2j, np.sqrt(out['dEtaJet12']**2 + out['dPhiJet12']**2), np.float32(SENTINEL)).astype(np.float32)
    out['dPhiJet13'] = np.where(has_3j, _dphi(j1phi, j3phi),    np.float32(SENTINEL)).astype(np.float32)

    # Dijet 4-vector masses + kinematics (use jet mass=0 sentinel → treat as massless when -9)
    j1m = np.where(j1pt > 0, jet_lead.get('_j1m', np.zeros_like(j1pt)), 0.0).astype(np.float32)
    j2m = np.where(j2pt > 0, jet_lead.get('_j2m', np.zeros_like(j2pt)), 0.0).astype(np.float32)
    j3m = np.where(j3pt > 0, jet_lead.get('_j3m', np.zeros_like(j3pt)), 0.0).astype(np.float32)

    px1, py1, pz1, e1 = _4vec(j1pt, j1eta, j1phi, j1m)
    px2, py2, pz2, e2 = _4vec(j2pt, j2eta, j2phi, j2m)
    px3, py3, pz3, e3 = _4vec(j3pt, j3eta, j3phi, j3m)

    m12, pt12, eta12, phi12 = _dijet(px1, py1, pz1, e1, px2, py2, pz2, e2)
    out['M_Jet1Jet2']   = np.where(has_2j, m12,   np.float32(SENTINEL)).astype(np.float32)
    out['pT_Jet1Jet2']  = np.where(has_2j, pt12,  np.float32(SENTINEL)).astype(np.float32)
    out['eta_Jet1Jet2'] = np.where(has_2j, eta12, np.float32(SENTINEL)).astype(np.float32)
    out['phi_Jet1Jet2'] = np.where(has_2j, phi12, np.float32(SENTINEL)).astype(np.float32)

    m13, _, _, _ = _dijet(px1, py1, pz1, e1, px3, py3, pz3, e3)
    out['M_Jet1Jet3'] = np.where(has_3j, m13, np.float32(SENTINEL)).astype(np.float32)

    # min dPhi(jet, MET) over all jets — needs jagged jets, handled separately via _dphi_jet_met
    return out


def _dphi_jet_met(objects: Dict[str, Any], met_phi: np.ndarray) -> np.ndarray:
    """Min |delta phi| between any jet and MET, wrapped to [0, pi]."""
    jets = objects.get('jets', ak.Array([]))
    raw = np.abs(ak.to_numpy(ak.fill_none(
        ak.min(np.abs(jets.phi - met_phi.astype(np.float32)), axis=1),
        np.float32(SENTINEL),
    )))
    return np.where(raw >= 0, np.minimum(raw, np.float32(2 * np.pi) - raw), raw).astype(np.float32)


_SIGNAL_XSEC_JSON = Path(__file__).resolve().parent.parent / "data" / "cross-section" / "xsection_signal.json"


@lru_cache(maxsize=1)
def _signal_grid_branches() -> Tuple[str, ...]:
    """GenModel_<point> branch name for every signal grid point in
    xsection_signal.json ({model: {point: xsec}}; keys starting with '_' are
    comments)."""
    try:
        with open(_SIGNAL_XSEC_JSON) as f:
            table = json.load(f)
    except (OSError, ValueError) as e:
        logging.warning(f"Could not read signal grid points from {_SIGNAL_XSEC_JSON}: {e}")
        return ()
    return tuple(sorted(
        f"GenModel_{point}"
        for points in table.values() if isinstance(points, dict)
        for point in points if not point.startswith("_")
    ))


def _genmodel_variables(events: ak.Array) -> Dict[str, np.ndarray]:
    """GenModel_* grid-point flags (signal only; no-op for bkg/data).

    A NanoAOD signal file only has GenModel_ branches for the grid points it
    contains, so outputs would differ in branch set between files and hadd
    silently drops the events of files whose branch set differs. Every signal
    output therefore gets a flag for every grid point in xsection_signal.json,
    0 for points absent from this file.
    """
    out: Dict[str, np.ndarray] = {}
    present = [f for f in events.fields if f.startswith('GenModel_')]
    if not present:
        return out
    for field in present:
        try:
            out[field] = ak.to_numpy(events[field]).astype(np.int8)
        except Exception:
            pass
    for branch in _signal_grid_branches():
        if branch not in out:
            out[branch] = np.zeros(len(events), dtype=np.int8)
    return out


# Raw-level inputs needed to re-derive/audit JEC, jet-ID and Type-1 MET from
# EVENTSELECTION.root without rerunning from raw NanoAOD. CorrT1METJet_* is
# NanoAOD's separate low-pT MET-specific jet collection (not part of the
# analysis's "jets" object, so it doesn't go through _jagged_variables'
# objects-dict path above) and fixedGridRhoFastjetAll is an event-level
# scalar (pileup density input to the L1FastJet correction) — both passed
# through directly from raw `events` here, same style as
# _genmodel_variables above.
_CORRT1METJET_FIELDS = (
    "rawPt", "eta", "phi", "area", "muonSubtrFactor", "muonSubtrDeltaPhi", "EmEF",
)


def _raw_jec_inputs_variables(events: ak.Array) -> Dict[str, Any]:
    """CorrT1METJet_* passthrough (jagged) + fixedGridRhoFastjetAll (scalar).
    Rho's branch name varies by NanoAOD version (flat
    "fixedGridRhoFastjetAll" vs NanoAODv15's nested "Rho_fixedGridRhoFastjetAll")
    — same candidate list as CorrectionManager._get_rho, saved under one
    consistent output name regardless of which one the input file has."""
    out: Dict[str, Any] = {}
    for field in _CORRT1METJET_FIELDS:
        branch = f"CorrT1METJet_{field}"
        if branch in events.fields:
            out[f"corrt1metjet_{field.lower()}"] = events[branch]
    for rho_branch in ("fixedGridRhoFastjetAll", "Rho_fixedGridRhoFastjetAll", "rho"):
        if rho_branch in events.fields:
            try:
                out["fixedGridRhoFastjetAll"] = ak.to_numpy(events[rho_branch]).astype(np.float32)
            except Exception:
                pass
            break
    return out


def _jagged_variables(objects: Dict[str, Any]) -> Dict[str, ak.Array]:
    """Full jagged (per-event vector) branches for all objects."""
    out: Dict[str, ak.Array] = {}
    _jagged_spec = [
        ('muons',     'pt',           'muon_pt'),
        ('muons',     'eta',          'muon_eta'),
        ('muons',     'phi',          'muon_phi'),
        ('electrons', 'pt',           'electron_pt'),
        ('electrons', 'eta',          'electron_eta'),
        ('electrons', 'phi',          'electron_phi'),
        # ('taus',      'pt',           'tau_pt'),
        # ('taus',      'eta',          'tau_eta'),
        # ('taus',      'phi',          'tau_phi'),
        ('jets',      'pt',           'jet_pt'),
        ('jets',      'eta',          'jet_eta'),
        ('jets',      'phi',          'jet_phi'),
        ('jets',      'btagScore',    'jet_btag'),
        # Raw-level jet inputs (undefined on data-era files without e.g.
        # Jet_muonSubtrFactor, or years where build_jet_collection didn't
        # carry a given field through — _jagged_spec's "skip if missing"
        # loop below handles that transparently) — saved so JEC/jet-ID/
        # Type-1 MET can be re-derived or audited from EVENTSELECTION.root
        # directly, without rerunning from raw NanoAOD. See
        # objects.py:build_jet_collection's raw_field loop, which is the
        # single source of truth for which of these actually get zipped in.
        ('jets',      'rawFactor',          'jet_rawFactor'),
        ('jets',      'area',               'jet_area'),
        ('jets',      'muonSubtrFactor',    'jet_muonSubtrFactor'),
        ('jets',      'muonSubtrDeltaPhi',  'jet_muonSubtrDeltaPhi'),
        ('jets',      'chHEF',              'jet_chHEF'),
        ('jets',      'neHEF',              'jet_neHEF'),
        ('jets',      'chEmEF',             'jet_chEmEF'),
        ('jets',      'neEmEF',             'jet_neEmEF'),
        ('jets',      'muEF',               'jet_muEF'),
        ('jets',      'chMultiplicity',     'jet_chMultiplicity'),
        ('jets',      'neMultiplicity',     'jet_neMultiplicity'),
        ('bjets',     'pt',           'bjet_pt'),
        ('bjets',     'eta',          'bjet_eta'),
        ('bjets',     'phi',          'bjet_phi'),
        # ('fatjets',   'pt',           'fatjet_pt'),
        # ('fatjets',   'eta',          'fatjet_eta'),
        # ('fatjets',   'phi',          'fatjet_phi'),
        # ('fatjets',   'mass',         'fatjet_mass'),
    ]
    for obj_key, field, branch in _jagged_spec:
        obj = objects.get(obj_key)
        if obj is None:
            continue
        try:
            if hasattr(obj, 'fields') and field in obj.fields:
                out[branch] = obj[field]
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Empty-schema helper (for zero-event ROOT files)
# ---------------------------------------------------------------------------

# Scalar branches: name → numpy dtype
_SCALAR_BRANCHES: Dict[str, Any] = {
    'event': np.int64, 'run': np.int64, 'luminosityBlock': np.int64,
    'Pileup_nTrueInt': np.float32, 'Pileup_nPU': np.int32,
    'PV_npvsGood': np.int32, 'PV_npvs': np.int32,
    # Pileup density input to the L1FastJet JEC correction — saved so JEC
    # can be re-derived/audited from EVENTSELECTION.root without rerunning
    # from raw NanoAOD (see _raw_jec_inputs_variables).
    'fixedGridRhoFastjetAll': np.float32,
    'MET_pt': np.float32, 'MET_phi': np.float32, 'MET_significance': np.float32,
    'Recoil': np.float32, 'RecoilPhi': np.float32,
    'Recoil_JERUp': np.float32, 'Recoil_JERDown': np.float32,
    # Static schema (used for empty-event placeholder files, no runtime
    # config/year available here) — covers every year this framework has a
    # configs/<year>.yaml for, so whichever year a given run actually uses,
    # its Recoil_JEC<source>_<year>Up/Down branch names are already present.
    **{
        f"Recoil_JEC{src}{direction}": np.float32
        for _year in (2022, 2023, 2024, 2025, 2026)
        for src in jes_sources(_year) for direction in ("Up", "Down")
    },
    'costheta_star': np.float32,
    'njets': np.int32, 'n_bjets': np.int32, 'n_muons': np.int32,
    'n_electrons': np.int32, 'n_tight_muons': np.int32, 'n_tight_electrons': np.int32,
    'n_taus': np.int32, 'b_flavor_count': np.int32,
    'is_lead_bjet': np.int32, 'is_sublead_bjet': np.int32,
    'Bjet1bCond': np.int32, 'Bjet2bCond': np.int32,
    'n_z_muons': np.int32, 'n_z_electrons': np.int32, 'mll': np.float32, 'Zpt': np.float32,
    'mt': np.float32,
    'Jet1Pt': np.float32, 'Jet1Eta': np.float32, 'Jet1Phi': np.float32,
    'Jet2Pt': np.float32, 'Jet2Eta': np.float32, 'Jet2Phi': np.float32,
    'Jet3Pt': np.float32, 'Jet3Eta': np.float32, 'Jet3Phi': np.float32,
    'JetHT': np.float32,
    'ratioJet1PtMET': np.float32, 'ratioPtJet21': np.float32,
    'dEtaJet12': np.float32, 'dPhiJet12': np.float32, 'dRJet12': np.float32,
    'dPhiJet13': np.float32,
    'M_Jet1Jet2': np.float32, 'pT_Jet1Jet2': np.float32,
    'eta_Jet1Jet2': np.float32, 'phi_Jet1Jet2': np.float32,
    'M_Jet1Jet3': np.float32,
    'dPhi_jetMET': np.float32,
    'del_plus': np.float32, 'del_minus': np.float32,
    'muon_lep1_pt': np.float32, 'muon_lep1_phi': np.float32, 'muon_lep1_eta': np.float32,
    'muon_lep2_pt': np.float32, 'muon_lep2_phi': np.float32, 'muon_lep2_eta': np.float32,
    'electron_lep1_pt': np.float32, 'electron_lep1_phi': np.float32, 'electron_lep1_eta': np.float32,
    'electron_lep2_pt': np.float32, 'electron_lep2_phi': np.float32, 'electron_lep2_eta': np.float32,
    'full_event_weight': np.float32, 'weight_noPileup': np.float32,
    'weight_gen': np.float32,
    'weight_pu': np.float32, 'weight_btag': np.float32,
    'weight_muon': np.float32, 'weight_electron': np.float32,
    'weight_ele_hlt': np.float32, 'weight_met_hlt': np.float32,
    'weight_pdfUP': np.float32, 'weight_pdfDOWN': np.float32,
    'weight_scaleUP': np.float32, 'weight_scaleDOWN': np.float32,
    'pass_met_trigger': np.int32,
    'pass_ele_trigger': np.int32,
}

# Jagged branches: name → numpy dtype of elements
_JAGGED_BRANCHES: Dict[str, Any] = {
    'muon_pt': np.float32, 'muon_eta': np.float32, 'muon_phi': np.float32,
    'electron_pt': np.float32, 'electron_eta': np.float32, 'electron_phi': np.float32,
    'jet_pt': np.float32, 'jet_eta': np.float32, 'jet_phi': np.float32,
    'jet_btag': np.float32,
    'jet_rawFactor': np.float32, 'jet_area': np.float32,
    'jet_muonSubtrFactor': np.float32, 'jet_muonSubtrDeltaPhi': np.float32,
    'jet_chHEF': np.float32, 'jet_neHEF': np.float32,
    'jet_chEmEF': np.float32, 'jet_neEmEF': np.float32, 'jet_muEF': np.float32,
    'jet_chMultiplicity': np.float32, 'jet_neMultiplicity': np.float32,
    'bjet_pt': np.float32, 'bjet_eta': np.float32, 'bjet_phi': np.float32,
    'corrt1metjet_rawpt': np.float32, 'corrt1metjet_eta': np.float32,
    'corrt1metjet_phi': np.float32, 'corrt1metjet_area': np.float32,
    'corrt1metjet_muonsubtrfactor': np.float32, 'corrt1metjet_muonsubtrdeltaphi': np.float32,
    'corrt1metjet_emef': np.float32,
}


def get_empty_branch_types() -> Dict[str, Any]:
    """Return uproot mktree-compatible branch type dict for an empty Events TTree.

    Used to write an empty Events TTree when no events pass selection, so that
    hadd can merge files regardless of whether any chunk had selected events.
    """
    types: Dict[str, Any] = {}
    for name, dtype in _SCALAR_BRANCHES.items():
        types[name] = np.dtype(dtype)
    for name in _JAGGED_BRANCHES:
        types[name] = 'var * float32'
    for idx in (1, 2, 3):
        types[f'Jet{idx}BTagScore'] = np.dtype(np.float32)
    return types


def prepare_event_branches(branches: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise compute_event_variables() output for writing, the same way for
    every executor (iterative, futures, dask), so all of them save the same
    branches with the same types.

    - list of lists (materialised jagged)  -> awkward float32 jagged
    - empty list (zero events, jagged)     -> kept; write_events_tree types it
    - flat list                            -> numpy (floats as float32)
    - awkward array                        -> kept as is
    - numpy: unsigned -> int64, non-numeric -> float64, others kept
    """
    out: Dict[str, Any] = {}
    for k, v in branches.items():
        if isinstance(v, list) and not v:
            out[k] = v
        elif isinstance(v, list):
            try:
                arr = ak.Array(v)
                if arr.ndim > 1:
                    out[k] = ak.values_astype(arr, np.float32)
                else:
                    a = np.asarray(arr)
                    out[k] = a.astype(np.float32) if a.dtype.kind == 'f' else a
            except Exception:
                out[k] = v
        elif isinstance(v, ak.Array):
            out[k] = v
        else:
            a = np.asarray(v)
            if a.dtype.kind == 'u':
                a = a.astype(np.int64)
            elif a.dtype.kind not in ('i', 'f'):
                a = a.astype(np.float64)
            out[k] = a
    return out


def write_events_tree(f, branches: Dict[str, Any], name: str = "Events") -> int:
    """Write *branches* into the open uproot file *f* as a classic TTree.

    Always use this instead of ``f[name] = branches``: from uproot 5.7 dict
    assignment writes an RNTuple, which hadd (ROOT 6.38) cannot merge. mktree +
    extend gives a TTree whatever uproot's default is, and with zero-length
    arrays it writes an empty tree with the full schema, so files with no
    selected events still merge with the rest.

    Branch values may be 1-D numpy arrays, or awkward arrays / lists of lists for
    jagged branches. Returns the number of entries written.
    """
    types: Dict[str, Any] = {}
    data: Dict[str, Any] = {}
    n_entries = None
    for k, v in branches.items():
        if isinstance(v, np.ndarray) and v.dtype != object:
            arr = v
            types[k] = arr.dtype
        else:
            arr = v if isinstance(v, ak.Array) else ak.Array(v)
            if arr.ndim > 1 or "unknown" in str(arr.type):
                # Every jagged branch is float32 (_JAGGED_BRANCHES), as the
                # iterative path has always written them. Cast so the type does
                # not depend on how values arrived (Python floats -> float64,
                # NanoAOD multiplicities -> int64, empty list -> unknown).
                arr = ak.values_astype(arr, np.float32) if len(arr) else arr
                types[k] = "var * float32"
            else:
                arr = np.asarray(arr)
                types[k] = arr.dtype
        data[k] = arr
        if n_entries is None:
            n_entries = len(arr)
        elif len(arr) != n_entries:
            raise ValueError(f"branch {k!r} has {len(arr)} entries, expected {n_entries}")
    f.mktree(name, types)
    if n_entries:
        f[name].extend(data)
    return n_entries or 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_event_variables(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any],
    event_weights: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Compute all output branches for the Events TTree.

    Returns flat dict: str → np.ndarray (scalar) or ak.Array (jagged).
    Sentinel -9.0 used for variables undefined on an event (e.g. < N jets).
    """
    n_ev = len(events)
    out: Dict[str, Any] = {}

    # --- Event identifiers ---
    for field in ('event', 'run', 'luminosityBlock'):
        try:
            out[field] = ak.to_numpy(events[field]).astype(np.int64)
        except Exception:
            out[field] = np.zeros(n_ev, dtype=np.int64)

    # --- Pileup truth (MC only; absent on data → SENTINEL / -1) ---
    if 'Pileup_nTrueInt' in events.fields:
        out['Pileup_nTrueInt'] = ak.to_numpy(events['Pileup_nTrueInt']).astype(np.float32)
    else:
        out['Pileup_nTrueInt'] = np.full(n_ev, SENTINEL, dtype=np.float32)
    if 'Pileup_nPU' in events.fields:
        out['Pileup_nPU'] = ak.to_numpy(events['Pileup_nPU']).astype(np.int32)
    else:
        out['Pileup_nPU'] = np.full(n_ev, -1, dtype=np.int32)

    # --- Reconstructed primary vertices (data + MC; pileup data/MC validation) ---
    # PV_npvsGood plotted with PU-reweighted MC (full_event_weight) and un-reweighted
    # MC (weight_noPileup, via pseudo-variable PV_npvsGood_noPU) against data.
    for _pv in ('PV_npvsGood', 'PV_npvs'):
        if _pv in events.fields:
            out[_pv] = ak.to_numpy(events[_pv]).astype(np.int32)
        else:
            out[_pv] = np.full(n_ev, -1, dtype=np.int32)

    # --- Per-trigger-group decisions (stored for per-region trigger requirement) ---
    # MET trigger group: MET only (SingleMuon is for the trigger-efficiency study only)
    # EGamma trigger group: EGamma (used for electron CRs)
    if config.get('data', {}).get('is_signal', False):
        # Fastsim signal (BBDM-2HDMa) has no HLT branches — force trigger pass.
        out['pass_met_trigger'] = np.ones(n_ev, dtype=np.int32)
        out['pass_ele_trigger'] = np.ones(n_ev, dtype=np.int32)
    else:
        _met_trig_paths  = config.get('triggers', {}).get('MET', [])
        _ele_trig_paths  = config.get('triggers', {}).get('EGamma', [])
        def _trig_decision(paths):
            mask = np.zeros(n_ev, dtype=np.int32)
            for p in paths:
                if p in events.fields:
                    try:
                        mask = np.maximum(mask, ak.to_numpy(events[p]).astype(np.int32))
                    except Exception:
                        pass
            return mask
        out['pass_met_trigger'] = _trig_decision(_met_trig_paths)
        out['pass_ele_trigger'] = _trig_decision(_ele_trig_paths)

    # --- MET ---
    met_vars = _met_variables(events, objects)
    out.update(met_vars)

    # --- Recoil ---
    out.update(_recoil_variables(objects, config))

    # --- cos(theta*) ---
    try:
        out['costheta_star'] = ak.to_numpy(ak.fill_none(objects['costheta_star'], np.float32(SENTINEL))).astype(np.float32)
    except Exception:
        out['costheta_star'] = np.full(n_ev, SENTINEL, dtype=np.float32)

    # --- Multiplicities ---
    out.update(_multiplicity_variables(objects))

    # --- Positional b-tag flags (lead / sublead jet passes b-tag WP) ---
    # Region-level cuts require the pt-leading (1b) or pt-leading+subleading (2b)
    # jets to be the b-tagged ones. NanoAOD jets are pt-sorted and cleaning
    # preserves order, so index 0/1 are the leading/subleading jets.
    out.update(_positional_btag_flags(objects, config))

    # --- Leading jet scalars ---
    jets = objects.get('jets', ak.Array([]))
    jet_lead = _jet_lead_variables(objects)
    # stash jet masses for composite computation
    if hasattr(jets, 'mass'):
        jet_lead['_j1m'] = _lead(jets.mass, 0, default=0.0)
        jet_lead['_j2m'] = _lead(jets.mass, 1, default=0.0)
        jet_lead['_j3m'] = _lead(jets.mass, 2, default=0.0)
    out.update({k: v for k, v in jet_lead.items() if not k.startswith('_')})

    # --- Composite jet + MET variables ---
    out.update(_jet_composite_variables(jet_lead, met_vars))
    out['dPhi_jetMET'] = _dphi_jet_met(objects, met_vars['MET_phi'])

    # --- Derived topological variables (same as dnn/feature_engineering.py) ---
    # SENTINEL-in -> SENTINEL-out: dPhiJet12 is SENTINEL whenever there's no
    # 2nd jet, so arithmetic on it (e.g. dphi_jmet + dphi_j12 - pi) would
    # otherwise produce a large-but-finite non-SENTINEL number that downstream
    # sentinel masking (exact == SENTINEL) can't catch.
    dphi_jmet = out.get('dPhi_jetMET', np.zeros(n_ev, dtype=np.float32))
    dphi_j12  = out.get('dPhiJet12',  np.zeros(n_ev, dtype=np.float32))
    _del_valid = (dphi_jmet != SENTINEL) & (dphi_j12 != SENTINEL)
    out['del_plus']  = np.where(_del_valid, np.abs(dphi_jmet + dphi_j12 - np.pi), SENTINEL).astype(np.float32)
    out['del_minus'] = np.where(_del_valid, dphi_jmet - dphi_j12, SENTINEL).astype(np.float32)

    # --- Leading lepton scalar branches (needed for MT/Mll in CR cuts) ---
    for lep_key, prefix in (('muons', 'muon'), ('electrons', 'electron')):
        lep = objects.get(lep_key, ak.Array([]))
        for idx, suffix in ((0, 'lep1'), (1, 'lep2')):
            try:
                if hasattr(lep, 'pt'):
                    out[f'{prefix}_{suffix}_pt']  = _lead(lep.pt,  idx, default=SENTINEL).astype(np.float32)
                    out[f'{prefix}_{suffix}_phi'] = _lead(lep.phi, idx, default=SENTINEL).astype(np.float32)
                    out[f'{prefix}_{suffix}_eta'] = _lead(lep.eta, idx, default=SENTINEL).astype(np.float32)
            except Exception:
                pass

    # --- Jagged object branches ---
    out.update(_jagged_variables(objects))

    # --- Raw JEC/Type-1-MET inputs not carried by the "jets" object (see
    #     objects.py:build_jet_collection for the per-jet raw fields that
    #     ARE on "jets" and already saved via _jagged_variables above) ---
    out.update(_raw_jec_inputs_variables(events))

    # --- GenModel grid-point flags (signal only; no-op for bkg/data) ---
    out.update(_genmodel_variables(events))

    # --- Event weights ---
    if event_weights:
        for name, val in event_weights.items():
            if isinstance(val, dict):
                for var, arr in val.items():
                    if isinstance(arr, np.ndarray):
                        out[f'{name}_{var}'] = arr
            elif isinstance(val, np.ndarray):
                out[name] = val

    # Materialise any remaining ak.Array values to plain Python lists so the
    # returned dict is fully pickle-safe across loky worker process boundaries.
    # NanoEvents-derived arrays are lazy / tied to the uproot file handle;
    # they cannot be deserialized in the main process after the file is closed.
    for _k, _v in list(out.items()):
        if isinstance(_v, ak.Array):
            try:
                out[_k] = ak.to_list(_v)
            except Exception:
                out.pop(_k, None)

    return out
