"""
Event-level variable computation for DarkBottomLine output trees.

All physics variables saved to the output ROOT TTree are defined here.
To add a new variable: implement a function and register it in compute_event_variables().

Return value of compute_event_variables(): flat dict[str, np.ndarray | ak.Array]
  - 1D numpy arrays  → scalar branches (one value per event)
  - 2D ak.Arrays     → jagged branches (vector<float> in ROOT)
  - SENTINEL (-9.0)  → variable undefined for this event (e.g. < N jets required)
"""

import numpy as np
import awkward as ak
from typing import Dict, Any
from darkbottomline.objects import SENTINEL


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
    """Build Cartesian 4-vector from (pt, eta, phi, mass)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
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

def _met_variables(events: ak.Array) -> Dict[str, np.ndarray]:
    """MET kinematics. Supports both NanoAOD v12 (MET_*) and v15 (PFMET_*)."""
    def _get(v15, v12):
        if v15 in events.fields:
            return ak.to_numpy(events[v15])
        elif v12 in events.fields:
            return ak.to_numpy(events[v12])
        return np.zeros(len(events), dtype=np.float32)

    return {
        'PFMET_pt':           _get('PFMET_pt',           'MET_pt'),
        'PFMET_phi':          _get('PFMET_phi',           'MET_phi'),
        'pfMetCorrSig':       _get('PFMET_significance',  'MET_significance'),
    }


def _recoil_variables(objects: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Recoil pT and phi (precomputed in build_objects)."""
    recoil     = objects.get('recoil')
    recoil_phi = objects.get('recoil_phi')
    n = len(recoil) if recoil is not None else 0
    _zeros = ak.Array(np.zeros(n, dtype=np.float32))
    return {
        'Recoil':    _scalar(recoil     if recoil     is not None else _zeros),
        'RecoilPhi': _scalar(recoil_phi if recoil_phi is not None else _zeros),
    }


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

    return {
        'Njets_PassID':   _num('jets'),
        'n_bjets':        _num('bjets'),
        'n_muons':        _num('muons'),
        'n_electrons':    _num('electrons'),
        'n_taus':         _num('taus'),
        'b_flavor_count': b_flavor_count,
    }


def _jet_lead_variables(objects: Dict[str, Any], btag_algo: str) -> Dict[str, np.ndarray]:
    """Leading 3 jet kinematics + btag scores."""
    jets = objects.get('jets', ak.Array([]))
    btag = jets.btagDeepFlavB if hasattr(jets, 'btagDeepFlavB') else ak.zeros_like(jets.pt)

    return {
        'Jet1Pt':              _lead(jets.pt,  0),
        'Jet1Eta':             _lead(jets.eta, 0),
        'Jet1Phi':             _lead(jets.phi, 0),
        f'Jet1{btag_algo}':    _lead(btag,     0, default=SENTINEL),
        'Jet2Pt':              _lead(jets.pt,  1),
        'Jet2Eta':             _lead(jets.eta, 1),
        'Jet2Phi':             _lead(jets.phi, 1),
        f'Jet2{btag_algo}':    _lead(btag,     1, default=SENTINEL),
        'Jet3Pt':              _lead(jets.pt,  2),
        'Jet3Eta':             _lead(jets.eta, 2),
        'Jet3Phi':             _lead(jets.phi, 2),
        f'Jet3{btag_algo}':    _lead(btag,     2, default=SENTINEL),
        'JetHT':               _scalar(ak.sum(jets.pt, axis=1)),
    }


def _jet_composite_variables(
    jet_lead: Dict[str, np.ndarray],
    met_vars: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Derived dijet + jet-MET variables. Requires jet_lead and met_vars already computed."""
    j1pt  = jet_lead['Jet1Pt'];  j2pt  = jet_lead['Jet2Pt'];  j3pt  = jet_lead['Jet3Pt']
    j1eta = jet_lead['Jet1Eta']; j2eta = jet_lead['Jet2Eta']; j3eta = jet_lead['Jet3Eta']
    j1phi = jet_lead['Jet1Phi']; j2phi = jet_lead['Jet2Phi']; j3phi = jet_lead['Jet3Phi']
    met_pt  = met_vars['PFMET_pt'].astype(np.float32)
    met_phi = met_vars['PFMET_phi'].astype(np.float32)

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
        ('jets',      'btagDeepFlavB','jet_btag'),
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
    btag_algo = config.get('btagging', {}).get('algorithm', 'deepJet')
    n_ev = len(events)
    out: Dict[str, Any] = {}

    # --- Event identifiers ---
    for field in ('event', 'run', 'luminosityBlock'):
        try:
            out[field] = ak.to_numpy(events[field]).astype(np.int64)
        except Exception:
            out[field] = np.zeros(n_ev, dtype=np.int64)

    # --- MET ---
    met_vars = _met_variables(events)
    out.update(met_vars)

    # --- Recoil ---
    out.update(_recoil_variables(objects))

    # --- cos(theta*) ---
    try:
        out['costheta_star'] = ak.to_numpy(ak.fill_none(objects['costheta_star'], np.float32(SENTINEL))).astype(np.float32)
    except Exception:
        out['costheta_star'] = np.full(n_ev, SENTINEL, dtype=np.float32)

    # --- Multiplicities ---
    out.update(_multiplicity_variables(objects))

    # --- Leading jet scalars ---
    jets = objects.get('jets', ak.Array([]))
    jet_lead = _jet_lead_variables(objects, btag_algo)
    # stash jet masses for composite computation
    if hasattr(jets, 'mass'):
        jet_lead['_j1m'] = _lead(jets.mass, 0, default=0.0)
        jet_lead['_j2m'] = _lead(jets.mass, 1, default=0.0)
        jet_lead['_j3m'] = _lead(jets.mass, 2, default=0.0)
    out.update({k: v for k, v in jet_lead.items() if not k.startswith('_')})

    # --- Composite jet + MET variables ---
    out.update(_jet_composite_variables(jet_lead, met_vars))
    out['dPhi_jetMET'] = _dphi_jet_met(objects, met_vars['PFMET_phi'])

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
