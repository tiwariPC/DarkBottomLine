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

    def _scalar_int(key):
        arr = objects.get(key)
        if arr is None:
            return None
        try:
            return ak.to_numpy(ak.fill_none(ak.values_astype(arr, np.int32), np.int32(0)))
        except Exception:
            return None

    out = {
        'Njets_PassID':   _num('jets'),
        'n_bjets':        _num('bjets'),
        'n_muons':        _num('muons'),
        'n_electrons':    _num('electrons'),
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
                ak.where(ak.fill_none(ak.values_astype(nze, np.int32), 0) == 2, mll_el, 0.0)
            )
            out['mll'] = ak.to_numpy(ak.fill_none(ak.values_astype(mll, np.float32), np.float32(0.0)))
        except Exception:
            pass
    return out


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
# Empty-schema helper (for zero-event ROOT files)
# ---------------------------------------------------------------------------

# Scalar branches: name → numpy dtype
_SCALAR_BRANCHES: Dict[str, Any] = {
    'event': np.int64, 'run': np.int64, 'luminosityBlock': np.int64,
    'PFMET_pt': np.float32, 'PFMET_phi': np.float32, 'pfMetCorrSig': np.float32,
    'Recoil': np.float32, 'RecoilPhi': np.float32,
    'costheta_star': np.float32,
    'Njets_PassID': np.int32, 'n_bjets': np.int32, 'n_muons': np.int32,
    'n_electrons': np.int32, 'n_taus': np.int32, 'b_flavor_count': np.int32,
    'n_z_muons': np.int32, 'n_z_electrons': np.int32, 'mll': np.float32,
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
    'muon_lep1_pt': np.float32, 'muon_lep1_phi': np.float32, 'muon_lep1_eta': np.float32,
    'muon_lep2_pt': np.float32, 'muon_lep2_phi': np.float32, 'muon_lep2_eta': np.float32,
    'electron_lep1_pt': np.float32, 'electron_lep1_phi': np.float32, 'electron_lep1_eta': np.float32,
    'electron_lep2_pt': np.float32, 'electron_lep2_phi': np.float32, 'electron_lep2_eta': np.float32,
    'full_event_weight': np.float32,
    'pass_met_trigger': np.int32,
    'pass_ele_trigger': np.int32,
}

# Jagged branches: name → numpy dtype of elements
_JAGGED_BRANCHES: Dict[str, Any] = {
    'muon_pt': np.float32, 'muon_eta': np.float32, 'muon_phi': np.float32,
    'electron_pt': np.float32, 'electron_eta': np.float32, 'electron_phi': np.float32,
    'jet_pt': np.float32, 'jet_eta': np.float32, 'jet_phi': np.float32,
    'jet_btag': np.float32,
    'bjet_pt': np.float32, 'bjet_eta': np.float32, 'bjet_phi': np.float32,
}


def get_empty_branch_types(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return uproot mktree-compatible branch type dict for an empty Events TTree.

    Used to write an empty Events TTree when no events pass selection, so that
    hadd can merge files regardless of whether any chunk had selected events.
    The btag branch name is config-driven; all other names are fixed.
    """
    btag_algo = (config or {}).get('btagging', {}).get('algorithm', 'deepJet')
    types: Dict[str, Any] = {}
    for name, dtype in _SCALAR_BRANCHES.items():
        types[name] = np.dtype(dtype)
    for name in _JAGGED_BRANCHES:
        types[name] = 'var * float32'
    for idx in (1, 2, 3):
        types[f'Jet{idx}{btag_algo}'] = np.dtype(np.float32)
    return types


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

    # --- Per-trigger-group decisions (stored for per-region trigger requirement) ---
    # MET trigger group: MET + SingleMuon (used for SR, muon CRs)
    # EGamma trigger group: EGamma (used for electron CRs)
    _met_trig_paths  = (config.get('triggers', {}).get('MET', [])
                        + config.get('triggers', {}).get('SingleMuon', []))
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
