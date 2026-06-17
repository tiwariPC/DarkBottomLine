"""
Region definition and management for DarkBottomLine analysis.
"""

import warnings
import awkward as ak
import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Union
import logging

warnings.filterwarnings("ignore", message="invalid value encountered in sqrt",
                        category=RuntimeWarning)


class Region:
    """
    Represents a single analysis region with its cuts and properties.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize region from configuration.

        Args:
            name: Region name
            config: Region configuration dictionary
        """
        self.name = name
        self.description = config.get("description", "")
        self.cuts = config.get("cuts", {})
        self.expected_backgrounds = config.get("expected_backgrounds", [])
        self.blind_data = config.get("blind_data", False)
        self.priority = config.get("priority", 1)
        self.transfer_factor_to_SR = config.get("transfer_factor_to_SR", None)

        # Parse cuts into evaluable expressions
        self.parsed_cuts = self._parse_cuts()

    def _parse_cuts(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse cut strings into evaluable expressions.

        Returns:
            Dictionary of parsed cuts
        """
        parsed = {}

        for var, cut_str in self.cuts.items():
            if ">=" in cut_str:
                value = float(cut_str.replace(">=", ""))
                parsed[var] = {"operator": ">=", "value": value}
            elif "<=" in cut_str:
                value = float(cut_str.replace("<=", ""))
                parsed[var] = {"operator": "<=", "value": value}
            elif ">" in cut_str:
                value = float(cut_str.replace(">", ""))
                parsed[var] = {"operator": ">", "value": value}
            elif "<" in cut_str:
                value = float(cut_str.replace("<", ""))
                parsed[var] = {"operator": "<", "value": value}
            elif "==" in cut_str:
                value = float(cut_str.replace("==", ""))
                parsed[var] = {"operator": "==", "value": value}
            elif "!=" in cut_str:
                value = float(cut_str.replace("!=", ""))
                parsed[var] = {"operator": "!=", "value": value}
            else:
                try:
                    value = float(cut_str)
                    parsed[var] = {"operator": "==", "value": value}
                except ValueError:
                    logging.warning(f"Could not parse cut: {var} {cut_str}")

        return parsed

    def _evaluate_cut_sequence(self, events: ak.Array, objects: Dict[str, Any]) -> tuple[ak.Array, Dict[str, int]]:
        """Return the final mask and ordered cumulative cutflow counts for this region."""
        _ref_field = "event" if "event" in events.fields else (events.fields[0] if events.fields else None)
        if _ref_field is not None:
            mask = ak.ones_like(events[_ref_field], dtype=bool)
        else:
            mask = ak.Array(np.ones(len(events), dtype=bool))
        n_initial = int(ak.sum(mask))
        logging.debug(f"Region {self.name}: initial (preselected) events: {n_initial}")

        cutflow: Dict[str, int] = {"Total events": n_initial}
        after_cuts = []
        for var, cut_info in self.parsed_cuts.items():
            operator = cut_info["operator"]
            value = cut_info["value"]

            var_value = self._get_variable_value(events, objects, var)
            if var_value is None:
                logging.warning(f"Variable {var} not found in region {self.name}, skipping cut")
                cutflow[f"SKIPPED {var}"] = int(ak.sum(mask))
                continue

            if operator == ">":
                cut_mask = var_value > value
            elif operator == ">=":
                cut_mask = var_value >= value
            elif operator == "<":
                cut_mask = var_value < value
            elif operator == "<=":
                cut_mask = var_value <= value
            elif operator == "==":
                cut_mask = var_value == value
            elif operator == "!=":
                cut_mask = var_value != value
            else:
                logging.warning(f"Unknown operator {operator!r} for {var} in region {self.name}")
                cutflow[f"SKIPPED {var}"] = int(ak.sum(mask))
                continue

            mask = mask & ak.fill_none(cut_mask, False, axis=0)
            n_pass = int(ak.sum(mask))
            cutflow[f"After {var}"] = n_pass
            after_cuts.append(f"{var}: {n_pass}")

        if after_cuts:
            logging.debug(" %s", ", ".join(after_cuts))

        return mask, cutflow

    def evaluate_cutflow(self, events: ak.Array, objects: Dict[str, Any]) -> Dict[str, int]:
        """Public helper returning the cumulative cutflow for this region."""
        _, cutflow = self._evaluate_cut_sequence(events, objects)
        return cutflow

    def apply_cuts(self, events: ak.Array, objects: Dict[str, Any]) -> ak.Array:
        """Apply region cuts to events and return the final mask."""
        mask, _ = self._evaluate_cut_sequence(events, objects)
        return mask

    def apply_cuts_with_yields(
        self, events: ak.Array, objects: Dict[str, Any], weight: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Apply cuts sequentially; return ordered {cut_label: weighted_yield_after_cut}."""
        _ref_field = "event" if "event" in events.fields else (events.fields[0] if events.fields else None)
        if _ref_field is not None:
            mask = ak.ones_like(events[_ref_field], dtype=bool)
        else:
            mask = ak.Array(np.ones(len(events), dtype=bool))

        yields: Dict[str, float] = {}
        for var, cut_info in self.parsed_cuts.items():
            operator = cut_info["operator"]
            value    = cut_info["value"]
            var_value = self._get_variable_value(events, objects, var)
            if var_value is None:
                continue
            if operator == ">":   cut_mask = var_value > value
            elif operator == ">=": cut_mask = var_value >= value
            elif operator == "<":  cut_mask = var_value < value
            elif operator == "<=": cut_mask = var_value <= value
            elif operator == "==": cut_mask = var_value == value
            elif operator == "!=": cut_mask = var_value != value
            else: continue
            mask = mask & ak.fill_none(cut_mask, False, axis=0)
            mask_np = np.asarray(mask, dtype=bool)
            if weight is not None:
                yields[f"{var}{operator}{value}"] = float(weight[mask_np].sum())
            else:
                yields[f"{var}{operator}{value}"] = float(mask_np.sum())
        return yields

    def _zeros_like_events(self, events: ak.Array, n_ev: int, dtype=float) -> ak.Array:
        """Return a per-event zeros array regardless of whether events has an 'event' field."""
        _f = "event" if "event" in events.fields else (events.fields[0] if events.fields else None)
        if _f is not None:
            try:
                return ak.zeros_like(events[_f], dtype=dtype)
            except Exception:
                pass
        return ak.Array(np.zeros(n_ev, dtype=dtype))

    def _safe_num_axis1(self, arr, n_ev: int):
        """Per-event count; avoid axis=1 on depth-1 arrays (e.g. edge-case structures)."""
        if arr is None or (isinstance(arr, ak.Array) and len(arr) == 0):
            return np.zeros(n_ev, dtype=np.int64)
        try:
            return ak.num(arr, axis=1)
        except (Exception, BaseException):
            return np.zeros(n_ev, dtype=np.int64)

    def _get_variable_value(self, events: ak.Array, objects: Dict[str, Any], var: str) -> Optional[ak.Array]:
        """
        Get variable value from events or objects.
        """
        n_ev = len(events)
        # Special variables
        if var == "MET":
            # Raw MET (no lepton correction) — used for W CR MET > 100
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            return ak.fill_none(met_pt, 0.0)
        if var == "Recoil":
            # Global recoil = |-(MET_vec + sum pT(loose leptons))|, precomputed in build_objects
            if "recoil" in objects:
                return ak.fill_none(objects["recoil"], 0.0)
            # Flat-branch fallback (event-selected files store "Recoil" as a scalar branch)
            for _fname in ("Recoil", "recoil"):
                if _fname in events.fields:
                    return ak.fill_none(events[_fname], 0.0)
            return self._zeros_like_events(events, n_ev, dtype=float)
        if var == "Nbjets":
            if "bjets" in objects:
                return self._safe_num_axis1(objects["bjets"], n_ev)
            # Flat-branch fallback: n_bjets scalar per event
            if "n_bjets" in events.fields:
                return events["n_bjets"]
            return ak.zeros(n_ev, dtype=np.int64)
        if var == "Njets":
            if "jets" in objects:
                return self._safe_num_axis1(objects["jets"], n_ev)
            for _fname in ("Njets_PassID", "n_jets"):
                if _fname in events.fields:
                    return events[_fname]
            return ak.zeros(n_ev, dtype=np.int64)
        if var == "NjetsMin":
            # Lower bound on jet multiplicity — same variable as Njets, distinct key for clarity
            if "jets" in objects:
                return self._safe_num_axis1(objects["jets"], n_ev)
            for _fname in ("Njets_PassID", "n_jets"):
                if _fname in events.fields:
                    return events[_fname]
            return ak.zeros(n_ev, dtype=np.int64)
        if var == "Jet1Pt":
            jets = objects.get("jets", ak.Array([]))
            if len(ak.flatten(jets)) == 0:
                return self._zeros_like_events(events, n_ev, dtype=float)
            try:
                return ak.fill_none(ak.max(jets.pt, axis=1), 0.0, axis=0)
            except (Exception, BaseException):
                return self._zeros_like_events(events, n_ev, dtype=float)
        if var == "Nleptons":
            if "tight_muons" in objects or "tight_electrons" in objects or "tight_taus" in objects:
                n_muons = self._safe_num_axis1(objects.get("tight_muons", ak.Array([])), n_ev)
                n_electrons = self._safe_num_axis1(objects.get("tight_electrons", ak.Array([])), n_ev)
                n_taus = self._safe_num_axis1(objects.get("tight_taus", ak.Array([])), n_ev)
                return n_muons + n_electrons + n_taus
            # Flat-branch fallback
            _nm = events["n_muons"] if "n_muons" in events.fields else ak.zeros(n_ev, dtype=np.int64)
            _ne = events["n_electrons"] if "n_electrons" in events.fields else ak.zeros(n_ev, dtype=np.int64)
            _nt = events["n_taus"] if "n_taus" in events.fields else ak.zeros(n_ev, dtype=np.int64)
            return _nm + _ne + _nt
        if var == "Nmuons":
            if "tight_muons" in objects:
                return self._safe_num_axis1(objects["tight_muons"], n_ev)
            if "n_muons" in events.fields:
                return events["n_muons"]
            return ak.zeros(n_ev, dtype=np.int64)
        if var == "Nelectrons":
            if "tight_electrons" in objects:
                return self._safe_num_axis1(objects["tight_electrons"], n_ev)
            if "n_electrons" in events.fields:
                return events["n_electrons"]
            return ak.zeros(n_ev, dtype=np.int64)
        if var == "NmuonsZ":
            # Z CR: 2 OS leptons, leading tight pt>30, subleading loose pt>10
            if "n_z_muons" in objects:
                return objects["n_z_muons"]
            # Flat-branch fallback for event-selection files
            if "n_muons" in events.fields:
                return events["n_muons"]
            return self._zeros_like_events(events, n_ev, dtype=int)
        if var == "NelectronsZ":
            if "n_z_electrons" in objects:
                return objects["n_z_electrons"]
            # Flat-branch fallback for event-selection files
            if "n_electrons" in events.fields:
                return events["n_electrons"]
            return self._zeros_like_events(events, n_ev, dtype=int)
        if var == "Ntaus":
            if "tight_taus" in objects:
                return self._safe_num_axis1(objects["tight_taus"], n_ev)
            if "n_taus" in events.fields:
                return events["n_taus"]
            return ak.zeros(n_ev, dtype=np.int64)
        if var == "NAdditionalJets":
            if "jets" in objects or "bjets" in objects:
                n_jets = self._safe_num_axis1(objects.get("jets", ak.Array([])), n_ev)
                n_bjets = self._safe_num_axis1(objects.get("bjets", ak.Array([])), n_ev)
                return n_jets - n_bjets
            # Flat-branch fallback
            _nj = events["Njets_PassID"] if "Njets_PassID" in events.fields else (
                events["n_jets"] if "n_jets" in events.fields else ak.zeros(n_ev, dtype=np.int64))
            _nb = events["n_bjets"] if "n_bjets" in events.fields else ak.zeros(n_ev, dtype=np.int64)
            return _nj - _nb
        if var == "MT":
            # Flat-branch fallback — event-selected files store precomputed mt
            if "tight_muons" not in objects and "tight_electrons" not in objects:
                for _fname in ("mt", "MT", "w_mt"):
                    if _fname in events.fields:
                        return ak.fill_none(events[_fname], 0.0)
                # Compute from scalar leading-lepton branches + MET
                met_pt_f  = events["PFMET_pt"]  if "PFMET_pt"  in events.fields else None
                met_phi_f = events["PFMET_phi"] if "PFMET_phi" in events.fields else None
                if met_pt_f is not None and met_phi_f is not None:
                    for pt_f, phi_f in (
                        ("muon_lep1_pt",     "muon_lep1_phi"),
                        ("electron_lep1_pt", "electron_lep1_phi"),
                        ("muon_pt",          "muon_phi"),      # jagged fallback
                        ("electron_pt",      "electron_phi"),  # jagged fallback
                    ):
                        if pt_f in events.fields and phi_f in events.fields:
                            try:
                                lpt  = events[pt_f]
                                lphi = events[phi_f]
                                # jagged: take leading element; scalar: use directly
                                if hasattr(lpt, 'ndim') and lpt.ndim == 1:
                                    l1pt, l1phi = lpt, lphi
                                else:
                                    has1 = ak.num(lpt) >= 1
                                    l1pt  = ak.where(has1, lpt[:, 0],  0.0)
                                    l1phi = ak.where(has1, lphi[:, 0], 0.0)
                                valid = l1pt > 0
                                dphi = abs(l1phi - met_phi_f)
                                dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
                                mt_val = ak.where(valid, np.sqrt(2 * l1pt * met_pt_f * (1 - np.cos(dphi))), 0.0)
                                return ak.fill_none(mt_val, 0.0, axis=0)
                            except Exception:
                                pass
            # Transverse mass (tight pt>30 leptons for CR)
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            muons = objects.get("tight_muons", ak.Array([]))
            electrons = objects.get("tight_electrons", ak.Array([]))

            mt = ak.zeros_like(met_pt)
            try:
                # Muon MT (argsort axis=1 can fail on depth-1)
                has_muons = ak.num(muons) > 0
                leading_muon = ak.firsts(muons[ak.argsort(muons.pt, ascending=False, axis=1)])
                muon_pt = leading_muon.pt
                muon_phi = leading_muon.phi
                delta_phi_mu = abs(muon_phi - met_phi)
                delta_phi_mu = ak.where(delta_phi_mu > np.pi, 2 * np.pi - delta_phi_mu, delta_phi_mu)
                mt_mu = np.sqrt(2 * muon_pt * met_pt * (1 - np.cos(delta_phi_mu)))
                mt = ak.where(has_muons, mt_mu, mt)

                # Electron MT for events without muons
                has_electrons = ak.num(electrons) > 0
                leading_electron = ak.firsts(electrons[ak.argsort(electrons.pt, ascending=False, axis=1)])
                ele_pt = leading_electron.pt
                ele_phi = leading_electron.phi
                delta_phi_el = abs(ele_phi - met_phi)
                delta_phi_el = ak.where(delta_phi_el > np.pi, 2 * np.pi - delta_phi_el, delta_phi_el)
                mt_el = np.sqrt(2 * ele_pt * met_pt * (1 - np.cos(delta_phi_el)))
                mt = ak.where(~has_muons & has_electrons, mt_el, mt)
            except (Exception, BaseException):
                pass
            return ak.fill_none(mt, 0.0, axis=0)
        if var in ("Mll", "MllMin", "MllMax"):
            # Flat-branch fallback — event-selected files store precomputed mll
            if "n_z_muons" not in objects and "n_z_electrons" not in objects:
                for _fname in ("mll", "Mll", "z_mass"):
                    if _fname in events.fields:
                        return ak.fill_none(events[_fname], 0.0)
                # Compute from scalar leading-lepton branches if available
                for l1pt_f, l1eta_f, l1phi_f, l2pt_f, l2eta_f, l2phi_f in (
                    ("muon_lep1_pt",     "muon_lep1_eta",     "muon_lep1_phi",
                     "muon_lep2_pt",     "muon_lep2_eta",     "muon_lep2_phi"),
                    ("electron_lep1_pt", "electron_lep1_eta", "electron_lep1_phi",
                     "electron_lep2_pt", "electron_lep2_eta", "electron_lep2_phi"),
                ):
                    if all(f in events.fields for f in (l1pt_f, l1eta_f, l1phi_f, l2pt_f, l2eta_f, l2phi_f)):
                        try:
                            l1pt  = events[l1pt_f];  l1eta = events[l1eta_f]; l1phi = events[l1phi_f]
                            l2pt  = events[l2pt_f];  l2eta = events[l2eta_f]; l2phi = events[l2phi_f]
                            has2  = (l1pt > 0) & (l2pt > 0)
                            dphi  = abs(l1phi - l2phi)
                            dphi  = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
                            deta  = l1eta - l2eta
                            mll   = ak.where(
                                has2,
                                np.sqrt(2 * l1pt * l2pt * (np.cosh(deta) - np.cos(dphi))),
                                0.0,
                            )
                            return ak.fill_none(mll, 0.0, axis=0)
                        except Exception:
                            pass
            # Z candidate mass: muon pair if NmuonsZ==2 else electron pair if NelectronsZ==2
            n_z_mu = objects.get("n_z_muons", self._zeros_like_events(events, n_ev, dtype=int))
            n_z_el = objects.get("n_z_electrons", self._zeros_like_events(events, n_ev, dtype=int))
            mll_mu = objects.get("mll_mu", self._zeros_like_events(events, n_ev, dtype=float))
            mll_el = objects.get("mll_el", self._zeros_like_events(events, n_ev, dtype=float))
            mll = ak.where(n_z_mu == 2, mll_mu, ak.where(n_z_el == 2, mll_el, 0.0))
            # axis=0 to avoid axis=-1 exceeding depth (1) on record/1D arrays
            try:
                return ak.fill_none(mll, 0.0, axis=0)
            except (Exception, BaseException):
                try:
                    return np.asarray(ak.to_numpy(ak.ravel(mll)), dtype=np.float64)
                except (Exception, BaseException):
                    return np.zeros(n_ev, dtype=np.float64)
        if var == "Zpt":
            # pT of dilepton system from scalar lep1/lep2 branches
            for (l1pt_f, l1phi_f, l2pt_f, l2phi_f) in (
                ("muon_lep1_pt",     "muon_lep1_phi",     "muon_lep2_pt",     "muon_lep2_phi"),
                ("electron_lep1_pt", "electron_lep1_phi", "electron_lep2_pt", "electron_lep2_phi"),
            ):
                if all(f in events.fields for f in (l1pt_f, l1phi_f, l2pt_f, l2phi_f)):
                    try:
                        l1pt  = events[l1pt_f];  l1phi = events[l1phi_f]
                        l2pt  = events[l2pt_f];  l2phi = events[l2phi_f]
                        has2  = (l1pt > 0) & (l2pt > 0)
                        px    = l1pt * np.cos(l1phi) + l2pt * np.cos(l2phi)
                        py    = l1pt * np.sin(l1phi) + l2pt * np.sin(l2phi)
                        zpt   = ak.where(has2, np.sqrt(px**2 + py**2), 0.0)
                        return ak.fill_none(zpt, 0.0, axis=0)
                    except Exception:
                        pass
            return np.zeros(n_ev, dtype=np.float64)
        if var == "DeltaPhi":
            jets = objects.get("jets", ak.Array([]))
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            # Check if jets array is empty or has no structure
            if len(ak.flatten(jets)) == 0 or len(jets) == 0:
                return self._zeros_like_events(events, n_ev, dtype=float)

            try:
                n_jets_per_event = self._safe_num_axis1(jets, n_ev)
                has_jets = np.any(np.asarray(n_jets_per_event) > 0)
                if has_jets:
                    jet_phi = jets.phi
                    delta_phi = ak.min(np.abs(jet_phi - met_phi), axis=1)
                    return ak.fill_none(delta_phi, 0.0, axis=0)
            except (Exception, BaseException):
                pass

            return self._zeros_like_events(events, n_ev, dtype=float)
        if var == "LeptonPt":
            muons = objects.get("tight_muons", ak.Array([]))
            electrons = objects.get("tight_electrons", ak.Array([]))
            taus = objects.get("tight_taus", ak.Array([]))
            try:
                all_leptons = ak.concatenate([muons, electrons, taus], axis=1)
                if len(ak.flatten(all_leptons)) == 0:
                    return self._zeros_like_events(events, n_ev, dtype=float)
                return ak.fill_none(ak.max(all_leptons.pt, axis=1), 0.0, axis=0)
            except (Exception, BaseException):
                return self._zeros_like_events(events, n_ev, dtype=float)
        if var == "metQuality":
            # MET Quality = (pfMET - caloMET) / Recoil
            pf_met = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            calo_met = events.get("CaloMET_pt", pf_met)  # Fallback to pfMET if caloMET not available
            recoil = self._get_variable_value(events, objects, "Recoil")

            # Avoid division by zero
            met_quality = ak.where(recoil > 0, (pf_met - calo_met) / recoil, 0.0)
            return ak.fill_none(met_quality, 0.0, axis=0)

        # Direct from events/objects (axis=0 to avoid depth-1 fill_none error)
        if hasattr(events, var):
            try:
                return ak.fill_none(getattr(events, var), 0, axis=0)
            except Exception:
                return getattr(events, var)
        for obj_name, obj_data in objects.items():
            if hasattr(obj_data, var):
                try:
                    return ak.fill_none(getattr(obj_data, var), 0, axis=0)
                except Exception:
                    return getattr(obj_data, var)
        return self._zeros_like_events(events, n_ev, dtype=float)


class RegionManager:
    """
    Manages multiple analysis regions and their application.
    """

    def __init__(self, config_path: str):
        """
        Initialize region manager from configuration file.

        Args:
            config_path: Path to regions configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.regions = {}
        self.settings = self.config.get("settings", {})
        self.validation = self.config.get("validation", {})

        # Create region objects
        for region_name, region_config in self.config.get("regions", {}).items():
            self.regions[region_name] = Region(region_name, region_config)

        logging.debug(f"Loaded {len(self.regions)} regions: {list(self.regions.keys())}")

    def get_region(self, name: str) -> Optional[Region]:
        return self.regions.get(name)

    def get_all_regions(self) -> Dict[str, Region]:
        return self.regions

    def apply_regions(self, events: ak.Array, objects: Dict[str, Any]) -> Dict[str, ak.Array]:
        region_masks = {}
        for region_name, region in self.regions.items():
            mask = region.apply_cuts(events, objects)
            region_masks[region_name] = ak.fill_none(mask, False, axis=0)
        return region_masks

    def get_region_cutflows(self, events: ak.Array, objects: Dict[str, Any], only_control_regions: bool = True) -> Dict[str, Dict[str, int]]:
        """Collect ordered cumulative cutflows for regions."""
        cutflows: Dict[str, Dict[str, int]] = {}
        for region_name, region in self.regions.items():
            if only_control_regions and "CR_" not in region_name:
                continue
            cutflows[region_name] = region.evaluate_cutflow(events, objects)
        return cutflows

    def validate_regions(self, events: ak.Array, objects: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validation.get("check_orthogonality", True):
            return {"status": "skipped"}
        region_masks = self.apply_regions(events, objects)
        validation_results = {
            "status": "completed",
            "regions": {},
            "overlaps": {},
            "warnings": []
        }
        for region_name, mask in region_masks.items():
            n_events = ak.sum(mask)
            validation_results["regions"][region_name] = {
                "n_events": n_events,
                "fraction": float(n_events) / len(events) if len(events) > 0 else 0.0
            }
            min_events = self.settings.get("min_events", 10)
            if n_events < min_events:
                validation_results["warnings"].append(
                    f"Region {region_name} has only {n_events} events (minimum: {min_events})"
                )
        if self.validation.get("check_overlap", True):
            max_overlap = self.settings.get("max_overlap", 0.1)
            names = list(region_masks.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    m1 = region_masks[names[i]]
                    m2 = region_masks[names[j]]
                    overlap = ak.sum(m1 & m2)
                    total1 = ak.sum(m1)
                    total2 = ak.sum(m2)
                    if total1 > 0 and total2 > 0:
                        frac = float(overlap) / min(total1, total2)
                        validation_results["overlaps"][f"{names[i]}_{names[j]}"] = {
                            "n_overlap": overlap,
                            "fraction": frac,
                        }
                        if frac > max_overlap:
                            validation_results["warnings"].append(
                                f"High overlap between {names[i]} and {names[j]}: {frac:.2f}"
                            )
        return validation_results

    def get_region_summary(self) -> Dict[str, Any]:
        summary = {"n_regions": len(self.regions), "regions": {}}
        for region_name, region in self.regions.items():
            summary["regions"][region_name] = {
                "description": region.description,
                "n_cuts": len(region.cuts),
                "expected_backgrounds": region.expected_backgrounds,
                "blind_data": region.blind_data,
                "priority": region.priority,
                "transfer_factor": region.transfer_factor_to_SR,
            }
        return summary

    def get_signal_regions(self) -> List[str]:
        return [name for name in self.regions if ("SR" in name)]

    def get_control_regions(self) -> List[str]:
        return [name for name in self.regions if ("CR" in name)]

    def get_validation_regions(self) -> List[str]:
        return [name for name in self.regions if ("VR" in name)]
