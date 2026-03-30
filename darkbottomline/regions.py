"""
Region definition and management for DarkBottomLine analysis.
"""

import awkward as ak
import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Union
import logging


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

    def apply_cuts(self, events: ak.Array, objects: Dict[str, Any]) -> ak.Array:
        """
        Apply region cuts to events. Logs initial count and one line with events after each cut.
        """
        mask = ak.ones_like(events.event, dtype=bool)
        n_initial = int(ak.sum(mask))
        logging.info(f"Region {self.name}: initial (preselected) events: {n_initial}")

        after_cuts = []
        for var, cut_info in self.parsed_cuts.items():
            operator = cut_info["operator"]
            value = cut_info["value"]

            # Get variable value
            var_value = self._get_variable_value(events, objects, var)

            if var_value is None:
                logging.warning(f"Variable {var} not found, skipping cut")
                continue

            # Apply cut
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
                logging.warning(f"Unknown operator: {operator}")
                continue

            mask = mask & ak.fill_none(cut_mask, False, axis=0)
            n_pass = int(ak.sum(mask))
            after_cuts.append(f"{var}: {n_pass}")

        if after_cuts:
            logging.info(" %s", ", ".join(after_cuts))

        return mask

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
            try:
                if "PFMET_pt" in events.fields:
                    return ak.fill_none(events["PFMET_pt"], 0.0)
                else:
                    return ak.fill_none(events["MET_pt"], 0.0)
            except Exception:
                if "PFMET_pt" in events.fields:
                    return events["PFMET_pt"]
                else:
                    return events["MET_pt"]
        if var == "Nbjets":
            return self._safe_num_axis1(objects.get("bjets", ak.Array([])), n_ev)
        if var == "Njets" or var == "NjetsMin":
            return self._safe_num_axis1(objects.get("jets", ak.Array([])), n_ev)
        if var == "Jet1Pt":
            jets = objects.get("jets", ak.Array([]))
            if len(ak.flatten(jets)) == 0:
                return ak.zeros_like(events.event, dtype=float)
            try:
                return ak.fill_none(ak.max(jets.pt, axis=1), 0.0, axis=0)
            except (Exception, BaseException):
                return ak.zeros_like(events.event, dtype=float)
        if var == "Nleptons":
            n_muons = self._safe_num_axis1(objects.get("tight_muons_pt30", ak.Array([])), n_ev)
            n_electrons = self._safe_num_axis1(objects.get("tight_electrons_pt30", ak.Array([])), n_ev)
            n_taus = self._safe_num_axis1(objects.get("tight_taus_pt30", ak.Array([])), n_ev)
            return n_muons + n_electrons + n_taus
        if var == "Nmuons":
            return self._safe_num_axis1(objects.get("tight_muons_pt30", ak.Array([])), n_ev)
        if var == "Nelectrons":
            return self._safe_num_axis1(objects.get("tight_electrons_pt30", ak.Array([])), n_ev)
        if var == "NmuonsZ":
            # Z CR: 2 OS leptons, leading tight pt>30, subleading loose pt>10
            return objects.get("n_z_muons", ak.zeros_like(events.event, dtype=int))
        if var == "NelectronsZ":
            return objects.get("n_z_electrons", ak.zeros_like(events.event, dtype=int))
        if var == "Ntaus":
            return self._safe_num_axis1(objects.get("tight_taus_pt30", ak.Array([])), n_ev)
        if var == "NAdditionalJets":
            n_jets = self._safe_num_axis1(objects.get("jets", ak.Array([])), n_ev)
            n_bjets = self._safe_num_axis1(objects.get("bjets", ak.Array([])), n_ev)
            return n_jets - n_bjets
        if var == "Recoil":
            # Recoil = | -( pTmiss + sum pT(leptons) ) | (tight pt>30 leptons for CR)
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            muons = objects.get("tight_muons_pt30", ak.Array([]))
            electrons = objects.get("tight_electrons_pt30", ak.Array([]))

            # Calculate lepton px/py components (axis=1 can fail on depth-1)
            lep_px = ak.zeros_like(met_pt)
            lep_py = ak.zeros_like(met_pt)
            try:
                if len(ak.flatten(muons)) > 0:
                    lep_px = lep_px + ak.sum(muons.pt * np.cos(muons.phi), axis=1)
                    lep_py = lep_py + ak.sum(muons.pt * np.sin(muons.phi), axis=1)
                if len(ak.flatten(electrons)) > 0:
                    lep_px = lep_px + ak.sum(electrons.pt * np.cos(electrons.phi), axis=1)
                    lep_py = lep_py + ak.sum(electrons.pt * np.sin(electrons.phi), axis=1)
            except (Exception, BaseException):
                pass

            # Calculate Recoil magnitude
            recoil_px = -(met_pt * np.cos(met_phi) + lep_px)
            recoil_py = -(met_pt * np.sin(met_phi) + lep_py)
            recoil = np.sqrt(recoil_px**2 + recoil_py**2)

            return ak.fill_none(recoil, 0.0)
        if var == "MT":
            # Transverse mass (tight pt>30 leptons for CR)
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            muons = objects.get("tight_muons_pt30", ak.Array([]))
            electrons = objects.get("tight_electrons_pt30", ak.Array([]))

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
            # Z candidate mass: muon pair if NmuonsZ==2 else electron pair if NelectronsZ==2
            n_z_mu = objects.get("n_z_muons", ak.zeros_like(events.event, dtype=int))
            n_z_el = objects.get("n_z_electrons", ak.zeros_like(events.event, dtype=int))
            mll_mu = objects.get("mll_mu", ak.zeros_like(events.event, dtype=float))
            mll_el = objects.get("mll_el", ak.zeros_like(events.event, dtype=float))
            mll = ak.where(n_z_mu == 2, mll_mu, ak.where(n_z_el == 2, mll_el, 0.0))
            # axis=0 to avoid axis=-1 exceeding depth (1) on record/1D arrays
            try:
                return ak.fill_none(mll, 0.0, axis=0)
            except (Exception, BaseException):
                try:
                    return np.asarray(ak.to_numpy(ak.ravel(mll)), dtype=np.float64)
                except (Exception, BaseException):
                    return np.zeros(n_ev, dtype=np.float64)
        if var == "RecoilZ":
            # Recoil from Z candidate pair (for Z CR): |-(MET + sum pT(Z leptons))|
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
            n_z_mu = objects.get("n_z_muons", ak.zeros_like(events.event, dtype=int))
            n_z_el = objects.get("n_z_electrons", ak.zeros_like(events.event, dtype=int))
            sx_mu = objects.get("z_lep_sum_x_mu", ak.zeros_like(events.event, dtype=float))
            sy_mu = objects.get("z_lep_sum_y_mu", ak.zeros_like(events.event, dtype=float))
            sx_el = objects.get("z_lep_sum_x_el", ak.zeros_like(events.event, dtype=float))
            sy_el = objects.get("z_lep_sum_y_el", ak.zeros_like(events.event, dtype=float))
            sx = ak.where(n_z_mu == 2, sx_mu, ak.where(n_z_el == 2, sx_el, 0.0))
            sy = ak.where(n_z_mu == 2, sy_mu, ak.where(n_z_el == 2, sy_el, 0.0))
            recoil_x = -(met_pt * np.cos(met_phi) + sx)
            recoil_y = -(met_pt * np.sin(met_phi) + sy)
            return ak.fill_none(np.sqrt(recoil_x**2 + recoil_y**2), 0.0)
        if var == "DeltaPhi":
            jets = objects.get("jets", ak.Array([]))
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            # Check if jets array is empty or has no structure
            if len(ak.flatten(jets)) == 0 or len(jets) == 0:
                return ak.zeros_like(events["event"], dtype=float)

            try:
                n_jets_per_event = self._safe_num_axis1(jets, n_ev)
                has_jets = np.any(np.asarray(n_jets_per_event) > 0)
                if has_jets:
                    jet_phi = jets.phi
                    delta_phi = ak.min(np.abs(jet_phi - met_phi), axis=1)
                    return ak.fill_none(delta_phi, 0.0, axis=0)
            except (Exception, BaseException):
                pass

            return ak.zeros_like(events["event"], dtype=float)
        if var == "LeptonPt":
            muons = objects.get("tight_muons_pt30", ak.Array([]))
            electrons = objects.get("tight_electrons_pt30", ak.Array([]))
            taus = objects.get("tight_taus_pt30", ak.Array([]))
            try:
                all_leptons = ak.concatenate([muons, electrons, taus], axis=1)
                if len(ak.flatten(all_leptons)) == 0:
                    return ak.zeros_like(events.event, dtype=float)
                return ak.fill_none(ak.max(all_leptons.pt, axis=1), 0.0, axis=0)
            except (Exception, BaseException):
                return ak.zeros_like(events.event, dtype=float)
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
        return ak.zeros_like(events.event, dtype=float)


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

        logging.info(f"Loaded {len(self.regions)} regions: {list(self.regions.keys())}")

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
