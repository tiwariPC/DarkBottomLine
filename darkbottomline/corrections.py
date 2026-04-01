"""
Correction and scale factor calculations using correctionlib.
"""

import awkward as ak
import gzip
import json
import logging
import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

try:
    from correctionlib import CorrectionSet
    CORRECTIONLIB_AVAILABLE = True
except ImportError:
    CORRECTIONLIB_AVAILABLE = False
    logging.warning("correctionlib not available. Corrections will be disabled.")

try:
    import correctionlib.schemav2 as _correctionlib_schemav2
    _SCHEMAV2_AVAILABLE = True
except ImportError:
    _SCHEMAV2_AVAILABLE = False


def _to_edge_float(x: Any) -> float:
    """Convert correctionlib edge value to float for ordering."""
    if isinstance(x, (int, float)):
        return float(x)
    if x == "inf" or x == "+inf":
        return float("inf")
    if x == "-inf":
        return float("-inf")
    return float(x)


def _make_edges_strictly_increasing(edges: List[Any]) -> List[Any]:
    """
    Return a new list of edges that is strictly increasing (no duplicates).
    Nudges duplicate finite values; skips duplicate inf/-inf so correctionlib's
    hi <= lo check passes (parsed "inf" becomes float('inf'), so duplicate inf fails).
    """
    if not edges or len(edges) < 2:
        return list(edges)
    try:
        sorted_edges = sorted(edges, key=_to_edge_float)
        out: List[Any] = [sorted_edges[0]]
        for i in range(1, len(sorted_edges)):
            prev_f = _to_edge_float(out[-1])
            curr_f = _to_edge_float(sorted_edges[i])
            if curr_f > prev_f:
                out.append(sorted_edges[i])
            elif curr_f <= prev_f and float("-inf") < prev_f < float("inf"):
                nudge = prev_f + max(1.0e-6, abs(prev_f) * 1.0e-10)
                out.append(nudge)
            # else: duplicate inf/-inf; skip (do not append) so list stays strictly increasing
        return out
    except (TypeError, ValueError):
        return list(edges)


def _fix_all_edges_in_place(node: Any) -> None:
    """
    Traverse entire JSON and fix every "edges" list to be strictly increasing.
    When duplicate inf/-inf are skipped, edges get shorter; trim content to match
    (content length must equal product of (len(edges[i])-1) for MultiBinning, len(edges)-1 for Binning).
    Modifies node in place.
    """
    if isinstance(node, dict):
        if "edges" in node:
            edges_val = node["edges"]
            if isinstance(edges_val, list):
                if len(edges_val) > 0 and isinstance(edges_val[0], list):
                    # MultiBinning: fix each dimension's edges
                    for dim in range(len(edges_val)):
                        if isinstance(edges_val[dim], list):
                            node["edges"][dim] = _make_edges_strictly_increasing(edges_val[dim])
                    # Content length must equal product of (len(edges[i])-1)
                    new_len = 1
                    for dim_edges in node["edges"]:
                        if isinstance(dim_edges, list):
                            new_len *= max(0, len(dim_edges) - 1)
                    content = node.get("content")
                    if isinstance(content, list) and len(content) > new_len:
                        node["content"] = content[:new_len]
                else:
                    # Binning: single edge array
                    node["edges"] = _make_edges_strictly_increasing(edges_val)
                    new_len = max(0, len(node["edges"]) - 1)
                    content = node.get("content")
                    if isinstance(content, list) and len(content) > new_len:
                        node["content"] = content[:new_len]
        for v in node.values():
            _fix_all_edges_in_place(v)
    elif isinstance(node, list):
        for item in node:
            _fix_all_edges_in_place(item)


def _fix_binning_edges(node: Any) -> None:
    """
    Recursively fix correction JSON so all binning edges are strictly increasing.
    Modifies node in place. Handles Binning (sort edges, reorder content, remove
    duplicate edges) and MultiBinning (sort each dimension's edges).
    Then runs a second pass that fixes every "edges" key in the tree (catches any
    structure we might have missed).
    See: https://cms-analysis-corrections.docs.cern.ch/ and correctionlib schema v2.
    """
    if not isinstance(node, dict):
        return
    nodetype = node.get("nodetype")
    if nodetype == "binning":
        edges = node.get("edges")
        content = node.get("content")
        if isinstance(edges, list) and isinstance(content, list) and len(content) == len(edges) - 1:
            try:
                order = sorted(range(len(edges)), key=lambda i: _to_edge_float(edges[i]))
                sorted_edges = [edges[i] for i in order]
                new_edges: List[Any] = [sorted_edges[0]]
                new_content: List[Any] = []
                for i in range(1, len(sorted_edges)):
                    if _to_edge_float(sorted_edges[i]) > _to_edge_float(new_edges[-1]):
                        new_edges.append(sorted_edges[i])
                        new_content.append(content[order[i - 1]])
                if len(new_edges) >= 2 and len(new_content) == len(new_edges) - 1:
                    node["edges"] = new_edges
                    node["content"] = new_content
                else:
                    node["edges"] = sorted_edges
                    node["content"] = [content[order[i]] for i in range(len(sorted_edges) - 1)]
                edges_final = node["edges"]
                for j in range(1, len(edges_final)):
                    try:
                        p, c = _to_edge_float(edges_final[j - 1]), _to_edge_float(edges_final[j])
                        if c <= p and float("-inf") < p < float("inf"):
                            edges_final[j] = p + max(1.0e-6, abs(p) * 1.0e-10)
                    except (TypeError, ValueError):
                        pass
            except (TypeError, ValueError):
                pass
        for c in node.get("content", []):
            _fix_binning_edges(c)
        flow = node.get("flow")
        if flow is not None and isinstance(flow, dict):
            _fix_binning_edges(flow)
    elif nodetype == "multibinning":
        edges_arr = node.get("edges")
        if isinstance(edges_arr, list):
            for dim, edges in enumerate(edges_arr):
                if isinstance(edges, list):
                    try:
                        node["edges"][dim] = _make_edges_strictly_increasing(edges)
                    except (TypeError, ValueError):
                        pass
        for c in node.get("content", []):
            if isinstance(c, dict):
                _fix_binning_edges(c)
        flow = node.get("flow")
        if flow is not None and isinstance(flow, dict):
            _fix_binning_edges(flow)
    else:
        for v in node.values():
            if isinstance(v, dict):
                _fix_binning_edges(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _fix_binning_edges(item)


def _resolve_correction_path(file_path: str) -> Optional[Path]:
    """Resolve correction file path: try as-is, then relative to package (project) root."""
    path = Path(file_path)
    if path.exists():
        return path
    # When running as module (e.g. python -m darkbottomline.cli), cwd may not be project root
    try:
        project_root = Path(__file__).resolve().parent.parent
        alt = project_root / file_path
        if alt.exists():
            return alt
    except Exception:
        pass
    return None


def _load_correction_file_with_edges_fix(file_path: str) -> Optional[CorrectionSet]:
    """Load a correction file; fix non-monotonic or duplicate bin edges then load via from_string."""
    path = _resolve_correction_path(file_path)
    if path is None:
        logging.debug("Correction file not found for edge fix: %s", file_path)
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.debug("Failed to read correction file for edge fix: %s", e)
        return None
    _fix_binning_edges(data)
    _fix_all_edges_in_place(data)  # Second pass: fix every "edges" in tree (any structure)
    try:
        # C library validates edges; pydantic-validated JSON (type-coerced) is accepted
        if _SCHEMAV2_AVAILABLE and data.get("schema_version") == 2:
            cs_py = _correctionlib_schemav2.CorrectionSet.parse_obj(data)
            json_str = cs_py.json(exclude_unset=True)
            return CorrectionSet.from_string(json_str)
        return CorrectionSet.from_string(json.dumps(data))
    except Exception as e:
        err_str = str(e)
        truncated = err_str[:200] + " [...]" if len(err_str) > 200 else err_str
        logging.warning(
            "CorrectionSet.from_string failed after bin-edge fix for %s (will fall back to from_file): %s",
            file_path,
            truncated,
        )
        return None


class CorrectionManager:
    """
    Manager class for loading and applying corrections using correctionlib.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize correction manager with configuration.

        Args:
            config: Configuration dictionary containing correction file paths
        """
        self.config = config
        self.corrections = {}
        self._load_corrections()

    def _load_corrections(self):
        """Load correction files using correctionlib."""
        if not CORRECTIONLIB_AVAILABLE:
            logging.warning("correctionlib not available. Skipping correction loading.")
            return

        correction_paths = self.config.get("corrections", {})
        # Only process keys that are correction file paths (skip electronSF_name, etc.)
        file_correction_keys = {k for k, v in correction_paths.items()
                                if isinstance(v, str) and v.endswith(('.json', '.json.gz'))}

        for correction_type in file_correction_keys:
            file_path = correction_paths[correction_type]
            if not file_path:
                continue
            loaded = False
            # For electron SF, try edge-fix load first (EGM JSON can have non-monotonic bin edges)
            if correction_type in ("electronSF", "electronHLT"):
                cs = _load_correction_file_with_edges_fix(file_path)
                if cs is not None:
                    self.corrections[correction_type] = cs
                    logging.info(
                        f"Loaded {correction_type} corrections from {file_path} "
                        "(bin edges fixed for correctionlib)"
                    )
                    loaded = True
            # For b-tag SF (.gz): load via gzip + from_string (BTV convention)
            if not loaded and correction_type == "btagSF" and file_path.endswith(".gz"):
                resolved_path = _resolve_correction_path(file_path)
                path_to_try = str(resolved_path) if resolved_path is not None else file_path
                try:
                    with gzip.open(path_to_try, "rt") as f:
                        data = f.read().strip()
                    self.corrections[correction_type] = CorrectionSet.from_string(data)
                    logging.info(f"Loaded {correction_type} corrections from {path_to_try} (gzip)")
                    loaded = True
                except Exception as e:
                    logging.warning(f"Failed to load {correction_type} (gzip): {e}")
            if not loaded:
                resolved_path = _resolve_correction_path(file_path)
                path_to_try = str(resolved_path) if resolved_path is not None else file_path
                try:
                    self.corrections[correction_type] = CorrectionSet.from_file(path_to_try)
                    logging.info(f"Loaded {correction_type} corrections from {path_to_try}")
                    loaded = True
                except Exception as e:
                    err_msg = str(e).lower()
                    if "monotone" in err_msg or "monotonic" in err_msg:
                        cs = _load_correction_file_with_edges_fix(file_path)
                        if cs is not None:
                            self.corrections[correction_type] = cs
                            logging.info(
                                f"Loaded {correction_type} corrections from {file_path} "
                                "(applied bin-edge fix for non-monotonic edges)"
                            )
                            loaded = True
                    if not loaded:
                        logging.warning(f"Failed to load {correction_type} corrections: {e}")

    def get_pileup_weight(
        self,
        events: ak.Array,
        systematic: str = "central"
    ) -> ak.Array:
        """
        Get pileup reweighting factors using correctionlib.

        Uses NanoAOD branch Pileup_nTrueInt and the first correction in the pileup
        CorrectionSet (e.g. Collisions*_goldenJSON). Variation: "nominal" for central,
        "up"/"down" if supported by the file.
        """
        if "pileup" not in self.corrections:
            logging.warning("Pileup corrections not available")
            return ak.ones_like(events.event, dtype=float)

        try:
            # NanoAOD: flat branch Pileup_nTrueInt (number of true interactions)
            pileup = getattr(events, "Pileup_nTrueInt", None)
            if pileup is None:
                pileup = events["Pileup_nTrueInt"] if "Pileup_nTrueInt" in events.fields else None
            if pileup is None:
                logging.warning("Pileup_nTrueInt not found in events, using unit weights")
                return ak.ones_like(events.event, dtype=float)

            cs = self.corrections["pileup"]
            # Use name from config (e.g. Collisions2022_355100_357900_eraBCD_GoldenJson) or first key
            corr_name = self.config.get("corrections", {}).get("pileup_name")
            if not corr_name or corr_name not in cs:
                corr_name = next(iter(cs.keys()), None)
            if corr_name is None:
                return ak.ones_like(pileup, dtype=float)
            corr = cs[corr_name]

            # Map "central" -> "nominal"; use "up"/"down" if correction supports it
            variation = "nominal" if systematic == "central" else systematic
            n_true_int = np.asarray(ak.to_numpy(pileup), dtype=float)
            try:
                weights = np.asarray(corr.evaluate(n_true_int, variation), dtype=float)
                return ak.Array(weights)
            except Exception:
                # Fallback to nominal if up/down not supported
                if variation != "nominal":
                    try:
                        weights = np.asarray(corr.evaluate(n_true_int, "nominal"), dtype=float)
                        return ak.Array(weights)
                    except Exception:
                        pass
                raise

        except Exception as e:
            logging.warning(f"Failed to apply pileup correction: {e}")
            return ak.ones_like(events.event, dtype=float)

    def _evaluate_correction_jagged_or_flat(self, corr, count_ref: ak.Array, *eval_args) -> Optional[ak.Array]:
        """
        Coffea/correctionlib pattern: try evaluate with jagged arrays first, else flatten -> evaluate -> unflatten.
        See: https://coffea-hep.readthedocs.io/en/latest/notebooks/applying_corrections.html
        count_ref: array (e.g. objects.pt) used to get per-event counts for unflatten.
        eval_args: arguments to corr.evaluate(*eval_args); can be jagged or scalars.
        Returns result as ak.Array matching count_ref structure, or None on failure.
        """
        if corr is None:
            return None
        try:
            result = corr.evaluate(*eval_args)
            if isinstance(result, ak.Array):
                return result
        except Exception:
            pass
        try:
            flat_args = []
            for a in eval_args:
                if isinstance(a, (ak.Array, np.ndarray)):
                    flat_args.append(np.asarray(ak.ravel(a)))
                else:
                    flat_args.append(a)
            result = np.asarray(corr.evaluate(*flat_args), dtype=float)
            try:
                counts = np.asarray(ak.num(count_ref, axis=1))
            except Exception:
                return ak.Array(result)
            if len(counts) > 0 and int(counts.sum()) == len(result):
                return ak.unflatten(result, counts)
            return ak.Array(result)
        except Exception:
            return None

    def _get_correction_by_name(self, cs_key: str, name_key: str, fallback_names: tuple = ()):
        """Return correction from CorrectionSet by config name, or first key. Avoids 'in cs' which can raise."""
        if cs_key not in self.corrections:
            return None
        cs = self.corrections[cs_key]
        name = self.config.get("corrections", {}).get(name_key)
        if name:
            try:
                return cs[name]
            except (IndexError, KeyError, TypeError):
                pass
        for fb in fallback_names:
            try:
                return cs[fb]
            except (IndexError, KeyError, TypeError):
                continue
        try:
            keys = list(cs.keys()) if hasattr(cs, "keys") else []
            if keys:
                return cs[keys[0]]
        except Exception:
            pass
        return None

    def _get_muonSF_highPT_correction(self):
        """Muon SF for pt>30: muon_Z_ID with NUM_TightPFIso_DEN_TightID."""
        return self._get_correction_by_name(
            "muonSF_highPT", "muonSF_highPT_name",
            fallback_names=("NUM_TightPFIso_DEN_TightID",)
        )

    def _get_muonSF_lowPT_correction(self):
        """Muon SF for pt<=30: muon_JPsi_LowpT with NUM_LooseID_DEN_TrackerMuons."""
        return self._get_correction_by_name(
            "muonSF_lowPT", "muonSF_lowPT_name",
            fallback_names=("NUM_LooseID_DEN_TrackerMuons",)
        )

    def _evaluate_muon_sf_per_pt_bin(
        self,
        muons: ak.Array,
        corr_highpt,
        corr_lowpt,
        systematic: str = "central",
    ) -> Optional[ak.Array]:
        """
        Evaluate muon SF: pt>30 use highpt correction, pt<=30 use lowpt.
        Muon SFs are pt and |eta| dependent; inputs passed as (abseta, pt) or (pt, abseta).
        Returns jagged array of SFs or None on failure.
        """
        flat_pt = np.asarray(ak.ravel(muons.pt), dtype=float)
        flat_abseta = np.asarray(np.abs(ak.ravel(muons.eta)), dtype=float)
        n_flat = len(flat_pt)
        if n_flat == 0:
            return ak.ones_like(muons.pt, dtype=float)
        result = np.ones(n_flat, dtype=float)
        mask_high = flat_pt > 30.0
        mask_low = ~mask_high
        for corr, mask, label in [(corr_highpt, mask_high, "highpt"), (corr_lowpt, mask_low, "lowpt")]:
            if corr is None or not np.any(mask):
                continue
            pt_sel = flat_pt[mask]
            eta_sel = flat_abseta[mask]
            # Try (abseta, pt) and (pt, abseta) orderings; optional systematic
            for args in [
                (eta_sel, pt_sel),
                (pt_sel, eta_sel),
                (eta_sel, pt_sel, systematic),
                (pt_sel, eta_sel, systematic),
                (systematic, eta_sel, pt_sel),
                (systematic, pt_sel, eta_sel),
            ]:
                try:
                    sf = np.asarray(corr.evaluate(*args), dtype=float)
                    if sf.shape == pt_sel.shape:
                        result[mask] = sf
                        break
                except Exception:
                    continue
        try:
            counts = np.asarray(ak.num(muons.pt, axis=1))
            return ak.unflatten(result, counts)
        except Exception:
            return ak.Array(result)

    def get_muon_sf(
        self,
        muons: ak.Array,
        systematic: str = "central"
    ) -> ak.Array:
        """
        Get muon scale factors: pt>30 use muon_Z_ID NUM_TightPFIso_DEN_TightID,
        pt<=30 use muon_JPsi_LowpT NUM_LooseID_DEN_TrackerMuons.
        """
        ones = ak.ones_like(muons.pt, dtype=float)
        if not muons.pt.layout or len(ak.ravel(muons.pt)) == 0:
            return ones
        corr_highpt = self._get_muonSF_highPT_correction()
        corr_lowpt = self._get_muonSF_lowPT_correction()
        if corr_highpt is None and corr_lowpt is None:
            return ones
        out = self._evaluate_muon_sf_per_pt_bin(muons, corr_highpt, corr_lowpt, systematic)
        if out is not None:
            return out
        logging.warning("Muon SF evaluate failed for all tried signatures")
        return ones

    def get_muon_sf_nominal_up_down(self, muons: ak.Array) -> Dict[str, ak.Array]:
        """Get muon SF as central, up, down (pt>30 highpt, pt<=30 lowpt)."""
        ones = ak.ones_like(muons.pt, dtype=float)
        out = {"central": ones, "up": ones, "down": ones}
        corr_highpt = self._get_muonSF_highPT_correction()
        corr_lowpt = self._get_muonSF_lowPT_correction()
        if corr_highpt is None and corr_lowpt is None:
            return out
        for key, syst in [("central", "central"), ("up", "up"), ("down", "down")]:
            val = self._evaluate_muon_sf_per_pt_bin(muons, corr_highpt, corr_lowpt, syst)
            if val is not None:
                out[key] = val
        return out

    def _get_electron_sf_correction(self):
        """Return the Electron-ID-SF correction evaluator from the loaded CorrectionSet."""
        if "electronSF" not in self.corrections:
            return None
        cs = self.corrections["electronSF"]
        for name in ("Electron-ID-SF", "electron_id_reco", "electron"):
            if name in cs:
                return cs[name]
        if hasattr(cs, "keys"):
            first = next(iter(cs.keys()), None)
            if first is not None:
                return cs[first]
        return None

    def _electron_sf_params(self) -> tuple:
        """Return (year_str, working_point) from config for Electron-ID-SF."""
        year = self.config.get("year", 2022)
        corr_cfg = self.config.get("corrections", {})
        year_str = corr_cfg.get("electronSF_name") or {
            2022: "2022Re-recoBCD",
            2023: "2022Re-recoBCD",
            2024: "2022Re-recoBCD",
        }.get(year, "2022Re-recoBCD")
        working_point = corr_cfg.get("electronSF_WP", "Tight")
        return str(year_str), str(working_point)

    def get_electron_sf(
        self,
        electrons: ak.Array,
        systematic: str = "central"
    ) -> ak.Array:
        """
        Get electron scale factors for ID and reconstruction (nominal, up, or down).
        Uses correction \"Electron-ID-SF\" with ValType sf / sfup / sfdown.
        """
        var = self.get_electron_sf_nominal_up_down(electrons)
        if systematic == "up":
            return var["up"]
        if systematic == "down":
            return var["down"]
        return var["central"]

    def get_electron_sf_nominal_up_down(
        self,
        electrons: ak.Array,
    ) -> Dict[str, ak.Array]:
        """
        Get electron scale factors as nominal, up, and down for systematics.
        Uses shared jagged-then-flat pattern (Coffea/correctionlib).
        Returns dict with keys \"central\", \"up\", \"down\" (same shape as electrons.pt).
        """
        ones = ak.ones_like(electrons.pt, dtype=float)
        out = {"central": ones, "up": ones, "down": ones}
        corr = self._get_electron_sf_correction()
        if corr is None:
            return out
        year_str, working_point = self._electron_sf_params()
        pt = electrons.pt
        eta = electrons.eta
        if len(ak.ravel(pt)) == 0:
            return out
        for val_type, key in [("sf", "central"), ("sfup", "up"), ("sfdown", "down")]:
            val = self._evaluate_correction_jagged_or_flat(
                corr, pt, year_str, val_type, working_point, eta, pt
            )
            out[key] = val if val is not None else ones
        return out

    def _get_electronHLT_sf_correction(self):
        """Return the Electron-HLT-SF correction evaluator from the loaded CorrectionSet."""
        if "electronHLT" not in self.corrections:
            return None
        cs = self.corrections["electronHLT"]
        for name in ("Electron-HLT-SF", "electron_hlt_sf", "electron_hlt"):
            try:
                return cs[name]
            except (IndexError, KeyError, TypeError):
                continue
        if hasattr(cs, "keys"):
            first = next(iter(cs.keys()), None)
            if first is not None:
                return cs[first]
        return None

    def get_electronHLT_sf_nominal_up_down(
        self,
        electrons: ak.Array,
    ) -> Dict[str, ak.Array]:
        """
        Get electron HLT trigger scale factors as central, up, and down.
        Uses Electron-HLT-SF with ValType sf/sfup/sfdown.
        Signature: evaluate(year, val_type, hlt_path, eta, pt).
        """
        ones = ak.ones_like(electrons.pt, dtype=float)
        out = {"central": ones, "up": ones, "down": ones}
        corr = self._get_electronHLT_sf_correction()
        if corr is None:
            return out
        corr_cfg = self.config.get("corrections", {})
        year_str = corr_cfg.get("electronHLT_name", "2022Re-recoBCD")
        hlt_path = corr_cfg.get("electronHLT_WP", "HLT_SF_Ele30_TightID")
        pt = electrons.pt
        eta = electrons.eta
        if len(ak.ravel(pt)) == 0:
            return out
        for val_type, key in [("sf", "central"), ("sfup", "up"), ("sfdown", "down")]:
            val = self._evaluate_correction_jagged_or_flat(
                corr, pt, year_str, val_type, hlt_path, eta, pt
            )
            out[key] = val if val is not None else ones
        return out

    def get_electronHLT_sf(
        self,
        electrons: ak.Array,
        systematic: str = "central",
    ) -> ak.Array:
        """Get electron HLT trigger scale factors (central, up, or down)."""
        var = self.get_electronHLT_sf_nominal_up_down(electrons)
        if systematic == "up":
            return var["up"]
        if systematic == "down":
            return var["down"]
        return var["central"]

    def _get_btag_sf_correction(self):
        """Return the b-tag SF correction (deepJet_shape). Safe lookup, no 'in cs'."""
        if "btagSF" not in self.corrections:
            return None
        cs = self.corrections["btagSF"]
        for name in ("deepJet_shape", "btagSF", "btag_sf", "BTV"):
            try:
                return cs[name]
            except (IndexError, KeyError, TypeError):
                continue
        try:
            keys = list(cs.keys()) if hasattr(cs, "keys") else []
            if keys:
                return cs[keys[0]]
        except Exception:
            pass
        return None

    def _evaluate_btag_sf(
        self,
        jets: ak.Array,
        corr,
        systematic: str,
    ) -> Optional[ak.Array]:
        """
        B-tag shape SF: evaluate(systematic, flavor, eta, pt, discriminator).
        Flavor: 0=udsg, 4=c, 5=b (hadronFlavour). Returns jagged SF array or None.
        """
        if corr is None:
            return None
        flavor = getattr(jets, "hadronFlavour", None)
        if flavor is None:
            flavor = ak.zeros_like(jets.pt, dtype=np.int32)
        eta = np.asarray(ak.ravel(np.abs(jets.eta)), dtype=float)
        pt = np.asarray(ak.ravel(jets.pt), dtype=float)
        discr = np.asarray(ak.ravel(jets.btagDeepFlavB), dtype=float)
        flavor_flat = np.asarray(ak.ravel(flavor), dtype=np.int32)
        n = len(pt)
        if n == 0:
            return ak.ones_like(jets.pt, dtype=float)
        try:
            sf = np.asarray(
                corr.evaluate(systematic, flavor_flat, eta, pt, discr),
                dtype=float,
            )
            if sf.shape != (n,):
                return None
            counts = np.asarray(ak.num(jets.pt, axis=1))
            return ak.unflatten(ak.Array(sf), counts)
        except Exception:
            return None

    def get_btag_sf(
        self,
        jets: ak.Array,
        systematic: str = "central"
    ) -> ak.Array:
        """
        Get b-tagging scale factors (deepJet_shape): evaluate(systematic, flavor, eta, pt, discriminator).
        Central/up/down via systematic string.
        """
        ones = ak.ones_like(jets.pt, dtype=float)
        if not jets.pt.layout or len(ak.ravel(jets.pt)) == 0:
            return ones
        corr = self._get_btag_sf_correction()
        out = self._evaluate_btag_sf(jets, corr, systematic)
        if out is not None:
            return out
        logging.warning("B-tag SF evaluate failed")
        return ones

    def get_btag_sf_nominal_up_down(self, jets: ak.Array) -> Dict[str, ak.Array]:
        """Get b-tag SF as central, up, down (deepJet_shape systematics)."""
        ones = ak.ones_like(jets.pt, dtype=float)
        out = {"central": ones, "up": ones, "down": ones}
        corr = self._get_btag_sf_correction()
        if corr is None:
            return out
        for key, syst in [("central", "central"), ("up", "up"), ("down", "down")]:
            val = self._evaluate_btag_sf(jets, corr, syst)
            if val is not None:
                out[key] = val
        return out

    def _get_rho(self, events: ak.Array) -> Optional[np.ndarray]:
        """Get per-event rho (median energy density) from NanoAOD events."""
        for field in ("fixedGridRhoFastjetAll", "Rho_fixedGridRhoFastjetAll", "rho"):
            try:
                val = getattr(events, field, None)
                if val is None and field in events.fields:
                    val = events[field]
                if val is not None:
                    return np.asarray(ak.to_numpy(val), dtype=float)
            except Exception:
                continue
        return None

    def get_jet_jec_weight_nominal_up_down(
        self,
        jets: ak.Array,
        events: ak.Array,
    ) -> Dict[str, ak.Array]:
        """
        Get per-event JEC weights (central, up, down) as product of per-jet JEC factors.
        Central: compound L1L2L3Res correction evaluated with (JetPt, JetEta, JetA, JetPhi, Rho).
        Up/down: central factor * (1 ± Total uncertainty).
        """
        n_events = len(events)
        ones = ak.Array(np.ones(n_events, dtype=float))
        out = {"central": ones, "up": ones, "down": ones}
        if "jetJEC" not in self.corrections:
            return out
        cs = self.corrections["jetJEC"]
        corr_cfg = self.config.get("corrections", {})
        tag = corr_cfg.get("jetJEC_tag", "")
        algo = corr_cfg.get("jetJEC_algo", "AK4PFPuppi")
        unc_name = corr_cfg.get("jetJEC_unc", "Total")

        n_flat = len(ak.ravel(jets.pt))
        if n_flat == 0:
            return out

        flat_pt = np.asarray(ak.ravel(jets.pt), dtype=float)
        flat_eta = np.asarray(ak.ravel(jets.eta), dtype=float)
        try:
            flat_phi = np.asarray(ak.ravel(jets.phi), dtype=float)
        except Exception:
            flat_phi = np.zeros(n_flat)
        try:
            flat_area = np.asarray(ak.ravel(jets.area), dtype=float)
        except Exception:
            flat_area = np.full(n_flat, 0.5)
        counts = np.asarray(ak.num(jets.pt, axis=1))
        rho_per_event = self._get_rho(events)
        flat_rho = np.repeat(rho_per_event, counts) if rho_per_event is not None else np.full(n_flat, 15.0)

        # Compound JEC correction (central)
        jec_central = np.ones(n_flat, dtype=float)
        compound_key = f"{tag}_L1L2L3Res_{algo}" if tag else None
        try:
            if compound_key:
                sf_obj = cs.compound[compound_key]
            else:
                keys = list(cs.compound.keys()) if hasattr(cs, "compound") else []
                sf_obj = cs.compound[keys[0]] if keys else None
            if sf_obj is not None:
                if isinstance(sf_obj, str):
                    sf_obj = cs.compound[sf_obj]
                jec_central = np.asarray(
                    sf_obj.evaluate(flat_pt, flat_eta, flat_area, flat_phi, flat_rho), dtype=float
                )
        except Exception as e:
            logging.warning(f"JEC compound correction evaluation failed: {e}")

        # JEC Total uncertainty (for up/down)
        jec_unc = np.zeros(n_flat, dtype=float)
        unc_key_full = f"{tag}_{unc_name}_{algo}" if tag else None
        try:
            unc_obj = cs[unc_key_full] if unc_key_full else None
            if unc_obj is not None:
                jec_unc = np.asarray(
                    unc_obj.evaluate(flat_pt * jec_central, flat_eta), dtype=float
                )
        except Exception as e:
            logging.warning(f"JEC uncertainty evaluation failed: {e}")

        for key, factors in [
            ("central", jec_central),
            ("up", jec_central * (1.0 + jec_unc)),
            ("down", jec_central * (1.0 - jec_unc)),
        ]:
            out[key] = self._per_event_product(ak.unflatten(ak.Array(factors), counts))
        return out

    def get_jet_jec_weight(
        self,
        jets: ak.Array,
        events: ak.Array,
        systematic: str = "central",
    ) -> ak.Array:
        """Get per-event JEC weight (central, up, or down)."""
        var = self.get_jet_jec_weight_nominal_up_down(jets, events)
        if systematic == "up":
            return var["up"]
        if systematic == "down":
            return var["down"]
        return var["central"]

    def _per_event_product(self, jagged_sf: ak.Array) -> ak.Array:
        """
        Reduce per-object scale factors to per-event: product over objects in each event.
        Events with no objects get 1.0 so they do not change the total weight.
        """
        try:
            prod = ak.prod(jagged_sf, axis=1)
            return ak.fill_none(prod, 1.0)
        except Exception:
            n_events = len(jagged_sf)
            return np.ones(n_events, dtype=float)

    def get_all_corrections(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        systematic: str = "central"
    ) -> Dict[str, Union[ak.Array, Dict[str, ak.Array]]]:
        """
        Get all corrections for an event sample.

        Each weight is the product over all selected objects in the event:
        - weight_btag: product of b-tag SF over all jets
        - weight_electron_id: product of electron ID SF over all electrons
        - weight_muon_id: product of muon ID (and iso if combined) SF over all muons
        Final event weight = pileup * generator * weight_muon_id * weight_electron_id * weight_btag * ...

        Returns:
            Dictionary of per-event weights. Values are either a single array (e.g. pileup)
            or a dict {"central", "up", "down"} so the weight calculator can combine
            central only for total weight and use up/down for systematics.
        """
        corrections = {}

        # Pileup (per-event, no systematic variations stored here)
        corrections["pileup"] = self.get_pileup_weight(events, systematic)

        # Muon: product of SF over all muons -> weight_muon_id (ID/iso as in correction file)
        if "tight_muons_pt30" in objects and len(ak.flatten(objects["tight_muons_pt30"])) > 0:
            mu_sf = self.get_muon_sf(objects["tight_muons_pt30"], systematic)
            mu_var = self.get_muon_sf_nominal_up_down(objects["tight_muons_pt30"])
            corrections["weight_muon_id"] = {
                "central": self._per_event_product(mu_sf),
                "up": self._per_event_product(mu_var["up"]),
                "down": self._per_event_product(mu_var["down"]),
            }

        # Electron: product of SF over all electrons -> weight_electron_id
        if "tight_electrons_pt30" in objects and len(ak.flatten(objects["tight_electrons_pt30"])) > 0:
            ele_sf = self.get_electron_sf(objects["tight_electrons_pt30"], systematic)
            ele_var = self.get_electron_sf_nominal_up_down(objects["tight_electrons_pt30"])
            corrections["weight_electron_id"] = {
                "central": self._per_event_product(ele_sf),
                "up": self._per_event_product(ele_var["up"]),
                "down": self._per_event_product(ele_var["down"]),
            }

        # B-tag: product of SF over all jets -> weight_btag
        if "jets" in objects and len(ak.flatten(objects["jets"])) > 0:
            btag_sf = self.get_btag_sf(objects["jets"], systematic)
            btag_var = self.get_btag_sf_nominal_up_down(objects["jets"])
            corrections["weight_btag"] = {
                "central": self._per_event_product(btag_sf),
                "up": self._per_event_product(btag_var["up"]),
                "down": self._per_event_product(btag_var["down"]),
            }

        # Electron HLT trigger SF: product of SF over all electrons -> weight_electronHLT
        if "tight_electrons_pt30" in objects and len(ak.flatten(objects["tight_electrons_pt30"])) > 0:
            hlt_sf = self.get_electronHLT_sf(objects["tight_electrons_pt30"], systematic)
            hlt_var = self.get_electronHLT_sf_nominal_up_down(objects["tight_electrons_pt30"])
            corrections["weight_electronHLT"] = {
                "central": self._per_event_product(hlt_sf),
                "up": self._per_event_product(hlt_var["up"]),
                "down": self._per_event_product(hlt_var["down"]),
            }

        # JEC: product of correction factors over all jets -> weight_JEC
        if "jets" in objects and len(ak.flatten(objects["jets"])) > 0:
            corrections["weight_JEC"] = self.get_jet_jec_weight_nominal_up_down(
                objects["jets"], events
            )

        return corrections

    def get_h_total_weight(self, events: ak.Array) -> float:
        """
        Compute the normalization sum from all events before any selection.
        For MC: sum of sign(genWeight) over all events (+1 or -1 per event).
        For data: always 1.0 (no generator weight normalization needed).
        Call this immediately after the input file is read (and lumi mask applied for data).
        """
        is_data = self.config.get("data", {}).get("is_data", False)
        if is_data:
            return 1.0
        try:
            _gw = np.asarray(ak.to_numpy(events.genWeight), dtype=float)
            return float(np.sum(np.sign(_gw)))
        except Exception:
            return float(len(events))

    def compute_event_weights(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
    ) -> Dict[str, ak.Array]:
        """
        Compute total event weight and per-systematic up/down variations.

        total_weight = genweight * weight_pileup * weight_btag * weight_muon
                       * weight_electron * weight_electronHLT * weight_JEC

        For each component X:
            weight_XUP   = (total_weight / weight_X_central) * weight_X_up
            weight_XDOWN = (total_weight / weight_X_central) * weight_X_down

        Returns dict with keys: total_weight, weight_pileupUP/DOWN,
        weight_btagUP/DOWN, weight_muonUP/DOWN, weight_electronUP/DOWN,
        weight_electronHLTUP/DOWN, weight_JECUP/DOWN.
        """
        n_events = len(events)
        ones = ak.Array(np.ones(n_events, dtype=float))

        # genWeight sign (+1/-1)
        try:
            _gw = np.asarray(ak.to_numpy(events.genWeight), dtype=float)
            genweight = ak.Array(np.sign(_gw).astype(float))
        except Exception:
            genweight = ones

        def _prod_or_ones(obj_key, sf_func, var_func):
            """Return (central, up, down) per-event product SFs, or ones if no objects."""
            obj = objects.get(obj_key)
            if obj is None or len(ak.ravel(obj.pt)) == 0:
                return ones, ones, ones
            var = var_func(obj)
            return (
                self._per_event_product(sf_func(obj)),
                self._per_event_product(var["up"]),
                self._per_event_product(var["down"]),
            )

        # Pileup (per-event scalar from correctionlib)
        w_pu = self.get_pileup_weight(events, "central")
        w_pu_up = self.get_pileup_weight(events, "up")
        w_pu_down = self.get_pileup_weight(events, "down")

        # B-tag
        w_btag, w_btag_up, w_btag_down = _prod_or_ones(
            "jets", self.get_btag_sf, self.get_btag_sf_nominal_up_down
        )

        # Muon
        w_muon, w_muon_up, w_muon_down = _prod_or_ones(
            "tight_muons_pt30", self.get_muon_sf, self.get_muon_sf_nominal_up_down
        )

        # Electron ID
        w_electron, w_electron_up, w_electron_down = _prod_or_ones(
            "tight_electrons_pt30", self.get_electron_sf, self.get_electron_sf_nominal_up_down
        )

        # Electron HLT
        w_ele_hlt, w_ele_hlt_up, w_ele_hlt_down = _prod_or_ones(
            "tight_electrons_pt30", self.get_electronHLT_sf, self.get_electronHLT_sf_nominal_up_down
        )

        # JEC
        jets = objects.get("jets")
        if jets is not None and len(ak.ravel(jets.pt)) > 0:
            jec_var = self.get_jet_jec_weight_nominal_up_down(jets, events)
            w_jec, w_jec_up, w_jec_down = jec_var["central"], jec_var["up"], jec_var["down"]
        else:
            w_jec = w_jec_up = w_jec_down = ones

        total_weight = genweight * w_pu * w_btag * w_muon * w_electron * w_ele_hlt * w_jec

        def _vary(total, central, variation):
            """Replace central component with variation: (total / central) * variation."""
            safe = ak.where(central != 0.0, central, ak.ones_like(central))
            return (total / safe) * variation

        return {
            "total_weight": total_weight,
            "weight_pileupUP": _vary(total_weight, w_pu, w_pu_up),
            "weight_pileupDOWN": _vary(total_weight, w_pu, w_pu_down),
            "weight_btagUP": _vary(total_weight, w_btag, w_btag_up),
            "weight_btagDOWN": _vary(total_weight, w_btag, w_btag_down),
            "weight_muonUP": _vary(total_weight, w_muon, w_muon_up),
            "weight_muonDOWN": _vary(total_weight, w_muon, w_muon_down),
            "weight_electronUP": _vary(total_weight, w_electron, w_electron_up),
            "weight_electronDOWN": _vary(total_weight, w_electron, w_electron_down),
            "weight_electronHLTUP": _vary(total_weight, w_ele_hlt, w_ele_hlt_up),
            "weight_electronHLTDOWN": _vary(total_weight, w_ele_hlt, w_ele_hlt_down),
            "weight_JECUP": _vary(total_weight, w_jec, w_jec_up),
            "weight_JECDOWN": _vary(total_weight, w_jec, w_jec_down),
        }

    def get_systematic_variations(self) -> list:
        """
        Get list of available systematic variations.

        Returns:
            List of systematic variation names
        """
        return ["central", "up", "down"]

    def is_available(self, correction_type: str) -> bool:
        """
        Check if a correction type is available.

        Args:
            correction_type: Type of correction to check

        Returns:
            True if correction is available
        """
        return correction_type in self.corrections
