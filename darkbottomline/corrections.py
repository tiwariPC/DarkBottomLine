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

from darkbottomline.selections import pass_triggers

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


# Reduced 11-source JES uncertainty scheme (JetMET POG grouping for analyses
# only mildly sensitive to JES): 6 sources fully correlated across all data-
# taking periods (no year suffix), plus 5 sources uncorrelated across
# periods (year-suffixed — e.g. "Absolute_2024" for a 2024 config,
# "Absolute_2023" for 2023; the year comes from config["year"], NOT a
# hardcoded constant, since this framework has a configs/*.yaml per year
# from 2022 through 2026). Each name must match a
# "{tag}_Regrouped_{name}_{algo}" correction in the jetJEC file (confirmed
# present as Summer24Prompt24_V5_MC_Regrouped_{name}_AK4PFPuppi for 2024;
# verify the equivalent exists in each other year's jet_jerc.json.gz before
# relying on this for a non-2024 config).
#
# Single source of truth: darkbottomline/objects.py (the 11x2=22-variant
# shifted-recoil rebuild), darkbottomline/variables.py (branch naming) and
# darkbottomline/plotting.py (_kin_systs) all call jes_sources(year) rather
# than redefining this list, so the three stay in sync by construction.
_JES_SOURCES_UNCORRELATED = ("Absolute", "BBEC1", "EC2", "HF", "FlavorQCD", "RelativeBal")
_JES_SOURCES_YEAR_SUFFIXED_BASE = ("Absolute", "BBEC1", "EC2", "HF", "RelativeSample")


def jes_sources(year) -> "tuple[str, ...]":
    """11 reduced-scheme JES uncertainty source names for the given data-taking
    year (int or str, e.g. 2024 or "2024") — 6 fully-correlated + 5 that are
    uncorrelated across years and therefore year-suffixed with the real year
    (e.g. "Absolute_2024"). The JetMET POG table writes this suffix
    generically as "_year" (meaning "gets a per-year suffix"), but both the
    jetJEC correction-file lookup ("{tag}_Regrouped_{name}_{algo}") and this
    framework's branch-naming convention use the REAL year number, not the
    literal word "year" — so these names double as both the correction-file
    lookup key and (with a "JEC" prefix) the branch-label source, no
    separate label-conversion step needed. See the module-level comment
    above for the full scheme description."""
    year_str = str(year)
    return _JES_SOURCES_UNCORRELATED + tuple(
        f"{src}_{year_str}" for src in _JES_SOURCES_YEAR_SUFFIXED_BASE
    )


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
    # Some correction JSONs use 'category' nodes that are structurally
    # equivalent to binning nodes for the purposes of edges/content fixing.
    # Treat 'category' as 'binning' here so we can repair non-monotonic edges
    # and make the JSON acceptable to correctionlib.from_string.
    if nodetype == "category":
        node["nodetype"] = "binning"
        nodetype = "binning"

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
        logging.debug(
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
        self._btag_efficiency_maps: Optional[Dict[str, Any]] = None
        self._btag_efficiency_maps_loaded = False

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
                    logging.debug(
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
                    logging.debug(f"Loaded {correction_type} corrections from {path_to_try} (gzip)")
                    loaded = True
                except Exception as e:
                    logging.warning(f"Failed to load {correction_type} (gzip): {e}")
            if not loaded:
                resolved_path = _resolve_correction_path(file_path)
                path_to_try = str(resolved_path) if resolved_path is not None else file_path
                try:
                    self.corrections[correction_type] = CorrectionSet.from_file(path_to_try)
                    logging.debug(f"Loaded {correction_type} corrections from {path_to_try}")
                    loaded = True
                except Exception as e:
                    err_msg = str(e).lower()
                    if "monotone" in err_msg or "monotonic" in err_msg:
                        cs = _load_correction_file_with_edges_fix(file_path)
                        if cs is not None:
                            self.corrections[correction_type] = cs
                            logging.debug(
                                f"Loaded {correction_type} corrections from {file_path} "
                                "(applied bin-edge fix for non-monotonic edges)"
                            )
                            loaded = True
                    if not loaded:
                        logging.warning(f"Failed to load {correction_type} corrections: {e}")

    def get_jet_id_mask(self, events: ak.Array) -> Optional[ak.Array]:
        """
        Per-jet tight jet-ID pass/fail mask from the BTV jetid CorrectionSet
        (config.corrections.jetid; config.corrections.jetid_name selects the WP,
        default AK4PUPPI_Tight — NOT TightLeptonVeto, since this analysis
        already does dedicated jet-lepton dR cleaning in objects.py
        (clean_jets_from_leptons), so TightLeptonVeto's built-in lepton veto
        would duplicate that; see https://twiki.cern.ch/twiki/bin/view/CMS/JetID13p6TeV).
        Evaluated on the raw per-jet energy-fraction/multiplicity branches
        (Jet_chHEF, Jet_neHEF, Jet_chEmEF, Jet_neEmEF, Jet_muEF,
        Jet_chMultiplicity, Jet_neMultiplicity — all present in NanoAODv15;
        "multiplicity" input is derived as chMultiplicity + neMultiplicity per
        the JSON's own description). Returns None if unconfigured/unavailable/
        any required branch is missing, so callers can fall back to no jet-ID
        cut (same as before this correction existed) rather than silently
        zeroing out jets.
        """
        corr = self._get_correction_by_name("jetid", "jetid_name", ("AK4PUPPI_Tight",))
        if corr is None:
            return None
        required = ("Jet_eta", "Jet_chHEF", "Jet_neHEF", "Jet_chEmEF",
                    "Jet_neEmEF", "Jet_muEF", "Jet_chMultiplicity", "Jet_neMultiplicity")
        if any(b not in events.fields for b in required):
            return None
        try:
            eta = events["Jet_eta"]
            chHEF = events["Jet_chHEF"]
            neHEF = events["Jet_neHEF"]
            chEmEF = events["Jet_chEmEF"]
            neEmEF = events["Jet_neEmEF"]
            muEF = events["Jet_muEF"]
            chMult = events["Jet_chMultiplicity"]
            neMult = events["Jet_neMultiplicity"]
            multiplicity = chMult + neMult
            result = self._evaluate_correction_jagged_or_flat(
                corr, events["Jet_pt"],
                eta, chHEF, neHEF, chEmEF, neEmEF, muEF, chMult, neMult, multiplicity,
            )
            if result is None:
                return None
            return ak.values_astype(result, bool) if not isinstance(result, bool) else result
        except Exception as e:
            logging.warning(f"Failed to evaluate jet ID correction: {e}")
            return None

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

    # ------------------------------------------------------------------
    # YAML-driven per-object scale factors (muon ID/iso, electron ID, electron HLT)
    #
    # Config block (top level of configs/{year}.yaml):
    #   lepton_sf:
    #     <sf_kind>:                 # muon | electron_id | electron_hlt
    #       <object_collection>:     # e.g. tight_muons, z_sublead_electrons
    #         - file: <key under corrections:>   # correction file path key
    #           name: <correction name inside that file>
    #           pt_range: [min, max]             # optional; SF = 1 outside (component not applicable)
    #           inputs: [...]                    # evaluate() args in order; "$pt", "$eta",
    #                                            # "$abseta", "$syst" are substituted, anything
    #                                            # else is passed through as a literal
    #           syst: {central: .., up: .., down: ..}   # string for "$syst"
    # Components for an object are multiplied. Every missing key / unknown
    # correction / failed evaluate raises (no silent fallback to 1.0).
    # ------------------------------------------------------------------
    _SF_COMPONENT_KEYS = ("file", "name", "inputs", "syst")

    def _sf_components(self, sf_kind: str, collection: str) -> list:
        try:
            components = self.config["lepton_sf"][sf_kind][collection]
        except (KeyError, TypeError) as e:
            raise KeyError(
                f"config is missing lepton_sf.{sf_kind}.{collection} "
                f"(YAML-driven scale factors, see CorrectionManager._sf_components)"
            ) from e
        if not isinstance(components, list):
            raise TypeError(f"lepton_sf.{sf_kind}.{collection} must be a list of components")
        for comp in components:
            missing = [k for k in self._SF_COMPONENT_KEYS if k not in comp]
            if missing:
                raise KeyError(f"lepton_sf.{sf_kind}.{collection} component {comp} missing keys {missing}")
        return components

    def _sf_correction(self, comp: dict):
        file_key, name = comp["file"], comp["name"]
        if file_key not in self.corrections:
            raise KeyError(
                f"correction file '{file_key}' (corrections.{file_key}) is not loaded; "
                f"check the path and that correctionlib can read it"
            )
        cs = self.corrections[file_key]
        try:
            return cs[name]
        except (IndexError, KeyError) as e:
            keys = list(cs.keys()) if hasattr(cs, "keys") else []
            raise KeyError(f"correction '{name}' not in {file_key}; available: {keys}") from e

    def _evaluate_sf_components(
        self, objects_: ak.Array, sf_kind: str, collection: str
    ) -> Dict[str, ak.Array]:
        """Return {"central","up","down"} jagged SFs (product over YAML components)."""
        components = self._sf_components(sf_kind, collection)
        pt_j = objects_.pt
        counts = np.asarray(ak.num(pt_j, axis=1))
        flat_pt = np.asarray(ak.ravel(pt_j), dtype=float)
        flat_eta = np.asarray(ak.ravel(objects_.eta), dtype=float)
        flat_abseta = np.abs(flat_eta)
        out = {k: np.ones(len(flat_pt), dtype=float) for k in ("central", "up", "down")}
        if len(flat_pt) > 0:
            for comp in components:
                corr = self._sf_correction(comp)
                lo, hi = comp.get("pt_range", (-np.inf, np.inf))
                mask = (flat_pt >= lo) & (flat_pt < hi)
                if not np.any(mask):
                    continue
                subst = {"$pt": flat_pt[mask], "$eta": flat_eta[mask], "$abseta": flat_abseta[mask]}
                for key in ("central", "up", "down"):
                    syst_str = comp["syst"][key]
                    args = [
                        syst_str if a == "$syst" else subst.get(a, a) if isinstance(a, str) else a
                        for a in comp["inputs"]
                    ]
                    out[key][mask] *= np.asarray(corr.evaluate(*args), dtype=float)
        return {k: ak.unflatten(v, counts) for k, v in out.items()}

    def get_muon_sf_nominal_up_down(
        self, muons: ak.Array, collection: str = "tight_muons"
    ) -> Dict[str, ak.Array]:
        return self._evaluate_sf_components(muons, "muon", collection)

    def get_muon_sf(
        self, muons: ak.Array, systematic: str = "central", collection: str = "tight_muons"
    ) -> ak.Array:
        return self.get_muon_sf_nominal_up_down(muons, collection)[systematic]

    def get_electron_sf_nominal_up_down(
        self, electrons: ak.Array, collection: str = "tight_electrons"
    ) -> Dict[str, ak.Array]:
        return self._evaluate_sf_components(electrons, "electron_id", collection)

    def get_electron_sf(
        self, electrons: ak.Array, systematic: str = "central", collection: str = "tight_electrons"
    ) -> ak.Array:
        return self.get_electron_sf_nominal_up_down(electrons, collection)[systematic]

    def get_electronHLT_sf_nominal_up_down(
        self, electrons: ak.Array, collection: str = "tight_electrons"
    ) -> Dict[str, ak.Array]:
        return self._evaluate_sf_components(electrons, "electron_hlt", collection)

    def get_electronHLT_sf(
        self, electrons: ak.Array, systematic: str = "central", collection: str = "tight_electrons"
    ) -> ak.Array:
        return self.get_electronHLT_sf_nominal_up_down(electrons, collection)[systematic]

    def _get_metHLT_sf_correction(self):
        """Return the MET-HLT-SF correction evaluator from the loaded CorrectionSet."""
        if "metHlt" not in self.corrections:
            return None
        cs = self.corrections["metHlt"]
        for name in ("MET-HLT-SF", "met_hlt_sf", "met_hlt"):
            try:
                return cs[name]
            except (IndexError, KeyError, TypeError):
                continue
        if hasattr(cs, "keys"):
            first = next(iter(cs.keys()), None)
            if first is not None:
                return cs[first]
        return None

    def get_metHLT_sf_nominal_up_down(self, recoil: ak.Array) -> Dict[str, ak.Array]:
        """
        Get MET trigger scale factors as central, up, and down.
        Uses MET-HLT-SF with ValType sf/sfup/sfdown. Signature: evaluate(val_type, recoil).
        Per-event (flat), unlike per-object electron/muon SFs.
        """
        ones = ak.ones_like(recoil, dtype=float)
        out = {"central": ones, "up": ones, "down": ones}
        corr = self._get_metHLT_sf_correction()
        if corr is None or len(ak.ravel(recoil)) == 0:
            return out
        flat_recoil = np.asarray(ak.ravel(recoil), dtype=float)
        for val_type, key in [("sf", "central"), ("sfup", "up"), ("sfdown", "down")]:
            try:
                val = np.asarray(corr.evaluate(val_type, flat_recoil), dtype=float)
                out[key] = ak.Array(val)
            except Exception as e:
                logging.warning(f"MET HLT SF evaluate failed for {val_type}: {e}")
        return out

    def get_metHLT_sf(
        self,
        recoil: ak.Array,
        systematic: str = "central",
    ) -> ak.Array:
        """Get MET trigger scale factors (central, up, or down)."""
        var = self.get_metHLT_sf_nominal_up_down(recoil)
        if systematic == "up":
            return var["up"]
        if systematic == "down":
            return var["down"]
        return var["central"]

    # sf_wp letter (config.btagging.sf_wp) -> efficiency-map histogram suffix
    # (scripts/btag_efficiency_slimmer.py's WORKING_POINTS: lwp/mwp/twp) and
    # -> the config.btagging discriminant-threshold key for that same WP.
    _EFF_MAP_WP_SUFFIX = {"L": "lwp", "M": "mwp", "T": "twp"}
    _WP_SCORE_KEY = {"L": "score_loose", "M": "score", "T": "score_tight"}

    def _load_btag_efficiency_maps(self) -> Optional[Dict[str, Any]]:
        """
        Load the MC-truth tagging-efficiency map (config.btagging.efficiency_map,
        a plain ROOT file from scripts/btag_efficiency_slimmer.py +
        btag_efficiency_maker.py — NOT a correctionlib JSON, loaded via uproot).

        Returns {"b": {"lwp": (eff_2d, eta_edges, pt_edges), "mwp": ..., "twp": ...},
                 "c": {...}, "light": {...}} or None if unavailable/unconfigured.
        Cached after first call (lazy, since this needs the full BTV fixed-WP
        reweighting formula, which isn't always exercised).
        """
        if self._btag_efficiency_maps_loaded:
            return self._btag_efficiency_maps
        self._btag_efficiency_maps_loaded = True

        file_path = self.config.get("btagging", {}).get("efficiency_map")
        if not file_path:
            return None
        resolved = _resolve_correction_path(file_path)
        path_to_open = str(resolved) if resolved is not None else file_path
        import uproot
        maps: Dict[str, Any] = {}
        with uproot.open(path_to_open) as f:
            for flavor in ("b", "c", "light"):
                maps[flavor] = {}
                for wp_suffix in ("lwp", "mwp", "twp"):
                    hist = f[f"hist_{flavor}_efficiency_{wp_suffix}"]
                    eff = hist.values()
                    eta_edges = hist.axis(0).edges()
                    pt_edges = hist.axis(1).edges()
                    maps[flavor][wp_suffix] = (eff, eta_edges, pt_edges)
        self._btag_efficiency_maps = maps
        logging.debug(f"Loaded b-tag efficiency maps from {path_to_open}")
        return self._btag_efficiency_maps

    def _lookup_btag_efficiency(
        self,
        flavor_flat: np.ndarray,
        eta: np.ndarray,
        pt: np.ndarray,
        wp_suffix: str,
    ) -> Optional[np.ndarray]:
        """Per-jet MC-truth tagging efficiency at the given working point, from
        the loaded efficiency map. `eta` is SIGNED (map binning is signed eta,
        matching the real Run2 production maps), `flavor_flat` is hadronFlavour
        (5=b, 4=c, else=light). Returns None if the map isn't available."""
        maps = self._load_btag_efficiency_maps()
        if maps is None:
            return None
        eff_out = np.zeros(len(pt), dtype=float)
        for flavor, flav_vals in (("b", (5,)), ("c", (4,)), ("light", None)):
            if flav_vals is not None:
                mask = np.isin(flavor_flat, flav_vals)
            else:
                mask = ~np.isin(flavor_flat, (4, 5))
            if not np.any(mask):
                continue
            eff_2d, eta_edges, pt_edges = maps[flavor][wp_suffix]
            eta_idx = np.clip(np.digitize(eta[mask], eta_edges) - 1, 0, len(eta_edges) - 2)
            pt_idx = np.clip(np.digitize(pt[mask], pt_edges) - 1, 0, len(pt_edges) - 2)
            eff_out[mask] = eff_2d[eta_idx, pt_idx]
        return eff_out

    def _btag_sf_config(self) -> Dict[str, Any]:
        """Read b-tag SF settings from config.btagging (config-driven, no hardcoding).

        sf_name: base correction name (e.g. 'particleNet', 'deepJet', 'UParTAK4').
        sf_type: 'shape'    -> continuous, correction '{sf_name}_shape',
                              evaluate(systematic, flavor, abseta, pt, discriminant).
                 'fixed_wp' -> per-WP, corrections '{sf_name}_comb' (flavor 4,5) and
                              '{sf_name}_light' (flavor 0),
                              evaluate(systematic, sf_wp, flavor, abseta, pt).
        sf_wp: WP letter (L/M/T/XT/XXT), required for fixed_wp.
        """
        btag = self.config.get("btagging", {})
        return {
            "name": btag.get("sf_name"),
            "type": btag.get("sf_type", "shape"),
            "wp": btag.get("sf_wp"),
        }

    def _get_btag_correction_by_name(self, name: str):
        """Look up a single correction inside the btagSF CorrectionSet by name."""
        if name is None or "btagSF" not in self.corrections:
            return None
        cs = self.corrections["btagSF"]
        try:
            return cs[name]
        except (IndexError, KeyError, TypeError):
            return None

    def _evaluate_btag_sf(
        self,
        jets: ak.Array,
        systematic: str,
        component: Optional[str] = None,
    ) -> Optional[ak.Array]:
        """
        B-tag SF for one systematic. Dispatches on config sf_type:
          shape    : one correction, evaluate(syst, flavor, abseta, pt, discr).
                     Returns the per-jet SF directly (continuous discriminant
                     already encodes tag status, no reweighting needed).
          fixed_wp : flavor {4,5} -> comb, flavor {0} -> light,
                     evaluate(syst, wp, flavor, abseta, pt); no discriminant.
                     Applies the full BTV fixed-WP reweighting formula when
                     an efficiency map is configured (config.btagging.
                     efficiency_map): per-jet factor is SF for tagged jets,
                     (1-SF*eff)/(1-eff) for untagged jets (same formula as the
                     Run2 reference, ExoPieUtils/scalefactortools/
                     btag_SFMaker.py::getJetWeight) — the per-event product
                     over ALL jets (not just tagged ones) is then the correct
                     event weight. Falls back to the raw per-jet SF (valid
                     only for tagged jets) if no efficiency map is available.

        `component` splits the per-jet factor into two independent weights,
        matching the Run2 reference's weightB/weightFakeB split
        (ExoPieAnalyzer/bbMETAnalyzer.py: `if flavlist[i] != 5: fakebWgt *=
        ... else: bWgt *= ...` — true b-jets vs everything else, each with
        its own product and its own up/down systematic, then multiplied
        together for the total b-tag weight):
          "true"   -> only true b-jets (hadronFlavour==5) get their real
                      factor; all other jets are set to 1.0 (no contribution).
          "mistag" -> only non-b jets (c and light) get their real factor;
                      b-jets are set to 1.0.
          None     -> no split, every jet contributes its own factor (used
                      internally when computing the un-split combined SF).
        Flavor: 0=udsg, 4=c, 5=b (hadronFlavour). Returns jagged per-jet
        factor array (multiply over all jets in the event) or None.
        """
        cfg = self._btag_sf_config()
        name = cfg["name"]
        if name is None:
            return None

        flavor = getattr(jets, "hadronFlavour", None)
        if flavor is None:
            flavor = ak.zeros_like(jets.pt, dtype=np.int32)
        eta_signed = np.asarray(ak.ravel(jets.eta), dtype=float)
        eta = np.abs(eta_signed)
        pt = np.asarray(ak.ravel(jets.pt), dtype=float)
        flavor_flat = np.asarray(ak.ravel(flavor), dtype=np.int32)
        n = len(pt)
        if n == 0:
            return ak.ones_like(jets.pt, dtype=float)
        counts = np.asarray(ak.num(jets.pt, axis=1))

        if cfg["type"] == "fixed_wp" and cfg["wp"] is not None:
            # Load outside the broad try/except below so a missing/corrupt
            # efficiency-map file raises instead of silently disabling the
            # fixed-WP reweighting formula.
            self._load_btag_efficiency_maps()

        try:
            if cfg["type"] == "shape":
                corr = self._get_btag_correction_by_name(f"{name}_shape")
                if corr is None:
                    return None
                discr = np.asarray(ak.ravel(jets.btagScore), dtype=float)
                sf = np.asarray(
                    corr.evaluate(systematic, flavor_flat, eta, pt, discr),
                    dtype=float,
                )
            elif cfg["type"] == "fixed_wp":
                wp = cfg["wp"]
                if wp is None:
                    return None
                comb = self._get_btag_correction_by_name(f"{name}_comb")
                light = self._get_btag_correction_by_name(f"{name}_light")
                if comb is None or light is None:
                    return None
                # comb handles heavy flavor (b=5, c=4); light handles udsg (0).
                # Evaluate the appropriate correction per jet flavor bucket.
                sf = np.ones(n, dtype=float)
                is_light = flavor_flat == 0
                is_heavy = ~is_light
                if np.any(is_heavy):
                    sf[is_heavy] = np.asarray(
                        comb.evaluate(
                            systematic, wp,
                            flavor_flat[is_heavy], eta[is_heavy], pt[is_heavy],
                        ),
                        dtype=float,
                    )
                if np.any(is_light):
                    sf[is_light] = np.asarray(
                        light.evaluate(
                            systematic, wp,
                            flavor_flat[is_light], eta[is_light], pt[is_light],
                        ),
                        dtype=float,
                    )

                wp_suffix = self._EFF_MAP_WP_SUFFIX.get(wp)
                score_key = self._WP_SCORE_KEY.get(wp)
                eff = (
                    self._lookup_btag_efficiency(flavor_flat, eta_signed, pt, wp_suffix)
                    if wp_suffix is not None and score_key is not None else None
                )
                if eff is not None:
                    discr = np.asarray(ak.ravel(jets.btagScore), dtype=float)
                    is_tagged = discr > self.config["btagging"][score_key]
                    safe_eff = np.clip(eff, 0.0, 0.999999)
                    untagged_factor = (1.0 - sf * safe_eff) / (1.0 - safe_eff)
                    sf = np.where(is_tagged, sf, untagged_factor)
            else:
                return None

            if sf.shape != (n,):
                return None

            if component == "true":
                sf = np.where(flavor_flat == 5, sf, 1.0)
            elif component == "mistag":
                sf = np.where(flavor_flat == 5, 1.0, sf)

            return ak.unflatten(ak.Array(sf), counts)
        except Exception:
            return None

    def get_btag_sf(
        self,
        jets: ak.Array,
        systematic: str = "central",
        component: Optional[str] = None,
    ) -> ak.Array:
        """
        Get b-tagging scale factors for one systematic (central/up/down).
        SF type (shape vs fixed_wp) and correction name driven by config.btagging.
        """
        ones = ak.ones_like(jets.pt, dtype=float)
        if not jets.pt.layout or len(ak.ravel(jets.pt)) == 0:
            return ones
        out = self._evaluate_btag_sf(jets, systematic, component=component)
        if out is not None:
            return out
        logging.warning("B-tag SF evaluate failed")
        return ones

    def get_btag_sf_nominal_up_down(
        self, jets: ak.Array, component: Optional[str] = None
    ) -> Dict[str, ak.Array]:
        """Get b-tag SF as central, up, down. `component`: "true" (true
        b-jets only) or "mistag" (c+light jets only) — see _evaluate_btag_sf."""
        ones = ak.ones_like(jets.pt, dtype=float)
        out = {"central": ones, "up": ones, "down": ones}
        if "btagSF" not in self.corrections:
            return out
        for key, syst in [("central", "central"), ("up", "up"), ("down", "down")]:
            val = self._evaluate_btag_sf(jets, syst, component=component)
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

    # ------------------------------------------------------------------
    # JEC: proper jet-pT/mass rescaling + Type-1 MET recompute.
    #
    # NanoAOD's stored Jet_pt/Jet_mass are ALREADY JEC-corrected at central
    # production. The old get_jet_jec_weight_nominal_up_down (removed) took
    # the OLD, already-corrected Jet_pt as input to the L1L2L3Res compound
    # correction and used the result as a per-event WEIGHT — this silently
    # double-counted the correction (measured +37% inflation on a real DY
    # sample, scaling with njets) and never touched jet pt, so njets/n_bjets
    # category assignment never reflected any JEC update either.
    #
    # Per the official JERC recommendation (user-provided): "undo the
    # existing JECs and reapply the recommended versions" — i.e. recover
    # raw pt via Jet_rawFactor, then reapply the full chain from scratch.
    # Jet ID is unaffected by any of this: its energy-fraction branches
    # (Jet_chHEF etc.) are ratios invariant under the jet's overall pt/mass
    # rescaling, and get_jet_id_mask already reads them straight from
    # NanoAOD without recomputing — no interaction with the code below.
    #
    # JER smearing is intentionally NOT implemented (TODO(JER) below) —
    # deriving it requires GenJet dR/pt-window matching (hybrid method) or
    # Gaussian stochastic smearing when unmatched, a separate, non-trivial
    # procedure from the JEC chain. Central jets here are JES-corrected but
    # unsmeared; see the project plan doc for the explicit scope decision.
    # ------------------------------------------------------------------

    def _evaluate_jec_chain(
        self,
        flat_raw_pt: np.ndarray,
        flat_eta: np.ndarray,
        flat_phi: np.ndarray,
        flat_area: np.ndarray,
        flat_rho: np.ndarray,
        flat_run: Optional[np.ndarray],
        is_data: bool,
        jes_source: Optional[str] = None,
        jes_direction: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Per-jet JEC chain (Type1MET.pdf steps 3-6) on an ALREADY raw,
        muon-subtracted flat pT array: L1FastJet -> L2Relative ->
        [L2L3Residual if data] -> [(1 +- source uncertainty) if
        jes_source given, else central/unvaried]. Returns the corrected
        flat pt array (same shape as flat_raw_pt), or None if the required
        individual (non-compound) correction entries aren't found.

        jes_source: None for central (JES-corrected, no variation). A
        regrouped-source name ("Absolute", "BBEC1_2024", ...) together with
        jes_direction ("up"/"down") applies that source's fractional
        uncertainty via Regrouped_{source} (correctionlib "up"/"down"
        systematic on that single correction object) instead of the
        unvaried chain — used only by the 11-source JES systematic rebuild,
        not by the nominal/central path.
        """
        corr_cfg = self.config.get("corrections", {})
        tag = corr_cfg.get("jetJEC_tag", "")
        algo = corr_cfg.get("jetJEC_algo", "AK4PFPuppi")
        if "jetJEC" not in self.corrections or not tag:
            return None
        cs = self.corrections["jetJEC"]
        correction_keys = list(cs.keys())
        flavor = "_DATA_" if is_data else "_MC_"

        def _pick_key(level_name: str) -> Optional[str]:
            exact = f"{tag}_{level_name}_{algo}"
            if exact in correction_keys:
                return exact
            for candidate in correction_keys:
                if flavor in candidate and candidate.endswith(f"{level_name}_{algo}"):
                    return candidate
            for candidate in correction_keys:
                if candidate.endswith(f"{level_name}_{algo}"):
                    return candidate
            return None

        try:
            l1_key = _pick_key("L1FastJet")
            l2_key = _pick_key("L2Relative")
            if l1_key is None or l2_key is None:
                return None
            l1_corr = cs[l1_key]
            l2_corr = cs[l2_key]

            def _eval_factor(corr, pt_in: np.ndarray) -> np.ndarray:
                """correctionlib JEC levels return a multiplicative FACTOR
                (close to 1.0), not a corrected pT — the running pT must be
                multiplied by this, not replaced with it."""
                input_names = [i.name for i in corr.inputs]
                arg_map = {
                    "JetA": flat_area, "JetEta": flat_eta, "JetPt": pt_in,
                    "Rho": flat_rho, "JetPhi": flat_phi,
                    "run": flat_run if flat_run is not None else np.ones_like(pt_in),
                }
                args = [arg_map[name] for name in input_names]
                return np.asarray(corr.evaluate(*args), dtype=float)

            p_l1 = flat_raw_pt * _eval_factor(l1_corr, flat_raw_pt)
            p_2 = p_l1 * _eval_factor(l2_corr, p_l1)

            if is_data:
                res_key = _pick_key("L2L3Residual")
                p_out = p_2 * _eval_factor(cs[res_key], p_2) if res_key is not None else p_2
            else:
                p_out = p_2
                if jes_source is not None and jes_direction is not None:
                    regrouped_key = f"{tag}_Regrouped_{jes_source}_{algo}"
                    if regrouped_key not in correction_keys:
                        for candidate in correction_keys:
                            if candidate.endswith(f"Regrouped_{jes_source}_{algo}"):
                                regrouped_key = candidate
                                break
                        else:
                            return None
                    unc_corr = cs[regrouped_key]
                    # Regrouped_{source} takes only (JetEta, JetPt) and
                    # returns the fractional uncertainty MAGNITUDE (no
                    # "up"/"down" argument) — the +/- direction is applied
                    # here, not passed into evaluate().
                    delta = np.asarray(
                        unc_corr.evaluate(flat_eta, p_2), dtype=float
                    )
                    p_out = p_2 * (1.0 + delta) if jes_direction == "up" else p_2 * (1.0 - delta)
            return p_out
        except Exception as e:
            logging.warning(f"JEC chain evaluation failed: {e}")
            return None

    def get_corrected_jet_pt_mass(
        self, events: ak.Array, jets: Optional[ak.Array] = None,
        jes_source: Optional[str] = None, jes_direction: Optional[str] = None,
    ) -> Optional["tuple[ak.Array, ak.Array]"]:
        """
        Recover raw pt (Jet_pt * (1 - Jet_rawFactor)), muon-subtract, and
        reapply the JEC chain from scratch (see module-level note above).
        Returns (corrected_pt, corrected_mass) as jagged arrays matching
        events["Jet_pt"]'s structure, or None if jetJEC isn't configured or
        the required branches are missing (callers fall back to raw
        Jet_pt/Jet_mass, same "no correction, old behavior" philosophy as
        get_jet_id_mask). TODO(JER): corrected_pt is JES-corrected but not
        JER-smeared (see module-level note above).

        jes_source/jes_direction: forwarded to _evaluate_jec_chain for the
        11-source JES systematic rebuild; leave both None for central.
        """
        if "jetJEC" not in self.corrections:
            return None
        required = ("Jet_pt", "Jet_eta", "Jet_phi", "Jet_rawFactor")
        if any(b not in events.fields for b in required):
            return None
        try:
            jet_pt = events["Jet_pt"]
            n_flat = len(ak.ravel(jet_pt))
            if n_flat == 0:
                return jet_pt, events["Jet_mass"] if "Jet_mass" in events.fields else ak.zeros_like(jet_pt)

            flat_pt = np.asarray(ak.ravel(jet_pt), dtype=float)
            flat_eta = np.asarray(ak.ravel(events["Jet_eta"]), dtype=float)
            flat_phi = np.asarray(ak.ravel(events["Jet_phi"]), dtype=float)
            flat_rawfactor = np.asarray(ak.ravel(events["Jet_rawFactor"]), dtype=float)
            flat_musub = (
                np.asarray(ak.ravel(events["Jet_muonSubtrFactor"]), dtype=float)
                if "Jet_muonSubtrFactor" in events.fields else np.zeros(n_flat)
            )
            flat_area = (
                np.asarray(ak.ravel(events["Jet_area"]), dtype=float)
                if "Jet_area" in events.fields else np.full(n_flat, 0.5)
            )
            counts = np.asarray(ak.num(jet_pt, axis=1))
            rho_per_event = self._get_rho(events)
            flat_rho = np.repeat(rho_per_event, counts) if rho_per_event is not None else np.full(n_flat, 15.0)
            is_data = bool(self.config.get("data", {}).get("is_data", False))
            flat_run = (
                np.repeat(np.asarray(ak.to_numpy(events["run"]), dtype=float), counts)
                if is_data and "run" in events.fields else None
            )

            flat_raw_pt = flat_pt * (1.0 - flat_rawfactor)
            flat_raw_pt_musub = flat_raw_pt * (1.0 - flat_musub)

            corrected_flat_pt = self._evaluate_jec_chain(
                flat_raw_pt_musub, flat_eta, flat_phi, flat_area, flat_rho,
                flat_run, is_data, jes_source=jes_source, jes_direction=jes_direction,
            )
            if corrected_flat_pt is None:
                return None

            corrected_pt = ak.unflatten(ak.Array(corrected_flat_pt), counts)
            # Scale mass by the same overall pt ratio to keep mass/pt
            # consistent (JEC rescales the full 4-vector uniformly).
            if "Jet_mass" in events.fields:
                flat_mass = np.asarray(ak.ravel(events["Jet_mass"]), dtype=float)
                safe_pt = np.where(flat_pt != 0.0, flat_pt, 1.0)
                corrected_flat_mass = flat_mass * (corrected_flat_pt / safe_pt)
                corrected_mass = ak.unflatten(ak.Array(corrected_flat_mass), counts)
            else:
                corrected_mass = ak.zeros_like(corrected_pt)
            return corrected_pt, corrected_mass
        except Exception as e:
            logging.warning(f"Failed to compute corrected jet pt/mass: {e}")
            return None

    def get_type1_met_shift(
        self, events: ak.Array,
        jes_source: Optional[str] = None, jes_direction: Optional[str] = None,
    ) -> Optional["tuple[np.ndarray, np.ndarray]"]:
        """
        Type-1 MET shift (Type1MET.pdf Sec 1.2-1.5): per-event
        (delta_met_x, delta_met_y) to ADD to baseline MET_x/y (sign already
        folded in below — see the per-jet accumulation). Propagates the
        Jet branch UNION CorrT1METJet branch (merged, uniformly processed)
        through the same JEC chain as get_corrected_jet_pt_mass, computing
        ΔpT = p_full - p_L1 per eligible jet (p_full > 15 GeV, |eta| < 5.2,
        EM fraction < 0.9 where available — CorrT1METJet has no
        chEmEF/neEmEF branches in this NanoAOD version, so that cut is
        skipped for those entries per the PDF's own footnote).
        Returns None if jetJEC isn't configured or MET/jet branches are
        missing. TODO(JER): p_full is JES-corrected but not JER-smeared.
        """
        if "jetJEC" not in self.corrections:
            return None
        if not all(b in events.fields for b in ("Jet_pt", "Jet_eta", "Jet_phi", "Jet_rawFactor")):
            return None
        try:
            is_data = bool(self.config.get("data", {}).get("is_data", False))
            n_events = len(events)

            def _jet_branch_inputs():
                """(raw_pt_musub, eta, phi, area, muonSubtrDeltaPhi, em_frac_or_none, counts) for Jet_*."""
                jet_pt = events["Jet_pt"]
                counts = np.asarray(ak.num(jet_pt, axis=1))
                flat_pt = np.asarray(ak.ravel(jet_pt), dtype=float)
                n_flat = len(flat_pt)
                flat_rawfactor = np.asarray(ak.ravel(events["Jet_rawFactor"]), dtype=float)
                flat_musub = (
                    np.asarray(ak.ravel(events["Jet_muonSubtrFactor"]), dtype=float)
                    if "Jet_muonSubtrFactor" in events.fields else np.zeros(n_flat)
                )
                flat_raw_pt = flat_pt * (1.0 - flat_rawfactor)
                flat_raw_pt_musub = flat_raw_pt * (1.0 - flat_musub)
                flat_eta = np.asarray(ak.ravel(events["Jet_eta"]), dtype=float)
                flat_phi = np.asarray(ak.ravel(events["Jet_phi"]), dtype=float)
                flat_area = (
                    np.asarray(ak.ravel(events["Jet_area"]), dtype=float)
                    if "Jet_area" in events.fields else np.full(n_flat, 0.5)
                )
                flat_dphi = (
                    np.asarray(ak.ravel(events["Jet_muonSubtrDeltaPhi"]), dtype=float)
                    if "Jet_muonSubtrDeltaPhi" in events.fields else np.zeros(n_flat)
                )
                if "Jet_chEmEF" in events.fields and "Jet_neEmEF" in events.fields:
                    flat_emfrac = (
                        np.asarray(ak.ravel(events["Jet_chEmEF"]), dtype=float)
                        + np.asarray(ak.ravel(events["Jet_neEmEF"]), dtype=float)
                    )
                else:
                    flat_emfrac = None
                return flat_raw_pt_musub, flat_eta, flat_phi, flat_area, flat_dphi, flat_emfrac, counts

            def _corrt1_branch_inputs():
                """Same tuple shape as above, for CorrT1METJet_* (EM fraction always None)."""
                if "CorrT1METJet_rawPt" not in events.fields:
                    return None
                pt_ref = events["CorrT1METJet_rawPt"]
                counts = np.asarray(ak.num(pt_ref, axis=1))
                flat_rawpt = np.asarray(ak.ravel(pt_ref), dtype=float)
                n_flat = len(flat_rawpt)
                flat_musub = (
                    np.asarray(ak.ravel(events["CorrT1METJet_muonSubtrFactor"]), dtype=float)
                    if "CorrT1METJet_muonSubtrFactor" in events.fields else np.zeros(n_flat)
                )
                flat_raw_pt_musub = flat_rawpt * (1.0 - flat_musub)
                flat_eta = np.asarray(ak.ravel(events["CorrT1METJet_eta"]), dtype=float)
                flat_phi = np.asarray(ak.ravel(events["CorrT1METJet_phi"]), dtype=float)
                flat_area = (
                    np.asarray(ak.ravel(events["CorrT1METJet_area"]), dtype=float)
                    if "CorrT1METJet_area" in events.fields else np.full(n_flat, 0.5)
                )
                flat_dphi = (
                    np.asarray(ak.ravel(events["CorrT1METJet_muonSubtrDeltaPhi"]), dtype=float)
                    if "CorrT1METJet_muonSubtrDeltaPhi" in events.fields else np.zeros(n_flat)
                )
                return flat_raw_pt_musub, flat_eta, flat_phi, flat_area, flat_dphi, None, counts

            parts = [_jet_branch_inputs()]
            corrt1 = _corrt1_branch_inputs()
            if corrt1 is not None:
                parts.append(corrt1)

            rho_per_event = self._get_rho(events)
            flat_run_full = (
                np.asarray(ak.to_numpy(events["run"]), dtype=float)
                if is_data and "run" in events.fields else None
            )

            delta_met_x = np.zeros(n_events, dtype=float)
            delta_met_y = np.zeros(n_events, dtype=float)
            for raw_pt_musub, eta, phi, area, dphi, emfrac, counts in parts:
                n_flat = len(raw_pt_musub)
                if n_flat == 0:
                    continue
                flat_rho = np.repeat(rho_per_event, counts) if rho_per_event is not None else np.full(n_flat, 15.0)
                flat_run = np.repeat(flat_run_full, counts) if flat_run_full is not None else None

                p_l1 = self._eval_l1_only(raw_pt_musub, eta, phi, area, flat_rho, flat_run, is_data)
                if p_l1 is None:
                    continue
                p_full = self._evaluate_jec_chain(
                    raw_pt_musub, eta, phi, area, flat_rho, flat_run, is_data,
                    jes_source=jes_source, jes_direction=jes_direction,
                )
                if p_full is None:
                    continue

                eligible = (p_full > 15.0) & (np.abs(eta) < 5.2)
                if emfrac is not None:
                    eligible &= (emfrac < 0.9)

                delta_pt = np.where(eligible, p_full - p_l1, 0.0)
                event_idx = np.repeat(np.arange(len(counts)), counts)
                np.subtract.at(delta_met_x, event_idx, delta_pt * np.cos(phi + dphi))
                np.subtract.at(delta_met_y, event_idx, delta_pt * np.sin(phi + dphi))

            return delta_met_x, delta_met_y
        except Exception as e:
            logging.warning(f"Failed to compute Type-1 MET shift: {e}")
            return None

    def _eval_l1_only(
        self, flat_raw_pt, flat_eta, flat_phi, flat_area, flat_rho, flat_run, is_data,
    ) -> Optional[np.ndarray]:
        """L1FastJet-only correction (Type1MET.pdf's p_L1, the ΔpT baseline)."""
        corr_cfg = self.config.get("corrections", {})
        tag = corr_cfg.get("jetJEC_tag", "")
        algo = corr_cfg.get("jetJEC_algo", "AK4PFPuppi")
        if "jetJEC" not in self.corrections or not tag:
            return None
        cs = self.corrections["jetJEC"]
        correction_keys = list(cs.keys())
        flavor = "_DATA_" if is_data else "_MC_"

        def _pick_key(level_name: str) -> Optional[str]:
            exact = f"{tag}_{level_name}_{algo}"
            if exact in correction_keys:
                return exact
            for candidate in correction_keys:
                if flavor in candidate and candidate.endswith(f"{level_name}_{algo}"):
                    return candidate
            for candidate in correction_keys:
                if candidate.endswith(f"{level_name}_{algo}"):
                    return candidate
            return None

        try:
            l1_key = _pick_key("L1FastJet")
            if l1_key is None:
                return None
            l1_corr = cs[l1_key]
            input_names = [i.name for i in l1_corr.inputs]
            arg_map = {
                "JetA": flat_area, "JetEta": flat_eta, "JetPt": flat_raw_pt,
                "Rho": flat_rho, "JetPhi": flat_phi,
                "run": flat_run if flat_run is not None else np.ones_like(flat_raw_pt),
            }
            args = [arg_map[name] for name in input_names]
            l1_factor = np.asarray(l1_corr.evaluate(*args), dtype=float)
            return flat_raw_pt * l1_factor
        except Exception as e:
            logging.warning(f"L1FastJet-only evaluation failed: {e}")
            return None

    def get_jet_veto_mask(self, events: ak.Array) -> Optional[ak.Array]:
        """
        Per-EVENT jet-veto-map mask (True = event falls in a vetoed
        detector region). NOTE: not yet wired into event selection —
        this method exists, is callable, and is unit-testable, but
        build_objects/processor.py deliberately do NOT call it this pass
        (see project plan doc for the explicit scope decision). Mandatory
        for Run 3 analyses per JME recommendations once activated.

        Uses the manual tightLepVeto-equivalent jet-ID WP (NanoAODv15 has
        no Jet_jetId branch) + (chEmEF + neEmEF) < 0.9 + jet pt_min as the
        "minimal selection" gating which jets are checked against the map
        (config.corrections.jetvetomaps / jetvetomaps_name).
        """
        if "jetvetomaps" not in self.corrections:
            return None
        required = ("Jet_pt", "Jet_eta", "Jet_phi", "Jet_chEmEF", "Jet_neEmEF",
                    "Jet_chHEF", "Jet_neHEF", "Jet_muEF", "Jet_chMultiplicity", "Jet_neMultiplicity")
        if any(b not in events.fields for b in required):
            return None
        try:
            jetid_corr = self._get_correction_by_name(
                "jetid", "jetvetomaps_jetid_name", ("AK4PUPPI_TightLeptonVeto",)
            )
            pt_mask = events["Jet_pt"] > self.config.get("objects", {}).get("jets", {}).get("pt_min", 15.0)
            em_mask = (events["Jet_chEmEF"] + events["Jet_neEmEF"]) < 0.9
            minimal_mask = pt_mask & em_mask
            if jetid_corr is not None:
                id_result = self._evaluate_correction_jagged_or_flat(
                    jetid_corr, events["Jet_pt"],
                    events["Jet_eta"], events["Jet_chHEF"], events["Jet_neHEF"],
                    events["Jet_chEmEF"], events["Jet_neEmEF"], events["Jet_muEF"],
                    events["Jet_chMultiplicity"], events["Jet_neMultiplicity"],
                    events["Jet_chMultiplicity"] + events["Jet_neMultiplicity"],
                )
                if id_result is not None:
                    minimal_mask = minimal_mask & ak.values_astype(id_result, bool)

            cs = self.corrections["jetvetomaps"]
            name = self.config.get("corrections", {}).get("jetvetomaps_name")
            if not name:
                keys = list(cs.keys()) if hasattr(cs, "keys") else []
                name = keys[0] if keys else None
            if name is None:
                return None
            vetomap_corr = cs[name]

            eta = events["Jet_eta"][minimal_mask]
            phi = events["Jet_phi"][minimal_mask]
            flat_eta = np.asarray(ak.ravel(eta), dtype=float)
            flat_phi = np.asarray(ak.ravel(phi), dtype=float)
            n_flat = len(flat_eta)
            if n_flat == 0:
                return ak.Array(np.zeros(len(events), dtype=bool))
            flat_veto = np.asarray(
                vetomap_corr.evaluate("jetvetomap", flat_eta, flat_phi), dtype=float
            )
            counts = np.asarray(ak.num(eta, axis=1))
            per_jet_veto = ak.unflatten(flat_veto != 0.0, counts)
            return ak.values_astype(ak.any(per_jet_veto, axis=1), bool)
        except Exception as e:
            logging.warning(f"Failed to evaluate jet veto map: {e}")
            return None

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

    def get_weighted_total_events(self, events: ak.Array) -> float:
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

    def get_pdf_scale_weights(self, events: ak.Array) -> Dict[str, np.ndarray]:
        """
        PDF (envelope) and QCD scale (7-point) weights from LHEPdfWeight / LHEScaleWeight.

        PDF: NNPDF envelope — max/min over replicas 1-100 (index 0 = nominal already = 1.0).
        Scale: 7-point — indices [0,1,2,3,4,6,8], skip 5 (muR=2,muF=0.5) and 7 (muR=0.5,muF=2).
        Returns ratio weights (multiplicative factor on full_event_weight), ones if branches absent.
        """
        n = len(events)
        ones = np.ones(n, dtype=np.float32)

        # --- PDF envelope ---
        pdf_up = ones.copy()
        pdf_down = ones.copy()
        if "LHEPdfWeight" in events.fields:
            try:
                pdf_w = ak.to_numpy(events["LHEPdfWeight"])  # shape (n, 103)
                # replicas are indices 1-100; index 0 = nominal (=1.0 already normalized)
                replicas = pdf_w[:, 1:101]
                pdf_up   = np.max(replicas, axis=1).astype(np.float32)
                pdf_down = np.min(replicas, axis=1).astype(np.float32)
            except Exception as e:
                logging.debug("PDF weight failed: %s", e)

        # --- QCD scale 7-point envelope ---
        # Variations: (muR,muF) = (1,1)(1,2)(1,0.5)(2,1)(2,2)(skip:2,0.5)(0.5,1)(skip:0.5,2)(0.5,0.5)
        # Indices:       0      1      2      3      4       5              6         7              8
        # Skip 5 (muR=2,muF=0.5) and 7 (muR=0.5,muF=2). Exclude 0 (nominal=1.0) from envelope.
        scale_up = ones.copy()
        scale_down = ones.copy()
        if "LHEScaleWeight" in events.fields:
            try:
                sc_w = ak.to_numpy(events["LHEScaleWeight"])  # shape (n, 9)
                _idx = [1, 2, 3, 4, 6, 8]  # 6 non-central variations; skip 0 (nominal), 5, 7
                _idx = [i for i in _idx if i < sc_w.shape[1]]
                sc_sel = sc_w[:, _idx]
                scale_up   = np.max(sc_sel, axis=1).astype(np.float32)
                scale_down = np.min(sc_sel, axis=1).astype(np.float32)
            except Exception as e:
                logging.debug("Scale weight failed: %s", e)

        return {
            "pdf_up": pdf_up,
            "pdf_down": pdf_down,
            "scale_up": scale_up,
            "scale_down": scale_down,
        }

    def compute_event_weights(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
    ) -> Dict[str, ak.Array]:
        """
        Compute total event weight and per-systematic up/down variations.

        full_event_weight = genweight * weight_pileup * weight_btag * weight_muon
                            * weight_electron * weight_electronHLT * weight_metHLT

        For each component X:
            weight_XUP   = (full_event_weight / weight_X_central) * weight_X_up
            weight_XDOWN = (full_event_weight / weight_X_central) * weight_X_down

        Returns dict with keys: full_event_weight, weight_pileupUP/DOWN,
        weight_btagUP/DOWN (combined), weight_btag_trueUP/DOWN (true b-jets
        only), weight_btag_mistagUP/DOWN (c+light jets only), weight_muonUP/
        DOWN, weight_electronUP/DOWN, weight_electronHLTUP/DOWN,
        weight_metHLTUP/DOWN.

        JEC is NOT an event weight (see the module-level note above
        get_corrected_jet_pt_mass): it rescales jet pt/mass directly
        (objects.py's build_jet_collection/select_jets), so there's no
        weight_JEC component here. The JES systematic (11-source reduced
        scheme) is a kinematic variation — shifted Recoil_JEC<Source>Up/Down
        branches, not a weight — see objects.py/variables.py.
        """
        n_events = len(events)
        ones = ak.Array(np.ones(n_events, dtype=float))

        # genWeight sign (+1/-1)
        try:
            _gw = np.asarray(ak.to_numpy(events.genWeight), dtype=float)
            genweight = ak.Array(np.sign(_gw).astype(float))
        except Exception:
            genweight = ones

        def _prod_or_ones(obj_key, sf_func, var_func, **kwargs):
            """Return (central, up, down) per-event product SFs, or ones if no objects."""
            obj = objects.get(obj_key)
            if obj is None or len(ak.ravel(obj.pt)) == 0:
                return ones, ones, ones
            var = var_func(obj, **kwargs)
            return (
                self._per_event_product(sf_func(obj, **kwargs)),
                self._per_event_product(var["up"]),
                self._per_event_product(var["down"]),
            )

        # Pileup (per-event scalar from correctionlib)
        w_pu = self.get_pileup_weight(events, "central")
        w_pu_up = self.get_pileup_weight(events, "up")
        w_pu_down = self.get_pileup_weight(events, "down")

        # B-tag: full BTV fixed-WP reweighting over ALL selected jets (SF on
        # tagged jets, (1-SF*eff)/(1-eff) on untagged jets — see
        # _evaluate_btag_sf) when an efficiency map is configured
        # (config.btagging.efficiency_map). Split into two independent
        # components, matching the Run2 reference's weightB/weightFakeB
        # split (ExoPieAnalyzer/bbMETAnalyzer.py): "true" = true b-jets
        # only, "mistag" = c+light jets only — each gets its own product and
        # its own up/down systematic, then the two are multiplied together
        # for the total b-tag weight. Restricting to bjets only (the
        # previous fallback, before the efficiency map existed) would drop
        # the untagged-jet term the formula needs and is no longer correct
        # now that the map is available.
        w_btag_true, w_btag_true_up, w_btag_true_down = _prod_or_ones(
            "jets", self.get_btag_sf, self.get_btag_sf_nominal_up_down,
            component="true",
        )
        w_btag_mistag, w_btag_mistag_up, w_btag_mistag_down = _prod_or_ones(
            "jets", self.get_btag_sf, self.get_btag_sf_nominal_up_down,
            component="mistag",
        )
        w_btag = w_btag_true * w_btag_mistag
        w_btag_up = w_btag_true_up * w_btag_mistag_up
        w_btag_down = w_btag_true_down * w_btag_mistag_down

        # Muon
        w_muon, w_muon_up, w_muon_down = _prod_or_ones(
            "tight_muons", self.get_muon_sf, self.get_muon_sf_nominal_up_down,
            collection="tight_muons",
        )
        # Z-candidate subleading muon is loose-only (not in tight_muons) and
        # would otherwise never get an ID SF; fold it in using its own
        # lepton_sf.muon.z_sublead_muons components (loose ID).
        w_z_sub_mu, w_z_sub_mu_up, w_z_sub_mu_down = _prod_or_ones(
            "z_sublead_muons", self.get_muon_sf, self.get_muon_sf_nominal_up_down,
            collection="z_sublead_muons",
        )
        w_muon = w_muon * w_z_sub_mu
        w_muon_up = w_muon_up * w_z_sub_mu_up
        w_muon_down = w_muon_down * w_z_sub_mu_down

        # Electron ID
        w_electron, w_electron_up, w_electron_down = _prod_or_ones(
            "tight_electrons", self.get_electron_sf, self.get_electron_sf_nominal_up_down,
            collection="tight_electrons",
        )
        # Z-candidate subleading electron is loose-only (not in
        # tight_electrons) and would otherwise never get an ID SF; fold it in
        # using the Loose-WP correction (matches its Loose cutBased ID).
        z_sub_el = objects.get("z_sublead_electrons")
        if z_sub_el is not None and len(ak.ravel(z_sub_el.pt)) > 0:
            sub_var = self.get_electron_sf_nominal_up_down(
                z_sub_el, collection="z_sublead_electrons"
            )
            w_electron = w_electron * self._per_event_product(sub_var["central"])
            w_electron_up = w_electron_up * self._per_event_product(sub_var["up"])
            w_electron_down = w_electron_down * self._per_event_product(sub_var["down"])

        # Electron HLT. Same trigger-OR issue as MET HLT: an event can have a
        # tight electron but have fired only via the MET leg (trigger_cut is
        # MET|EGamma at preselection), so this per-electron HLT-efficiency SF
        # must not apply unless the event actually fired the EGamma trigger.
        w_ele_hlt, w_ele_hlt_up, w_ele_hlt_down = _prod_or_ones(
            "tight_electrons", self.get_electronHLT_sf, self.get_electronHLT_sf_nominal_up_down,
            collection="tight_electrons",
        )
        ele_trig_paths = self.config.get("triggers", {}).get("EGamma", [])
        fired_ele_trigger = np.asarray(ak.to_numpy(pass_triggers(events, ele_trig_paths)), dtype=bool)
        w_ele_hlt = ak.where(fired_ele_trigger, w_ele_hlt, ones)
        w_ele_hlt_up = ak.where(fired_ele_trigger, w_ele_hlt_up, ones)
        w_ele_hlt_down = ak.where(fired_ele_trigger, w_ele_hlt_down, ones)

        # MET HLT (per-event, on hadronic recoil). Only valid for events that
        # actually fired the MET trigger — electron-only-triggered events
        # (e.g. Wenu/Zee/Topenu CRs) must not get a MET-trigger-efficiency
        # correction folded in, so gate per-event on the real trigger bit.
        recoil = objects.get("recoil")
        if recoil is not None and len(ak.ravel(recoil)) > 0:
            met_hlt_var = self.get_metHLT_sf_nominal_up_down(recoil)
            met_trig_paths = self.config.get("triggers", {}).get("MET", [])
            fired_met_trigger = np.asarray(ak.to_numpy(pass_triggers(events, met_trig_paths)), dtype=bool)
            w_met_hlt = ak.where(fired_met_trigger, met_hlt_var["central"], ones)
            w_met_hlt_up = ak.where(fired_met_trigger, met_hlt_var["up"], ones)
            w_met_hlt_down = ak.where(fired_met_trigger, met_hlt_var["down"], ones)
        else:
            w_met_hlt = w_met_hlt_up = w_met_hlt_down = ones

        sf_weight = w_pu * w_btag * w_muon * w_electron * w_ele_hlt * w_met_hlt
        full_event_weight = genweight * sf_weight

        def _vary(total, central, variation):
            """Replace central component with variation: (total / central) * variation."""
            safe = ak.where(central != 0.0, central, ak.ones_like(central))
            return (total / safe) * variation

        # PDF and QCD scale weights (multiplicative on full_event_weight)
        pdf_scale = self.get_pdf_scale_weights(events)
        _fw = ak.to_numpy(full_event_weight).astype(np.float32)

        return {
            "sf_weight": sf_weight,
            "full_event_weight": full_event_weight,
            # full_event_weight with the pileup SF removed (all other SFs kept).
            # For the "no PU reweight" Pileup_nTrueInt cross-check plot.
            "weight_noPileup": _vary(full_event_weight, w_pu, ones),
            "weight_pileupUP": _vary(full_event_weight, w_pu, w_pu_up),
            "weight_pileupDOWN": _vary(full_event_weight, w_pu, w_pu_down),
            "weight_btagUP": _vary(full_event_weight, w_btag, w_btag_up),
            "weight_btagDOWN": _vary(full_event_weight, w_btag, w_btag_down),
            # Independent b-tag components (Run2 weightB/weightFakeB split):
            # heavy = true b-jets only, mistag = c+light jets only. Each
            # varies only its own factor, holding the other at central.
            "weight_btag_trueUP": _vary(full_event_weight, w_btag_true, w_btag_true_up),
            "weight_btag_trueDOWN": _vary(full_event_weight, w_btag_true, w_btag_true_down),
            "weight_btag_mistagUP": _vary(full_event_weight, w_btag_mistag, w_btag_mistag_up),
            "weight_btag_mistagDOWN": _vary(full_event_weight, w_btag_mistag, w_btag_mistag_down),
            "weight_muonUP": _vary(full_event_weight, w_muon, w_muon_up),
            "weight_muonDOWN": _vary(full_event_weight, w_muon, w_muon_down),
            "weight_electronUP": _vary(full_event_weight, w_electron, w_electron_up),
            "weight_electronDOWN": _vary(full_event_weight, w_electron, w_electron_down),
            "weight_electronHLTUP": _vary(full_event_weight, w_ele_hlt, w_ele_hlt_up),
            "weight_electronHLTDOWN": _vary(full_event_weight, w_ele_hlt, w_ele_hlt_down),
            "weight_metHLTUP": _vary(full_event_weight, w_met_hlt, w_met_hlt_up),
            "weight_metHLTDOWN": _vary(full_event_weight, w_met_hlt, w_met_hlt_down),
            "weight_pdfUP":   (_fw * pdf_scale["pdf_up"]).astype(np.float32),
            "weight_pdfDOWN": (_fw * pdf_scale["pdf_down"]).astype(np.float32),
            "weight_scaleUP":   (_fw * pdf_scale["scale_up"]).astype(np.float32),
            "weight_scaleDOWN": (_fw * pdf_scale["scale_down"]).astype(np.float32),
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
