"""
DNN inference for applying trained models to new events.

Uses dnn.model / dnn.scaler / dnn.feature_engineering primitives.
Public API (DNNInference) preserved for cli.py and tests.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dnn.model import load_checkpoint
from dnn.scaler import StandardScaler as _StandardScaler
from dnn.feature_engineering import build_feature_frame_from_tree
from dnn.common import sanitize_feature_frame


def _parse_masspoint_label(label: str) -> Optional[Tuple[float, float]]:
    """Parse "MH3_<a>_MH4_<b>_Mchi_<c>" -> (MH3, MH4) floats, or None if unparseable."""
    import re
    m = re.match(r"MH3_(\d+(?:\.\d+)?)_MH4_(\d+(?:\.\d+)?)_Mchi_\d+(?:\.\d+)?$", str(label))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def _mass_branch_name(score_branch: str, mh3: float, mh4: float) -> str:
    def _fmt(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else str(v).replace(".", "p")
    return f"{score_branch}_mh3_{_fmt(mh3)}_mh4_{_fmt(mh4)}"


def _resolve_mass_scan(spec: Optional[str], inference) -> Optional[List[Tuple[float, float]]]:
    """Resolve a --dnn-mass-scan CLI value against a DNNInference's mass_grid.

    None (flag omitted) -> None (single-benchmark scoring, unchanged default).
    "all" -> every point in inference.mass_grid.
    "mh3_600_mh4_300,mh3_800_mh4_400" -> only those parsed points.
    Not applicable (non-parametric model) -> None regardless of spec.
    """
    if not getattr(inference, "_parametric", False) or spec is None:
        return None
    if str(spec).strip().lower() == "all":
        if not inference.mass_grid:
            raise ValueError("--dnn-mass-scan all requested but model has no mass_grid.")
        return [tuple(m) for m in inference.mass_grid]
    points = []
    for label in str(spec).split(","):
        label = label.strip()
        if not label:
            continue
        parsed = _parse_masspoint_label(label if label.startswith("MH3_") else label.upper())
        if parsed is None:
            raise ValueError(f"Could not parse masspoint label {label!r} for --dnn-mass-scan.")
        points.append(parsed)
    if not points:
        raise ValueError(f"--dnn-mass-scan={spec!r} did not yield any masspoints.")
    return points


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DNNInference:
    """
    DNN inference — loads a checkpoint produced by DNNTrainer and scores events.

    Wraps dnn.model.load_checkpoint + dnn.scaler.StandardScaler; exposes the
    same public API as before.
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path

        self.device = _resolve_device()

        # Load net + spec from checkpoint
        self._net, self._spec = load_checkpoint(model_path, map_location="cpu")
        self._net.to(self.device)
        self._net.eval()

        # Load scaler sidecar
        scaler_path = str(model_path).replace(".pt", "_scaler.json")
        self._scaler: Optional[_StandardScaler] = None
        if Path(scaler_path).exists():
            self._scaler = _StandardScaler.from_jsonable(
                json.loads(Path(scaler_path).read_text())
            )

        # Optional yaml config for feature/model info
        self.config: Dict[str, Any] = {}
        if config_path:
            import yaml
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        # parametric-ness and the mass grid are properties of the trained
        # checkpoint, not the (possibly unrelated) yaml passed at inference
        # time — prefer the checkpoint's own spec, falling back to config
        # only for checkpoints saved before this field existed.
        if self._spec.mass_grid is not None or self._spec.parametric:
            self._parametric = bool(self._spec.parametric)
        else:
            self._parametric = bool(self.config.get("model", {}).get("parametric_input", False))
        self.mass_grid: List[List[float]] = list(self._spec.mass_grid or [])
        if not self.mass_grid:
            cfg_grid = self.config.get("mass_grid")
            if isinstance(cfg_grid, list) and cfg_grid:
                self.mass_grid = [list(m) for m in cfg_grid]

        # Feature list, in priority order: dnn.yaml config -> features.json
        # sidecar (written by the trainer next to the checkpoint, the true
        # record of what this specific model was trained on). No further
        # fallback — an unresolvable feature list is a config problem, not
        # something to silently guess around.
        self.features: List[str] = list(self.config.get("features") or [])
        if not self.features:
            features_path = Path(model_path).parent / "features.json"
            if features_path.exists():
                try:
                    self.features = json.loads(features_path.read_text())
                except Exception:
                    self.features = []
        if not self.features:
            raise ValueError(
                f"Could not determine feature list for {model_path}: "
                f"no 'features' in config_path={config_path!r} and no "
                f"features.json next to the checkpoint. Pass --dnn-config "
                f"with a features: list, or ensure features.json exists."
            )

        logging.info(
            "DNNInference loaded from %s  (n_inputs=%d, n_features=%d, device=%s)",
            model_path, self._spec.n_inputs, len(self.features), self.device,
        )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_numpy(self, X: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
        """Scale X, optionally append (MH3, MH4) mass columns, run net, return scores."""
        X = X.astype("f8")
        if self._scaler is not None:
            X = self._scaler.transform(X)

        Xt = torch.from_numpy(X.astype("float32")).to(self.device)

        if self._parametric:
            if masses is None:
                if not self.mass_grid:
                    raise ValueError(
                        "Parametric model has no mass_grid (old checkpoint?) and no "
                        "masses were passed — cannot score without a mass hypothesis."
                    )
                benchmark = np.asarray(self.mass_grid[0], dtype="f8")
                masses = np.tile(benchmark, (len(X), 1))
            masses = np.asarray(masses, dtype="f8")
            if masses.ndim == 1:
                masses = np.tile(masses.reshape(-1, 1), (1, 2))  # legacy 1-column callers
            Mt = torch.from_numpy(masses.astype("float32")).to(self.device)
            Xt = torch.cat([Xt, Mt], dim=-1)

        with torch.no_grad():
            scores = torch.sigmoid(self._net(Xt).squeeze(1)).cpu().numpy()
        return scores

    # ------------------------------------------------------------------
    # Feature extraction from uproot trees (ppbbchichi ROOT format)
    # ------------------------------------------------------------------

    def extract_features(self, tree, max_events: Optional[int] = None) -> np.ndarray:
        """
        Extract this model's feature set (self.features) from an uproot tree.

        Args:
            tree: uproot TTree object
            max_events: optional event limit

        Returns:
            numpy array shape (N, len(self.features))
        """
        df, _, _ = build_feature_frame_from_tree(
            tree, self.features, max_events=max_events
        )
        df = sanitize_feature_frame(df)
        return df.to_numpy(dtype="f8")

    # ------------------------------------------------------------------
    # Public predict API
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Score pre-extracted feature array.

        Args:
            features: shape (N, n_features) numpy array
            masses: optional mass parameter, shape (N,)

        Returns:
            scores array shape (N,)
        """
        return self._score_numpy(features, masses)

    def predict_batch(
        self,
        features_list: List[np.ndarray],
        masses_list: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """
        Score multiple feature arrays.

        Args:
            features_list: list of (N_i, n_features) arrays
            masses_list: optional list of mass arrays

        Returns:
            list of score arrays
        """
        results = []
        for i, X in enumerate(features_list):
            masses = masses_list[i] if masses_list else None
            results.append(self._score_numpy(X, masses))
        return results

    def predict_from_tree(
        self, tree, masses: Optional[np.ndarray] = None, max_events: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract features from uproot tree and score in one call.

        Args:
            tree: uproot TTree
            masses: optional mass parameter array
            max_events: optional event limit

        Returns:
            scores array
        """
        X = self.extract_features(tree, max_events=max_events)
        return self._score_numpy(X, masses)

    # ------------------------------------------------------------------
    # Feature importance (permutation)
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        features: np.ndarray,
        masses: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Permutation importance over pre-extracted features.

        Args:
            features: (N, n_features) array
            masses: optional mass array
            feature_names: names for output dict; defaults to self.features
                (the model's resolved feature list from config/features.json)

        Returns:
            dict of {feature_name: importance_score}
        """
        if feature_names is None:
            feature_names = list(self.features[: features.shape[1]])

        baseline = float(np.mean(self._score_numpy(features, masses)))

        importance: Dict[str, float] = {}
        for i, name in enumerate(feature_names):
            perm = features.copy()
            np.random.shuffle(perm[:, i])
            permuted_score = float(np.mean(self._score_numpy(perm, masses)))
            importance[name] = abs(baseline - permuted_score)

        return importance

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def save_predictions(self, predictions: np.ndarray, output_path: str):
        """Save predictions to .npy, .npz, or text file."""
        if output_path.endswith(".npy"):
            np.save(output_path, predictions)
        elif output_path.endswith(".npz"):
            np.savez(output_path, predictions=predictions)
        else:
            np.savetxt(output_path, predictions)
        logging.info("Predictions saved to %s", output_path)

    def get_model_info(self) -> Dict[str, Any]:
        """Return dict with model metadata."""
        return {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "device": str(self.device),
            "n_inputs": self._spec.n_inputs,
            "features": list(self.features),
            "hidden_layers": list(self._spec.hidden_layers),
            "dropout": self._spec.dropout,
            "n_parameters": sum(p.numel() for p in self._net.parameters()),
            "parametric_input": self._parametric,
            "mass_grid": self.mass_grid,
            "scaler_loaded": self._scaler is not None,
        }
