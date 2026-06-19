"""
DNN inference for applying trained models to new events.

Uses dnn.model / dnn.scaler / dnn.feature_engineering primitives.
Public API (DNNInference) preserved for cli.py and tests.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from dnn.model import load_checkpoint
from dnn.scaler import StandardScaler as _StandardScaler
from dnn.feature_engineering import REQUESTED_FEATURES_25, build_feature_frame_from_tree
from dnn.common import sanitize_feature_frame


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

        self._parametric = bool(self.config.get("model", {}).get("parametric_input", True))

        logging.info(
            "DNNInference loaded from %s  (n_inputs=%d, device=%s)",
            model_path, self._spec.n_inputs, self.device,
        )

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def _score_numpy(self, X: np.ndarray, masses: Optional[np.ndarray] = None) -> np.ndarray:
        """Scale X, optionally append mass column, run net, return scores."""
        X = X.astype("f8")
        if self._scaler is not None:
            X = self._scaler.transform(X)

        Xt = torch.from_numpy(X.astype("float32")).to(self.device)

        if self._parametric:
            if masses is None:
                masses = np.ones(len(X)) * 1000.0
            Mt = torch.from_numpy(masses.astype("float32")).unsqueeze(-1).to(self.device)
            Xt = torch.cat([Xt, Mt], dim=-1)

        with torch.no_grad():
            scores = torch.sigmoid(self._net(Xt).squeeze(1)).cpu().numpy()
        return scores

    # ------------------------------------------------------------------
    # Feature extraction from uproot trees (ppbbchichi ROOT format)
    # ------------------------------------------------------------------

    def extract_features(self, tree, max_events: Optional[int] = None) -> np.ndarray:
        """
        Extract the canonical 25 features from an uproot tree.

        Args:
            tree: uproot TTree object
            max_events: optional event limit

        Returns:
            numpy array shape (N, 25)
        """
        df, _, _ = build_feature_frame_from_tree(
            tree, REQUESTED_FEATURES_25, max_events=max_events
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
            feature_names: names for output dict; defaults to REQUESTED_FEATURES_25

        Returns:
            dict of {feature_name: importance_score}
        """
        if feature_names is None:
            feature_names = list(REQUESTED_FEATURES_25[: features.shape[1]])

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
            "hidden_layers": list(self._spec.hidden_layers),
            "dropout": self._spec.dropout,
            "n_parameters": sum(p.numel() for p in self._net.parameters()),
            "parametric_input": self._parametric,
            "scaler_loaded": self._scaler is not None,
        }
