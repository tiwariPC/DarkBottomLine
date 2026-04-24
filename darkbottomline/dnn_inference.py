"""
DNN inference for applying trained models to events.
"""

import torch
import numpy as np
import awkward as ak
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import yaml

from .dnn_trainer import ParametricDNN


class DNNInference:
    """
    DNN inference for applying trained models to events.
    """

    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize DNN inference.

        Args:
            model_path: Path to trained model file
            config_path: Path to DNN configuration file (optional)
        """
        self.model_path = model_path
        self.config_path = config_path

        # Load model and configuration
        self.model, self.config, self.scaler = self._load_model()

        # Get device
        self.device = self._get_device()
        self.model.to(self.device)
        self.model.eval()

        # Feature configuration
        self.feature_config = self.config.get("features", {})

        logging.info(f"DNN inference initialized with model from {model_path}")

    def _load_model(self) -> tuple:
        """
        Load trained model and configuration.

        Returns:
            Tuple of (model, config, scaler)
        """
        # Load model checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')

        # Get model configuration
        model_config = checkpoint.get('model_config', {})

        # Create model
        model = ParametricDNN(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get scaler
        scaler = checkpoint.get('scaler')

        # Load additional configuration if provided
        if self.config_path:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = model_config

        return model, config, scaler

    def _get_device(self) -> torch.device:
        """Get computation device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def extract_features(self, events: ak.Array, objects: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from events and objects.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects

        Returns:
            Feature array
        """
        features = []

        # Jet features
        jet_features = self._extract_jet_features(events, objects)
        if jet_features is not None:
            features.append(jet_features)

        # MET features
        met_features = self._extract_met_features(events)
        if met_features is not None:
            features.append(met_features)

        # Event features
        event_features = self._extract_event_features(events, objects)
        if event_features is not None:
            features.append(event_features)

        # Derived features
        derived_features = self._extract_derived_features(events, objects)
        if derived_features is not None:
            features.append(derived_features)

        # Combine all features
        if features:
            all_features = np.hstack(features)
        else:
            # Fallback: create dummy features
            n_events = len(events)
            n_features = self.config.get("model", {}).get("input_features", 20)
            all_features = np.random.randn(n_events, n_features)

        return all_features

    def _extract_jet_features(self, events: ak.Array, objects: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract jet features."""
        jet_config = self.feature_config.get("jet_features", [])
        if not jet_config:
            return None

        jets = objects.get("jets", ak.Array([]))
        if len(ak.flatten(jets)) == 0:
            return None

        features = []

        for feature in jet_config:
            if feature == "jet_pt":
                # Leading jet pT
                leading_jet_pt = ak.max(jets.pt, axis=1)
                features.append(ak.to_numpy(leading_jet_pt))
            elif feature == "jet_eta":
                # Leading jet eta
                leading_jet_eta = ak.max(jets.eta, axis=1)
                features.append(ak.to_numpy(leading_jet_eta))
            elif feature == "jet_phi":
                # Leading jet phi
                leading_jet_phi = ak.max(jets.phi, axis=1)
                features.append(ak.to_numpy(leading_jet_phi))
            elif feature == "jet_mass":
                # Leading jet mass
                if hasattr(jets, 'mass'):
                    leading_jet_mass = ak.max(jets.mass, axis=1)
                    features.append(ak.to_numpy(leading_jet_mass))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "jet_btag_score":
                # Leading jet b-tag score
                if hasattr(jets, 'btagDeepFlavB'):
                    leading_jet_btag = ak.max(jets.btagDeepFlavB, axis=1)
                    features.append(ak.to_numpy(leading_jet_btag))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "jet_id":
                # Leading jet ID
                if hasattr(jets, 'jetId'):
                    leading_jet_id = ak.max(jets.jetId, axis=1)
                    features.append(ak.to_numpy(leading_jet_id))
                else:
                    features.append(np.zeros(len(events)))

        if features:
            return np.column_stack(features)
        return None

    def _extract_met_features(self, events: ak.Array) -> Optional[np.ndarray]:
        """Extract MET features."""
        met_config = self.feature_config.get("met_features", [])
        if not met_config:
            return None

        features = []

        def _get_met_field_value(field_name_v15, field_name_v12, default_value_if_missing):
            if field_name_v15 in events.fields:
                return ak.to_numpy(events[field_name_v15])
            elif field_name_v12 in events.fields:
                return ak.to_numpy(events[field_name_v12])
            else:
                # Return an empty array of appropriate size if neither field exists
                return np.full(len(events), default_value_if_missing)

        for feature in met_config:
            if feature == "met_pt":
                features.append(_get_met_field_value("PFMET_pt", "MET_pt", 0.0))
            elif feature == "met_phi":
                features.append(_get_met_field_value("PFMET_phi", "MET_phi", 0.0))
            elif feature == "met_significance":
                features.append(_get_met_field_value("PFMET_significance", "MET_significance", 0.0))

        if features:
            return np.column_stack(features)
        return None

    def _extract_event_features(self, events: ak.Array, objects: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract event features."""
        event_config = self.feature_config.get("event_features", [])
        if not event_config:
            return None

        features = []

        for feature in event_config:
            if feature == "n_jets":
                n_jets = ak.num(objects.get("jets", ak.Array([])), axis=1)
                features.append(ak.to_numpy(n_jets))
            elif feature == "n_bjets":
                n_bjets = ak.num(objects.get("bjets", ak.Array([])), axis=1)
                features.append(ak.to_numpy(n_bjets))
            elif feature == "n_muons":
                n_muons = ak.num(objects.get("tight_muons", ak.Array([])), axis=1)
                features.append(ak.to_numpy(n_muons))
            elif feature == "n_electrons":
                n_electrons = ak.num(objects.get("tight_electrons", ak.Array([])), axis=1)
                features.append(ak.to_numpy(n_electrons))
            elif feature == "n_taus":
                n_taus = ak.num(objects.get("tight_taus", ak.Array([])), axis=1)
                features.append(ak.to_numpy(n_taus))
            elif feature == "ht":
                # HT calculation
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    ht = ak.sum(jets.pt, axis=1)
                    features.append(ak.to_numpy(ht))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "st":
                # ST calculation (HT + MET)
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    ht = ak.sum(jets.pt, axis=1)
                    met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
                    st = ht + met_pt
                    features.append(ak.to_numpy(st))
                else:
                    met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
                    features.append(ak.to_numpy(met_pt))

        if features:
            return np.column_stack(features)
        return None

    def _extract_derived_features(self, events: ak.Array, objects: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract derived features."""
        derived_config = self.feature_config.get("derived_features", [])
        if not derived_config:
            return None

        features = []

        for feature in derived_config:
            if feature == "delta_phi_met_jet":
                # DeltaPhi between MET and leading jet
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
                    jet_phi = jets.phi
                    delta_phi = ak.min(abs(jet_phi - met_phi), axis=1)
                    delta_phi = ak.fill_none(delta_phi, 0.0)
                    features.append(ak.to_numpy(delta_phi))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "delta_phi_met_jet2":
                # DeltaPhi between MET and second leading jet
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
                    jet_phi = jets.phi
                    # Sort jets by pT and get second leading
                    sorted_indices = ak.argsort(jets.pt, axis=1, ascending=False)
                    if ak.any(ak.num(jets, axis=1) >= 2):
                        second_jet_phi = ak.flatten(jet_phi[sorted_indices[:, 1:2]])
                        delta_phi = abs(second_jet_phi - met_phi)
                        delta_phi = ak.fill_none(delta_phi, 0.0)
                        features.append(ak.to_numpy(delta_phi))
                    else:
                        features.append(np.zeros(len(events)))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "delta_phi_met_jet3":
                # DeltaPhi between MET and third leading jet
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
                    jet_phi = jets.phi
                    # Sort jets by pT and get third leading
                    sorted_indices = ak.argsort(jets.pt, axis=1, ascending=False)
                    if ak.any(ak.num(jets, axis=1) >= 3):
                        third_jet_phi = ak.flatten(jet_phi[sorted_indices[:, 2:3]])
                        delta_phi = abs(third_jet_phi - met_phi)
                        delta_phi = ak.fill_none(delta_phi, 0.0)
                        features.append(ak.to_numpy(delta_phi))
                    else:
                        features.append(np.zeros(len(events)))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "jet1_pt":
                # Leading jet pT
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    leading_jet_pt = ak.max(jets.pt, axis=1)
                    features.append(ak.to_numpy(leading_jet_pt))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "jet2_pt":
                # Second leading jet pT
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    sorted_indices = ak.argsort(jets.pt, axis=1, ascending=False)
                    if ak.any(ak.num(jets, axis=1) >= 2):
                        second_jet_pt = ak.flatten(jets.pt[sorted_indices[:, 1:2]])
                        features.append(ak.to_numpy(second_jet_pt))
                    else:
                        features.append(np.zeros(len(events)))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "jet3_pt":
                # Third leading jet pT
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    sorted_indices = ak.argsort(jets.pt, axis=1, ascending=False)
                    if ak.any(ak.num(jets, axis=1) >= 3):
                        third_jet_pt = ak.flatten(jets.pt[sorted_indices[:, 2:3]])
                        features.append(ak.to_numpy(third_jet_pt))
                    else:
                        features.append(np.zeros(len(events)))
                else:
                    features.append(np.zeros(len(events)))
            elif feature == "jet4_pt":
                # Fourth leading jet pT
                jets = objects.get("jets", ak.Array([]))
                if len(ak.flatten(jets)) > 0:
                    sorted_indices = ak.argsort(jets.pt, axis=1, ascending=False)
                    if ak.any(ak.num(jets, axis=1) >= 4):
                        fourth_jet_pt = ak.flatten(jets.pt[sorted_indices[:, 3:4]])
                        features.append(ak.to_numpy(fourth_jet_pt))
                    else:
                        features.append(np.zeros(len(events)))
                else:
                    features.append(np.zeros(len(events)))

        if features:
            return np.column_stack(features)
        return None

    def predict(self, events: ak.Array, objects: Dict[str, Any], masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Make predictions on events.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            masses: Mass parameter array (optional)

        Returns:
            Predictions array
        """
        # Extract features
        features = self.extract_features(events, objects)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Convert to tensors
        features_tensor = torch.FloatTensor(features_scaled)

        # Handle mass parameter
        if masses is not None:
            masses_tensor = torch.FloatTensor(masses)
        else:
            # Default mass (signal mass)
            masses_tensor = torch.ones(len(events)) * 1000.0  # 1 TeV default

        # Make predictions
        with torch.no_grad():
            outputs = self.model(features_tensor.to(self.device), masses_tensor.to(self.device))
            predictions = outputs.cpu().numpy().flatten()

        return predictions

    def predict_batch(self, events_list: List[ak.Array], objects_list: List[Dict[str, Any]],
                     masses_list: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
        """
        Make predictions on multiple event batches.

        Args:
            events_list: List of event arrays
            objects_list: List of object dictionaries
            masses_list: List of mass arrays (optional)

        Returns:
            List of prediction arrays
        """
        predictions_list = []

        for i, (events, objects) in enumerate(zip(events_list, objects_list)):
            masses = masses_list[i] if masses_list else None
            predictions = self.predict(events, objects, masses)
            predictions_list.append(predictions)

        return predictions_list

    def get_feature_importance(self, events: ak.Array, objects: Dict[str, Any],
                              masses: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate feature importance using permutation method.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            masses: Mass parameter array (optional)

        Returns:
            Dictionary of feature importance scores
        """
        # Get baseline predictions
        baseline_predictions = self.predict(events, objects, masses)
        baseline_score = np.mean(baseline_predictions)

        # Extract features
        features = self.extract_features(events, objects)
        features_scaled = self.scaler.transform(features)

        # Calculate importance for each feature
        importance_scores = {}
        feature_names = self._get_feature_names()

        for i, feature_name in enumerate(feature_names):
            # Permute feature
            features_permuted = features_scaled.copy()
            np.random.shuffle(features_permuted[:, i])

            # Make predictions with permuted feature
            features_tensor = torch.FloatTensor(features_permuted)
            masses_tensor = torch.FloatTensor(masses) if masses is not None else torch.ones(len(events)) * 1000.0

            with torch.no_grad():
                outputs = self.model(features_tensor.to(self.device), masses_tensor.to(self.device))
                permuted_predictions = outputs.cpu().numpy().flatten()

            # Calculate importance as difference in score
            permuted_score = np.mean(permuted_predictions)
            importance = abs(baseline_score - permuted_score)
            importance_scores[feature_name] = importance

        return importance_scores

    def _get_feature_names(self) -> List[str]:
        """Get feature names from configuration."""
        feature_names = []

        # Jet features
        jet_features = self.feature_config.get("jet_features", [])
        feature_names.extend(jet_features)

        # MET features
        met_features = self.feature_config.get("met_features", [])
        feature_names.extend(met_features)

        # Event features
        event_features = self.feature_config.get("event_features", [])
        feature_names.extend(event_features)

        # Derived features
        derived_features = self.feature_config.get("derived_features", [])
        feature_names.extend(derived_features)

        return feature_names

    def save_predictions(self, predictions: np.ndarray, output_path: str):
        """
        Save predictions to file.

        Args:
            predictions: Predictions array
            output_path: Output file path
        """
        if output_path.endswith('.npy'):
            np.save(output_path, predictions)
        elif output_path.endswith('.npz'):
            np.savez(output_path, predictions=predictions)
        else:
            # Save as text file
            np.savetxt(output_path, predictions)

        logging.info(f"Predictions saved to {output_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "config_path": self.config_path,
            "device": str(self.device),
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
            "model_config": self.config.get("model", {}),
            "feature_config": self.feature_config
        }
