"""
Unit tests for DNN functionality.
"""

import pytest
import torch
import numpy as np
import tempfile
import yaml
from pathlib import Path

from darkbottomline.dnn_trainer import ParametricDNN, DNNTrainer
from darkbottomline.dnn_inference import DNNInference


class TestParametricDNN:
    """Test ParametricDNN class."""

    def test_dnn_initialization(self):
        """Test DNN initialization."""
        config = {
            "input_features": 20,
            "hidden_layers": [128, 64, 32],
            "activation": "relu",
            "dropout": 0.2,
            "parametric_input": True,
            "output_activation": "sigmoid"
        }

        model = ParametricDNN(config)

        assert model.input_features == 20
        assert model.hidden_layers == [128, 64, 32]
        assert model.activation == "relu"
        assert model.dropout == 0.2
        assert model.parametric_input == True
        assert model.output_activation == "sigmoid"

    def test_forward_pass(self):
        """Test forward pass through the network."""
        config = {
            "input_features": 10,
            "hidden_layers": [64, 32],
            "activation": "relu",
            "dropout": 0.1,
            "parametric_input": True,
            "output_activation": "sigmoid"
        }

        model = ParametricDNN(config)

        # Create test data
        batch_size = 32
        n_features = 10
        features = torch.randn(batch_size, n_features)
        masses = torch.randn(batch_size, 2)  # (MH3, MH4)

        # Forward pass
        output = model(features, masses)

        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0)  # Sigmoid output should be >= 0
        assert torch.all(output <= 1)  # Sigmoid output should be <= 1

    def test_forward_pass_no_parametric(self):
        """Test forward pass without parametric input."""
        config = {
            "input_features": 10,
            "hidden_layers": [64, 32],
            "activation": "relu",
            "dropout": 0.1,
            "parametric_input": False,
            "output_activation": "sigmoid"
        }

        model = ParametricDNN(config)

        # Create test data
        batch_size = 32
        n_features = 10
        features = torch.randn(batch_size, n_features)

        # Forward pass without mass parameter
        output = model(features, None)

        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_activation_functions(self):
        """Test different activation functions."""
        activations = ["relu", "leaky_relu", "tanh", "sigmoid", "gelu"]

        for activation in activations:
            config = {
                "input_features": 5,
                "hidden_layers": [16],
                "activation": activation,
                "dropout": 0.0,
                "parametric_input": False,
                "output_activation": "sigmoid"
            }

            model = ParametricDNN(config)
            features = torch.randn(10, 5)
            output = model(features, None)

            assert output.shape == (10, 1)
            assert torch.all(output >= 0)
            assert torch.all(output <= 1)


class TestDNNTrainer:
    """Test DNNTrainer class."""

    @pytest.fixture
    def dnn_config(self):
        """Create temporary DNN configuration."""
        config = {
            "model": {
                "input_features": 10,
                "hidden_layers": [64, 32],
                "activation": "relu",
                "dropout": 0.1,
                "parametric_input": True,
                "output_activation": "sigmoid"
            },
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 5,
                "early_stopping": {
                    "patience": 3,
                    "min_delta": 0.001
                },
                "optimizer": "adam",
                "loss_function": "binary_crossentropy"
            },
            "preprocessing": {
                "feature_scaling": "standard",
                "train_validation_split": 0.8,
                "random_seed": 42,
                "shuffle_data": True
            },
            "output": {
                "model_path": "test_model.pt"
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_trainer_initialization(self, dnn_config):
        """Test DNNTrainer initialization."""
        trainer = DNNTrainer(dnn_config)

        assert trainer.model_config["input_features"] == 10
        assert trainer.training_config["batch_size"] == 32
        assert trainer.preprocessing_config["feature_scaling"] == "standard"

    def test_load_data(self, dnn_config):
        """Test data loading."""
        trainer = DNNTrainer(dnn_config)

        # Mock signal and background files
        signal_files = ["signal1.root", "signal2.root"]
        background_files = ["bkg1.root", "bkg2.root"]

        # This will use the placeholder implementation
        features, labels, masses = trainer.load_data(signal_files, background_files)

        assert len(features) > 0
        assert len(labels) > 0
        assert len(masses) > 0
        assert features.shape[1] == 10  # input_features
        assert len(labels) == len(features)
        assert len(masses) == len(features)

    def test_preprocess(self, dnn_config):
        """Test data preprocessing."""
        trainer = DNNTrainer(dnn_config)

        # Create mock data
        n_samples = 1000
        features = np.random.randn(n_samples, 10)
        labels = np.random.randint(0, 2, n_samples)
        masses = np.random.uniform(1000, 2000, n_samples)

        # Preprocess data
        train_features, train_labels, train_masses, val_features, val_labels, val_masses = trainer.preprocess(
            features, labels, masses
        )

        assert len(train_features) > 0
        assert len(val_features) > 0
        assert len(train_features) + len(val_features) == n_samples
        assert len(train_labels) == len(train_features)
        assert len(val_labels) == len(val_features)

    def test_train(self, dnn_config):
        """Test training loop."""
        trainer = DNNTrainer(dnn_config)

        # Create mock training data
        n_train = 100
        n_val = 20

        train_features = torch.randn(n_train, 10)
        train_labels = torch.randint(0, 2, (n_train,)).float()
        train_masses = torch.randn(n_train)

        val_features = torch.randn(n_val, 10)
        val_labels = torch.randint(0, 2, (n_val,)).float()
        val_masses = torch.randn(n_val)

        # Train model
        results = trainer.train(
            train_features, train_labels, train_masses,
            val_features, val_labels, val_masses
        )

        assert "best_val_loss" in results
        assert "final_epoch" in results
        assert "history" in results
        assert len(trainer.history["train_loss"]) > 0
        assert len(trainer.history["val_loss"]) > 0

    def test_evaluate(self, dnn_config):
        """Test model evaluation."""
        trainer = DNNTrainer(dnn_config)

        # Create mock data
        n_samples = 50
        features = torch.randn(n_samples, 10)
        labels = torch.randint(0, 2, (n_samples,)).float()
        masses = torch.randn(n_samples, 2)  # (MH3, MH4)

        # Evaluate model
        metrics = trainer.evaluate(features, labels, masses)

        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics

        # Check that metrics are reasonable
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_save_load_model(self, dnn_config):
        """Test model saving and loading."""
        trainer = DNNTrainer(dnn_config)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name

        try:
            # Save model
            trainer.save_model(model_path)
            assert Path(model_path).exists()

            # Load model
            trainer2 = DNNTrainer(dnn_config)
            trainer2.load_model(model_path)

            # Check that model state is preserved
            assert trainer2.model_config == trainer.model_config

        finally:
            # Cleanup
            Path(model_path).unlink()

    def test_predict(self, dnn_config):
        """Test model prediction."""
        trainer = DNNTrainer(dnn_config)

        # Create mock data
        n_samples = 20
        features = np.random.randn(n_samples, 10)
        masses = np.random.uniform(1000, 2000, (n_samples, 2))  # (MH3, MH4)

        # Make predictions
        predictions = trainer.predict(features, masses)

        assert len(predictions) == n_samples
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)


class TestDNNInference:
    """Test DNNInference class."""

    @pytest.fixture
    def mock_model_file(self):
        """Create a mock model file."""
        # Create a simple model
        config = {
            "input_features": 10,
            "hidden_layers": [32, 16],
            "activation": "relu",
            "dropout": 0.1,
            "parametric_input": True,
            "output_activation": "sigmoid"
        }

        model = ParametricDNN(config)

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': config,
            'scaler': None,
            'history': {}
        }, model_path)

        yield model_path

        # Cleanup
        Path(model_path).unlink()

    def test_inference_initialization(self, mock_model_file):
        """Test DNNInference initialization."""
        inference = DNNInference(mock_model_file)

        assert inference.model_path == mock_model_file
        assert inference.model_config["input_features"] == 10
        assert inference.device.type in ["cpu", "cuda", "mps"]

    def test_extract_features(self, mock_model_file):
        """Test feature extraction."""
        inference = DNNInference(mock_model_file)

        # Create mock events and objects
        events = {
            "MET": {"pt": [200, 300, 250]},
            "event": [1, 2, 3]
        }

        objects = {
            "jets": [
                [{"pt": 100, "eta": 0.1, "phi": 0.0}],
                [{"pt": 150, "eta": 0.2, "phi": 0.1}],
                [{"pt": 200, "eta": 0.3, "phi": 0.2}]
            ],
            "bjets": [
                [{"pt": 100, "eta": 0.1, "phi": 0.0}],
                [{"pt": 150, "eta": 0.2, "phi": 0.1}],
                []
            ]
        }

        # Extract features
        features = inference.extract_features(events, objects)

        assert features.shape[0] == 3  # Number of events
        assert features.shape[1] == 10  # Number of features

    def test_predict(self, mock_model_file):
        """Test model prediction."""
        inference = DNNInference(mock_model_file)

        # Create mock data
        n_events = 10
        features = np.random.randn(n_events, 10)
        masses = np.random.uniform(1000, 2000, n_events)

        # Make predictions
        predictions = inference.predict(features, masses)

        assert len(predictions) == n_events
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)

    def test_predict_batch(self, mock_model_file):
        """Test batch prediction."""
        inference = DNNInference(mock_model_file)

        # Create mock batch data
        n_batches = 3
        features_list = [np.random.randn(5, 10) for _ in range(n_batches)]
        masses_list = [np.random.uniform(1000, 2000, 5) for _ in range(n_batches)]

        # Make batch predictions
        predictions_list = inference.predict_batch(features_list, masses_list)

        assert len(predictions_list) == n_batches
        for predictions in predictions_list:
            assert len(predictions) == 5
            assert np.all(predictions >= 0)
            assert np.all(predictions <= 1)

    def test_get_feature_importance(self, mock_model_file):
        """Test feature importance calculation."""
        inference = DNNInference(mock_model_file)

        # Create mock data
        n_events = 20
        features = np.random.randn(n_events, 10)
        masses = np.random.uniform(1000, 2000, n_events)

        # Calculate feature importance
        importance = inference.get_feature_importance(features, masses)

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # All importance values should be non-negative
        assert all(v >= 0 for v in importance.values())

    def test_get_model_info(self, mock_model_file):
        """Test getting model information."""
        inference = DNNInference(mock_model_file)

        info = inference.get_model_info()

        assert "model_path" in info
        assert "device" in info
        assert "n_parameters" in info
        assert "model_config" in info
        assert "feature_config" in info

        assert info["model_path"] == mock_model_file
        assert info["n_parameters"] > 0


class TestDNNIntegration:
    """Integration tests for DNN functionality."""

    def test_full_training_workflow(self):
        """Test complete DNN training workflow."""
        # Create comprehensive DNN config
        config = {
            "model": {
                "input_features": 15,
                "hidden_layers": [128, 64, 32],
                "activation": "relu",
                "dropout": 0.2,
                "parametric_input": True,
                "output_activation": "sigmoid"
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 10,
                "early_stopping": {
                    "patience": 5,
                    "min_delta": 0.001
                },
                "optimizer": "adam",
                "loss_function": "binary_crossentropy"
            },
            "preprocessing": {
                "feature_scaling": "standard",
                "train_validation_split": 0.8,
                "random_seed": 42,
                "shuffle_data": True
            },
            "features": {
                "jet_features": ["jet_pt", "jet_eta", "jet_phi"],
                "met_features": ["met_pt", "met_phi"],
                "event_features": ["n_jets", "n_bjets", "ht"],
                "derived_features": ["delta_phi_met_jet", "jet1_pt"]
            },
            "output": {
                "model_path": "test_model.pt"
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            # Initialize trainer
            trainer = DNNTrainer(temp_path)

            # Create mock data
            n_samples = 500
            features = np.random.randn(n_samples, 15)
            labels = np.random.randint(0, 2, n_samples)
            masses = np.random.uniform(1000, 2000, n_samples)

            # Preprocess data
            train_features, train_labels, train_masses, val_features, val_labels, val_masses = trainer.preprocess(
                features, labels, masses
            )

            # Train model
            results = trainer.train(
                train_features, train_labels, train_masses,
                val_features, val_labels, val_masses
            )

            # Evaluate model
            metrics = trainer.evaluate(val_features, val_labels, val_masses)

            # Test inference
            inference = DNNInference(temp_path)
            predictions = inference.predict(features[:10], masses[:10])

            # Check results
            assert results["best_val_loss"] >= 0
            assert metrics["accuracy"] >= 0
            assert len(predictions) == 10
            assert np.all(predictions >= 0)
            assert np.all(predictions <= 1)

        finally:
            # Cleanup
            Path(temp_path).unlink()







