"""
Unit tests for processor module.
"""

import pytest
import awkward as ak
import numpy as np
from darkbottomline.processor import DarkBottomLineProcessor


class TestDarkBottomLineProcessor:
    """Test DarkBottomLine processor."""

    def test_processor_initialization(self):
        """Test processor initialization."""
        config = {
            "year": 2023,
            "lumi": 35.9,
            "corrections": {
                "pileup": "data/corrections/pileup_2023.json.gz",
                "btagSF": "data/corrections/btagging_2023.json.gz",
                "muonSF": "data/corrections/muonSF_2023.json.gz",
                "electronSF": "data/corrections/electronSF_2023.json.gz"
            },
            "triggers": {
                "MET": ["HLT_PFMET120_PFMHT120_IDTight"]
            },
            "met_filters": ["Flag_goodVertices"],
            "objects": {
                "muons": {"pt_min": 20.0, "eta_max": 2.4, "id": "tight", "iso": "tight"},
                "electrons": {"pt_min": 20.0, "eta_max": 2.5, "id": "tight", "iso": "tight"},
                "taus": {"pt_min": 20.0, "eta_max": 2.3, "id": "tight", "decay_modes": [0, 1, 2, 10, 11]},
                "jets": {"pt_min": 30.0, "eta_max": 2.4, "id": "tight"},
                "fatjets": {"pt_min": 200.0, "eta_max": 2.4, "id": "tight"}
            },
            "btagging": {"algorithm": "deepJet", "wp": "medium"},
            "event_selection": {
                "min_muons": 0,
                "max_muons": 2,
                "min_electrons": 0,
                "max_electrons": 2,
                "min_taus": 0,
                "max_taus": 2,
                "min_jets": 2,
                "max_jets": 6,
                "min_bjets": 1,
                "met_min": 50.0
            },
            "cleaning": {
                "dr_muon_jet": 0.4,
                "dr_electron_jet": 0.4,
                "dr_tau_jet": 0.4
            }
        }

        processor = DarkBottomLineProcessor(config)

        # Check that processor is initialized
        assert processor.config == config
        assert processor.correction_manager is not None
        assert processor.histogram_manager is not None
        assert "histograms" in processor.accumulator
        assert "cutflow" in processor.accumulator
        assert "metadata" in processor.accumulator

    def test_processor_process(self):
        """Test processor event processing."""
        # Create mock events
        events = ak.Array({
            "event": [1, 2, 3, 4],
            "run": [1, 1, 1, 1],
            "luminosityBlock": [1, 1, 1, 1],
            "MET": {"pt": [60.0, 40.0, 80.0, 30.0], "phi": [0.0, 1.0, 2.0, 3.0]},
            "genWeight": [1.0, 1.0, 1.0, 1.0],
            "HLT": {
                "PFMET120_PFMHT120_IDTight": [True, False, True, True]
            },
            "Flag_goodVertices": [True, True, False, True],
            "Muon": {
                "pt": [[25.0, 30.0], [20.0], [35.0, 25.0], [15.0]],
                "eta": [[1.0, 2.0], [1.5], [0.8, 1.2], [2.5]],
                "phi": [[0.0, 1.0], [0.5], [1.5, 2.0], [3.0]],
                "tightId": [[1, 0], [1], [1, 1], [0]],
                "pfIsoId": [[3, 2], [3], [3, 3], [1]]
            },
            "Electron": {
                "pt": [[30.0], [25.0, 20.0], [40.0], [20.0]],
                "eta": [[1.2], [1.8, 2.1], [0.9], [2.6]],
                "phi": [[0.2], [0.7, 1.2], [1.7], [3.2]],
                "cutBased": [[4], [4, 2], [4], [2]],
                "pfIsoId": [[3], [3, 2], [3], [1]]
            },
            "Tau": {
                "pt": [[], [30.0], [35.0], []],
                "eta": [[], [1.6], [0.9], []],
                "phi": [[], [0.6], [1.6], []],
                "idDeepTau2017v2p1VSjet": [[], [16], [16], []],
                "decayMode": [[], [0], [1], []]
            },
            "Jet": {
                "pt": [[40.0, 35.0], [30.0], [45.0, 40.0], [25.0]],
                "eta": [[1.0, 2.0], [1.5], [0.8, 1.2], [2.5]],
                "phi": [[0.0, 1.0], [0.5], [1.5, 2.0], [3.0]],
                "jetId": [[2, 1], [2], [2, 2], [1]],
                "btagDeepFlavB": [[0.1, 0.5], [0.3], [0.8, 0.2], [0.1]],
                "hadronFlavour": [[5, 0], [5], [5, 0], [0]]
            },
            "FatJet": {
                "pt": [[250.0, 150.0], [200.0], [300.0], [100.0]],
                "eta": [[1.0, 2.0], [1.5], [0.8], [2.5]],
                "phi": [[0.0, 1.0], [0.5], [1.5], [3.0]],
                "jetId": [[2, 1], [2], [2], [1]]
            }
        })

        config = {
            "year": 2023,
            "lumi": 35.9,
            "corrections": {
                "pileup": "data/corrections/pileup_2023.json.gz",
                "btagSF": "data/corrections/btagging_2023.json.gz",
                "muonSF": "data/corrections/muonSF_2023.json.gz",
                "electronSF": "data/corrections/electronSF_2023.json.gz"
            },
            "triggers": {
                "MET": ["HLT_PFMET120_PFMHT120_IDTight"]
            },
            "met_filters": ["Flag_goodVertices"],
            "objects": {
                "muons": {"pt_min": 20.0, "eta_max": 2.4, "id": "tight", "iso": "tight"},
                "electrons": {"pt_min": 20.0, "eta_max": 2.5, "id": "tight", "iso": "tight"},
                "taus": {"pt_min": 20.0, "eta_max": 2.3, "id": "tight", "decay_modes": [0, 1, 2, 10, 11]},
                "jets": {"pt_min": 30.0, "eta_max": 2.4, "id": "tight"},
                "fatjets": {"pt_min": 200.0, "eta_max": 2.4, "id": "tight"}
            },
            "btagging": {"algorithm": "deepJet", "wp": "medium"},
            "event_selection": {
                "min_muons": 0,
                "max_muons": 2,
                "min_electrons": 0,
                "max_electrons": 2,
                "min_taus": 0,
                "max_taus": 2,
                "min_jets": 2,
                "max_jets": 6,
                "min_bjets": 1,
                "met_min": 50.0
            },
            "cleaning": {
                "dr_muon_jet": 0.4,
                "dr_electron_jet": 0.4,
                "dr_tau_jet": 0.4
            },
            "save_skims": False
        }

        processor = DarkBottomLineProcessor(config)
        results = processor.process(events)

        # Check that results are returned
        assert "histograms" in results
        assert "cutflow" in results
        assert "metadata" in results

        # Check metadata
        metadata = results["metadata"]
        assert "n_events_processed" in metadata
        assert "n_events_selected" in metadata
        assert "processing_time" in metadata
        assert "weight_statistics" in metadata

        # Check cutflow
        cutflow = results["cutflow"]
        assert "Total events" in cutflow
        assert "Final selection" in cutflow

    def test_processor_get_cutflow_summary(self):
        """Test cutflow summary generation."""
        config = {
            "year": 2023,
            "lumi": 35.9,
            "corrections": {},
            "triggers": {},
            "met_filters": [],
            "objects": {},
            "btagging": {},
            "event_selection": {},
            "cleaning": {}
        }

        processor = DarkBottomLineProcessor(config)

        # Test with empty cutflow
        summary = processor.get_cutflow_summary()
        assert "No cutflow data available" in summary

        # Test with cutflow data
        processor.accumulator["cutflow"] = {
            "Total events": 1000,
            "Pass trigger": 800,
            "Pass filters": 750,
            "Final selection": 500
        }

        summary = processor.get_cutflow_summary()
        assert "Total events" in summary
        assert "Pass trigger" in summary
        assert "Pass filters" in summary
        assert "Final selection" in summary

    def test_processor_get_processing_summary(self):
        """Test processing summary generation."""
        config = {
            "year": 2023,
            "lumi": 35.9,
            "corrections": {},
            "triggers": {},
            "met_filters": [],
            "objects": {},
            "btagging": {},
            "event_selection": {},
            "cleaning": {}
        }

        processor = DarkBottomLineProcessor(config)

        # Test with empty metadata
        summary = processor.get_processing_summary()
        assert "No processing data available" in summary

        # Test with metadata
        processor.accumulator["metadata"] = {
            "n_events_processed": 1000,
            "n_events_selected": 500,
            "processing_time": 10.5,
            "weight_statistics": {
                "mean": 1.0,
                "std": 0.1,
                "sum": 500.0
            }
        }

        summary = processor.get_processing_summary()
        assert "Events processed" in summary
        assert "Events selected" in summary
        assert "Processing time" in summary
        assert "Weight statistics" in summary


if __name__ == "__main__":
    pytest.main([__file__])
