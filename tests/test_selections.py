"""
Unit tests for selections module.
"""

import pytest
import awkward as ak
import numpy as np
from darkbottomline.selections import (
    pass_triggers, pass_met_filters, select_events, get_cutflow, apply_selection
)


class TestEventSelection:
    """Test event selection functions."""

    def test_pass_triggers(self):
        """Test trigger selection."""
        # Create mock events with HLT branches
        events = ak.Array({
            "event": [1, 2, 3],
            "HLT": {
                "PFMET120_PFMHT120_IDTight": [True, False, True],
                "IsoMu24": [False, True, True]
            }
        })

        trigger_paths = ["HLT_PFMET120_PFMHT120_IDTight", "HLT_IsoMu24"]
        mask = pass_triggers(events, trigger_paths)

        # Check trigger selection
        assert mask[0] == True   # First event passes MET trigger
        assert mask[1] == True   # Second event passes muon trigger
        assert mask[2] == True   # Third event passes both triggers

    def test_pass_triggers_empty(self):
        """Test trigger selection with empty paths."""
        events = ak.Array({
            "event": [1, 2, 3],
            "HLT": {
                "PFMET120_PFMHT120_IDTight": [True, False, True]
            }
        })

        trigger_paths = []
        mask = pass_triggers(events, trigger_paths)

        # All events should pass if no triggers specified
        assert all(mask)

    def test_pass_met_filters(self):
        """Test MET filter selection."""
        # Create mock events with MET filter branches
        events = ak.Array({
            "event": [1, 2, 3],
            "Flag_goodVertices": [True, True, False],
            "Flag_HBHENoiseFilter": [True, False, True],
            "Flag_BadPFMuonFilter": [True, True, True]
        })

        filter_names = ["Flag_goodVertices", "Flag_HBHENoiseFilter", "Flag_BadPFMuonFilter"]
        mask = pass_met_filters(events, filter_names)

        # Check MET filter selection
        assert mask[0] == True   # First event passes all filters
        assert mask[1] == False  # Second event fails HBHE filter
        assert mask[2] == False  # Third event fails goodVertices filter

    def test_pass_met_filters_empty(self):
        """Test MET filter selection with empty filters."""
        events = ak.Array({
            "event": [1, 2, 3],
            "Flag_goodVertices": [True, True, False]
        })

        filter_names = []
        mask = pass_met_filters(events, filter_names)

        # All events should pass if no filters specified
        assert all(mask)

    def test_select_events(self):
        """Test event selection."""
        # Create mock events and objects
        events = ak.Array({
            "event": [1, 2, 3, 4],
            "MET": {"pt": [60.0, 40.0, 80.0, 30.0]}
        })

        objects = {
            "muons": ak.Array({
                "pt": [[25.0, 30.0], [20.0], [35.0, 25.0], [15.0]]
            }),
            "electrons": ak.Array({
                "pt": [[30.0], [25.0, 20.0], [40.0], [20.0]]
            }),
            "taus": ak.Array({
                "pt": [[], [30.0], [35.0], []]
            }),
            "jets": ak.Array({
                "pt": [[40.0, 35.0], [30.0], [45.0, 40.0], [25.0]]
            }),
            "bjets": ak.Array({
                "pt": [[40.0], [30.0], [45.0], []]
            })
        }

        config = {
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
            }
        }

        mask = select_events(events, objects, config)

        # Check event selection
        assert mask[0] == True   # First event: 2 muons, 1 electron, 0 taus, 2 jets, 1 bjet, MET > 50
        assert mask[1] == False  # Second event: 1 muon, 2 electrons, 1 tau, 1 jet, 1 bjet, MET < 50
        assert mask[2] == True   # Third event: 2 muons, 1 electron, 1 tau, 2 jets, 1 bjet, MET > 50
        assert mask[3] == False  # Fourth event: 1 muon, 1 electron, 0 taus, 1 jet, 0 bjets, MET < 50

    def test_get_cutflow(self):
        """Test cutflow calculation."""
        # Create mock events and objects
        events = ak.Array({
            "event": [1, 2, 3, 4, 5],
            "MET": {"pt": [60.0, 40.0, 80.0, 30.0, 70.0]},
            "HLT": {
                "PFMET120_PFMHT120_IDTight": [True, False, True, True, False]
            },
            "Flag_goodVertices": [True, True, False, True, True]
        })

        objects = {
            "muons": ak.Array({
                "pt": [[25.0, 30.0], [20.0], [35.0, 25.0], [15.0], [40.0]]
            }),
            "electrons": ak.Array({
                "pt": [[30.0], [25.0, 20.0], [40.0], [20.0], [35.0]]
            }),
            "taus": ak.Array({
                "pt": [[], [30.0], [35.0], [], [30.0]]
            }),
            "jets": ak.Array({
                "pt": [[40.0, 35.0], [30.0], [45.0, 40.0], [25.0], [50.0, 45.0]]
            }),
            "bjets": ak.Array({
                "pt": [[40.0], [30.0], [45.0], [], [50.0]]
            })
        }

        config = {
            "triggers": {"MET": ["HLT_PFMET120_PFMHT120_IDTight"]},
            "met_filters": ["Flag_goodVertices"],
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
            }
        }

        cutflow = get_cutflow(events, objects, config)

        # Check cutflow structure
        assert "Total events" in cutflow
        assert "Pass trigger" in cutflow
        assert "Pass filters" in cutflow
        assert "Has muons" in cutflow
        assert "Has electrons" in cutflow
        assert "Has taus" in cutflow
        assert "Has jets" in cutflow
        assert "Has bjets" in cutflow
        assert "Final selection" in cutflow

        # Check cutflow values
        assert cutflow["Total events"] == 5
        assert cutflow["Pass trigger"] == 3  # Events 1, 3, 4 pass trigger
        assert cutflow["Pass filters"] == 2  # Events 1, 4 pass both trigger and filters
        assert cutflow["Final selection"] == 1  # Only event 1 passes all cuts

    def test_apply_selection(self):
        """Test complete selection application."""
        # Create mock events and objects
        events = ak.Array({
            "event": [1, 2, 3, 4],
            "MET": {"pt": [60.0, 40.0, 80.0, 30.0]},
            "HLT": {
                "PFMET120_PFMHT120_IDTight": [True, False, True, True]
            },
            "Flag_goodVertices": [True, True, False, True]
        })

        objects = {
            "muons": ak.Array({
                "pt": [[25.0, 30.0], [20.0], [35.0, 25.0], [15.0]]
            }),
            "electrons": ak.Array({
                "pt": [[30.0], [25.0, 20.0], [40.0], [20.0]]
            }),
            "taus": ak.Array({
                "pt": [[], [30.0], [35.0], []]
            }),
            "jets": ak.Array({
                "pt": [[40.0, 35.0], [30.0], [45.0, 40.0], [25.0]]
            }),
            "bjets": ak.Array({
                "pt": [[40.0], [30.0], [45.0], []]
            })
        }

        config = {
            "triggers": {"MET": ["HLT_PFMET120_PFMHT120_IDTight"]},
            "met_filters": ["Flag_goodVertices"],
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
            }
        }

        selected_events, selected_objects, cutflow = apply_selection(events, objects, config)

        # Check that selection is applied
        assert len(selected_events) <= len(events)
        assert "muons" in selected_objects
        assert "electrons" in selected_objects
        assert "taus" in selected_objects
        assert "jets" in selected_objects
        assert "bjets" in selected_objects
        assert "Total events" in cutflow
        assert "Final selection" in cutflow


if __name__ == "__main__":
    pytest.main([__file__])
