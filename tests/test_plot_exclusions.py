"""
Test script to verify region_variables whitelist works correctly.
"""

import unittest
from darkbottomline.plotting import PlotManager


class TestPlotExclusions(unittest.TestCase):
    """Test region variable whitelist logic."""

    def _make_pm(self, region_variables=None):
        cfg = {}
        if region_variables is not None:
            cfg["region_variables"] = region_variables
        return PlotManager(config=cfg)

    def test_1b_sr_exclusions(self):
        """1b SR whitelist keeps jet2 vars, no jet3, no leptons."""
        pm = self._make_pm()
        all_vars = ["Recoil", "Jet1Pt", "Jet2Pt", "Jet3Pt", "lep1_pt", "n_muons"]
        # No whitelist configured → return all_vars unchanged
        result = pm._get_allowed_variables_for_region("1b:SR", all_vars)
        self.assertEqual(result, all_vars)

    def test_whitelist_filters_to_listed_vars(self):
        """With whitelist, only listed vars are returned."""
        pm = self._make_pm(region_variables={
            "1b:SR": ["Recoil", "Jet1Pt", "Jet2Pt"],
        })
        all_vars = ["Recoil", "Jet1Pt", "Jet2Pt", "Jet3Pt", "lep1_pt", "n_muons"]
        result = pm._get_allowed_variables_for_region("1b:SR", all_vars)
        self.assertEqual(result, ["Recoil", "Jet1Pt", "Jet2Pt"])

    def test_whitelist_drops_unknown_names(self):
        """Vars in whitelist but absent from all_vars are silently dropped."""
        pm = self._make_pm(region_variables={
            "1b:SR": ["Recoil", "NoSuchBranch", "Jet1Pt"],
        })
        all_vars = ["Recoil", "Jet1Pt"]
        result = pm._get_allowed_variables_for_region("1b:SR", all_vars)
        self.assertEqual(result, ["Recoil", "Jet1Pt"])

    def test_whitelist_preserves_order(self):
        """Order follows whitelist, not all_vars."""
        pm = self._make_pm(region_variables={
            "2b:SR": ["Jet3Pt", "Recoil", "Jet1Pt"],
        })
        all_vars = ["Recoil", "Jet1Pt", "Jet3Pt"]
        result = pm._get_allowed_variables_for_region("2b:SR", all_vars)
        self.assertEqual(result, ["Jet3Pt", "Recoil", "Jet1Pt"])

    def test_substring_key_matches(self):
        """Substring key in region_variables matches regions containing that string."""
        pm = self._make_pm(region_variables={
            "CR_Wmunu": ["Recoil", "muon_lep1_pt"],
        })
        all_vars = ["Recoil", "muon_lep1_pt", "Jet3Pt"]
        result = pm._get_allowed_variables_for_region("1b:CR_Wmunu", all_vars)
        self.assertEqual(result, ["Recoil", "muon_lep1_pt"])

    def test_exact_key_takes_priority_over_substring(self):
        """Exact region key wins over substring when both would match."""
        pm = self._make_pm(region_variables={
            "1b:CR_Wmunu": ["Recoil", "Jet1Pt"],
            "CR_Wmunu":    ["Recoil", "Jet1Pt", "muon_lep1_pt"],
        })
        all_vars = ["Recoil", "Jet1Pt", "muon_lep1_pt"]
        result = pm._get_allowed_variables_for_region("1b:CR_Wmunu", all_vars)
        # Exact key matched first → only 2 vars
        self.assertEqual(result, ["Recoil", "Jet1Pt"])

    def test_no_match_returns_all_vars(self):
        """Region with no whitelist entry returns all_vars unchanged."""
        pm = self._make_pm(region_variables={
            "1b:SR": ["Recoil"],
        })
        all_vars = ["Recoil", "Jet1Pt", "Jet2Pt"]
        result = pm._get_allowed_variables_for_region("2b:SR", all_vars)
        self.assertEqual(result, all_vars)

    def test_top_cr_exclusions(self):
        """With yaml config, Top CR whitelist excludes z_mass, z_pt (not listed)."""
        pm = self._make_pm(region_variables={
            "2b:CR_Topmunu": ["Recoil", "Jet1Pt", "muon_lep1_pt"],
        })
        all_vars = ["Recoil", "Jet1Pt", "muon_lep1_pt", "z_mass", "z_pt", "lep1_pt"]
        result = pm._get_allowed_variables_for_region("2b:CR_Topmunu", all_vars)
        self.assertNotIn("z_mass", result)
        self.assertNotIn("z_pt", result)
        self.assertIn("muon_lep1_pt", result)

    def test_w_cr_exclusions(self):
        """With yaml config, W CR whitelist excludes z_mass (not listed)."""
        pm = self._make_pm(region_variables={
            "1b:CR_Wmunu": ["Recoil", "Jet1Pt", "muon_lep1_pt"],
        })
        all_vars = ["Recoil", "Jet1Pt", "muon_lep1_pt", "z_mass", "z_pt"]
        result = pm._get_allowed_variables_for_region("1b:CR_Wmunu", all_vars)
        self.assertNotIn("z_mass", result)
        self.assertIn("muon_lep1_pt", result)

    def test_custom_exclusions(self):
        """Whitelist-based custom config."""
        pm = self._make_pm(region_variables={
            "1b:SR": ["Recoil", "Jet1Pt"],
        })
        all_vars = ["Recoil", "Jet1Pt", "custom_var1", "custom_var2"]
        result = pm._get_allowed_variables_for_region("1b:SR", all_vars)
        self.assertNotIn("custom_var1", result)
        self.assertNotIn("custom_var2", result)
        self.assertIn("Recoil", result)


if __name__ == "__main__":
    unittest.main()
