BTagging workflow
=================

The BTagging workflow is a slightly changed base workflow. The main difference is that the BTagging workflow includes derivation of BTagging efficiencies for a given analysis as recommended by [BTV](https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/#b-tagging-efficiencies-in-simulation). The processor that implements the main operations performed can be found at ``bottom_line/workflows/btagging:BTaggingProcessor``.

To use b-jet related variables and weights, one has to adhere to the following procedures:

- Variables
  * Execute first `pull_files.py --target bTag`
  * Choose an appropriate btagging MVA: `deepJet` (all NanoAOD versions \>= v11), `particleNet` (NanoAOD \>= v12) and `robustParticleTransformer` (NanoAOD \>= v12)
    * Set it in the processor variable `self.bjet_mva`
  * Choose the Working Point: `L` (Loose), `M` (Medium), `T` (Tight), `XT` (extra Tight) or `XXT` (extra extra Tight)
    * Set it in the processor variable `self.bjet_wp`
- Weights
  * Execute first `pull_files.py --target bTag`
  * Since the btagging efficiency weights have to be computed **per analysis**, we have to produce them first with the `BTagging` processor
      * Example `Btagging` processor found in `./bottom_line/workflows`
         * Important: You have to apply your selections **before** the indicated MANDATORY PART. The latter must not be changed.
      * Select in your `runner.json` for the workflow `BTagging` and run the processor with no systematics (they are not necessary for the `BTagging` processor) over all your samples of your analysis
      * Pickle `.pkl` files are produced that contain the btagging efficiencies binned as per recommendation by [BTV](https://btv-wiki.docs.cern.ch/PerformanceCalibration/fixedWPSFRecommendations/)
        * pT in `[20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]`
        * abs(eta): No Binning
        * hadronFlavour: `[0, 4, 5]`
      * Once `.pkl` files are produced, generate the correctionlib file by calling `btagging_eff.py --input path/to/BTagging/output --output-name analysis-name` , where `analysis-name` is the name the correctionlib will be saved as (per default in `./bottom_line/systematics/JSONs/bTagEff/year/analysis-name.json.gz` )
  * When the correctionlib is produced and stored in `./bottom_line/systematics/JSONs/bTagEff/year` , we have to add an additional string to the dictionary in `runner.json`:     `"bTagEffFileName": "analysis-name.json.gz"` (**Important**: Only state the name of the file. Do not add the path, as HiggsDNA looks for it in `./bottom_line/systematics/JSONs/bTagEff/` by itself!)
  * Now, you can add the desired corrections and systematics to your `runner.json` and launch the production of your samples
    * Currently only `bTagFixedWP_PNetTight` (MVA: ParticleNet, Tight WP) is implemented. But feel free to add more with the new function for fixed btagging working points `bTagFixedWP` found in `bottom_line/systematics/event_weight_systematics.py`
  * If a systematic or correction from `bTagFixedWP` is used, the weights `weight` and `weight_central` are **not** containing the weight of the btagging scale factors. Instead it is added separately as `weight_bTagFixedWP` to be used later.


An example JSON for the `BTaggingProcessor` could look like this:

.. code-block:: json
    {
        "samplejson": "./samples.json",
        "workflow": "BTagging",
        "metaconditions": "Era2022_v1",
        "year": {
            "Channel_postEE": ["2022postEE"],
            "Channel_preEE": ["2022preEE"]
        },
        "corrections": {
            "Channel_postEE": ["Pileup", "Et_dependent_Smearing", "energyErrShift"],
            "Channel_preEE":  ["Pileup", "Et_dependent_Smearing", "energyErrShift"]
        },
        "systematics": {
            "Channel_postEE": [],
            "Channel_preEE":  []
        }
    }

Then after the production of the `.pkl` files, the final `runner.json` for your analyis involving the Particle Net MVA with a tight working point could look like this:

.. code-block:: json
    {
        "samplejson": "./samples.json",
        "workflow": "base",
        "metaconditions": "Validation_Plots_22_23",
        "bTagEffFileName": "DY_PNetT_Base.json.gz",
        "year": {
            "Channel_postEE": ["2022postEE"],
            "Channel_preEE": ["2022preEE"]
        },
        "corrections": {
            "Channel_postEE": ["bTagFixedWP_PNetTight", "jerc_jet_syst", "Pileup", "Et_dependent_Smearing", "energyErrShift"],
            "Channel_preEE":  ["bTagFixedWP_PNetTight", "jerc_jet_syst", "Pileup", "Et_dependent_Smearing", "energyErrShift"]
        },
        "systematics": {
            "Channel_postEE": ["bTagFixedWP_PNetTight", "Pileup", "Et_dependent_ScaleEB", "Et_dependent_ScaleEE", "Et_dependent_Smearing", "energyErrShift"],
            "Channel_preEE":  ["bTagFixedWP_PNetTight", "Pileup", "Et_dependent_ScaleEB", "Et_dependent_ScaleEE", "Et_dependent_Smearing", "energyErrShift"]
        }
    }


Right now, the following btagging corrections and systematics are implemented:

    - Particle Net
        - `bTagFixedWP_PNetLoose` (Loose WP)
        - `bTagFixedWP_PNetMedium` (Medium WP)
        - `bTagFixedWP_PNetTight` (Tight WP)
        - `bTagFixedWP_PNetExtraTight` (Extra Tight WP)
        - `bTagFixedWP_PNetExtraExtraTight` (Extra Extra Tight WP)

    - Deep Jet
        - `bTagFixedWP_deepJetLoose` (Loose WP)
        - `bTagFixedWP_deepJetMedium` (Medium WP)
        - `bTagFixedWP_deepJetTight` (Tight WP)
        - `bTagFixedWP_deepJetExtraTight` (Extra Tight WP)
        - `bTagFixedWP_deepJetExtraExtraTight` (Extra Extra Tight WP)

    - Robust Particle Transformer
        - `bTagFixedWP_robustParticleTransformerLoose` (Loose WP)
        - `bTagFixedWP_robustParticleTransformerMedium` (Medium WP)
        - `bTagFixedWP_robustParticleTransformerTight` (Tight WP)
        - `bTagFixedWP_robustParticleTransformerExtraTight` (Extra Tight WP)
        - `bTagFixedWP_robustParticleTransformerExtraExtraTight` (Extra Extra Tight WP)


IMPORTANT: When the systematics are applied, the weights `weight` and `weight_central` are **not** containing the weight of the btagging scale factors. Instead it is added separately as `weight_bTagFixedWP` to be used later.
