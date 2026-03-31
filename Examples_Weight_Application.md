# Examples_Weight_Application:

## Electron Weight:

```
## example how to read the electron format in 2023 Prompt (eta, pT, phi)

evaluator = _core.CorrectionSet.from_file(
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-23DSep23-Summer23BPix-NanoAODv12/latest/electron.json.gz"
)

valsf = evaluator["Electron-ID-SF"].evaluate(
    "2023PromptD", "sf", "RecoBelow20", 1.1, 15.0, 2.0
)
print("sf is:" + str(valsf))

valsf = evaluator["Electron-ID-SF"].evaluate(
    "2023PromptD", "sf", "Reco20to75", 1.1, 25.0, 2.0
)
print("sf is:" + str(valsf))

valsf = evaluator["Electron-ID-SF"].evaluate(
    "2023PromptD", "sf", "Medium", 1.1, 34.0, -1.0
)
print("sf is:" + str(valsf))

valsystup = evaluator["Electron-ID-SF"].evaluate(
    "2023PromptD", "sfup", "Medium", 1.1, 34.0, -1.0
)
print("systup is:" + str(valsystup))

valsystdown = evaluator["Electron-ID-SF"].evaluate(
    "2023PromptD", "sfdown", "Medium", 1.1, 34.0, -1.0
)
print("systdown is:" + str(valsystdown))
```

## Electron hlt weight:

```
## example how to read the electronHlt format v2
from correctionlib import _core

evaluator = _core.CorrectionSet.from_file(
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/EGM/Run3-22CDSep23-Summer22-NanoAODv12/latest/electronHlt.json.gz"
)

valsf = evaluator["Electron-HLT-SF"].evaluate(
    "2022Re-recoBCD", "sf", "HLT_SF_Ele30_TightID", 1.1, 45.0
)
print("sf is:" + str(valsf))

valsystup = evaluator["Electron-HLT-SF"].evaluate(
    "2022Re-recoBCD", "sfup", "HLT_SF_Ele30_TightID", 1.1, 45.0
)
print("systup is:" + str(valsystup))

valsystdown = evaluator["Electron-HLT-SF"].evaluate(
    "2022Re-recoBCD", "sfdown", "HLT_SF_Ele30_TightID", 1.1, 45.0
)
print("systdown is:" + str(valsystdown))

valeffdata = evaluator["Electron-HLT-DataEff"].evaluate(
    "2022Re-recoBCD", "nom", "HLT_SF_Ele30_MVAiso90ID", 1.1, 45.0
)
print("sf is:" + str(valeffdata))

valeffMCup = evaluator["Electron-HLT-McEff"].evaluate(
    "2022Re-recoBCD", "up", "HLT_SF_Ele30_MVAiso90ID", 1.1, 45.0
)
print("sf is:" + str(valeffMCup))
```

## Muon Weight:

```
## example how to read the muon format v2
## (Adapted from JMAR and EGM examples)
from correctionlib import _core

############################
## Example A: 2016postVFP ##
############################

# Load CorrectionSet
fname = "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run2-2016postVFP-UL-NanoAODv9/latest/muon_Z.json.gz"
if fname.endswith(".json.gz"):
    import gzip

    with gzip.open(fname, "rt") as file:
        data = file.read().strip()
        evaluator = _core.CorrectionSet.from_string(data)
else:
    evaluator = _core.CorrectionSet.from_file(fname)

# TrackerMuon Reconstruction UL scale factor ==> NOTE the year key has been removed, for consistency with Run 3
valsf = evaluator["NUM_TrackerMuons_DEN_genTracks"].evaluate(-1.1, 50.0, "nominal")
print("sf is: " + str(valsf))

# Medium ID UL scale factor, down variation ==> NOTE the year key has been removed, for consistency with Run 3
valsf = evaluator["NUM_MediumID_DEN_TrackerMuons"].evaluate(0.8, 35.0, "systdown")
print("systdown is: " + str(valsf))

# Medium ID UL scale factor, up variation ==> NOTE the year key has been removed, for consistency with Run 3
valsf = evaluator["NUM_MediumID_DEN_TrackerMuons"].evaluate(0.8, 35.0, "systup")
print("systup is: " + str(valsf))

# Trigger UL systematic uncertainty only ==> NOTE the year key has been removed, for consistency with Run 3
valsyst = evaluator[
    "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight"
].evaluate(1.8, 54.0, "syst")
print("syst is: " + str(valsyst))

##########################
## Example B: 2022preEE ##
##########################

fname = "/cvmfs/cms-griddata.cern.ch/cat/metadata/MUO/Run3-22CDSep23-Summer22-NanoAODv12/latest/muon_Z.json.gz"
if fname.endswith(".json.gz"):
    import gzip

    with gzip.open(fname, "rt") as file:
        data = file.read().strip()
        evaluator = _core.CorrectionSet.from_string(data)
else:
    evaluator = _core.CorrectionSet.from_file(fname)

# Medium ID 2022 scale factor using eta as input
valsf_eta = evaluator["NUM_MediumID_DEN_TrackerMuons"].evaluate(-1.1, 45.0, "nominal")
print("sf for eta = -1.1: " + str(valsf_eta))

# Medium ID 2022 scale factor using eta as input ==> Note that this value should be the same
# as the previous one, since even though the input can be signed eta, the SFs for 2022 were
# computed for |eta|. This is valid for ALL the years and jsons
valsf_eta = evaluator["NUM_MediumID_DEN_TrackerMuons"].evaluate(1.1, 45.0, "nominal")
print("sf for eta = 1.1 " + str(valsf_eta))

# Trigger 2022 systematic uncertainty only
valsyst = evaluator["NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium"].evaluate(
    -1.8, 54.0, "syst"
)
print("syst is: " + str(valsyst))
```

## Btagging weight:

```
import os

import correctionlib
import numpy as np

sfDir = os.path.join(".", "..", "POG", "BTV", "2018_UL")
btvjson = correctionlib.CorrectionSet.from_file(
    "/cvmfs/cms-griddata.cern.ch/cat/metadata/BTV/Run2-2018-UL-NanoAODv9/latest/btagging.json.gz"
)

# generate 20 dummy jet features
jet_pt = np.random.exponential(50.0, 20)
jet_eta = np.random.uniform(0.0, 2.4, 20)
jet_flav = np.random.choice([0, 4, 5], 20)
jet_discr = np.random.uniform(0.0, 1.0, 20)

# separate light and b/c jets
light_jets = np.where(jet_flav == 0)
bc_jets = np.where(jet_flav != 0)

# case 1: fixedWP correction with mujets (here medium WP)
# evaluate('systematic', 'working_point', 'flavor', 'abseta', 'pt')
bc_jet_sf = btvjson["deepJet_mujets"].evaluate(
    "central", "M", jet_flav[bc_jets], jet_eta[bc_jets], jet_pt[bc_jets]
)
light_jet_sf = btvjson["deepJet_incl"].evaluate(
    "central", "M", jet_flav[light_jets], jet_eta[light_jets], jet_pt[light_jets]
)
print("\njet SFs for mujets at medium WP:")
print(f"SF b/c: {bc_jet_sf}")
print(f"SF light: {light_jet_sf}")

# case 2: fixedWP correction uncertainty (here tight WP and comb SF)
# evaluate('systematic', 'working_point', 'flavor', 'abseta', 'pt')
bc_jet_sf = btvjson["deepJet_comb"].evaluate(
    "up_correlated", "T", jet_flav[bc_jets], jet_eta[bc_jets], jet_pt[bc_jets]
)
light_jet_sf = btvjson["deepJet_incl"].evaluate(
    "up_correlated", "T", jet_flav[light_jets], jet_eta[light_jets], jet_pt[light_jets]
)
print("\njet SF up_correlated for comb at tight WP:")
print(f"SF b/c: {bc_jet_sf}")
print(f"SF light: {light_jet_sf}")

# case 3: shape correction SF
# evaluate('systematic', 'flavor', 'eta', 'pt', 'discriminator')
jet_sf = btvjson["deepJet_shape"].evaluate(
    "central", jet_flav, jet_eta, jet_pt, jet_discr
)
print("\njet SF for shape correction:")
print(f"SF: {jet_sf}")

# case 4: shape correction SF uncertainties
# evaluate('systematic', 'flavor', 'eta', 'pt', 'discriminator')
c_jets = np.where(jet_flav == 4)
blight_jets = np.where(jet_flav != 4)
blight_jet_sf = btvjson["deepJet_shape"].evaluate(
    "up_hfstats2",
    jet_flav[blight_jets],
    jet_eta[blight_jets],
    jet_pt[blight_jets],
    jet_discr[blight_jets],
)
c_jet_sf = btvjson["deepJet_shape"].evaluate(
    "up_cferr1", jet_flav[c_jets], jet_eta[c_jets], jet_pt[c_jets], jet_discr[c_jets]
)
print("\njet SF up_hfstats2 for shape correction b/light jets:")
print(f"SF b/light: {blight_jet_sf}")
print("jet SF up_cferr1 for shape correction c jets:")
print(f"SF c: {c_jet_sf}")
```

## Jet weights

```
#! /usr/bin/env python
# Example of how to read the JME-JERC JSON files
# For more information, see the README in
# https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/JME
# For a comparison to the CMSSW-syntax refer to
# https://github.com/cms-jet/JECDatabase/blob/master/scripts/JERC2JSON/minimalDemo.py

import os

import correctionlib._core as core

# path to directory of this script
__this_dir__ = os.path.dirname(__file__)

#
# helper functions
#


def get_corr_inputs(input_dict, corr_obj):
    """
    Helper function for getting values of input variables
    given a dictionary and a correction object.
    """
    input_values = [input_dict[inp.name] for inp in corr_obj.inputs]
    return input_values


#
# values of input variables
#

example_value_dict = {
    # jet transverse momentum
    "JetPt": 100.0,
    # jet pseudorapidity
    "JetEta": 0.0,
    # jet azimuthal angle
    "JetPhi": 0.2,
    # jet area
    "JetA": 0.5,
    # median energy density (pileup)
    "Rho": 15.0,
    # systematic variation (only for JER SF)
    "systematic": "nom",
    # pT of matched gen-level jet (only for JER smearing)
    "GenPt": 80.0,  # or -1 if no match
    # unique event ID used for deterministic
    # pseudorandom number generation (only for JER smearing)
    "EventID": 12345,
}


#
# JEC-related examples
#

# JEC base tag
jec = "Summer19UL16_V7_MC"

# jet algorithms
algo = "AK4PFchs"
algo_ak8 = "AK8PFPuppi"

# jet energy correction level
lvl = "L2Relative"

# jet energy correction level
lvl_compound = "L1L2L3Res"

# jet energy uncertainty
unc = "Total"

# print input information
print("\n\nJEC parameters")
print("##############\n")

print("jec = {}".format(jec))
print("algo = {}".format(algo))
print("algo_ak8 = {}".format(algo_ak8))
for v in ("JetPt", "JetEta", "JetA", "JetPhi", "JetA", "Rho"):
    print("{} = {}".format(v, example_value_dict[v]))


#
# load JSON files using correctionlib
#

# AK4
fname = "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016postVFP-UL-NanoAODv9/latest/jet_jerc.json.gz"
print("\nLoading JSON file: {}".format(fname))
cset = core.CorrectionSet.from_file(os.path.join(fname))

# AK8
fname_ak8 = "/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/Run2-2016postVFP-UL-NanoAODv9/latest/fatJet_jerc.json.gz"
print("\nLoading JSON file: {}".format(fname_ak8))
cset_ak8 = core.CorrectionSet.from_file(os.path.join(fname_ak8))

# tool for JER smearing
# fname_jersmear = os.path.join(__this_dir__, "../POG/JME/jer_smear.json.gz")
# print("\nLoading JSON file: {}".format(fname_jersmear))
# cset_jersmear = core.CorrectionSet.from_file(os.path.join(fname_jersmear))


#
# example 1: getting a single JEC level
#

print("\n\nExample 1: single JEC level\n===================")

key = "{}_{}_{}".format(jec, lvl, algo)
key_ak8 = "{}_{}_{}".format(jec, lvl, algo_ak8)
print("JSON access to keys: '{}' and '{}'".format(key, key_ak8))
sf = cset[key]
sf_ak8 = cset_ak8[key_ak8]

sf_input_names = [inp.name for inp in sf.inputs]
print("Inputs: " + ", ".join(sf_input_names))

inputs = get_corr_inputs(example_value_dict, sf)
print("JSON result AK4: {}".format(sf.evaluate(*inputs)))

inputs = get_corr_inputs(example_value_dict, sf_ak8)
print("JSON result AK8: {}".format(sf_ak8.evaluate(*inputs)))


#
# example 2: accessing the JEC as a CompoundCorrection
#

print("\n\nExample 2: compound JEC level\n===================")

key = "{}_{}_{}".format(jec, lvl_compound, algo)
key_ak8 = "{}_{}_{}".format(jec, lvl_compound, algo_ak8)
print("JSON access to keys: '{}' and '{}'".format(key, key_ak8))
sf = cset.compound[key]
sf_ak8 = cset_ak8.compound[key_ak8]

sf_input_names = [inp.name for inp in sf.inputs]
print("Inputs: " + ", ".join(sf_input_names))

inputs = get_corr_inputs(example_value_dict, sf)
print("JSON result AK4: {}".format(sf.evaluate(*inputs)))

inputs = get_corr_inputs(example_value_dict, sf_ak8)
print("JSON result AK8: {}".format(sf_ak8.evaluate(*inputs)))

```

## Pileup weight:

```
#!/usr/bin/env python3
import correctionlib
import numpy as np

fileName = "/cvmfs/cms-griddata.cern.ch/cat/metadata/LUM/Run2-2018-UL-NanoAODv9/2021-09-10/puWeights.json.gz"
evaluator = correctionlib.CorrectionSet.from_file(fileName)

correction = evaluator["Collisions18_UltraLegacy_goldenJSON"]


NumTrueInteractions = np.array([22.0, 9.0], dtype=float)
pu_weight = correction.evaluate(NumTrueInteractions, "nominal")

print(f"PU weight: {pu_weight}")
```