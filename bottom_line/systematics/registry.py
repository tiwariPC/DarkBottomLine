from .photon_systematics import (
    photon_pt_scale_dummy,
    Scale_Trad,
    Smearing_Trad,
    Scale_IJazZ,
    Smearing_IJazZ,
    energyErrShift,
    FNUF,
    ShowerShape,
    Material,
)
from .event_weight_systematics import (
    Pileup,
    SF_photon_ID,
    LooseMvaSF,
    ElectronVetoSF,
    PreselSF,
    TriggerSF,
    NNLOPS,
    AlphaS,
    PartonShower,
    cTagSF,
    bTagShapeSF,
    bTagFixedWP,
    Zpt,
    muonSFs
)
from .jet_systematics import (
    jet_pt_scale_dummy,
    JERC_jet,
)

from .electron_systematics import (
    Electron_Scale_Trad,
    Electron_Smearing_Trad,
    Electron_Scale_IJazZ,
    Electron_Smearing_IJazZ,
)

from .muon_systematics import (
    muon_pt_scare
)

from functools import partial
import logging

logger = logging.getLogger(__name__)

# using add_systematic function of coffea.nanoevents.methods.nanoaod objects as Photon to store object systematics in addition to nominal objects
object_systematics = {
    "PhotonPtScale_dummy": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": photon_pt_scale_dummy,
        },
    },
    # Traditional EGM scale and smearing
    "Scale_Trad": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_Trad, is_correction=False),
        },
    },
    "ScaleEB_Trad": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_Trad, is_correction=False, restriction="EB"),
        },
    },
    "ScaleEE_Trad": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_Trad, is_correction=False, restriction="EE"),
        },
    },
    "Smearing_Trad": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Smearing_Trad, is_correction=False),
        },
    },
    # IJazZ (Fabrice et al Saclay) scale and smearing
    "Scale_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_IJazZ, is_correction=False, gaussians="1G", restriction=None),
        },
    },
    "ScaleEB_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_IJazZ, is_correction=False, gaussians="1G", restriction="EB"),
        },
    },
    "ScaleEE_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_IJazZ, is_correction=False, gaussians="1G", restriction="EE"),
        },
    },
    "Smearing_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Smearing_IJazZ, is_correction=False, gaussians="1G"),
        },
    },
    "Scale2G_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_IJazZ, is_correction=False, gaussians="2G", restriction=None),
        },
    },
    "ScaleEB2G_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_IJazZ, is_correction=False, gaussians="2G", restriction="EB"),
        },
    },
    "ScaleEE2G_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Scale_IJazZ, is_correction=False, gaussians="2G", restriction="EE"),
        },
    },
    "Smearing2G_IJazZ": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Smearing_IJazZ, is_correction=False, gaussians="2G"),
        },
    },
    # IJazZ corrections for Electrons
    "Electron_Scale_IJazZ": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Scale_IJazZ, is_correction=False, gaussians="1G", restriction=None),
        },
    },
    "Electron_Scale2G_IJazZ": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Scale_IJazZ, is_correction=False, gaussians="2G", restriction=None),
        },
    },
    "Electron_Smearing_IJazZ": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Smearing_IJazZ, is_correction=False, gaussians="1G"),
        },
    },
    "Electron_Smearing2G_IJazZ": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Smearing_IJazZ, is_correction=False, gaussians="2G"),
        },
    },
    # Traditional corrections for Electrons
    "Electron_Scale_Trad": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Scale_Trad, is_correction=False),
        },
    },
    "Electron_Smearing_Trad": {
        "object": "Electron",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Electron_Smearing_Trad, is_correction=False),
        },
    },
    # Muon scale and smearing
    "MuonScaRe": {
        "object": "Muon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(muon_pt_scare, is_correction=False),
        },
    },
    "energyErrShift": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "energyErr",
            "varying_function": partial(energyErrShift, is_correction=False),
        },
    },
    "FNUF": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(FNUF, year="2017", is_correction=False),
        },
    },
    "ShowerShape": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(ShowerShape, year="2017", is_correction=False),
        },
    },
    "Material": {
        "object": "Photon",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(Material, year="2017", is_correction=False),
        },
    },
    "JetPtScale_dummy": {
        "object": "Jet",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(
                jet_pt_scale_dummy, year=None, is_correction=False
            ),
        },
    },
    "JES": {
        "object": "Jet",
        "args": {
            "kind": "UpDownSystematic",
            "what": "pt",
            "varying_function": partial(
                JERC_jet,
                year="2022postEE",
                skip_JER=False,
                skip_JEC=False,
                is_correction=False,
            ),
        },
    },
}

# functions correcting nominal object quantities to be placed here
# dict containing "name": varying_function
object_corrections = {
    "Scale_Trad": partial(Scale_Trad, pt=None, is_correction=True),
    "Smearing_Trad": partial(Smearing_Trad, pt=None, is_correction=True),
    "Electron_Scale_Trad": partial(Electron_Scale_Trad, pt=None, is_correction=True),
    "Electron_Smearing_Trad": partial(Electron_Smearing_Trad, pt=None, is_correction=True),
    "Scale_IJazZ": partial(Scale_IJazZ, pt=None, is_correction=True, gaussians="1G"),
    "Smearing_IJazZ": partial(Smearing_IJazZ, pt=None, is_correction=True, gaussians="1G"),
    "Scale2G_IJazZ": partial(Scale_IJazZ, pt=None, is_correction=True, gaussians="2G"),
    "Smearing2G_IJazZ": partial(Smearing_IJazZ, pt=None, is_correction=True, gaussians="2G"),
    "Electron_Scale_IJazZ": partial(Electron_Scale_IJazZ, pt=None, is_correction=True, gaussians="1G"),
    "Electron_Smearing_IJazZ": partial(Electron_Smearing_IJazZ, pt=None, is_correction=True, gaussians="1G"),
    "Electron_Scale2G_IJazZ": partial(Electron_Scale_IJazZ, pt=None, is_correction=True, gaussians="2G"),
    "Electron_Smearing2G_IJazZ": partial(Electron_Smearing_IJazZ, pt=None, is_correction=True, gaussians="2G"),
    "MuonScaRe": partial(muon_pt_scare, pt=None, is_correction=True),
    "energyErrShift": partial(energyErrShift, energyErr=None, is_correction=True),
    "FNUF": partial(FNUF, pt=None, is_correction=True),
    "ShowerShape": partial(ShowerShape, pt=None, is_correction=True),
    "Material": partial(Material, pt=None, is_correction=True),
    "JetPtScale_dummy": partial(
        jet_pt_scale_dummy, pt=None, year=None, is_correction=True
    ),
    "JES": partial(
        JERC_jet,
        pt=None,
        year="2022postEE",
        skip_JER=False,
        skip_JEC=False,
        is_correction=True,
    ),
}

# functions adding systematic variations to event weights to be placed here
# dict containing "name": varying_function
weight_systematics = {
    "Pileup": partial(Pileup, is_correction=False),
    "SF_photon_ID": partial(SF_photon_ID, is_correction=False),
    "LooseMvaSF": partial(LooseMvaSF, is_correction=False),
    "ElectronVetoSF": partial(ElectronVetoSF, is_correction=False),
    "PreselSF": partial(PreselSF, is_correction=False),
    "TriggerSF": partial(TriggerSF, is_correction=False),
    "cTagSF": partial(cTagSF, is_correction=False),
    "deepJet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="deepJet_shape", is_correction=False),
    "PNet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="particleNet_shape", is_correction=False),
    "bTagFixedWP_PNetLoose": partial(bTagFixedWP, mva_name="particleNet", wp="L", is_correction=False),
    "bTagFixedWP_PNetMedium": partial(bTagFixedWP, mva_name="particleNet", wp="M", is_correction=False),
    "bTagFixedWP_PNetTight": partial(bTagFixedWP, mva_name="particleNet", wp="T", is_correction=False),
    "bTagFixedWP_PNetExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XT", is_correction=False),
    "bTagFixedWP_PNetExtraExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XXT", is_correction=False),
    "bTagFixedWP_deepJetLoose": partial(bTagFixedWP, mva_name="deepJet", wp="L", is_correction=False),
    "bTagFixedWP_deepJetMedium": partial(bTagFixedWP, mva_name="deepJet", wp="M", is_correction=False),
    "bTagFixedWP_deepJetTight": partial(bTagFixedWP, mva_name="deepJet", wp="T", is_correction=False),
    "bTagFixedWP_deepJetExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XT", is_correction=False),
    "bTagFixedWP_deepJetExtraExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XXT", is_correction=False),
    "bTagFixedWP_robustParticleTransformerLoose": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="L", is_correction=False),
    "bTagFixedWP_robustParticleTransformerMedium": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="M", is_correction=False),
    "bTagFixedWP_robustParticleTransformerTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="T", is_correction=False),
    "bTagFixedWP_robustParticleTransformerExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XT", is_correction=False),
    "bTagFixedWP_robustParticleTransformerExtraExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XXT", is_correction=False),
    "ParT_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="robustParticleTransformer_shape", is_correction=False),
    "AlphaS": partial(AlphaS),
    "PartonShower": partial(PartonShower),
    # LHEScale and LHEPdf are implemented in the processors themselves (special behavior, taking variations directly from nanoAOD)
    "LHEScale": None,
    "LHEPdf": None,
    "Zpt": partial(Zpt),
    "MuonIdMediumSF": partial(muonSFs, SF_name="NUM_MediumID_DEN_TrackerMuons", is_correction=False),
    "MuonIsoTightSF_IdMedium": partial(muonSFs, SF_name="NUM_TightPFIso_DEN_MediumID", is_correction=False),
    "MuonIsoLooseSF_IdMedium": partial(muonSFs, SF_name="NUM_LoosePFIso_DEN_MediumID", is_correction=False),
}

# functions correcting nominal event weights to be placed here
# dict containing "name": varying_function
weight_corrections = {
    "Pileup": partial(Pileup, is_correction=True),
    "SF_photon_ID": partial(SF_photon_ID, is_correction=True),
    "LooseMvaSF": partial(LooseMvaSF, is_correction=True),
    "ElectronVetoSF": partial(ElectronVetoSF, is_correction=True),
    "PreselSF": partial(PreselSF, is_correction=True),
    "TriggerSF": partial(TriggerSF, is_correction=True),
    "cTagSF": partial(cTagSF, is_correction=True),
    "deepJet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="deepJet_shape", is_correction=True),
    "PNet_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="particleNet_shape", is_correction=True),
    "bTagFixedWP_PNetLoose": partial(bTagFixedWP, mva_name="particleNet", wp="L", is_correction=True),
    "bTagFixedWP_PNetMedium": partial(bTagFixedWP, mva_name="particleNet", wp="M", is_correction=True),
    "bTagFixedWP_PNetTight": partial(bTagFixedWP, mva_name="particleNet", wp="T", is_correction=True),
    "bTagFixedWP_PNetExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XT", is_correction=True),
    "bTagFixedWP_PNetExtraExtraTight": partial(bTagFixedWP, mva_name="particleNet", wp="XXT", is_correction=True),
    "bTagFixedWP_deepJetLoose": partial(bTagFixedWP, mva_name="deepJet", wp="L", is_correction=True),
    "bTagFixedWP_deepJetMedium": partial(bTagFixedWP, mva_name="deepJet", wp="M", is_correction=True),
    "bTagFixedWP_deepJetTight": partial(bTagFixedWP, mva_name="deepJet", wp="T", is_correction=True),
    "bTagFixedWP_deepJetExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XT", is_correction=True),
    "bTagFixedWP_deepJetExtraExtraTight": partial(bTagFixedWP, mva_name="deepJet", wp="XXT", is_correction=True),
    "bTagFixedWP_robustParticleTransformerLoose": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="L", is_correction=True),
    "bTagFixedWP_robustParticleTransformerMedium": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="M", is_correction=True),
    "bTagFixedWP_robustParticleTransformerTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="T", is_correction=True),
    "bTagFixedWP_robustParticleTransformerExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XT", is_correction=True),
    "bTagFixedWP_robustParticleTransformerExtraExtraTight": partial(bTagFixedWP, mva_name="robustParticleTransformer", wp="XXT", is_correction=True),
    "ParT_bTagShapeSF": partial(bTagShapeSF, ShapeSF_name="robustParticleTransformer_shape", is_correction=True),
    "NNLOPS": partial(NNLOPS, is_correction=True),
    "Zpt": partial(Zpt, is_correction=True),
    "MuonIdMediumSF": partial(muonSFs, SF_name="NUM_MediumID_DEN_TrackerMuons", is_correction=True),
    "MuonIsoTightSF_IdMedium": partial(muonSFs, SF_name="NUM_TightPFIso_DEN_MediumID", is_correction=True),
    "MuonIsoLooseSF_IdMedium": partial(muonSFs, SF_name="NUM_LoosePFIso_DEN_MediumID", is_correction=True),
}
