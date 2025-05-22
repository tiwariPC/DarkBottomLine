import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema


def test_nanoAOD_v11():
    """
    Provides a test of the content of an example nanoAOD v11 to ensure that all necessary variables are there.
    """
    fname = "./tests/samples/skimmed_nano/GJet_v11_Skim.root"
    events = NanoEventsFactory.from_root(
        fname,
        schemaclass=NanoAODSchema,
    ).events()

    global_fields = events.fields
    photon_fields = events.Photon.fields
    needed_global_fields = ["event", "luminosityBlock", "run", "Rho", "GenVtx", "PV", "Photon"]
    needed_photon_fields = ["pfChargedIsoPFPV", "sieie", "r9", "pfPhoIso03", "pfRelIso03_chg_quadratic"]

    # Check the "global" fields
    for needed_global_field in needed_global_fields:
        assert needed_global_field in global_fields
    # Check fixedGridRhoAll
    assert "fixedGridRhoAll" in events.Rho.fields
    # Check GenVtx and PV z
    assert "z" in events.GenVtx.fields
    assert "z" in events.PV.fields
    # Check photon variables
    for needed_photon_field in needed_photon_fields:
        assert needed_photon_field in photon_fields