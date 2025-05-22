from bottom_line.selections.object_selections import delta_r_mask
import awkward as ak
from coffea import nanoevents


def test_delta_r_mask():
    """
    Check that delta_r_mask returns an array with the correct shape.
    """
    events = nanoevents.NanoEventsFactory.from_root(
        "tests/samples/skimmed_nano/ttH_M125_2017.root"
    ).events()
    photons = events.Photon
    electrons = events.Electron
    mask = delta_r_mask(photons, electrons, 0.2)

    assert ak.all(ak.num(mask) == ak.num(photons))
