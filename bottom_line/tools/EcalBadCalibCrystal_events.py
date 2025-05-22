import awkward as ak
from bottom_line.selections.object_selections import delta_phi_mask


def remove_EcalBadCalibCrystal_events(
        events: ak.highlevel.Array
) -> ak.highlevel.Array:

    """
    A peak appears in the pT spectrum of photons due to EcalBadCalibCrystal. The events affected by this has to be rejected.
    Check here for more details and for the recipe to mitigate this:
    https://indico.cern.ch/event/1397512/contributions/5873976/attachments/2828733/4942264/EcalBadCalibFilterRun3.pdf
    """

    # select the affected runs
    run_mask = ((events.run >= 362433) & (events.run <= 367144))

    # MET cut
    MET_cut = (events.PuppiMET.pt > 100)

    # Jet cuts
    jet_cuts = (
        (events.Jet.pt > 50)
        & ((events.Jet.eta > -0.5) & (events.Jet.eta < -0.1))
        & ((events.Jet.phi > -2.1) & (events.Jet.phi < -1.8))
        & ((events.Jet.neEmEF > 0.9) | (events.Jet.chEmEF > 0.9))
        & (delta_phi_mask(events.PuppiMET.phi, events.Jet.phi, 2.9))
    )
    jet_cuts_any = ak.any(jet_cuts, axis=1)

    events_to_remove = run_mask & MET_cut & jet_cuts_any
    events = events[~events_to_remove]

    return events
