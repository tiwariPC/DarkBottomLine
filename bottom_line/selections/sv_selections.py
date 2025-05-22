from bottom_line.selections.object_selections import delta_r_mask
import awkward as ak


def match_sv(
    self,
    jets: ak.highlevel.Array,
    sv: ak.highlevel.Array,
    lead_only: bool
) -> ak.highlevel.Array:
    if lead_only:
        jets = ak.firsts(jets)
    dr_max = 0.4
    dr_dipho_cut = delta_r_mask(sv, jets, dr_max)

    return (
        ~dr_dipho_cut
    )
