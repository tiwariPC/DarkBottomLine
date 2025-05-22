import awkward as ak


class ptTagger:

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "ptTagger"

    # lower priority is better
    # first decimal point is category within tag (in case for untagged)
    @property
    def priority(self) -> int:
        return 20

    def GetCategory(self, ievt: int) -> int:

        evt_pt = self.pt[ievt][0]

        Demarcation = [5, 10, 15, 20, 25, 30, 35, 45, 60, 80, 100, 120, 140, 170, 200, 250, 350, 450]
        for index, elem in enumerate(Demarcation):
            if evt_pt < elem:
                cat = index
                break
        else:
            cat = 18

        return cat

    def __call__(self, events: ak.Array, diphotons: ak.Array) -> ak.Array:
        """
        We can classify events according to it's pt of diphotons.
        """

        self.pt = events.diphotons.pt

        nDiphotons = ak.num(
            events.diphotons.pt, axis=1
        )  # Number of entries per row. (N diphotons per row)

        ievts_by_dipho = ak.flatten(
            ak.Array([nDipho * [evt_i] for evt_i, nDipho in enumerate(nDiphotons)])
        )

        cat_vals = ak.Array(map(self.GetCategory, ievts_by_dipho))
        cats = ak.unflatten(cat_vals, nDiphotons)  # Back to size of events.
        cats_by_diphoEvt = self.priority + cats

        return (cats_by_diphoEvt, {})
