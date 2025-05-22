import awkward as ak
import numpy as np


class yTagger:

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "yTagger"

    # lower priority is better
    # first decimal point is category within tag (in case for untagged)
    @property
    def priority(self) -> int:
        return 20

    def GetCategory(self, ievt: int) -> int:

        evt_pz = self.pz[ievt][0]
        evt_energy = self.energy[ievt][0]

        evt_y = np.arctanh(evt_pz / evt_energy)

        Demarcation = [0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.90, 2.5]
        for index, elem in enumerate(Demarcation):
            if abs(evt_y) < elem:
                cat = index
                break
        else:
            raise KeyError

        return cat

    def __call__(self, events: ak.Array, diphotons: ak.Array) -> ak.Array:
        """
        We can classify events according to it's y of diphotons.
        """

        self.pz = events.diphotons.pz
        self.energy = events.diphotons.energy

        nDiphotons = ak.num(
            events.diphotons.pz, axis=1
        )  # Number of entries per row. (N diphotons per row)

        ievts_by_dipho = ak.flatten(
            ak.Array([nDipho * [evt_i] for evt_i, nDipho in enumerate(nDiphotons)])
        )

        cat_vals = ak.Array(map(self.GetCategory, ievts_by_dipho))
        cats = ak.unflatten(cat_vals, nDiphotons)  # Back to size of events.
        cats_by_diphoEvt = self.priority + cats

        return (cats_by_diphoEvt, {})
