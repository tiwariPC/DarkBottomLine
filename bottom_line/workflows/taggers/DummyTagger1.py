import awkward as ak
import numpy


class DummyTagger1:
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "DummyTagger1"

    # lower priority is better
    # first decimal point is category within tag (in case for untagged)
    @property
    def priority(self) -> int:
        return 20

    def __call__(
        self, events: ak.Array, diphotons: ak.Array
    ) -> ak.Array:
        # Baseline example for subcategorization:
        return (
            self.priority * ak.ones_like(diphotons.pt, dtype=numpy.int32),
            {},
        )
