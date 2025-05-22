import awkward as ak
import numpy


class DummyTagger2:
    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "DummyTagger2"

    # lower priority is better
    # first decimal point is category within tag (in case for untagged)
    @property
    def priority(self) -> int:
        return 10

    def __call__(
        self, events: ak.Array, diphotons: ak.Array
    ) -> ak.Array:
        counts = ak.num(events.diphotons, axis=1)
        x = numpy.random.exponential(size=ak.sum(counts))
        return ak.unflatten(self.priority * (x > 2), counts), {}
