from bottom_line.workflows.dystudies import (
    DYStudiesProcessor,
    TagAndProbeProcessor,
)
from bottom_line.workflows.taggers import taggers
from bottom_line.workflows.HHbbgg import HHbbggProcessor
from bottom_line.workflows.particleLevel import ParticleLevelProcessor
from bottom_line.workflows.top import TopProcessor
from bottom_line.workflows.Zmmy import ZmmyProcessor, ZmmyHist, ZmmyZptHist
from bottom_line.workflows.hpc_processor import HplusCharmProcessor
from bottom_line.workflows.zee_processor import ZeeProcessor
from bottom_line.workflows.lowmass import lowmassProcessor
from bottom_line.workflows.btagging import BTaggingEfficienciesProcessor
from bottom_line.workflows.stxs import STXSProcessor
from bottom_line.workflows.diphoton_training import DiphoTrainingProcessor

workflows = {}

workflows["base"] = DYStudiesProcessor
workflows["tagandprobe"] = TagAndProbeProcessor
workflows["HHbbgg"] = HHbbggProcessor
workflows["particleLevel"] = ParticleLevelProcessor
workflows["top"] = TopProcessor
workflows["zmmy"] = ZmmyProcessor
workflows["zmmyHist"] = ZmmyHist
workflows["zmmyZptHist"] = ZmmyZptHist
workflows["hpc"] = HplusCharmProcessor
workflows["zee"] = ZeeProcessor
workflows["lowmass"] = lowmassProcessor
workflows["BTagging"] = BTaggingEfficienciesProcessor
workflows["stxs"] = STXSProcessor
workflows["diphotonID"] = DiphoTrainingProcessor

__all__ = ["workflows", "taggers", "DYStudiesProcessor"]
