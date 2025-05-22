from .registry import object_systematics, weight_systematics, object_corrections, weight_corrections
from .utils import apply_systematic_variations_object_level, check_corr_syst_combinations
from .factories import add_jme_corr_syst
import logging

logger = logging.getLogger(__name__)

object_corrections, object_systematics = add_jme_corr_syst(
    object_corrections, object_systematics, logger
)

__all__ = [
    "object_systematics",
    "weight_systematics",
    "object_corrections",
    "weight_corrections",
    "apply_systematic_variations_object_level",
    "check_corr_syst_combinations",
    "add_jme_corr_syst",
]
