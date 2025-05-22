import warnings
import logging
import sys
from functools import partial
from .registry import object_corrections, weight_corrections

logger = logging.getLogger(__name__)


def apply_systematic_variations_object_level(systematic_names, events, dataset_year, logger, available_object_systematics, available_weight_systematics, collections):
    """
    Apply systematic variations to the provided object collections.

    Parameters:
        systematic_names (list): List of systematic variation names to process.
        events (awkward.Array): The events collection.
        dataset_year (str): Year information for the dataset.
        logger (logging.Logger): Logger for output messages.
        available_object_systematics (dict): Dictionary mapping systematic names to definitions for objects.
        available_weight_systematics (dict): Dictionary mapping systematic names to definitions for weights.
        collections (dict): Dictionary mapping object names (e.g., "Photon", "Electron", "Muon") to their collections.

    Returns:
        dict: The updated collections with systematics applied.
    """

    for syst in systematic_names:
        if syst in available_object_systematics:
            syst_def = available_object_systematics[syst]
            obj_type = syst_def["object"]
            if obj_type in collections:
                logger.info(f"Adding systematic {syst} to {obj_type} collection of dataset.")
                collections[obj_type].add_systematic(
                    name=syst,
                    kind=syst_def["args"]["kind"],
                    what=syst_def["args"]["what"],
                    varying_function=partial(
                        syst_def["args"]["varying_function"],
                        events=events,
                        year=dataset_year
                    )
                )
            else:
                warnings.warn(
                    f"Systematic '{syst}' is defined for object '{obj_type}' "
                    "but no corresponding collection was provided."
                )
        elif syst in available_weight_systematics:
            # Weight systematics are handled later.
            continue
        else:
            warnings.warn(f"Could not process systematic variation '{syst}'.")
    return collections


def check_corr_syst_combinations(corrections_dict, systematics_dict, logger):
    """
    This function is a sanity check for the choice of systematics and corrections which the user wants to process.
    It ensures that systematic variations of a correction can only be processed when the correction itself is applied.
    """
    for dataset in systematics_dict.keys():
        for chosen_syst in systematics_dict[dataset]:
            if (
                chosen_syst in weight_corrections.keys()
                and chosen_syst not in corrections_dict[dataset]
            ) or (
                chosen_syst in object_corrections.keys()
                and chosen_syst not in corrections_dict[dataset]
            ):
                # scale unc. will be applied to MC while the correction is applied to data. Exception.
                if (
                    "scale" in chosen_syst.lower()
                    or "jec" in chosen_syst.lower()
                    or "jer" in chosen_syst.lower()
                ):
                    continue
                logger.info(
                    f"Requested to evaluate systematic variation {chosen_syst} for dataset {dataset} without applying the corresponding correction. \nThis is not intended.\nExiting."
                )
                sys.exit(1)
