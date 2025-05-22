import glob
import os
import sys
import numpy as np
import pickle
from collections import defaultdict
import correctionlib.schemav2 as cs
import correctionlib
import gzip
import errno
from bottom_line.utils.logger_utils import setup_logger
import argparse

import importlib.resources as resources
resource_dir = resources.files("bottom_line")

def get_fetcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate correctionlib for b-tagging efficiencies."
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Path to the output folder (Btagging processor) of HiggsDNA.",
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output-name",
        help="Name of the output correctionlib."
        + "Must be entered as bTagEffFileName in runner.",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        help="Name of the optional output path containing the correctionlib.",
        default=None,
        required=False,
    )

    return parser.parse_args()


# Function to safely create a directory
def safe_mkdir(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def extract_eras(paths):
    """
    Extracts the list of available eras from a list of paths.
    An era is defined as the last string separated by the underscore (_)
    in the last element of the path (after the final slash).

    Args:
        paths (list): List of paths (strings).

    Returns:
        list: A list of unique eras extracted from the paths.
    """
    eras = set()
    for path in paths:
        # Get the last component of the path
        last_component = os.path.basename(path)
        # Split by underscore and get the last part
        if "_" in last_component:
            era = last_component.split("_")[-1]
            eras.add(era)
    return list(eras)


def group_by_eras_dict(paths):
    """
    Groups the paths into a dictionary by eras.
    An era is defined as the last string separated by the
    underscore in the last element of the path.

    Args:
        paths (list): List of paths (strings).

    Returns:
        dict: A dictionary where keys are eras and values are
        lists of paths belonging to those eras.
    """
    era_groups = defaultdict(list)

    for path in paths:
        # Get the last component of the path
        last_component = os.path.basename(path)
        # Split by underscore and get the last part
        if "_" in last_component:
            era = last_component.split("_")[-1]
            # Group the paths by era
            era_groups[era].append(path)

    # Convert defaultdict to a standard dictionary
    return era_groups


def create_hadron_flavour_correction(hadron_data, pt_edges):
    return [
        {"key": int(hadronFlavour), "value": cs.Binning(
            nodetype="binning",
            input="pt",
            edges=pt_edges,
            content=_efficiencies,
            flow="clamp",
        )}
        for hadronFlavour, _efficiencies in hadron_data.items()
    ]


def main():
    args = get_fetcher_args()

    logger = setup_logger(level="INFO")

    all_procs_eras = glob.glob(os.path.join(args.input, "*"))

    considered_eras = extract_eras(all_procs_eras)
    separated_procs_eras_dict = group_by_eras_dict(all_procs_eras)

    if args.output_dir is None:
        output_dir = os.path.join(resource_dir, "systematics/JSONs/bTagEff")
    else:
        output_dir = args.output_dir

    outputDir_era_dict = {
        "preEE": "2022_Summer22",
        "postEE": "2022_Summer22EE",
        "2022preEE": "2022_Summer22",
        "2022postEE": "2022_Summer22EE",
        "preBPix": "2023_Summer23",
        "postBPix": "2023_Summer23BPix",
        "2023preBPix": "2023_Summer23",
        "2023postBPix": "2023_Summer23BPix",
    }

    pt_bin_edges = [20, 30, 50, 70, 100, 140, 200, 300, 600, 1000]

    for current_era in considered_eras:
        all_efficiencies = {}
        for current_folder in separated_procs_eras_dict[current_era]:
            current_proc_long = current_folder.split("/")[-1]
            filepaths = glob.glob(os.path.join(current_folder, "nominal", "*.pkl"))
            genBJet_recoBJet = []
            genBJet_recoJet = []
            genCJet_recoBJet = []
            genCJet_recoJet = []
            genLJet_recoBJet = []
            genLJet_recoJet = []

            # Adding all the lists
            for current_file in filepaths:
                with open(current_file, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
                genBJet_recoBJet += list(data["genBJet_recoBJet"])
                genBJet_recoJet += list(data["genBJet_recoJet"])
                genCJet_recoBJet += list(data["genCJet_recoBJet"])
                genCJet_recoJet += list(data["genCJet_recoJet"])
                genLJet_recoBJet += list(data["genLJet_recoBJet"])
                genLJet_recoJet += list(data["genLJet_recoJet"])

            genBJet_recoBJet_histo, _ = np.histogram(genBJet_recoBJet, bins=pt_bin_edges)
            genBJet_recoJet_histo, _ = np.histogram(genBJet_recoJet, bins=pt_bin_edges)

            genCJet_recoBJet_histo, _ = np.histogram(genCJet_recoBJet, bins=pt_bin_edges)
            genCJet_recoJet_histo, _ = np.histogram(genCJet_recoJet, bins=pt_bin_edges)

            genLJet_recoBJet_histo, _ = np.histogram(genLJet_recoBJet, bins=pt_bin_edges)
            genLJet_recoJet_histo, _ = np.histogram(genLJet_recoJet, bins=pt_bin_edges)

            genBJet_bEfficiency = []
            genCJet_bEfficiency = []
            genLJet_bEfficiency = []

            for i in range(0, len(genBJet_recoJet_histo)):
                if genBJet_recoJet_histo[i] == 0:
                    genBJet_bEfficiency.append(0)
                else:
                    genBJet_bEfficiency.append(genBJet_recoBJet_histo[i] / genBJet_recoJet_histo[i])
                if genCJet_recoJet_histo[i] == 0:
                    genCJet_bEfficiency.append(0)
                else:
                    genCJet_bEfficiency.append(genCJet_recoBJet_histo[i] / genCJet_recoJet_histo[i])
                if genLJet_recoJet_histo[i] == 0:
                    genLJet_bEfficiency.append(0)
                else:
                    genLJet_bEfficiency.append(genLJet_recoBJet_histo[i] / genLJet_recoJet_histo[i])

            genBJet_bEfficiency_uncert = []
            genCJet_bEfficiency_uncert = []
            genLJet_bEfficiency_uncert = []

            for i in range(0, len(genBJet_recoJet_histo)):
                if genBJet_recoJet_histo[i] == 0:
                    genBJet_bEfficiency_uncert.append(0)
                else:
                    genBJet_bEfficiency_uncert.append(np.sqrt((genBJet_bEfficiency[i] * (1 - genBJet_bEfficiency[i])) / genBJet_recoJet_histo[i]))
                if genCJet_recoJet_histo[i] == 0:
                    genCJet_bEfficiency_uncert.append(0)
                else:
                    genCJet_bEfficiency_uncert.append(np.sqrt((genCJet_bEfficiency[i] * (1 - genCJet_bEfficiency[i])) / genCJet_recoJet_histo[i]))
                if genLJet_recoJet_histo[i] == 0:
                    genLJet_bEfficiency_uncert.append(0)
                else:
                    genLJet_bEfficiency_uncert.append(np.sqrt((genLJet_bEfficiency[i] * (1 - genLJet_bEfficiency[i])) / genLJet_recoJet_histo[i]))

            efficiencies = {
                5: genBJet_bEfficiency,
                4: genCJet_bEfficiency,
                0: genLJet_bEfficiency
            }

            all_efficiencies[current_proc_long] = efficiencies

        # Creation of the correctionlib file
        process_corrections = [
            {"key": process, "value": cs.Category(
                nodetype="category",
                input="hadronFlavour",
                content=create_hadron_flavour_correction(hadron_data, pt_bin_edges),
            )}
            for process, hadron_data in all_efficiencies.items()
        ]

        btagging_efficiencies = cs.Correction(
            name="btagging_efficiencies",
            description="Correction that contains the b-tagging efficiencies as a function of the proces, the jet hadron flavour and the transverse momentum.",
            version=1,
            inputs=[
                cs.Variable(name="process", type="string", description="Simulation process"),
                cs.Variable(name="hadronFlavour", type="int", description="Jet hadron flavour"),
                cs.Variable(name="pt", type="real", description="Jet transverse momentum"),
            ],
            output=cs.Variable(name="efficiency", type="real", description="B-tagging efficiency"),
            data=cs.Category(
                nodetype="category",
                input="process",
                content=process_corrections,
            ),
        )

        cset = correctionlib.schemav2.CorrectionSet(
            schema_version=2,
            description="Correction set for b-tagging efficiencies",
            corrections=[
                btagging_efficiencies
            ],
        )

        safe_mkdir(os.path.join(output_dir, outputDir_era_dict[current_era]))

        # Write the correctionlib data to the output file
        try:
            with gzip.open(os.path.join(output_dir, outputDir_era_dict[current_era], args.output_name + ".json.gz"), "wt") as fout:
                fout.write(cset.json(exclude_unset=True))

            logger.info("Successfully wrote data to the correctionlib.")
        except Exception as e:
            logger.error(f"Error writing to correctionlib file : {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
