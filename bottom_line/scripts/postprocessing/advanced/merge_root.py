#!/usr/bin/env python
import argparse
import json
import ast
import os
import glob
import awkward
from bottom_line.utils.logger_utils import setup_logger
import pyarrow.parquet as pq
import numpy as np
import uproot
from importlib import resources
from bottom_line.scripts.postprocessing.tools.Btag_WeightSum_Calculation import Get_WeightSum_Btag, Renormalize_BTag_Weights


def extract_tuples(input_string):
    tuples = []
    # Remove leading and trailing parentheses and split by comma
    tuple_strings = input_string.strip("()").split(";")
    for tuple_str in tuple_strings:
        # Remove leading and trailing whitespace and parentheses
        tuple_elements = tuple_str.strip("()").split(",")
        # Strip each element and append to the list of tuples
        tuples.append(tuple(map(str.strip, tuple_elements)))
    return tuples


def extract_filter(dataset, additionalConditionTuple):
    variable, operator, value = additionalConditionTuple

    if operator == ">":
        return dataset[variable] > float(value)
    elif operator == ">=":
        return dataset[variable] >= float(value)
    elif operator == "<":
        return dataset[variable] < float(value)
    elif operator == "<=":
        return dataset[variable] <= float(value)
    elif operator == "==":
        if ("True" in value) or ("False" in value):
            value = bool(value)
            return dataset[variable] == value
        else:
            return dataset[variable] == float(value)


def filter_and_set_diff_variable(dataset, ranges_dict, selectionVariableName="GenPTH", diffVariableName="diffVariable_GenPTH"):
    # Initialize diff variable in the awkward array
    dataset[diffVariableName] = 0

    # Specify variables which need the absolute value for the selection (eg. rapidity)
    absolute_value_vars = ["GenYH"]

    if len(dataset) > 0:

        for range_min, range_max, fiducialTag, additionalConditions in ranges_dict.keys():
            diffId = ranges_dict[(range_min, range_max, fiducialTag, additionalConditions)]

            if fiducialTag == "in":
                condition = (dataset["fiducialGeometricFlag"] == True)
            else:
                condition = (dataset["fiducialGeometricFlag"] == False)

            if selectionVariableName in absolute_value_vars:
                condition = condition & (np.abs(dataset[selectionVariableName]) >= range_min) & (np.abs(dataset[selectionVariableName]) < range_max)
            else:
                condition = condition & (dataset[selectionVariableName] >= range_min) & (dataset[selectionVariableName] < range_max)

            if additionalConditions != "":
                tuple_list = extract_tuples(additionalConditions)
                for additionalCondition in tuple_list:
                    condition = condition & extract_filter(dataset, additionalCondition)
            dataset[diffVariableName] = awkward.where(condition, diffId, dataset[diffVariableName])

    return dataset


def get_dataset(_args, folder_path, cat, is_data, is_syst, source_path, target_path, cat_dict, gen_binning, logger, rename_dict):

    renamed_dict = {}

    # TODO: is it possible to read all files metadata with the ParquetDataset function. Currently extracting norm outside
    if (not is_data) & (not _args.skip_normalisation):
        logger.info(
            "Extracting sum of gen weights (before selection) from metadata of files to be merged."
        )
        sum_genw_beforesel_arr = []
        source_files = glob.glob("%s/*.parquet" % folder_path)
        sum_genw_beforesel = 0
        for f in source_files:
            sum_genw_beforesel += float(pq.read_table(f).schema.metadata[b'sum_genw_presel'])
        sum_genw_beforesel_arr.append(sum_genw_beforesel)
        logger.info(
            "Successfully extracted sum of gen weights (before selection)"
        )

    logger.info("-" * 125)
    logger.info(
        f"INFO: Starting parquet file merging. Attempting to read ParquetDataset from {folder_path}, for category: {cat}"
    )
    if is_data and _args.merge_data:
        path = glob.glob(os.path.join(source_path, '**', '*.parquet'), recursive=True)
    else:
        path = folder_path
    dataset = pq.ParquetDataset(path, filters=cat_dict[cat]["cat_filter"])
    logger.info("ParquetDataset read successfully.")
    logger.info(
        f"Attempting to merge ROOT file and save to {target_path}."
    )

    # If syst, read less branches to save memory
    if is_syst:
        columns_to_read = ["mass", "weight", "fiducialGeometricFlag"]
        table = dataset.read(columns=columns_to_read)

    # Load the piece into an Awkward Array
    table = dataset.read()
    eve = awkward.from_arrow(table)

    print("Successfully read from parquet piece with awkward.")

    # '''
    logger.info(
        f"Success! Merged ROOT file is located in {target_path}."
    )
    # If MC then open the merged dataset and add normalised weight column (sumw = efficiency)
    # TODO: can we add column before writing table and prevent re-reading in as awkward array
    if (not is_data) & (not _args.skip_normalisation):
        # Add filtering for differentials here
        if gen_binning != None:
            for keys in gen_binning:
                var_dict = {ast.literal_eval(key): value for key, value in gen_binning[keys].items()}
                eve = filter_and_set_diff_variable(eve, var_dict, keys, "diffVariable_" + keys)
        # Add column for unnormalised weight
        eve['weight_nominal'] = eve['weight']
        if len(eve) > 0:
            eve['weight'] = eve['weight'] / sum_genw_beforesel_arr[0]
            logger.info(
                "Successfully added normalised weight column to dataset"
            )
        else:
            logger.info(
                "No events survived category selection. Skipping normalisation step."
            )

    # Rename fields and store them in the dictionary
    for field_name in eve.fields:
        new_field_name = rename_dict.get(field_name, field_name)
        renamed_dict[new_field_name] = eve[field_name]
    logger.info("-" * 125)
    return renamed_dict


def main():
    parser = argparse.ArgumentParser(
        description="Simple utility script to merge all parquet files in one folder."
    )
    parser.add_argument(
    "--source",
    type=str,
    default="",
    help="Comma separated paths (with trailing slash) to folder where multiple parquet files are located. Careful: Folder should ONLY contain parquet files!",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Comma separated paths (with trailing slash) to desired folder. Resulting merged file is placed there.",
    )
    parser.add_argument(
        "--cats",
        type=str,
        dest="cats_dict",
        default="",
        help="Dictionary containing category selections.",
    )
    parser.add_argument(
        "--type",
        type=str,
        dest="type",
        default="",
        help="Type of dataset (data or mc).",
    )
    parser.add_argument(
        "--notag",
        dest="notag",
        action="store_true",
        default=False,
        help="create NOTAG dataset as well.",
    )
    parser.add_argument("--process", type=str, default="", help="Production mode.")
    parser.add_argument(
        "--vars",
        type=str,
        dest="vars_dict",
        default="",
        help="Dictionary containing variations.",
    )
    parser.add_argument(
        "--do-syst",
        dest="do_syst",
        action="store_true",
        default=False,
        help="create branches for systematic variations",
    )
    parser.add_argument(
        "--merge-all-data",
        dest="merge_data",
        action="store_true",
        default=False,
        help="Flag if all eras should be merged.",
    )
    parser.add_argument(
        "--skip-normalisation",
        default=False,
        action="store_true",
        help="Independent of file type, skip normalisation step",
    )
    parser.add_argument(
        "--abs",
        dest="abs",
        action="store_true",
        default=False,
        help="Uses absolute path for the dictionary files.",
    )
    parser.add_argument(
        "--genBinning",
        type=str,
        dest="genBinning",
        default="",
        help="Optional: Path to the JSON containing the binning at gen-level.",
    )
    parser.add_argument(
    "--do-b-weight-normalisation",
    default=False,
    action="store_true",
    help="Perform the bweight normalization to make sure the number of event remain the same before and after apling the b tagging weights",
   )

    args = parser.parse_args()
    source_path = args.source
    target_path = args.target

    # Create target directory if it does not exist
    os.makedirs("/".join(target_path.split('/')[:-1]), exist_ok=True)

    notag = True if (args.type == "mc" and args.notag == True) else False
    process = args.process if (args.process != "") else "data"
    is_data = (args.type == "data") or (args.type == "Data")

    BASEDIR = resources.files("bottom_line").joinpath("")

    if args.genBinning != "":
        if args.abs:
            genBinning_path = args.genBinning
        else:
            genBinning_path = os.path.join(BASEDIR, "scripts/postprocessing/sample_gen_binning.json")
        with open(genBinning_path, 'r') as json_file:
            gen_binning = json.load(json_file)
    else:
        gen_binning = None

    logger = setup_logger(level="INFO")

    logger.warning("Renormalize_BTag_Weights is not implemented yet for the new version of merge_root.py")

    rename_dict = {
        "mass": "CMS_hgg_mass"
    }

    outfiles = {
        "ch": target_path.replace(
            "merged.root", "output_cHToGG_M125_13TeV_amcatnloFXFX_pythia8.root"
        ),
        "ggh": target_path.replace(
            "merged.root", "output_GluGluHToGG_M125_13TeV_amcatnloFXFX_pythia8.root"
        ),
        "ggh_125": target_path.replace(
            "merged.root", "output_GluGluHToGG_M125_13TeV_amcatnloFXFX_pythia8.root"
        ),
        "ggh_120": target_path.replace(
            "merged.root", "output_GluGluHToGG_M120_13TeV_amcatnloFXFX_pythia8.root"
        ),
        "ggh_130": target_path.replace(
            "merged.root", "output_GluGluHToGG_M130_13TeV_amcatnloFXFX_pythia8.root"
        ),
        "vbf": target_path.replace(
            "merged.root", "output_VBFHToGG_M125_13TeV_amcatnlo_pythia8.root"
        ),
        "vbf_125": target_path.replace(
            "merged.root", "output_VBFHToGG_M125_13TeV_amcatnlo_pythia8.root"
        ),
        "vbf_120": target_path.replace(
            "merged.root", "output_VBFHToGG_M120_13TeV_amcatnlo_pythia8.root"
        ),
        "vbf_130": target_path.replace(
            "merged.root", "output_VBFHToGG_M130_13TeV_amcatnlo_pythia8.root"
        ),
        "vh": target_path.replace(
            "merged.root", "output_VHToGG_M125_13TeV_amcatnlo_pythia8.root"
        ),
        "vh_125": target_path.replace(
            "merged.root", "output_VHToGG_M125_13TeV_amcatnlo_pythia8.root"
        ),
        "vh_120": target_path.replace(
            "merged.root", "output_VHToGG_M120_13TeV_amcatnlo_pythia8.root"
        ),
        "vh_130": target_path.replace(
            "merged.root", "output_VHToGG_M130_13TeV_amcatnlo_pythia8.root"
        ),
        "tth": target_path.replace(
            "merged.root", "output_TTHToGG_M125_13TeV_amcatnlo_pythia8.root"
        ),
        "tth_125": target_path.replace(
            "merged.root", "output_TTHToGG_M125_13TeV_amcatnlo_pythia8.root"
        ),
        "tth_120": target_path.replace(
            "merged.root", "output_TTHToGG_M120_13TeV_amcatnlo_pythia8.root"
        ),
        "tth_130": target_path.replace(
            "merged.root", "output_TTHToGG_M130_13TeV_amcatnlo_pythia8.root"
        ),
        "dy": target_path.replace(
            "merged.root", "output_DYto2L.root"
        ),
        "ggbox": target_path.replace(
            "merged.root", "output_GG-Box-3Jets_MGG-80.root"
        ),
        "gjet": target_path.replace(
            "merged.root", "output_GJet_DoubleEMEnriched_MGG-80.root"
        ),
        "data": target_path.replace("merged.root", "allData.root"),
    }

    if args.cats_dict != "":
        if args.abs:
            cats_path = args.cats_dict
        else:
            cats_path = os.path.join(BASEDIR, "category.json")
        with open(cats_path) as pf:
            cat_dict = json.load(pf)
        for cat in cat_dict:
            logger.info(f"Found category: {cat}")
    else:
        logger.info(
            "You provided an invalid dictionary containing categories information, have a look at your version of prepare_output_file.py"
        )
        logger.info(
            "An inclusive NOTAG category is used as default"
        )
        cat_dict = {"NOTAG": {"cat_filter": [("pt", ">", -1.0)]}}

    # Loading variation informations (used for naming of files to read/write)
    # Active object systematics, weight systematics are just different sets of weights contained in the nominal file
    if args.vars_dict != "":
        if args.abs:
            vars_path = args.vars_dict
        else:
            vars_path = os.join(BASEDIR, args.vars_dict)
        with open(vars_path) as pf:
            variation_dict = json.load(pf)
        for var in variation_dict:
            logger.debug(f"Found variation: {var}")
    else:
        if args.do_syst:
            raise Exception(
                "You provided an invalid dictionary containing systematic variations information, have a look at your version of merge_root.py"
            )

    df_dict = {}
    if not args.merge_data:
        if args.do_syst or is_data:
            for var, var_value in variation_dict.items():
                df_dict[var] = {}
                logger.info(
                f"Variation: {var}"
                )
                for subfolder in os.listdir(source_path):
                    subfolder_path = os.path.join(source_path, subfolder)
                    if var_value in subfolder_path:
                        if os.path.isdir(subfolder_path):
                            for cat, _ in cat_dict.items():
                                if args.do_syst and not is_data:
                                    is_syst = True
                                else:
                                    is_syst = False
                                dict = get_dataset(args, subfolder_path, cat, is_data, is_syst, source_path, target_path, cat_dict, gen_binning, logger, rename_dict)
                                df_dict[var][cat] = dict

                    else:
                        continue
        else:
            df_dict["NOMINAL"] = {}
            for cat, _ in cat_dict.items():
                dict = get_dataset(args, source_path, cat, is_data, False, source_path, target_path, cat_dict, gen_binning, logger, rename_dict)
                df_dict["NOMINAL"][cat] = dict

    else:
        for var, var_value in variation_dict.items():
            df_dict[var] = {}
            for cat, _ in cat_dict.items():
                if var == "NOMINAL":
                    is_syst = False
                else:
                    is_syst = True
                dict = get_dataset(args, source_path, cat, is_data, is_syst, source_path, target_path, cat_dict, gen_binning, logger, rename_dict)
                df_dict[var][cat] = dict


    labels = {}
    names = {}
    if args.type == "mc":
        for cat in cat_dict:
            if len(process.split("_"))>1:
                # If process of the form {process}_{mass}
                names[
                    cat
                ] = f"DiphotonTree/{process.split('_')[0]}_{process.split('_')[-1]}_13TeV_{cat}"
            else:
                names[
                cat
                ] = f"DiphotonTree/{process}_125_13TeV_{cat}"
            labels[cat] = []
        if len(process.split("_"))>1:
            name_notag = "DiphotonTree/" + process.split('_')[0] + f"_{process.split('_')[-1]}_13TeV_NOTAG"
        else:
            name_notag = "DiphotonTree/" + process + "_125_13TeV_NOTAG"
        # flashggFinalFit needs to have each systematic variation in a different branch
        if args.do_syst:
            for var in variation_dict:
                for cat in cat_dict:
                    # for object systematics we have different files storing the variated collections with the nominal weights
                    syst_ = var
                    logger.info("found syst: %s for category: %s" % (syst_, cat))
                    if len(process.split("_"))>1:
                        labels[cat].append(
                            [
                                "DiphotonTree/" + process.split('_')[0] + f"_{process.split('_')[-1]}_13TeV_{cat}_" + syst_,
                                "weight",
                                syst_,
                                cat,
                            ]
                        )
                    else:
                        labels[cat].append(
                        [
                            "DiphotonTree/" + process + f"_125_13TeV_{cat}_" + syst_,
                            "weight",
                            syst_,
                            cat,
                        ]
                )

    else:
        for cat in cat_dict:
            labels[cat] = []
            labels[cat].append([f"DiphotonTree/Data_13TeV_{cat}", cat])
            names[cat] = f"DiphotonTree/Data_13TeV_{cat}"

    # Now we want to write the dictionary to a root file, since object systematics don't come from
    # the nominal file we have to separate again the treatment of them from the object ones
    with uproot.recreate(outfiles[process]) as file:
        logger.debug(outfiles[process])
        # Final fit want a separate tree for each category and variation,
        # the naming of the branches are quite rigid:
        # For MC: {inputTreeDir}/{production-mode}_{mass}_{sqrts}_{category}_{syst}
        # For data: {inputTreeDir}/Data_{sqrts}_{category}
        for cat in cat_dict:
            logger.debug(f"writing category: {cat}")

            if args.do_syst:
                # check that the category actually contains something, otherwise the flattening step will make the script crash,
                # an improvement (not sure if needed) may be to also write an empty TTree to not confuse FinalFit
                if len(df_dict["NOMINAL"][cat]["weight"]):
                    for branch in df_dict["NOMINAL"][cat]:
                        # here I had to add a flattening step to help uproot with the type of the awkward arrays,
                        # if you don't flatten (event if you don't have a nested field) you end up having a type like (len_of_array) * ?type, which make uproot very mad apparently
                        df_dict["NOMINAL"][cat][branch] = awkward.flatten(df_dict["NOMINAL"][cat][branch], axis=0)
                    file[names[cat]] = df_dict["NOMINAL"][cat]
                    for syst_name, weight, syst_, c in labels[cat]:
                        # Skip "NOMINAL" as information included in nominal tree
                        if syst_ == "NOMINAL":
                            continue
                        logger.debug(f"{syst_name}, {weight}, {syst_}, {c}")
                        # If the name is not in the variation dictionary it is assumed to be a weight systematic
                        var_list = [
                                ["CMS_hgg_mass", "CMS_hgg_mass"],
                                [weight, "weight"],
                                # ["PTH", "PTH"],
                                # ["YH", "YH"],
                                ["fiducialGeometricFlag", "fiducialGeometricFlag"]
                            ]
                        if gen_binning != None:
                            for keys in gen_binning:
                                var_list.append(["diffVariable_" + keys, "diffVariable_" + keys])

                        if syst_ not in variation_dict:
                            logger.debug(f"found weight syst {syst_}")
                            red_dict = {}
                            for key, new_key in var_list:
                                if "NOMINAL" in df_dict and cat in df_dict["NOMINAL"] and key in df_dict["NOMINAL"][cat]:
                                    red_dict[new_key] = df_dict["NOMINAL"][cat][key]

                            logger.info(f"Adding {syst_name}01sigma to out tree...")
                            file[syst_name + "01sigma"] = red_dict
                        else:
                            red_dict = {}
                            for key, new_key in var_list:
                                if syst_ in df_dict and cat in df_dict[syst_] and key in df_dict[syst_][cat]:
                                    if len(df_dict[syst_][cat][key]) > 0:
                                        red_dict[new_key] = awkward.flatten(df_dict[syst_][cat][key], 0)
                                    else: # Handle cases where the array is empty
                                        red_dict[new_key] = awkward.Array(np.array([], dtype=np.float64))

                            logger.info(f"Adding {syst_name}01sigma to out tree...")
                            file[syst_name + "01sigma"] = red_dict
                else:
                    logger.info(f"no events survived category selection for cat: {cat}")

            else:
                # if there are no syst there is no df_dict["NOMINAL"] entry in the dict
                if len(df_dict["NOMINAL"][cat][[*df_dict["NOMINAL"][cat]][0]]):
                    # same as before
                    for branch in df_dict["NOMINAL"][cat]:
                        df_dict["NOMINAL"][cat][branch] = awkward.flatten(df_dict["NOMINAL"][cat][branch], axis=0)
                    file[names[cat]] = df_dict["NOMINAL"][cat]
                    if notag:
                        file[name_notag] = df_dict["NOMINAL"][cat]  # this is wrong, to be fixed
                else:
                    logger.info(f"no events survived category selection for cat: {cat}")

        logger.info(
            f"Successfully converted parquet file to ROOT file for process {process}."
        )

if __name__ == "__main__":
    main()
