#!/usr/bin/env python
import argparse
import json
import ast
import os
import glob
import awkward as ak
from bottom_line.utils.logger_utils import setup_logger
import pyarrow.parquet as pq
import numpy as np
from importlib import resources
from bottom_line.scripts.postprocessing.tools.Btag_WeightSum_Calculation import Get_WeightSum_Btag, Renormalize_BTag_Weights
from collections import defaultdict


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
        dataset[diffVariableName] = ak.where(condition, diffId, dataset[diffVariableName])

    return dataset


def process_custom_accumulator(source_path, logger):
    # Get the custom accumulator from all files in the source path
    accumulator = defaultdict(float)
    source_files = glob.glob("%s/*.parquet" % source_path)
    for f in source_files:
        try:
            file_accumulator = pq.read_table(f).schema.metadata[b'custom_accumulator']
        except KeyError as e:
            logger.error(f"Custom accumulator requested but not found in file {f}")
            raise e
        file_accumulator = file_accumulator.decode("utf-8")
        file_accumulator = ast.literal_eval(file_accumulator)
        for key, value in file_accumulator.items():
            accumulator[key] += value
    return dict(accumulator)


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
        "--is-data",
        default=False,
        action="store_true",
        help="Files to be merged are data and therefore do not require normalisation.",
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
    parser.add_argument(
        "--custom-accumulator",
        default=False,
        action="store_true",
        dest="custom_accumulator",
        help="If set, the script will process the custom accumulator from the parquet files.",
    )

    args = parser.parse_args()
    source_paths = args.source.split(",")
    target_paths = args.target.split(",")

    BASEDIR = resources.files("bottom_line").joinpath("")

    if args.genBinning != "":
        if args.abs:
            genBinning_path = os.path.realpath(args.genBinning)
        else:
            genBinning_path = os.path.join(BASEDIR, "scripts/postprocessing/sample_gen_binning.json")
        with open(genBinning_path, 'r') as json_file:
            gen_binning = json.load(json_file)
    else:
        gen_binning = None

    logger = setup_logger(level="INFO")


    if (
        (len(source_paths) != len(target_paths))
        or (args.source == "")
        or (args.target == "")
    ):
        logger.info("You gave a different number of sources and targets")
        exit


    if args.cats_dict != "":
        if args.abs:
            cats_path = os.path.realpath(args.cats_dict)
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


# TODO: is it possible to read all files metadata with the ParquetDataset function. Currently extracting norm outside
    if (not args.is_data) & (not args.skip_normalisation):
        logger.info(
            "Extracting sum of gen weights (before selection) from metadata of files to be merged."
        )
        if(args.do_b_weight_normalisation): IsBtagNorm_sys_arr,WeightSum_preBTag_arr,WeightSum_postBTag_arr,WeightSum_postBTag_sys_arr = Get_WeightSum_Btag(source_paths,logger)

        sum_genw_beforesel_arr = []
        for i, source_path in enumerate(source_paths):
            source_files = glob.glob("%s/*.parquet" % source_path)
            sum_genw_beforesel = 0
            for f in source_files:
                sum_genw_beforesel += float(pq.read_table(f).schema.metadata[b'sum_genw_presel'])
            sum_genw_beforesel_arr.append(sum_genw_beforesel)
        logger.info(
            "Successfully extracted sum of gen weights (before selection)"
        )

    for i, source_path in enumerate(source_paths):
        # Process custom accumulator
        if args.custom_accumulator:
            logger.info(f"Processing custom accumulator for {source_path}")
            custom_accumulator = process_custom_accumulator(source_path, logger)

        for cat in cat_dict:
            logger.info("-" * 125)
            logger.info(
                f"INFO: Starting parquet file merging. Attempting to read ParquetDataset from {source_path}, for category: {cat}"
            )
            dataset = pq.ParquetDataset(source_path, filters=cat_dict[cat]["cat_filter"])
            logger.info("ParquetDataset read successfully.")
            logger.info(
                f"Attempting to merge ParquetDataset and save to {target_paths[i]}."
            )
            if "Data" in target_paths[i]:
                os.makedirs("/".join(target_paths[i].split("/")[:-1]), exist_ok=True)
            else:
                os.makedirs(target_paths[i], exist_ok=True) # Create target directory if it does not exist
            pq.write_table(
                dataset.read(), target_paths[i] + cat + "_merged.parquet"
            )  # dataset.read() is a pyarrow table
            logger.info(
                f"Success! Merged parquet file is located in {target_paths[i]}{cat}_merged.parquet."
            )
            # If MC then open the merged dataset and add normalised weight column (sumw = efficiency)
            # TODO: can we add column before writing table and prevent re-reading in as awkward array
            if (not args.is_data) & (not args.skip_normalisation):
                # Remove ParquetDataset from memory and read file in as awkward array
                del dataset
                dataset_arr = ak.from_parquet(target_paths[i] + cat + "_merged.parquet")
                # Add filtering for differentials here

                if gen_binning != None:
                    for keys in gen_binning:
                        var_dict = {ast.literal_eval(key): value for key, value in gen_binning[keys].items()}
                        dataset_arr = filter_and_set_diff_variable(dataset_arr, var_dict, keys, "diffVariable_" + keys)

                # Add column for unnormalised weight
                dataset_arr['weight_nominal'] = dataset_arr['weight']
                # normalise nominal and systematics weights by sum of gen weights before selection
                syst_weight_fields = [field for field in dataset_arr.fields if (("weight_" in field) and ("Up" in field or "Down" in field))]
                for weight_field in ["weight"] + syst_weight_fields:
                    dataset_arr[weight_field] = dataset_arr[weight_field] / sum_genw_beforesel_arr[i]
                if(args.do_b_weight_normalisation):
                    if((WeightSum_preBTag_arr[i]/WeightSum_postBTag_arr[i])!=1):
                        dataset_arr = Renormalize_BTag_Weights(dataset_arr,target_paths[i],cat,WeightSum_preBTag_arr[i],WeightSum_postBTag_arr[i],WeightSum_postBTag_sys_arr[i],IsBtagNorm_sys_arr[i],logger)
                ak.to_parquet(dataset_arr, target_paths[i] + cat + "_merged.parquet")
                logger.info(
                    "Successfully added normalised weight column to dataset"
                )

            # Add custom accumulator as metadata
            if args.custom_accumulator:
                logger.info(f"Adding custom accumulator")
                table = pq.read_table(target_paths[i] + cat + "_merged.parquet")
                table = table.replace_schema_metadata({b'custom_accumulator': json.dumps(custom_accumulator).encode('utf-8')})
                pq.write_table(table, target_paths[i] + cat + "_merged.parquet")
                logger.info(f"Custom accumulator added successfully")
            logger.info("-" * 125)

if __name__ == "__main__":
    main()
