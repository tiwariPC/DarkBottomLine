#!/usr/bin/env python
import argparse
from bottom_line.utils.logger_utils import setup_logger
import uproot
import awkward as ak
import os
import json
from importlib import resources


def split_awkward_arrays_by_length(d, target_length=5000):
    split_dicts = []
    max_len = max(len(arr) for arr in d.values())
    num_chunks = (max_len + target_length - 1) // target_length  # ceiling division

    for i in range(num_chunks):
        start = i * target_length
        end = min((i + 1) * target_length, max_len)
        current_split = {}
        for key, arr in d.items():
            current_split[key] = arr.__getitem__(slice(start, end))

        split_dicts.append(current_split)

    return split_dicts


def main():
    parser = argparse.ArgumentParser(
        description="Simple utility script to convert one parquet file into one ROOT file."
    )
    parser.add_argument("source", type=str, help="Path to input file.")
    parser.add_argument("target", type=str, help="Path to desired output file.")
    parser.add_argument("type", type=str, help="Type of dataset (data or mc).")
    parser.add_argument(
        "--log", dest="log", type=str, default="INFO", help="Logger info level"
    )
    parser.add_argument("--process", type=str, default="", help="Production mode.")
    parser.add_argument(
        "--notag",
        dest="notag",
        action="store_true",
        default=False,
        help="create NOTAG dataset as well.",
    )
    parser.add_argument(
        "--do-syst",
        dest="do_syst",
        action="store_true",
        default=False,
        help="create branches for systematic variations",
    )
    parser.add_argument(
        "--cats",
        type=str,
        dest="cats_dict",
        default="",
        help="Dictionary containing category selections.",
    )
    parser.add_argument(
        "--vars",
        type=str,
        dest="vars_dict",
        default="",
        help="Dictionary containing variations.",
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
        "--tbasket-length",
        type=int,
        dest="tbasket_length",
        default=5000,
        help="Length of the tbasket in the ROOT file.",
    )
    args = parser.parse_args()
    source_path = args.source
    target_path = args.target
    notag = True if (args.type == "mc" and args.notag == True) else False
    process = args.process if (args.process != "") else "data"

    logger = setup_logger(level=args.log)

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

# Dictionary for renaming variables in ROOT tree output for final fits
    rename_dict = {
        "mass": "CMS_hgg_mass"
    }

    # Ensure that the target directory exists
    os.makedirs('/'.join(target_path.split("/")[:-1]), exist_ok=True)

    df_dict = {}
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
        "data": target_path.replace("merged.root", "allData_2017.root"),
    }
# Loading category informations (used for naming of files to read/write)
    if args.cats_dict != "":
        if args.abs:
            cats_path = os.path.realpath(args.cats_dict)
        else:
            cats_path = os.path.join(BASEDIR, args.cats_dict)
        with open(cats_path) as pf:
        # with resources.open_text("bottom_line", args.cats_dict) as pf:
            cat_dict = json.load(pf)
        for cat in cat_dict:
            logger.debug(f"Found category: {cat}")
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
            vars_path = os.path.realpath(args.vars_dict)
        else:
            vars_path = os.path.join(BASEDIR, args.vars_dict)
        # with resources.open_text("bottom_line", args.vars_dict) as pf:
        with open(vars_path) as pf:
            variation_dict = json.load(pf)
        for var in variation_dict:
            logger.debug(f"Found variation: {var}")
    else:
        if args.do_syst:
            raise Exception(
                "You provided an invalid dictionary containing systematic variations information, have a look at your version of prepare_output_file.py"
            )

    if args.do_syst:
        # object systematics come from a different file (you are supposed to have merged .parquet with the merge_parquet.py script)
        for var in variation_dict:
            df_dict[var] = {}
            for cat in cat_dict:
                var_path = source_path.replace(
                    "merged.parquet", f"{variation_dict[var]}/{cat}_merged.parquet"
                )
                logger.info(
                    f"Starting conversion of one parquet file to ROOT. Attempting to read file {var_path} for category: {cat}."
                )

                eve = ak.from_parquet(var_path)

                logger.info("Successfully read from parquet file with awkward.")

                dict = {}
                for i in eve.fields:
                    i_re = rename_dict[i] if i in rename_dict else i
                    dict[i_re] = eve[i]

                df_dict[var][cat] = dict

                logger.debug(
                    f"Successfully created dict from awkward arrays for {var} variation for category: {cat}."
                )

        logger.info(f"Attempting to write dict to ROOT file {target_path}.")
    else:
        for cat in cat_dict:
            var_path = source_path.replace("merged.parquet", f"{cat}_merged.parquet")
            logger.info(
                f"Starting conversion of one parquet file to ROOT. Attempting to read file {var_path}."
            )

            eve = ak.from_parquet(var_path)

            logger.info("Successfully read from parquet file with awkward.")

            dict = {}
            for i in eve.fields:
                i_re = rename_dict[i] if i in rename_dict else i
                dict[i_re] = eve[i]

            df_dict[cat] = dict

            logger.debug(
                f"Successfully created dict from awkward arrays without variation for category: {cat}."
            )

    cat_postfix = {"ggh": "GG2H", "vbf": "VBF", "tth": "TTH", "vh": "VH", "dy": "DY"}

# For MC: {inputTreeDir}/{production-mode}_{mass}_{sqrts}_{category}
# For data: {inputTreeDir}/Data_{sqrts}_{category}
    labels = {}
    names = {}
    if args.type == "mc":
        for cat in cat_dict:
            if len(process.split("_"))>1:
                # If process of the form {process}_{mass}
                names[
                    cat
                ] = f"DiphotonTree/{process.split('_')[0]}_{process.split('_')[-1]}_13TeV_{cat}"  # _"+cat_postfix[process]
            else:
                names[
                cat
                ] = f"DiphotonTree/{process}_125_13TeV_{cat}"  # _"+cat_postfix[process]
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
                        df_dict["NOMINAL"][cat][branch] = ak.flatten(df_dict["NOMINAL"][cat][branch], axis=0)
                    file[names[cat]] = df_dict["NOMINAL"][cat]
                    if notag:
                        file[name_notag] = df_dict["NOMINAL"][cat]  # this is wrong, to be fixed
                    for syst_name, weight, syst_, c in labels[cat]:
                        # Skip "NOMINAL" as information included in nominal tree
                        if syst_ == "NOMINAL":
                            continue
                        logger.debug(f"{syst_name}, {weight}, {syst_}, {c}")
                        # If the name is not in the variation dictionary it is assumed to be a weight systematic
                        var_list = [
                                ["CMS_hgg_mass", "CMS_hgg_mass"],
                                [weight, "weight"],
                                ["HTXS_Higgs_pt", "HTXS_Higgs_pt"],
                                ["HTXS_Higgs_y", "HTXS_Higgs_y"],
                                ["PTH", "PTH"],
                                ["YH", "YH"],
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
                                    red_dict[new_key] = ak.flatten(df_dict[syst_][cat][key], 0)

                            logger.info(f"Adding {syst_name}01sigma to out tree...")
                            file[syst_name + "01sigma"] = red_dict
                else:
                    logger.info(f"no events survived category selection for cat: {cat}")

            else:
                # if there are no syst there is no df_dict["NOMINAL"] entry in the dict
                if len(df_dict[cat][[*df_dict[cat]][0]]):
                    # same as before
                    for branch in df_dict[cat]:
                        df_dict[cat][branch] = ak.flatten(df_dict[cat][branch], axis=0)

                    logger.info(f"Adding cat {cat} to ROOT file...")

                    split_dict = split_awkward_arrays_by_length(df_dict[cat], target_length=int(args.tbasket_length))

                    for i, current_dict in enumerate(split_dict):
                        logger.debug(f"Adding {i + 1}th dict out of {len(split_dict)}")

                        if args.log == "DEBUG":
                            array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                            logger.debug(f"Size of current_dict: {sum(array_sizes.values())}")

                        if i == 0:
                            file[names[cat]] = current_dict
                        else:
                            file[names[cat]].extend(current_dict)

                    if notag:
                        # this is wrong, to be fixed
                        logger.info("Adding also NOTAG to ROOT file...")

                        split_dict = split_awkward_arrays_by_length(df_dict[cat], target_length=int(args.tbasket_length))
                        for i, current_dict in enumerate(split_dict):
                            logger.debug(f"Adding {i + 1}th dict out of {len(split_dict)}")

                            if args.log == "DEBUG":
                                array_sizes = {key: arr.nbytes for key, arr in current_dict.items()}
                                logger.debug(f"Size of current_dict: {sum(array_sizes.values())}")

                            if i == 0:
                                file[names[cat]] = current_dict
                            else:
                                file[names[cat]].extend(current_dict)
                else:
                    logger.info(f"no events survived category selection for cat: {cat}")

        logger.info(
            f"Successfully converted parquet file to ROOT file for process {process}."
        )

if __name__ == "__main__":
    main()
