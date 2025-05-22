import glob
import awkward as ak
import pyarrow.parquet as pq


def Get_WeightSum_Btag(source_paths,logger):
    # All systematic variation list
    bTag_sys_variation = ['lfstats1', 'hfstats2' , 'jes', 'cferr2', 'lf', 'hf', 'lfstats2', 'hfstats1', 'cferr1']
    sum_weight_central_arr, sum_weight_central_wo_bTagSF_arr = [], []
    sum_weight_bTagSF_sys_arr = []
    flag_bWeight_sys_array = []
    


    for i, source_path in enumerate(source_paths):
        # create array to store the sum of the weights for all systematic vartions
        dataset_check_fields = ak.from_parquet(glob.glob("%s/*.parquet" % source_path)[0])
        # check if systamtic vatiation are stored by acessing one field of the parquet file
        flag_bWeight_sys = "weight_bTagSF_sys_jesDown" in dataset_check_fields.fields
        del dataset_check_fields
        if (flag_bWeight_sys):
            logger.info(
                f"Attampeting Extracting sum of central weights and bweight systematics from metadata of files to be merged from {source_path}"
            )
        else:
            logger.info(
                "Skiping the renormalization of b-tagging systematic weights. Please check if you have stored the weights for bTag systematic variation. Dont worry if you are not evaluating btaging systematic for now"
            )
        source_files = glob.glob("%s/*.parquet" % source_path)
        sum_weight_central,sum_weight_central_wo_bTagSF = 0,0
        sum_weight_bTagSF_sys_dct = {}
        if (flag_bWeight_sys):
            # dictionory to store up and down variation together
            for numSys in range(0, len(bTag_sys_variation)):
                sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Up"] = 0
                sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Down"] = 0

        for f in source_files:
            try:
                # read the sum of the weights from metadata without any systematic variation
                sum_weight_central += float(pq.read_table(f).schema.metadata[b'sum_weight_central'])
                sum_weight_central_wo_bTagSF += float(pq.read_table(f).schema.metadata[b'sum_weight_central_wo_bTagSF'])
            except:
                logger.info(
                    "Skiping the renormalization of weights from b-tagging systematics. Please check if you have stored sum of the weights after applying the b-weight systematics in the metadata with proper naming. Example: sum_weight_bTagSF_jesUp, sum_weight_bTagSF_jesDown."
                )
                # return sum of the weights before and after b-weight to 1 so that the ration will be one  and merge_parquet.py will not process renormalization
                sum_weight_central,sum_weight_central_wo_bTagSF = 1.0,1.0 
            if (flag_bWeight_sys):
                for numSys in range(0, len(bTag_sys_variation)):
                    try:
                        # read the sum of the weights from metadata for all systematic variation
                        sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Up"] += float(pq.read_table(f).schema.metadata[bytes('sum_weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Up',encoding='utf8')])
                        sum_weight_bTagSF_sys_dct["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Down"] += float(pq.read_table(f).schema.metadata[bytes('sum_weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Down',encoding='utf8')])
                    except:
                        logger.info(
                            "Skiping the renormalization of weights from btagging systematics. Please check if you have stored sum of the weights after appling the bweight systematics in the metadata with proper nameing : example: sum_weight_bTagSF_jesUp, sum_weight_bTagSF_jesDown"
                        )
                        flag_bWeight_sys = False
                        break

        sum_weight_central_arr.append(sum_weight_central)
        sum_weight_central_wo_bTagSF_arr.append(sum_weight_central_wo_bTagSF)

        flag_bWeight_sys_array.append(flag_bWeight_sys)
        sum_weight_bTagSF_sys_arr.append(sum_weight_bTagSF_sys_dct)
        
    logger.info(
        "Successfully extracted sum of weights with and without b-tag weights."
    )
    if (flag_bWeight_sys):
        logger.info(
            "Successfully extracted sum of systematic weights with and without b-tag SF"
        )

    IsBtagNorm_sys_arr,WeightSum_preBTag_arr,WeightSum_postBTag_arr,dir_WeightSum_postBTag_sys_arr = flag_bWeight_sys_array, sum_weight_central_wo_bTagSF_arr, sum_weight_central_arr,sum_weight_bTagSF_sys_arr
    return IsBtagNorm_sys_arr,WeightSum_preBTag_arr,WeightSum_postBTag_arr,dir_WeightSum_postBTag_sys_arr

def Renormalize_BTag_Weights(dataset,target_path,cat,WeightSum_preBTag,WeightSum_postBTag,WeightSum_postBTag_sys,IsBtagNorm_sys,logger):
    bTag_sys_variation = ['lfstats1', 'hfstats2' , 'jes', 'cferr2', 'lf', 'hf', 'lfstats2', 'hfstats1', 'cferr1']
    logger.info(
        f"Attempting to renormalize the weights wrt no b-tag SF from {target_path}{cat}_merged.parquet"
    )
    # Modify existing column

    if (WeightSum_preBTag != 0 and WeightSum_postBTag != 0):
        dataset['weight'] = dataset['weight'] * (WeightSum_preBTag / WeightSum_postBTag)
        logger.info(
            f"Successfully renormalised weights wrt no b-tag SF from {target_path}{cat}_merged.parquet"
        )
    else:
        logger.info(
            f"Skipping weights renormalisation wrt No bTagSF from {target_path}{cat}_merged.parquet"
        )
    if (IsBtagNorm_sys):
        for numSys in range(0, len(bTag_sys_variation)):
            dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Up'] = dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Up'] * (WeightSum_preBTag / WeightSum_postBTag_sys["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Up"])
            dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Down'] = dataset['weight_bTagSF_sys_' + bTag_sys_variation[numSys] + 'Down'] * (WeightSum_preBTag / WeightSum_postBTag_sys["sum_weight_bTagSF_" + bTag_sys_variation[numSys] + "Down"])
        logger.info(
            f"Successfully renormalised weights wrt no b-tag SF from {target_path}{cat}_merged.parquet"
                )
    else:
        logger.info(
            f"Skipping systematic weights renormalisation wrt no b-tag SF from {target_path}{cat}_merged.parquet"
        )
    return dataset
