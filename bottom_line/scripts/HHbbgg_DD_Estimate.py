import awkward
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from scipy import optimize as opt

from bottom_line.utils.logger_utils import setup_logger
logger = setup_logger(level='INFO')

#############################################################################################################
#  A parquet file containing the events from the data-driven method can be produced with the command:       #
#                                                                                                           #
#       ipython3 scripts/DD_Estimate.py -- -i ${INPUTFILE} -o ${OUTPATH} --run_era ${RUN_ERA}               #
#                                                                                                           #
#  For the run era, choose a run era for which there is a PDF available (Currently 2022preEE and            #
#  2022postEE). For the input file, please use a yaml file formatted like the one linked below:             #
#                                                                                                           #
#       https://gitlab.cern.ch/hhbbgg/docs/-/blob/master/v1/samples_v1_Run3_2022postEE.yaml?ref_type=heads  #
#                                                                                                           #
#############################################################################################################

pdf_coeffs = {
    "Run3_2022preEE":{
        "EE": [0.3745739369285325, -0.26155191192118127, 0.11398241596508048, 0.5117120026643069, 0.5610293197232438, -1.3923778516490706, 0.6717545764886738, 0.42480298623979806, -0.4740730156553807, -1.1744532213062604, 0.9311427400226765],
        "EB": [0.3902701455451318, -0.5115656696175526, -0.002384306796159553, 1.0900049177717233, 1.1879835363688696, -2.089155161196582]
        },
    "Run3_2022postEE":{
        "EE": [0.39422362871848493, -0.37434203553043915, -0.5389902158493887, 0.8262942159487295, 3.0630025721767993, -2.7858422084474195, -3.660114997786721, 3.6462932432417685, 6.799674103365295, -4.764110799261395, -6.583793780495415, 2.1130404204293103, 2.12283155450363],
        "EB": [0.39259062005763345, -0.4095276669800686, 0.1644570327734382, 1.021480044884044, 0.1399577902061723, -1.931989490230088, 1.6157883788638956, 0.7978700324002087, -1.9115037823819594, -1.443304761732576, 1.7363122773450523]
        }
}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="path to yaml mapping of samples"
        )

    parser.add_argument(
        "--output",
        "-o",
        default = './DDQCD_Output/',
        type=str,
        help="where to output files"
        )

    parser.add_argument(
        "--run-era",
        action = 'store',
        default = 'Run3_2022postEE',
        choices = ['Run3_2022preEE','Run3_2022postEE'],
        help="Choose PDF from run era"
        )

    parser.add_argument(
        "--unrestricted",
        action = 'store_true',
        help="Use alternate sampling method"
        )


    return parser.parse_args()


def sample(pdfcoeffs, maxMVAIDs, idcut, args, tag = '', mode = ''):
    ppf_res = 10000

    # Define PPF(Inverse of CDF) for Sampling
    pdf = np.polynomial.Polynomial(pdfcoeffs)
    maxMVAIDs = np.array(maxMVAIDs)
    cdf = pdf.integ(lbnd = -0.9)
    denom = cdf(idcut)
    maxodds = cdf(maxMVAIDs)
    minodds = cdf(idcut)

    y = np.linspace(-0.9, 1, ppf_res)
    ppfdist = cdf(y)
    ppf = lambda x: np.interp(x, ppfdist, y)

    #Initial min MVA IDs Sample
    probs = np.random.random(size = len(maxMVAIDs))
    probs = ((maxodds - minodds)*probs) + minodds
    minMVAIDs = ppf(probs)
    #Resample until min MVA IDs satisfy requirements, truncate if too many attempts used
    #Should not engage, exists as a failsafe
    if args.unrestricted:
        idx = (minMVAIDs > 1) | (minMVAIDs < idcut)
    else:
        idx = (minMVAIDs > maxMVAIDs) | (minMVAIDs < idcut)
    if any(idx):
        N = 100000
        for att in tqdm(range(N)):
            if not any(idx):
                break
            probs[idx] = np.random.random(size = np.sum(idx))
            minMVAIDs = ppf(probs)
            if args.unrestricted:
                idx = (minMVAIDs > 1) | (minMVAIDs < idcut)
            else:
                idx = (minMVAIDs > maxMVAIDs) | (minMVAIDs < idcut)
        if att + 1 == N:
            logger.info('Attempts maxed out.')
            logger.info(str(np.sum(minMVAIDs > maxMVAIDs)) + 'unweighted events failed the max MVA ID cut.')
            logger.info(str(np.sum(minMVAIDs < idcut)) + 'unweighted events failed the preselection cut.')


    # set weights based on max MVA ID, PDF
    if args.unrestricted:
        weights = np.ones_like(maxMVAIDs)
        newmin = np.min([minMVAIDs, maxMVAIDs], axis = 0)
        newmax = np.max([minMVAIDs, maxMVAIDs], axis = 0)
        minMVAIDs = newmin
        maxMVAIDs = newmax
    weights = (cdf(maxMVAIDs) - denom)/denom

    return minMVAIDs, maxMVAIDs, weights

def loadData(files, proc):
    #Load Sample and apply correct lumi, xs, and create new useful variables
    md = files[proc]

    #Load Sample File
    path = md['path']
    logger.info('Loading ' + path)
    df = awkward.from_parquet(path)

    #Adding necessary variables
    logger.info('Adding Min/Max MVAID Variables')
    df['Max_mvaID'] = np.max([df.lead_mvaID, df.sublead_mvaID], axis = 0)
    df['Min_mvaID'] = np.min([df.lead_mvaID, df.sublead_mvaID], axis = 0)
    idx = df.lead_mvaID == df.Max_mvaID
    for var in ['isScEtaEB', 'isScEtaEE']:
        df['Max_mvaID_'+var] = awkward.where(
            idx,
            df['lead_'+var],
            df['sublead_'+var],
        )
        df['Min_mvaID_'+var] = awkward.where(
            ~idx,
            df['lead_'+var],
            df['sublead_'+var],
        )
    return df

def main(args):

    #Load File Mapping
    with open(args.input, 'r') as f:
        files = yaml.load(f, Loader = yaml.Loader)

    #Load appropriate PDF Coeffs
    coeffs = pdf_coeffs[args.run_era]
    coeffs_EE = coeffs['EE']
    coeffs_EB = coeffs['EB']
    idcut = -0.7


    #Get sideband data events
    for i, proc in enumerate([f for f in files if 'Data' in f]):
        if i == 0:
            data = loadData(files, proc)
        else:
            temp = loadData(files, proc)
            data = awkward.concatenate([data, temp])
    sdbd_events = data[(data.Max_mvaID > idcut) & (data.Min_mvaID < idcut)]
    logger.info('Total SBD Yield: ' + str(np.sum(sdbd_events.weight)))
    del data

    #Generate new min MVA IDs using EE, EB PDFs
    Min_mvaID_EE, Max_mvaID_EE, Min_weights_EE = sample(coeffs_EE, sdbd_events.Max_mvaID, idcut, args, tag = 'EE', mode = 'fake')
    Min_mvaID_EB, Max_mvaID_EB, Min_weights_EB = sample(coeffs_EB, sdbd_events.Max_mvaID, idcut, args, tag = 'EB', mode = 'fake')

    #Collect proper mva IDs
    newmin = np.ones_like(Min_mvaID_EB)
    newweight = np.ones_like(Min_mvaID_EB)
    newmin[sdbd_events.Min_mvaID_isScEtaEB] = Min_mvaID_EB[sdbd_events.Min_mvaID_isScEtaEB]
    newmin[sdbd_events.Min_mvaID_isScEtaEE] = Min_mvaID_EE[sdbd_events.Min_mvaID_isScEtaEE]
    newweight[sdbd_events.Min_mvaID_isScEtaEB] = Min_weights_EB[sdbd_events.Min_mvaID_isScEtaEB]
    newweight[sdbd_events.Min_mvaID_isScEtaEE] = Min_weights_EE[sdbd_events.Min_mvaID_isScEtaEE]
    logger.info('DD EE Yields: ' + str(np.sum(newweight[sdbd_events.Min_mvaID_isScEtaEE])))
    logger.info('DD EB Yields: ' + str(np.sum(newweight[sdbd_events.Min_mvaID_isScEtaEB])))

    #Stitch min MVA IDs to lead, sublead photons
    leadismax = sdbd_events.lead_mvaID == sdbd_events.Max_mvaID
    sdbd_events['lead_mvaID'] = awkward.where(
        ~leadismax,
        newmin,
        sdbd_events.lead_mvaID
    )
    sdbd_events['sublead_mvaID'] = awkward.where(
        leadismax,
        newmin,
        sdbd_events.sublead_mvaID
    )
    sdbd_events['weight'] = newweight

    #Drop extra variables and save parquet file
    dropper = [f for f in sdbd_events.fields if 'Min_mvaID' in f and 'Max_mvaID' in f]
    keeper_fields = [f for f in sdbd_events.fields if not f in dropper]
    sdbd_events = sdbd_events[keeper_fields]
    filename = args.output + "/DDQCDGJET.parquet"
    logger.info('Saving ' + filename)
    awkward.to_parquet(sdbd_events, filename)
    files['DDQCDGJets'] = {'path':filename, 'xs':1}
    filename = args.output + args.input.split('/')[-1].replace('.yaml', '_updatedDDQCDGJets.yaml')
    logger.info('Saving ' + filename)
    with open(filename, 'w') as f:
        yaml.dump(files, f)
    return sdbd_events


if __name__ == "__main__":
    args = parse_arguments()
    df = main(args)
