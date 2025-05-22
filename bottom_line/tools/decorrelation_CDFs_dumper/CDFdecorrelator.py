from bottom_line.tools.decorrelator import cdfCalc
import bottom_line.tools.decorrelator as decorr
import matplotlib.pyplot as plt
import concurrent.futures
import awkward as ak
import mplhep as hep
import pandas as pd
import numpy as np
import argparse
import mplhep
import pickle
import os
import glob
import gzip

plt.style.use(mplhep.style.CMS)


def load_pkl_gz(file_path):
    with gzip.open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def read_parquet_file(file_path):
    return pd.read_parquet(file_path)


def generate_output_filename(var, era=None, outFile=None):
    if outFile:
        filename = outFile
    else:
        filename = f"{var}"
    if era:
        filename += f"_{era}"

    return f"{filename}"


def printProgressBar(iteration,total,prefix='',suffix='',decimals=1,length=100,fill=chr(9608),printEnd="\r"):

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


def diphoton_ak_array(diphotons: ak.Array) -> ak.Array:

    output = {}
    for field in ak.fields(diphotons):
        output[field] = diphotons[field]
    return ak.Array(output)


def getArrayBranchName(branchname, fieldname, index):
    if index != ():
        return '{}{}'.format(branchname, index[0])
    return '{}'.format(branchname)


def weighted_quantiles_interpolate(values, weights, quantiles=0.5):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]


def calculate_bins_position(mass):

    sorted_data = np.sort(mass)
    num_elements = len(sorted_data) // 40
    bins,x_bins = [],[]
    x_bins.append(100)

    for i in range(0, len(sorted_data), num_elements):
        bin_data = sorted_data[i:i + num_elements]
        bins.append(bin_data)
        x_bins.append(float(np.max(bin_data)))

    x_bins.pop()
    x_bins.pop()
    x_bins.append(180)

    return x_bins


def plot_sigma_over_m_profile(position, mean_value, mean_value_decorr, mean_value_10, mean_value_decorr_10, mean_value_90, mean_value_decorr_90, filename):
    filename = filename
    plt.close()

    # 10% quantile
    plt.plot(position, mean_value_decorr_10, linestyle='dashed', linewidth=4, color='orange')
    plt.plot(position, mean_value_10, linestyle='dashed', linewidth=4, color='blue')
    plt.plot([], [], ' ', label="Diphoton samples")

    # 50 quantile
    plt.plot(position, mean_value_decorr ,linewidth=4, color='orange', label="decorrelated")
    plt.plot(position, mean_value , linewidth=4, color='blue', label="not decorrelated")

    # 90% quantile
    plt.plot(position, mean_value_decorr_90, linestyle='dashed', linewidth=4, color='orange')
    plt.plot(position, mean_value_90, linestyle='dashed', linewidth=4, color='blue')

    # x = [120,130]
    # y = [0.0104,0.0104]
    # plt.plot(x, y, linewidth = 3, linestyle='--'color = 'red')
    plt.ylim(0.0085, 0.0255)
    plt.xlim(95, 185)
    plt.xlabel('Diphoton Mass [GeV]')
    plt.ylabel(r'$\sigma_{m}/m$')

    # plt.text(0.5, 0.75, "Diphoton MC samples", fontsize=22)

    hep.cms.label(data=True, loc=0, label="Work in progress", com=13.6, lumi=21.7)
    plt.tight_layout()
    plt.legend(fontsize=16, loc='upper right', facecolor='white', borderaxespad=1.).set_zorder(2)
    # plt.savefig("plots/" + filename + "_success_decorr.png")
    plt.savefig("plots/" + filename + "_success_decorr.pdf")


def plot_CDFS(mass, sigma_over_m, sigma_over_m_decorr, mc_weights, filename):

    mass_mask_1 = np.logical_and(mass > 100, mass < 100.5)
    mass_mask_2 = np.logical_and(mass > 125, mass < 125.5)
    mass_mask_3 = np.logical_and(mass > 170, mass < 170.5)

    great_mask = [mass_mask_1, mass_mask_2, mass_mask_3]
    great_CDF = []

    # First, the plot of the sigma_over_m correlated CDF in the three bins
    for i in range(3):
        val = sigma_over_m[great_mask[i]]
        dBins = np.linspace(0.,0.5,1001)
        hist, _ = np.histogram(val, weights=mc_weights[great_mask[i]], bins=dBins)
        rightEdge = dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum = bCum / float(bCum.max())
        cdfBinned = np.vstack((bCum,rightEdge))
        great_CDF.append(cdfBinned)

    # Ploting:
    plt.close()
    plt.plot(great_CDF[0][1], great_CDF[0][0], color='blue', alpha=0.7, linewidth=2, label=r' 100 < $m_{\gamma\gamma}$ < 100.5')
    plt.plot(great_CDF[1][1], great_CDF[1][0], color='red', alpha=0.7, linewidth=2, label=r' 125 < $m_{\gamma\gamma}$ < 125.5')
    plt.plot(great_CDF[2][1], great_CDF[2][0], color='purple', alpha=0.7, linewidth=2, label=r' 170 < $m_{\gamma\gamma}$ < 170.5')

    hep.cms.label(data=True, loc=0, label="Work in progress", com=13.6, lumi=21.7)

    plt.legend(fontsize=22, loc='lower right')
    plt.xlim(0, 0.035)
    plt.ylabel(r"$cdf(\sigma_{m}/m)$")
    plt.xlabel(r"$\sigma_{m}/m$")
    # plt.savefig("plots/" + filename + "_CDF.png")
    plt.savefig("plots/" + filename + "_CDF.pdf")

    # Now, the same plot but using the decorrelated sigma_over_m
    great_CDF = []

    # First, the plot of the sigma_over_m correlated CDF in the three bins
    for i in range(3):

        val = sigma_over_m_decorr[great_mask[i]]

        dBins = np.linspace(0., 0.5, 1001)

        hist, _ = np.histogram(val, weights=mc_weights[great_mask[i]], bins=dBins)
        rightEdge = dBins[1:]
        bCum = np.cumsum(hist)
        bCum[bCum < 0.] = 0
        bCum = bCum / float(bCum.max())
        cdfBinned = np.vstack((bCum,rightEdge))

        great_CDF.append(cdfBinned)

    # Ploting:
    plt.close()
    plt.plot(great_CDF[0][1], great_CDF[0][0],color='blue', alpha=0.7, linewidth=2, label=r' 100 < $m_{\gamma\gamma}$ < 100.5 ')
    plt.plot(great_CDF[1][1], great_CDF[1][0],color='red', alpha=0.7, linewidth=2, label=r' 125 < $m_{\gamma\gamma}$ < 125.5 ')
    plt.plot(great_CDF[2][1], great_CDF[2][0],color='purple', alpha=0.7, linewidth=2, label=r' 170 < $m_{\gamma\gamma}$ < 170.5 ')
    plt.legend(fontsize=22, loc='lower right')

    hep.cms.label(data=True, loc=0, label="Work in progress", com=13.6, lumi=21.7)

    plt.xlim(0,0.035)
    plt.ylabel(r'$cdf(\sigma_{m}^{Decorr}/m)$')
    plt.xlabel(r'$\sigma_{m}^{Decorr}/m$')
    # plt.savefig("plots/" + filename + "_CDF_decorr.png")
    plt.savefig("plots/" + filename + "_CDF_decorr.pdf")
    plt.close()

    # plot the transformatiom
    mass_mask_1 = np.logical_and(mass > 100 ,mass < 100.5)
    sig_m = sigma_over_m[mass_mask_1]
    sig_m_decorr = sigma_over_m_decorr[mass_mask_1]

    plt.scatter(sig_m, sig_m_decorr, label=' 100 - 100.5 ')
    plt.xlim(0.004,0.025)
    plt.ylim(0.004,0.025)

    # x == y line
    lims = [
        np.min([0.004, 0.025]),  # min of both axes
        np.max([0.004, 0.025]),  # max of both axes
    ]

    # now plot both limits against eachother
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel('Input Value')
    plt.ylabel('Corrected Value')
    plt.legend()
    plt.tight_layout()
    # plt.savefig("plots/" + filename + "_transformation.png")
    plt.savefig("plots/" + filename + "_transformation.pdf")
    plt.close()

    # plot the transformatiom - on signal range xD
    mass_mask_1 = np.logical_and(mass > 160 , mass < 160.5)
    sig_m = sigma_over_m[mass_mask_1]
    sig_m_decorr = sigma_over_m_decorr[mass_mask_1]

    plt.scatter(sig_m, sig_m_decorr, color='orange', label=' 160 - 160.5 ')
    plt.xlim(0.004,0.025)
    plt.ylim(0.004,0.025)

    # x == y line
    lims = [
        np.min([0.004, 0.025]),  # min of both axes
        np.max([0.004, 0.025]),  # max of both axes
    ]

    # now plot both limits against eachother
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel('Input Value')
    plt.ylabel('Corrected Value')
    plt.legend()
    plt.tight_layout()
    # plt.savefig("plots/" + filename + "_transformation_signal.png")
    plt.savefig("plots/" + filename + "_transformation_signal.pdf")
    plt.close()

    mass_mask_1 = np.logical_and(mass > 125, mass < 125.5)
    sig_m = sigma_over_m[mass_mask_1]
    sig_m_decorr = sigma_over_m_decorr[mass_mask_1]

    plt.scatter(sig_m, sig_m_decorr, label=' 125 - 125.5 ', color='red')
    plt.xlim(0.004,0.025)
    plt.ylim(0.004,0.025)

    lims = [
        np.min([0.004, 0.025]),
        np.max([0.004, 0.025])
    ]

    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlabel('Input Value')
    plt.ylabel('Corrected Value')
    plt.legend()
    plt.tight_layout()
    # plt.savefig("plots/" + filename + "_transformation_signal_id.png")
    plt.savefig("plots/" + filename + "_transformation_signal_id.pdf")

    return None


def plotParquet(events, var, dVar, filename):
    vector = pd.concat([events], ignore_index=True)
    filename = filename
    sigma_m_over_m = np.array(vector[var])
    sigma_m_over_m_decorr = np.array(vector[var + "_decorr"])
    mass = np.array(vector[dVar])
    mc_weights = np.array(vector["weight"])

    mask_mass = np.logical_and(mass >= 100, mass <= 180)
    mask_mass = np.logical_and(mask_mass, ~np.isnan(mass))

    mass, sigma_m_over_m, sigma_m_over_m_decorr, mc_weights = mass[mask_mass], sigma_m_over_m[mask_mass], sigma_m_over_m_decorr[mask_mass], mc_weights[mask_mass]
    x_bins = calculate_bins_position(mass)
    position, mean_value, mean_value_decorr = [], [], []

    mean_value_10,mean_value_decorr_10 = [], []
    mean_value_90,mean_value_decorr_90 = [], []
    for i in range(len(x_bins) - 1):

        mass_window = np.logical_and(mass >= x_bins[i], mass <= x_bins[i + 1])
        sigma_m_over_m_decorr_inside_window = sigma_m_over_m_decorr[mass_window]
        sigma_m_over_m_inside_window = sigma_m_over_m[mass_window]
        w_inside_window = mc_weights[mass_window]

        if (i == len(x_bins) - 2):
            position.append(180)
        elif (i == 0):
            position.append(100)
        else:
            position.append(float((x_bins[i] + x_bins[i + 1]) / 2))
        mean_value.append(weighted_quantiles_interpolate(sigma_m_over_m_inside_window, w_inside_window))
        mean_value_decorr.append(weighted_quantiles_interpolate(sigma_m_over_m_decorr_inside_window, w_inside_window))

        mean_value_10.append(weighted_quantiles_interpolate(sigma_m_over_m_inside_window, w_inside_window, quantiles=0.10))
        mean_value_decorr_10.append(weighted_quantiles_interpolate(sigma_m_over_m_decorr_inside_window, w_inside_window, quantiles=0.10))

        mean_value_90.append(weighted_quantiles_interpolate(sigma_m_over_m_inside_window, w_inside_window, quantiles=0.9))
        mean_value_decorr_90.append(weighted_quantiles_interpolate(sigma_m_over_m_decorr_inside_window, w_inside_window, quantiles=0.9))

    os.makedirs("plots", exist_ok=True)
    plot_sigma_over_m_profile(position, mean_value, mean_value_decorr, mean_value_10, mean_value_decorr_10, mean_value_90, mean_value_decorr_90, filename)
    plot_CDFS(mass, sigma_m_over_m, sigma_m_over_m_decorr, mc_weights, filename)


def main(options):
    filename = generate_output_filename(options.var, options.era, options.outFile)
    # reading the parquet files
    if not options.infilepath.endswith('/'):
        print(f"WARNING: Please make sure that {options.infilepath} is a path to .parquet files and ends with /")
        options.infilepath = options.infilepath + "/"
        print(f"INFO: To help you out, the path is changed to: {options.infilepath}")

    df = pd.DataFrame()
    files = glob.glob(str(options.infilepath) + "*.parquet")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(read_parquet_file, files))
    events = pd.concat(data, ignore_index=True)
    print(f"INFO: found {len(events)} events")

    # helper and parquet column read in
    if options.var in events:
        df[options.var] = events[options.var].to_numpy()
    else:
        print(f"ERROR: var not in columns of parquet files from {options.infilepath}")
        print(f"Please choose e.g. from {[col for col in events.columns if col.startswith('sigma')]} or inspect columns for more information!")
        exit()
    df[options.dVar] = events[options.dVar].to_numpy()
    df["weight"] = events.weight.to_numpy()

    # handling of CDFs
    calc = cdfCalc(df, options.var, options.dVar, np.linspace(100, 180, 161))
    calc.calcCdfs()
    cdfs = calc.cdfs
    dummyDf = pd.DataFrame({'{}'.format(options.var): [0], '{}'.format(options.dVar): [0]})
    decl = decorr.decorrelator(dummyDf, options.var, options.dVar, np.linspace(100., 180., 161))
    decl.cdfs = cdfs
    print("INFO: var, dVar:", options.var,", ", options.dVar)

    # giving variables to dataframe and resetting index
    decl.df = df.loc[:, [options.var, options.dVar]]
    decl.df.reset_index(inplace=True)

    # doing the decorr
    decorrelated_var = decl.doDecorr(options.ref)

    # create pickle file and remove the old sigma_m_over_m variable
    calc = cdfCalc(df, options.var, options.dVar, np.linspace(100, 180, 161))
    print(f"INFO: CDF contains: {df.columns.tolist()}")
    calc.dumpCdfs(filename + "_CDFs.pkl.gz")
    print("INFO: Created pickle file!")

    # a parquet file with the decorrelated variable will not be created by default, but can with "-p" flag
    if options.parquetGenerationOn or options.plotParquet:
        print("INFO: new parquet will be created...!")
        events['{}_decorr'.format(options.var)] = decorrelated_var
        events.to_parquet(filename + ".parquet")
        print("INFO: Created parquet file!")
    else:
        print("INFO: New parquet file will not be created, only pickle, otherwise give -p flag!")

    if options.plotPickle:
        os.makedirs("plots", exist_ok=True)
        plt.plot(calc.cdfs["100.25"][1], calc.cdfs["100.25"][0], label="100.25")
        plt.plot(calc.cdfs["125.25"][1], calc.cdfs["125.25"][0], label="125.25")
        plt.plot(calc.cdfs["170.25"][1], calc.cdfs["170.25"][0], label="170.25")
        plt.xlim(0,0.035)
        plt.ylabel(r"$cdf(\sigma_{m}/m)$")
        plt.xlabel(r"$\sigma_{m}/m$")
        plt.legend()
        # plt.savefig("plots/" + filename + "_CDFs_plot.png")
        plt.savefig("plots/" + filename + "_CDFs_pickle_plot.pdf")
        print("INFO: Pickle plots done!")

    if options.plotParquet:
        plotParquet(events, options.var, options.dVar, filename)
        print("INFO: Parquet plots done!")


def checkPickle():
    print("INFO: Checking pickle file: ", options.checkPickle)
    data = load_pkl_gz(options.checkPickle)
    os.makedirs("plots", exist_ok=True)
    plt.plot(data['100.25'][1], data['100.25'][0], label='100.25')
    plt.plot(data['125.25'][1], data['125.25'][0], label='125.25')
    plt.plot(data['170.25'][1], data['170.25'][0], label='170.25')
    plt.xlim(0,0.035)
    plt.ylabel(r"$cdf(\sigma_{m}/m)$")
    plt.xlabel(r"$\sigma_{m}/m$")
    plt.legend()
    # plt.savefig("plots_checks/" + "pickle_plot_check.png")
    plt.savefig("plots/" + "check_pickle_plot_check.pdf")
    print("INFO: Check pickle plots done!")


def checkParquet():
    print("INFO: Checking parquet files: ", options.checkParquet)
    print(f"INFO: The plots will contain {options.var} and {options.var}_decorr from the parquet file. This can be changed with -v flag.")
    data = read_parquet_file(options.checkParquet)
    if (str(options.var) + "_decorr") in data:
        plotParquet(data, options.var, options.dVar, filename="check")
        print("INFO: Check parquet plots done!")
    else:
        print(f"ERROR: {options.var} and/or {options.var}_decorr not in columns of parquet files from {options.checkParquet}")
        print(f"Please choose e.g. from {[col for col in data.columns if col.startswith('sigma')]} or inspect columns for more information!")
        exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-i', '--infilepath', action='store', type=str, help="Input file path, e.g. /net/.../samples/")
    requiredArgs.add_argument('-v','--var', default='sigma_m_over_m', action='store', type=str, help="variable you want to decorrelate (default: sigma_m_over_m)")
    requiredArgs.add_argument('-d','--dVar', default='mass', action='store', type=str, help="variable you want to correlate against, most likely mass, (default: mass)")
    requiredArgs.add_argument('-o','--outFile', action='store', type=str, help="filename and path to the decorrelated files, default: ")
    optArgs = parser.add_argument_group('Optional Arguments')
    optArgs.add_argument('-r', '--ref', action='store', type=float, default=125., help="reference mass for decorrelation")
    optArgs.add_argument('-p', '--parquetGenerationOn', action='store_true', help="Set this flag to generate parquet file")
    optArgs.add_argument('--plotPickle', action='store_true', help="Set this flag to plot pickle content")
    optArgs.add_argument('--plotParquet', action='store_true', help="Set this flag to plot parquet content")
    optArgs.add_argument('--checkPickle', action='store', type=str, help="Check already existing pickle file")
    optArgs.add_argument('--checkParquet', action='store', help="Check already existing parquet file")
    optArgs.add_argument('-e','--era', action='store', type=str, help="optional: choose era to give suffix in file name")
    options = parser.parse_args()
    if options.infilepath:
        main(options)
    if options.checkPickle:
        checkPickle()
    if options.checkParquet:
        checkParquet()
    if not options.infilepath and not options.checkPickle and not options.checkParquet:
        print("ERROR: Please provide an input file path or check an existing pickle file or parquet file!")
