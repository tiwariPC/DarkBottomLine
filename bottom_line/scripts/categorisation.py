"""
Script for optimization of event categories based on boundaries on decorrelated mass resolution estimator and photon MVA ID cut.
The optimization is done by fitting signal and background histograms with appropriate functions, determining S/sqrt(B), and comparing the sensitivities achieved with different category partitionings.
This code can be run on 1 to 4 categories in mass resolution.

Usage:
    1. Run HiggsDNA on Diphoton, GJet and Hgg signal samples.
    2. Set variable `base_path` to point to the output of HiggsDNA.
    3. Define resolution and MVA ID boundaries in 'resboundaries' and 'MVAboundaries'. NB: the script will take a long time to run over 4 categories, so it is helpful to first run it on a coarse grid and then on a fine grid around the optimum.
    4. Define over how many resolution categories to run, e.g. cats = [1, 3] to test one vs. 3 resolution categories.
    5. Run the script.
"""

import numpy as np
import hist
import os
import glob
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import quad
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import mplhep as hep
import time
hep.style.use("CMS")


def exponential(x, a, b, c):
    return a * np.exp(- x / b) + c


def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2


def get_central_interval(y_vals, dx, area=0.68):
    sorted_indices = np.argsort(y_vals)[::-1]  # Sort in descending order
    cumulative_area = 0
    central_interval = []

    for idx in sorted_indices:
        cumulative_area += y_vals[idx] * dx
        central_interval.append(x_steps[idx])
        if cumulative_area >= area:
            break
    central_interval = [min(central_interval), max(central_interval)]
    return central_interval


def plot_category(hist_sig, hists_bkg, x_vals, y_vals_sig, y_vals_bkg, params_sig, ax, hist_unc_bkg=None):

    hep.histplot(
        hists_bkg,
        label=["GJet", "Diphoton"],
        histtype="fill",
        linewidth=3,
        stack=True,
        alpha=0.5,
        color=["tab:gray","tab:blue"],
        ax=ax
    )

    if hist_unc_bkg is not None:

        errors_bkg = np.sqrt(hist_unc_bkg.to_numpy()[0])
        lower_bound = hists_bkg[0].to_numpy()[0] + hists_bkg[1].to_numpy()[0] - errors_bkg
        upper_bound = hists_bkg[0].to_numpy()[0] + hists_bkg[1].to_numpy()[0] + errors_bkg
        _, bin_edges = hists_bkg[0].to_numpy()
        bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
        # Plot hatched region
        ax.fill_between(
            bin_centers,
            lower_bound,
            upper_bound,
            hatch='XXX',
            facecolor="none",
            edgecolor="tab:gray",
            linewidth=0
        )
    ax.plot(x_vals, y_vals_bkg, label="Background fit", color="black", linewidth=3, linestyle="--")
    hep.histplot(hist_sig, label=r"ggH+VBF+VH+ttH $(\times 10)$", histtype="fill", linewidth=3, color="tab:orange", ax=ax)
    ax.plot(x_vals, y_vals_sig, label=r"Signal fit $(\times 10)$", color="tab:red", linewidth=3, linestyle="--", alpha=0.7)
    ax.text(0.05, 0.73, r"Gauss 1: $\mu={:.2f}\,$GeV, $\sigma={:.2f}\,$GeV".format(params_sig[1], params_sig[2]), fontsize=18, transform=ax.transAxes)
    ax.text(0.05, 0.68, r"Gauss 2: $\mu={:.2f}\,$GeV, $\sigma={:.2f}\,$GeV".format(params_sig[4], params_sig[5]), fontsize=18, transform=ax.transAxes)
    ax.set_xlabel('Invariant diphoton mass [GeV]')
    ax.set_ylabel(r'Events / GeV')
    ax.set_ylim(0., 1.5 * ax.get_ylim()[1])
    ax.legend(ncol=2)
    ax.set_xlim(100.,180.)
    hep.cms.label(data=True, ax=ax, loc=0, label="Simulation Work in Progress", com=13.6, lumi=35.1, fontsize=22)


# values from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap and XSDB
dict_xSecs = {
    # in fb
    "ggH": 52.23e3 * 0.00227,
    "VBF": 4.078e3 * 0.00227,
    "VH": 2.4009e3 * 0.00227,
    "ttH": 0.5700e3 * 0.00227,
    "Diphoton": 89.14e3,
    "GJetPT20to40": 242.5e3,
    "GJetPT40": 919.1e3,
}
lumi_preEE = 8.1  # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#2022_Analysis_Summary_Table with normtag
lumi_postEE = 27.0  # https://twiki.cern.ch/twiki/bin/viewauth/CMS/PdmVRun3Analysis#2022_Analysis_Summary_Table with normtag

base_path = "/path/to/my/samples/processed/with/HiggsDNA/"

dict_paths_signal = {
    "ggH_preEE": base_path + "GluGluHtoGG_M-125_preEE/nominal/",
    "ggH_postEE": base_path + "GluGluHtoGG_M-125_postEE/nominal/",
    "VBF_preEE": base_path + "VBFHtoGG_M-125_preEE/nominal/",
    "VBF_postEE": base_path + "VBFHtoGG_M-125_postEE/nominal/",
    "VH_preEE": base_path + "VHtoGG_M-125_preEE/nominal/",
    "VH_postEE": base_path + "VHtoGG_M-125_postEE/nominal/",
    "ttH_preEE": base_path + "ttHtoGG_M-125_preEE/nominal/",
    "ttH_postEE": base_path + "ttHtoGG_M-125_postEE/nominal/"
}

dict_paths_bkg = {
    "Diphoton_preEE": base_path + "GG-Box-3Jets_MGG-80_preEE/nominal/",
    "Diphoton_postEE": base_path + "GG-Box-3Jets_MGG-80_postEE/nominal/",
    "GJetPT20to40_preEE": base_path + "GJet_PT-20to40_DoubleEMEnriched_MGG-80_preEE/nominal/",
    "GJetPT20to40_postEE": base_path + "GJet_PT-20to40_DoubleEMEnriched_MGG-80_postEE/nominal/",
    "GJetPT40_preEE": base_path + "GJet_PT-40_DoubleEMEnriched_MGG-80_preEE/nominal/",
    "GJetPT40_postEE": base_path + "GJet_PT-40_DoubleEMEnriched_MGG-80_postEE/nominal/"
}


signal_events_preEE = []
signal_events_postEE = []

for process in dict_paths_signal.keys():
    campaign = process.split('_')[-1]
    lumi = lumi_preEE if campaign == "preEE" else lumi_postEE

    files_signal = glob.glob(dict_paths_signal[process] + "/*.parquet")
    data_signal = [pd.read_parquet(f) for f in files_signal]
    events = pd.concat(data_signal, ignore_index=True)
    events = events[(events.mass > 100) & (events.mass < 180)]

    sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in files_signal)
    events["weight"] *= (lumi * dict_xSecs[process.split('_')[0]] / sum_genw_beforesel)

    if campaign == "preEE":
        signal_events_preEE.append(events)
    else:
        signal_events_postEE.append(events)

# Concatenate all events
signal_events_preEE = pd.concat(signal_events_preEE, ignore_index=True)
signal_events_preEE["min_mvaID"] = np.min([signal_events_preEE.lead_corr_mvaID_run3.values, signal_events_preEE.sublead_corr_mvaID_run3.values], axis=0)
signal_events_postEE = pd.concat(signal_events_postEE, ignore_index=True)
signal_events_postEE["min_mvaID"] = np.min([signal_events_postEE.lead_corr_mvaID_run3.values, signal_events_postEE.sublead_corr_mvaID_run3.values], axis=0)
signal_events = pd.concat([signal_events_preEE, signal_events_postEE], ignore_index=True)

bkg_events_preEE = []
bkg_events_postEE = []

for process in dict_paths_bkg.keys():
    campaign = process.split('_')[-1]
    lumi = lumi_preEE if campaign == "preEE" else lumi_postEE

    files_bkg = glob.glob(dict_paths_bkg[process] + "/*.parquet")
    data_bkg = [pd.read_parquet(f) for f in files_bkg]
    events = pd.concat(data_bkg, ignore_index=True)
    events = events[(events.mass > 100) & (events.mass < 180)]

    sum_genw_beforesel = sum(float(pq.read_table(file).schema.metadata[b'sum_genw_presel']) for file in files_bkg)
    events["weight"] *= (lumi * dict_xSecs[process.split('_')[0]] / sum_genw_beforesel)
    events["process"] = process.split('_')[0]

    if "GJet" in process:
        print("INFO: applying overlap removal for sample", process)
        events = events[events.lead_genPartFlav + events.sublead_genPartFlav != 2]

    if campaign == "preEE":
        bkg_events_preEE.append(events)
    else:
        bkg_events_postEE.append(events)

bkg_events_preEE = pd.concat(bkg_events_preEE, ignore_index=True)
bkg_events_preEE["min_mvaID"] = np.min([bkg_events_preEE.lead_corr_mvaID_run3.values, bkg_events_preEE.sublead_corr_mvaID_run3.values], axis=0)
bkg_events_postEE = pd.concat(bkg_events_postEE, ignore_index=True)
bkg_events_postEE["min_mvaID"] = np.min([bkg_events_postEE.lead_corr_mvaID_run3.values, bkg_events_postEE.sublead_corr_mvaID_run3.values], axis=0)
# remove events already here to make it faster...
bkg_events_preEE = bkg_events_preEE[bkg_events_preEE["min_mvaID"] > 0.]
bkg_events_postEE = bkg_events_postEE[bkg_events_postEE["min_mvaID"] > 0.]
bkg_events = pd.concat([bkg_events_preEE, bkg_events_postEE], ignore_index=True)

# we have several versions of the resolution estimator: sigma_m_over_m_corr_smeared_decorr, sigma_m_over_m_corr_decorr. Define this as extra variable to be chosen here:
resolution = "sigma_m_over_m_corr_smeared_decorr"
# resolution = "sigma_m_over_m_corr_decorr"

relevant_columns_sig = ['mass', 'weight', resolution, 'min_mvaID']
signal_events_preEE = signal_events_preEE[relevant_columns_sig]
signal_events_preEE["weight"] = 10 * signal_events_preEE["weight"].values
signal_events_postEE = signal_events_postEE[relevant_columns_sig]
signal_events_postEE["weight"] = 10 * signal_events_postEE["weight"].values
signal_events = signal_events[relevant_columns_sig]
signal_events["weight"] = 10 * signal_events["weight"].values

relevant_columns_bkg = ['mass', 'weight', resolution, 'min_mvaID', 'process']
bkg_events_preEE = bkg_events_preEE[relevant_columns_bkg]
bkg_events_postEE = bkg_events_postEE[relevant_columns_bkg]
bkg_events = bkg_events[relevant_columns_bkg]

scale_bkg = True
if scale_bkg:
    scale_diphoton = 1.42
    scale_GJet = 2.38
    print("INFO: scaling diphoton and GJet acording to chi2 fit to MVA ID with data.")
    bkg_events_preEE["weight"][bkg_events_preEE.process == "Diphoton"] *= scale_diphoton
    bkg_events_preEE["weight"][bkg_events_preEE.process != "Diphoton"] *= scale_GJet
    bkg_events_postEE["weight"][bkg_events_postEE.process == "Diphoton"] *= scale_diphoton
    bkg_events_postEE["weight"][bkg_events_postEE.process != "Diphoton"] *= scale_GJet
    bkg_events["weight"][bkg_events.process == "Diphoton"] *= scale_diphoton
    bkg_events["weight"][bkg_events.process != "Diphoton"] *= scale_GJet


cross_check_with_data = False
if cross_check_with_data:
    print("INFO: running categorisation as a cross-check using data sidebands to determine the bkg. \n This is only implemented for 3 categories!")
    # Paths for the actual data
    dict_paths_data = {
        "DataC_2022": base_path + "DataC_2022/nominal/",
        "DataD_2022": base_path + "DataD_2022/nominal/",
        "DataE_2022": base_path + "DataE_2022/nominal/",
        "DataF_2022": base_path + "DataF_2022/nominal/",
        "DataG_2022": base_path + "DataG_2022/nominal/"
    }

    events_data_preEE = []
    events_data_postEE = []
    for process, path in dict_paths_data.items():
        print("INFO: reading files for", process)
        files_data = glob.glob(path + "/*.parquet")
        data = [pd.read_parquet(f) for f in files_data]
        events = pd.concat(data, ignore_index=True)
        events = events[(events.mass > 100) & (events.mass < 180)]
        # blinding
        events = events[(events.mass < 120) | (events.mass > 130)]

        if process in ["DataC_2022", "DataD_2022"]:
            events_data_preEE.append(events)
        else:
            events_data_postEE.append(events)

    events_data_preEE = pd.concat(events_data_preEE, ignore_index=True)
    events_data_postEE = pd.concat(events_data_postEE, ignore_index=True)
    events_data_preEE["min_mvaID"] = np.min([events_data_preEE.lead_mvaID.values, events_data_preEE.sublead_mvaID.values], axis=0)
    events_data_postEE["min_mvaID"] = np.min([events_data_postEE.lead_mvaID.values, events_data_postEE.sublead_mvaID.values], axis=0)

    # remove events already here to make it faster...
    events_data_preEE = events_data_preEE[events_data_preEE["min_mvaID"] > 0.]
    events_data_postEE = events_data_postEE[events_data_postEE["min_mvaID"] > 0.]

    relevant_columns_data = ['mass', 'sigma_m_over_m_decorr', "sigma_m_over_m_smeared_decorr", 'min_mvaID']
    events_data_preEE = events_data_preEE[relevant_columns_data]
    events_data_postEE = events_data_postEE[relevant_columns_data]
    events_data = pd.concat([events_data_preEE, events_data_postEE], ignore_index=True)

# it is worth to run over a coarse grid first and then over a fine grid around the expected maximum, so adapt values accordingly!
if resolution == "sigma_m_over_m_corr_smeared_decorr":
    resboundaries = np.linspace(0.008, 0.02, 25)
elif resolution == "sigma_m_over_m_corr_decorr":
    resboundaries = np.linspace(0.005, 0.018, 14)

MVAboundaries = np.linspace(0.1, 0.3, 21)
relative_metrics = []
cats = [1, 2, 3, 4]  # MEOW


for n_cat in cats:

    print(f"\n INFO: starting calculation for {n_cat} categorie(s).\n")

    if n_cat == 1:
        path_plots_1_cat = "./Plots_1cat/"
        if not os.path.exists(path_plots_1_cat):
            os.makedirs(path_plots_1_cat)
        metrics = []

        for i, MVAbound in enumerate(MVAboundaries):
            # Filter events based on MVAbound
            signal_events_cat0 = signal_events[signal_events['min_mvaID'] > MVAbound]
            bkg_events_cat0 = bkg_events[bkg_events['min_mvaID'] > MVAbound]

            ##### background fit
            hist_bkg = hist.Hist(hist.axis.Regular(160, 100, 180))
            hist_bkg.fill(bkg_events_cat0.mass.values, weight=bkg_events_cat0.weight.values)
            histo, bin_edges = hist_bkg.to_numpy()
            bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(histo))]
            params_bkg, covariance = curve_fit(exponential, bin_centers, histo, p0=([10000, 40, 0]))

            ##### signal fit
            hist_sig = hist.Hist(hist.axis.Regular(160, 100, 180))
            hist_sig.fill(signal_events_cat0.mass.values, weight=signal_events_cat0.weight.values)
            histo, bin_edges = hist_sig.to_numpy()
            initial_guess = [5000, 124, 5, 5000, 125, 1]
            params_sig, covariance = curve_fit(double_gaussian, bin_centers, histo, p0=initial_guess)

            x_steps = np.linspace(bin_edges[0], bin_edges[-1], 10000)
            y_vals = double_gaussian(x_steps, *params_sig)

            # Normalize to form a PDF
            dx = x_steps[1] - x_steps[0]
            y_vals /= np.sum(y_vals * dx)
            central_interval = get_central_interval(y_vals, dx, area=0.68)
            signal_integral, _ = quad(double_gaussian, central_interval[0], central_interval[1], args=tuple(params_sig))
            background_integral, _ = quad(exponential, central_interval[0], central_interval[1], args=tuple(params_bkg))

            # Generating values for plotting
            y_vals_bkg = exponential(x_steps, params_bkg[0], params_bkg[1], params_bkg[2])
            y_vals_sig = double_gaussian(x_steps, params_sig[0], params_sig[1], params_sig[2], params_sig[3], params_sig[4], params_sig[5])

            metric = (signal_integral / np.sqrt(background_integral))
            metrics.append(metric)

            # Plotting
            hist_diphoton = hist.Hist(hist.axis.Regular(80, 100, 180))
            hist_diphoton.fill(bkg_events_cat0.mass[bkg_events_cat0.process == "Diphoton"].values, weight=bkg_events_cat0.weight[bkg_events_cat0.process == "Diphoton"].values)
            hist_GJet = hist.Hist(hist.axis.Regular(80, 100, 180))
            hist_GJet.fill(bkg_events_cat0.mass[bkg_events_cat0.process != "Diphoton"].values, weight=bkg_events_cat0.weight[bkg_events_cat0.process != "Diphoton"].values)
            # hist of squared weights for error bars of weighted hists. I am not aware of an easier method to get correct error bars with the hist package....
            hist_bkg_unc = hist.Hist(hist.axis.Regular(80, 100, 180))
            hist_bkg_unc.fill(bkg_events_cat0.mass.values, weight=bkg_events_cat0.weight.values**2)

            # put to figure
            fig, ax = plt.subplots(1, 1, figsize=(10 * n_cat, 10))
            plot_category(hist_sig[::2j], [hist_GJet, hist_diphoton], x_steps, y_vals_sig * 2, y_vals_bkg * 2, params_sig, ax=ax, hist_unc_bkg=hist_bkg_unc)
            fig.suptitle("MVA ID bound: {:.3f}, abs. metric: {:.4f}".format(MVAbound, metric), fontsize=24)
            fig.tight_layout()
            path = path_plots_1_cat
            if not os.path.exists(path):
                os.makedirs(path)
            fig.savefig(path + f"fit_MVA_{i}.pdf")
            plt.close()

        metric_1cat = np.max(np.array(metrics))
        relative_metrics.append(metric_1cat)
        print("Metric for 1 category:", metric_1cat)

    if 1 not in cats:
        metric_1cat = 42.81  # just for later comparison, adjust to your metric value
        relative_metrics.append(metric_1cat)

    if n_cat == 2:
        t0 = time.time()
        metrics = []
        for i_res, resbound in enumerate(resboundaries):
            print("Resolution bound", i_res)
            _metrics = []
            params = []  # for later plotting
            for i, MVAbound in enumerate(MVAboundaries):

                events_sig_cat0 = signal_events[(signal_events[resolution] < resbound) & (signal_events['min_mvaID'] > MVAbound)]
                events_sig_cat1 = signal_events[(signal_events[resolution] >= resbound) & (signal_events['min_mvaID'] > MVAbound)]
                events_bkg_cat0 = bkg_events[(bkg_events[resolution] < resbound) & (bkg_events['min_mvaID'] > MVAbound)]
                events_bkg_cat1 = bkg_events[(bkg_events[resolution] >= resbound) & (bkg_events['min_mvaID'] > MVAbound)]

                ##### background fit
                hist_bkg_cat0 = hist.Hist(hist.axis.Regular(160, 100, 180))
                hist_bkg_cat0.fill(events_bkg_cat0.mass.values, weight=events_bkg_cat0.weight.values)
                histo, bin_edges = hist_bkg_cat0.to_numpy()
                bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(histo))]
                params_bkg_cat0, _ = curve_fit(exponential, bin_centers, histo, p0=([10000, 40, 0]))

                hist_bkg_cat1 = hist.Hist(hist.axis.Regular(160, 100, 180))
                hist_bkg_cat1.fill(events_bkg_cat1.mass.values, weight=events_bkg_cat1.weight.values)
                histo, bin_edges = hist_bkg_cat1.to_numpy()
                bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(histo))]
                params_bkg_cat1, _ = curve_fit(exponential, bin_centers, histo, p0=([10000, 40, 0]))

                ##### signal fit
                hist_sig_cat0 = hist.Hist(hist.axis.Regular(160, 100, 180))
                hist_sig_cat0.fill(events_sig_cat0.mass.values, weight=events_sig_cat0.weight.values)
                histo, bin_edges = hist_sig_cat0.to_numpy()
                initial_guess = [5000, 124, 5, 5000, 125, 1]
                try:
                    params_sig_cat0, _ = curve_fit(double_gaussian, bin_centers, histo, p0=initial_guess)
                except RuntimeError:
                    print(f"WARNING: signal fit in cat. with res. bound [{resbound:.3f} and MVA bound {MVAbound:.3f} failed.")
                    params_sig_cat0 = [0., 124, 999, 0., 125, 999]

                hist_sig_cat1 = hist.Hist(hist.axis.Regular(160, 100, 180))
                hist_sig_cat1.fill(events_sig_cat1.mass.values, weight=events_sig_cat1.weight.values)
                histo, bin_edges = hist_sig_cat1.to_numpy()
                try:
                    params_sig_cat1, _ = curve_fit(double_gaussian, bin_centers, histo, p0=initial_guess)
                except RuntimeError:
                    print(f"WARNING: signal fit in cat. with res. bound [{resbound:.3f} and MVA bound {MVAbound:.3f} failed.")
                    params_sig_cat1 = [0., 124, 999, 0., 125, 999]

                params.append((params_sig_cat0, params_bkg_cat0, params_sig_cat1, params_bkg_cat1))

                # Generate a dense array of x values
                x_steps = np.linspace(bin_edges[0], bin_edges[-1], 10000)

                # Evaluate the fitted double Gaussian function at each x value
                y_vals_cat0 = double_gaussian(x_steps, *params_sig_cat0)
                y_vals_cat1 = double_gaussian(x_steps, *params_sig_cat1)

                # Normalize to form a PDF
                dx = x_steps[1] - x_steps[0]
                y_vals_cat0 /= np.sum(y_vals_cat0 * dx)
                y_vals_cat1 /= np.sum(y_vals_cat1 * dx)

                # Find the central interval
                central_interval_cat0 = get_central_interval(y_vals_cat0, dx, area=0.68)
                central_interval_cat1 = get_central_interval(y_vals_cat1, dx, area=0.68)

                signal_integral_cat0, _ = quad(double_gaussian, central_interval_cat0[0], central_interval_cat0[1], args=tuple(params_sig_cat0))
                signal_integral_cat1, _ = quad(double_gaussian, central_interval_cat1[0], central_interval_cat1[1], args=tuple(params_sig_cat1))

                background_integral_cat0, _ = quad(exponential, central_interval_cat0[0], central_interval_cat0[1], args=tuple(params_bkg_cat0))
                background_integral_cat1, _ = quad(exponential, central_interval_cat1[0], central_interval_cat1[1], args=tuple(params_bkg_cat1))

                metric = np.sqrt((signal_integral_cat0 / np.sqrt(background_integral_cat0))**2 + (signal_integral_cat1 / np.sqrt(background_integral_cat1))**2)
                _metrics.append(metric)

            # Plot best MVA category (if above threshold)
            if np.max(_metrics) / metric_1cat > 1.07:
                max_MVAbound = MVAboundaries[np.argmax(_metrics)]

                # select arrays for best MVA bound again
                events_sig_cat0 = signal_events[(signal_events[resolution] < resbound) & (signal_events['min_mvaID'] > max_MVAbound)]
                events_sig_cat1 = signal_events[(signal_events[resolution] >= resbound) & (signal_events['min_mvaID'] > max_MVAbound)]
                events_bkg_cat0 = bkg_events[(bkg_events[resolution] < resbound) & (bkg_events['min_mvaID'] > max_MVAbound)]
                events_bkg_cat1 = bkg_events[(bkg_events[resolution] >= resbound) & (bkg_events['min_mvaID'] > max_MVAbound)]

                # histogramming
                hists_signal = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(2)]
                hists_signal[0].fill(events_sig_cat0.mass.values, weight=events_sig_cat0.weight.values)
                hists_signal[1].fill(events_sig_cat1.mass.values, weight=events_sig_cat1.weight.values)

                # Assuming the process distinction is available in 'process' column in events dataframe
                hists_diphoton = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(2)]
                hists_diphoton[0].fill(events_bkg_cat0.mass[events_bkg_cat0.process == "Diphoton"].values, weight=events_bkg_cat0.weight[events_bkg_cat0.process == "Diphoton"].values)
                hists_diphoton[1].fill(events_bkg_cat1.mass[events_bkg_cat1.process == "Diphoton"].values, weight=events_bkg_cat1.weight[events_bkg_cat1.process == "Diphoton"].values)

                hists_GJet = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(2)]
                hists_GJet[0].fill(events_bkg_cat0.mass[events_bkg_cat0.process != "Diphoton"].values, weight=events_bkg_cat0.weight[events_bkg_cat0.process != "Diphoton"].values)
                hists_GJet[1].fill(events_bkg_cat1.mass[events_bkg_cat1.process != "Diphoton"].values, weight=events_bkg_cat1.weight[events_bkg_cat1.process != "Diphoton"].values)

                # extract fitted values
                params_sig_cat0, params_bkg_cat0, params_sig_cat1, params_bkg_cat1 = params[np.argmax(_metrics)]

                # generate x_steps based on the previously defined range
                x_steps = np.linspace(100, 180, 10000)

                y_vals_bkg_cat0 = exponential(x_steps, *params_bkg_cat0)
                y_vals_sig_cat0 = double_gaussian(x_steps, *params_sig_cat0)
                y_vals_bkg_cat1 = exponential(x_steps, *params_bkg_cat1)
                y_vals_sig_cat1 = double_gaussian(x_steps, *params_sig_cat1)

                # put to figure
                fig, axes = plt.subplots(1, n_cat, figsize=(10 * n_cat, 10))
                plot_category(hists_signal[0], [hists_GJet[0], hists_diphoton[0]], x_steps, y_vals_sig_cat0 * 2, y_vals_bkg_cat0 * 2, params_sig_cat0, ax=axes[0])
                plot_category(hists_signal[1], [hists_GJet[1], hists_diphoton[1]], x_steps, y_vals_sig_cat1 * 2, y_vals_bkg_cat1 * 2, params_sig_cat1, ax=axes[1])
                fig.suptitle("MVA ID bound: {:.3f}, resolution bound: {:.3f}, metric: {:.4f}".format(max_MVAbound, resbound, np.max(_metrics) / metric_1cat), fontsize=24)
                fig.tight_layout()
                path = "./Plots_2cats/"
                if not os.path.exists(path):
                    os.makedirs(path)
                fig.savefig(path + f"ResBound_{str(i_res)}.pdf")
                plt.close()

            metrics.append(np.max(_metrics))

        relative_metrics.append(np.max(metrics))

        fig, ax = plt.subplots()
        plt.plot(resboundaries, np.array(metrics) / metric_1cat)
        plt.xlabel("Boundary in sigma m / m")
        plt.ylabel("Relative sensitivity")
        plt.savefig("./sensitivity_2_cats.pdf")
        print("Metric 2 categories:", np.max(metrics))
        t1 = time.time()
        print(f"time for the whole 2 cat scan: {t1-t0:.2f}s.")

    if 2 not in cats:
        metric_2cats = 60.18  # just for later comparison
        relative_metrics.append(metric_2cats)

    if n_cat == 3 and not cross_check_with_data:
        t0 = time.time()
        metrics = np.zeros((len(resboundaries), len(resboundaries)))
        metrics_df = pd.DataFrame(columns=["resBound1", "resBound2", "MVA_bound", "relativeMetric"])

        for i, bound1 in enumerate(resboundaries):
            print("\n bound:", i)
            if bound1 > 0.02:
                print("WARNING: skipping cats with bound 1 > 0.02 for time reasons. Make sure that this fits your values!")  # these are worse anyways with the values we have here

            for j, bound2 in enumerate(resboundaries):
                if bound2 <= bound1:  # Ensuring increasing order of resboundaries, saving time
                    continue
                _metrics = []
                params = []
                for iMVA, MVAbound in enumerate(MVAboundaries):

                    events_sig_cats = [
                        signal_events[(signal_events[resolution] < bound1) & (signal_events['min_mvaID'] > MVAbound)],
                        signal_events[(signal_events[resolution] >= bound1) & (signal_events[resolution] < bound2) & (signal_events['min_mvaID'] > MVAbound)],
                        signal_events[(signal_events[resolution] >= bound2) & (signal_events['min_mvaID'] > MVAbound)]
                    ]

                    events_bkg_cats = [
                        bkg_events[(bkg_events[resolution] < bound1) & (bkg_events['min_mvaID'] > MVAbound)],
                        bkg_events[(bkg_events[resolution] >= bound1) & (bkg_events[resolution] < bound2) & (bkg_events['min_mvaID'] > MVAbound)],
                        bkg_events[(bkg_events[resolution] >= bound2) & (bkg_events['min_mvaID'] > MVAbound)]
                    ]

                    _params = []  # for later plotting
                    metric_val = 0

                    for k in range(3):
                        hist_bkg = hist.Hist(hist.axis.Regular(160, 100, 180))
                        hist_bkg.fill(events_bkg_cats[k].mass.values, weight=events_bkg_cats[k].weight.values)
                        histo, bin_edges = hist_bkg.to_numpy()
                        bin_centers = np.array([(bin_edges[_l] + bin_edges[_l + 1]) / 2 for _l in range(len(histo))])
                        params_bkg, _ = curve_fit(exponential, bin_centers, histo, p0=([10000, 40, 0]))

                        hist_sig = hist.Hist(hist.axis.Regular(160, 100, 180))
                        hist_sig.fill(events_sig_cats[k].mass.values, weight=events_sig_cats[k].weight.values)
                        histo, bin_edges = hist_sig.to_numpy()
                        initial_guess = [5000, 124, 5, 5000, 125, 1]

                        try:
                            params_sig, _ = curve_fit(double_gaussian, bin_centers, histo, p0=initial_guess)
                        except RuntimeError:
                            print(f"WARNING: signal fit in cat. with res. bounds [{bound1:.3f},{bound2:.3f}] and MVA bound {MVAbound:.3f} failed.")
                            params_sig = [0., 124, 999, 0., 125, 999]

                        _params.append(params_sig)
                        _params.append(params_bkg)

                        x_steps = np.linspace(bin_edges[0], bin_edges[-1], 10000)
                        y_vals = double_gaussian(x_steps, *params_sig)
                        dx = x_steps[1] - x_steps[0]
                        y_vals /= np.sum(y_vals * dx)

                        central_interval = get_central_interval(y_vals, dx, area=0.68)
                        signal_integral, _ = quad(double_gaussian, central_interval[0], central_interval[1], args=tuple(params_sig))
                        background_integral, _ = quad(exponential, central_interval[0], central_interval[1], args=tuple(params_bkg))

                        metric_val += (signal_integral / np.sqrt(background_integral)) ** 2

                    metric = np.sqrt(metric_val)
                    _metrics.append(metric)
                    params.append(_params)

                metrics[i, j] = np.max(_metrics)

                # Plot best MVA category for the current resolution category
                if np.max(_metrics) / metric_1cat > 1.07:
                    metrics_df.loc[len(metrics_df)] = [bound1, bound2, MVAboundaries[np.argmax(_metrics)], np.max(_metrics) / metric_1cat]
                    max_MVAbound = MVAboundaries[np.argmax(_metrics)]

                    # Select arrays for best MVA bound again
                    events_sig_cats = [
                        signal_events[(signal_events[resolution] < bound1) & (signal_events['min_mvaID'] > max_MVAbound)],
                        signal_events[(signal_events[resolution] >= bound1) & (signal_events[resolution] < bound2) & (signal_events['min_mvaID'] > max_MVAbound)],
                        signal_events[(signal_events[resolution] >= bound2) & (signal_events['min_mvaID'] > max_MVAbound)]
                    ]

                    events_bkg_cats = [
                        bkg_events[(bkg_events[resolution] < bound1) & (bkg_events['min_mvaID'] > max_MVAbound)],
                        bkg_events[(bkg_events[resolution] >= bound1) & (bkg_events[resolution] < bound2) & (bkg_events['min_mvaID'] > max_MVAbound)],
                        bkg_events[(bkg_events[resolution] >= bound2) & (bkg_events['min_mvaID'] > max_MVAbound)]
                    ]

                    # Histogramming
                    hists_signal = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    hists_diphoton = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    hists_GJet = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    # hist of squared weights for error bars of weighted hists. I am not aware of an easier method to get correct error bars with the hist package....
                    hists_bkg_unc = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    for i_, (event_sig, event_bkg) in enumerate(zip(events_sig_cats, events_bkg_cats)):
                        hists_signal[i_].fill(event_sig.mass.values, weight=event_sig.weight.values)
                        hists_diphoton[i_].fill(event_bkg.mass[event_bkg.process == "Diphoton"].values, weight=event_bkg.weight[event_bkg.process == "Diphoton"].values)
                        hists_GJet[i_].fill(event_bkg.mass[event_bkg.process != "Diphoton"].values, weight=event_bkg.weight[event_bkg.process != "Diphoton"].values)
                        hists_bkg_unc[i_].fill(event_bkg.mass.values, weight=event_bkg.weight.values**2)

                    # Extract fitted values
                    params_list = params[np.argmax(_metrics)]
                    y_vals_bkg = [exponential(x_steps, *params_list[_i * 2 + 1]) for _i in range(3)]
                    y_vals_sig = [double_gaussian(x_steps, *params_list[_i * 2]) for _i in range(3)]

                    # Put to figure
                    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
                    for _i in range(3):
                        plot_category(hists_signal[_i], [hists_GJet[_i], hists_diphoton[_i]], x_steps, y_vals_sig[_i] * 2, y_vals_bkg[_i] * 2, params_list[_i * 2], ax=axes[_i], hist_unc_bkg=hists_bkg_unc[_i])
                    fig.suptitle("MVA ID bound: {:.3f}, resolution bounds: [{:.3f},{:.3f}], metric: {:.4f}".format(max_MVAbound, bound1, bound2, np.max(_metrics) / metric_1cat), fontsize=24)
                    fig.tight_layout()
                    path = "./Plots_3cats/bound1_{:.3f}/".format(bound1)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    fig.savefig(path + "bound2_{}.pdf".format(j))
                    plt.close()

        relative_metrics.append(np.max(metrics[~np.isnan(metrics)]))
        t1 = time.time()
        print(f"time for the whole 3 cat scan: {t1-t0:.2f}s.")
        metrics_df.to_csv("./metrics_3cats.csv", float_format='%.5f')

        fig, ax = plt.subplots(figsize=(9,8))
        width = (resboundaries[-1] - resboundaries[0]) / (len(resboundaries) - 1)
        c = ax.imshow(metrics.T / metric_1cat, cmap='viridis', vmin=1, origin="lower", extent=[resboundaries[0], resboundaries[-1] + width, resboundaries[0], resboundaries[-1] + width])

        ax.set_xlabel('Lower resolution boundary', fontsize=22)
        ax.set_ylabel('Upper resolution boundary', fontsize=22)

        # Get the position of the current axis
        pos = ax.get_position()

        # Adjust the margin to a smaller value so the color bar is closer to the heatmap
        cbar_margin = 0.01  # Reduce the margin so the color bar is closer to the heatmap
        cbar_width = pos.width * 0.75 / len(resboundaries)  # Proportional width

        cbar_ax = fig.add_axes([pos.x1 + cbar_margin, pos.y0, cbar_width, pos.height])

        # Create the color bar in the new axis
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_label('Rel. significance w.r.t. 1 cat.', fontsize=20, labelpad=10)
        cbar.ax.tick_params(labelsize=16)

        # Set the tick labels for the color bar to be horizontal
        cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=0)

        ax.tick_params(axis='both', which='major', labelsize=18)

        num_rows, num_cols = metrics.T.shape
        for i in range(num_rows):
            for j in range(num_cols):
                value = metrics.T[i, j] / metric_1cat
                x = resboundaries[j] + width / 2
                y = resboundaries[i] + width / 2
                if value == 0:
                    continue
                fontsize = 14 if len(resboundaries) < 15 else 10
                ax.text(x, y, f"{value:.2f}", va='center', ha='center', color='white', fontsize=fontsize)
                print(x, y, f"{value:.2f}")

        hep.cms.label(data=True, ax=ax, loc=0, label="Work in Progress", com=13.6, fontsize=22)
        plt.savefig("./sensitivity_3_cats.pdf")

    if n_cat == 3 and cross_check_with_data:
        t0 = time.time()
        metrics = np.zeros((len(resboundaries), len(resboundaries)))
        metrics_df = pd.DataFrame(columns=["resBound1", "resBound2", "MVA_bound", "relativeMetric"])

        for i, bound1 in enumerate(resboundaries):
            print("\n bound:", i)
            if bound1 > 0.02:
                print("WARNING: skipping cats with bound 1 > 0.02 for time reasons. Make sure that this fits your values!")  # these are worse anyways with the values we have here

            for j, bound2 in enumerate(resboundaries):
                if bound2 <= bound1:  # Ensuring increasing order of resboundaries, saving time
                    continue
                _metrics = []
                params = []
                for iMVA, MVAbound in enumerate(MVAboundaries):

                    events_sig_cats = [
                        signal_events[(signal_events[resolution] < bound1) & (signal_events['min_mvaID'] > MVAbound)],
                        signal_events[(signal_events[resolution] >= bound1) & (signal_events[resolution] < bound2) & (signal_events['min_mvaID'] > MVAbound)],
                        signal_events[(signal_events[resolution] >= bound2) & (signal_events['min_mvaID'] > MVAbound)]
                    ]

                    events_bkg_cats = [
                        events_data[(events_data['sigma_m_over_m_smeared_decorr'] < bound1) & (events_data['min_mvaID'] > MVAbound)],
                        events_data[(events_data['sigma_m_over_m_smeared_decorr'] >= bound1) & (events_data['sigma_m_over_m_smeared_decorr'] < bound2) & (events_data['min_mvaID'] > MVAbound)],
                        events_data[(events_data['sigma_m_over_m_smeared_decorr'] >= bound2) & (events_data['min_mvaID'] > MVAbound)]
                    ]

                    _params = []  # for later plotting
                    metric_val = 0

                    for k in range(3):
                        hist_bkg = hist.Hist(hist.axis.Regular(160, 100, 180))
                        hist_bkg.fill(events_bkg_cats[k].mass.values)
                        histo, bin_edges = hist_bkg.to_numpy()
                        bin_centers = np.array([(bin_edges[_l] + bin_edges[_l + 1]) / 2 for _l in range(len(histo))])
                        # exclude signal window
                        blinding_mask = (bin_centers < 120) | (bin_centers > 130)
                        params_bkg, _ = curve_fit(exponential, bin_centers[blinding_mask], histo[blinding_mask], p0=([10000, 40, 0]))

                        hist_sig = hist.Hist(hist.axis.Regular(160, 100, 180))
                        hist_sig.fill(events_sig_cats[k].mass.values, weight=events_sig_cats[k].weight.values)
                        histo, bin_edges = hist_sig.to_numpy()
                        initial_guess = [5000, 124, 5, 5000, 125, 1]

                        try:
                            params_sig, _ = curve_fit(double_gaussian, bin_centers, histo, p0=initial_guess)
                        except RuntimeError:
                            print(f"WARNING: signal fit in cat. with res. bounds [{bound1:.3f},{bound2:.3f}] and MVA bound {MVAbound:.3f} failed.")
                            params_sig = [0., 124, 999, 0., 125, 999]

                        _params.append(params_sig)
                        _params.append(params_bkg)

                        x_steps = np.linspace(bin_edges[0], bin_edges[-1], 10000)
                        y_vals = double_gaussian(x_steps, *params_sig)
                        dx = x_steps[1] - x_steps[0]
                        y_vals /= np.sum(y_vals * dx)

                        central_interval = get_central_interval(y_vals, dx, area=0.68)
                        signal_integral, _ = quad(double_gaussian, central_interval[0], central_interval[1], args=tuple(params_sig))
                        background_integral, _ = quad(exponential, central_interval[0], central_interval[1], args=tuple(params_bkg))

                        metric_val += (signal_integral / np.sqrt(background_integral)) ** 2

                    metric = np.sqrt(metric_val)
                    _metrics.append(metric)
                    params.append(_params)

                metrics[i, j] = np.max(_metrics)
                print(np.max(_metrics))
                print("1cat", metric_1cat)

                # Plot best MVA category for the current resolution category
                if np.max(_metrics) / metric_1cat > 1.07:
                    metrics_df.loc[len(metrics_df)] = [bound1, bound2, MVAboundaries[np.argmax(_metrics)], np.max(_metrics) / metric_1cat]
                    max_MVAbound = MVAboundaries[np.argmax(_metrics)]

                    # Select arrays for best MVA bound again
                    events_sig_cats = [
                        signal_events[(signal_events[resolution] < bound1) & (signal_events['min_mvaID'] > max_MVAbound)],
                        signal_events[(signal_events[resolution] >= bound1) & (signal_events[resolution] < bound2) & (signal_events['min_mvaID'] > max_MVAbound)],
                        signal_events[(signal_events[resolution] >= bound2) & (signal_events['min_mvaID'] > max_MVAbound)]
                    ]

                    events_bkg_cats = [
                        bkg_events[(bkg_events[resolution] < bound1) & (bkg_events['min_mvaID'] > max_MVAbound)],
                        bkg_events[(bkg_events[resolution] >= bound1) & (bkg_events[resolution] < bound2) & (bkg_events['min_mvaID'] > max_MVAbound)],
                        bkg_events[(bkg_events[resolution] >= bound2) & (bkg_events['min_mvaID'] > max_MVAbound)]
                    ]

                    events_data_cats = [
                        events_data[(events_data['sigma_m_over_m_smeared_decorr'] < bound1) & (events_data['min_mvaID'] > max_MVAbound)],
                        events_data[(events_data['sigma_m_over_m_smeared_decorr'] >= bound1) & (events_data['sigma_m_over_m_smeared_decorr'] < bound2) & (events_data['min_mvaID'] > max_MVAbound)],
                        events_data[(events_data['sigma_m_over_m_smeared_decorr'] >= bound2) & (events_data['min_mvaID'] > max_MVAbound)]
                    ]

                    # Histogramming
                    hists_signal = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    hists_data = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    hists_diphoton = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    hists_GJet = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    # hist of squared weights for error bars of weighted hists. I am not aware of an easier method to get correct error bars with the hist package....
                    hists_bkg_unc = [hist.Hist(hist.axis.Regular(80, 100, 180)) for _ in range(3)]
                    for i_, (event_sig, event_bkg, event_data) in enumerate(zip(events_sig_cats, events_bkg_cats, events_data_cats)):
                        hists_signal[i_].fill(event_sig.mass.values, weight=event_sig.weight.values)
                        hists_data[i_].fill(event_data.mass.values)
                        hists_diphoton[i_].fill(event_bkg.mass[event_bkg.process == "Diphoton"].values, weight=event_bkg.weight[event_bkg.process == "Diphoton"].values)
                        hists_GJet[i_].fill(event_bkg.mass[event_bkg.process != "Diphoton"].values, weight=event_bkg.weight[event_bkg.process != "Diphoton"].values)
                        hists_bkg_unc[i_].fill(event_bkg.mass.values, weight=event_bkg.weight.values**2)

                    # Extract fitted values
                    params_list = params[np.argmax(_metrics)]
                    y_vals_bkg = [exponential(x_steps, *params_list[_i * 2 + 1]) for _i in range(3)]
                    y_vals_sig = [double_gaussian(x_steps, *params_list[_i * 2]) for _i in range(3)]

                    # Put to figure
                    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
                    for _i in range(3):
                        plot_category(hists_signal[_i], [hists_GJet[_i], hists_diphoton[_i]], x_steps, y_vals_sig[_i] * 2, y_vals_bkg[_i] * 2, params_list[_i * 2], ax=axes[_i], hist_unc_bkg=hists_bkg_unc[_i])
                        hep.histplot(hists_data[_i], label="Data 2022", ax=axes[_i], yerr=True, color="black", histtype='errorbar', markersize=12, elinewidth=3)
                        axes[_i].legend(ncol=2)
                    fig.suptitle("MVA ID bound: {:.3f}, resolution bounds: [{:.3f},{:.3f}], metric: {:.4f}".format(max_MVAbound, bound1, bound2, np.max(_metrics) / metric_1cat), fontsize=24)
                    fig.tight_layout()
                    path = "./Plots_3catsData/bound1_{:.3f}/".format(bound1)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    fig.savefig(path + "bound2_{}.pdf".format(j))
                    plt.close()

        relative_metrics.append(np.max(metrics))
        t1 = time.time()
        print(f"time for the whole 3 cat scan using actual data: {t1-t0:.2f}s.")
        metrics_df.to_csv("./metrics_3catsData.csv", float_format='%.5f')

    if 3 not in cats:
        metric_3cats = metric_2cats  # just for later comparison
        relative_metrics.append(metric_3cats)

    if n_cat == 4:
        path_plots_4_cats = "./Plots_4cats/"
        if not os.path.exists(path_plots_4_cats):
            os.makedirs(path_plots_4_cats)

        t0 = time.time()
        metrics = np.zeros((len(resboundaries), len(resboundaries), len(resboundaries)))
        metrics_df = pd.DataFrame(columns=["resBound1", "resBound2", "resBound3", "MVA_bound", "relativeMetric"])

        for i, bound1 in enumerate(resboundaries):
            print("\n bound 1:", i)
            if bound1 > 0.013:
                print("WARNING: skipping cats with bound 1 > 0.012 for time reasons. Make sure that this fits your values!")  # these are worse anyways with the values we have here
            for j, bound2 in enumerate(resboundaries):
                print("\t bound 2:", j)
                for k, bound3 in enumerate(resboundaries):
                    if bound2 <= bound1 or bound3 <= bound2:  # Ensuring increasing order of resboundaries, saving time
                        continue
                    _metrics = []
                    params = []
                    for iMVA, MVAbound in enumerate(MVAboundaries):
                        events_sig_cats = [
                            signal_events[(signal_events[resolution] < bound1) & (signal_events['min_mvaID'] > MVAbound)],
                            signal_events[(signal_events[resolution] >= bound1) & (signal_events[resolution] < bound2) & (signal_events['min_mvaID'] > MVAbound)],
                            signal_events[(signal_events[resolution] >= bound2) & (signal_events[resolution] < bound3) & (signal_events['min_mvaID'] > MVAbound)],
                            signal_events[(signal_events[resolution] >= bound3) & (signal_events['min_mvaID'] > MVAbound)]
                        ]

                        events_bkg_cats = [
                            bkg_events[(bkg_events[resolution] < bound1) & (bkg_events['min_mvaID'] > MVAbound)],
                            bkg_events[(bkg_events[resolution] >= bound1) & (bkg_events[resolution] < bound2) & (bkg_events['min_mvaID'] > MVAbound)],
                            bkg_events[(bkg_events[resolution] >= bound2) & (bkg_events[resolution] < bound3) & (bkg_events['min_mvaID'] > MVAbound)],
                            bkg_events[(bkg_events[resolution] >= bound3) & (bkg_events['min_mvaID'] > MVAbound)]
                        ]

                        metric_val = 0
                        _params = []  # for later plotting
                        for l_ in range(4):
                            hist_bkg = hist.Hist(hist.axis.Regular(160, 100, 180))
                            hist_bkg.fill(events_bkg_cats[l_].mass.values, weight=events_bkg_cats[l_].weight.values)
                            histo, bin_edges = hist_bkg.to_numpy()
                            bin_centers = np.array([(bin_edges[_l] + bin_edges[_l + 1]) / 2 for _l in range(len(histo))])
                            params_bkg, _ = curve_fit(exponential, bin_centers, histo, p0=([10000, 40, 0]))

                            hist_sig = hist.Hist(hist.axis.Regular(160, 100, 180))
                            hist_sig.fill(events_sig_cats[l_].mass.values, weight=events_sig_cats[l_].weight.values)
                            histo, bin_edges = hist_sig.to_numpy()
                            initial_guess = [5000, 124, 5, 5000, 125, 1]

                            try:
                                params_sig, _ = curve_fit(double_gaussian, bin_centers, histo, p0=initial_guess)
                            except RuntimeError:
                                print(f"WARNING: signal fit in cat. with res. bounds [{bound1:.3f},{bound2:.3f},{bound3:.3f}] and MVA bound {MVAbound:.3f} failed.")
                                params_sig = [0., 124, 999, 0., 125, 999]

                            _params.append(params_sig)
                            _params.append(params_bkg)

                            x_steps = np.linspace(bin_edges[0], bin_edges[-1], 10000)
                            y_vals = double_gaussian(x_steps, *params_sig)
                            dx = x_steps[1] - x_steps[0]
                            y_vals /= np.sum(y_vals * dx)

                            central_interval = get_central_interval(y_vals, dx, area=0.68)
                            signal_integral, _ = quad(double_gaussian, central_interval[0], central_interval[1], args=tuple(params_sig))
                            background_integral, _ = quad(exponential, central_interval[0], central_interval[1], args=tuple(params_bkg))

                            metric_val += (signal_integral / np.sqrt(background_integral)) ** 2

                        metric = np.sqrt(metric_val)
                        _metrics.append(metric)
                        params.append(_params)

                    metrics[i, j, k] = np.max(_metrics)
                    print(np.max(_metrics) / metric_1cat)

                    # Plot best MVA category for the current resolution category
                    if np.max(_metrics) / metric_1cat > 1.05:
                        metrics_df.loc[len(metrics_df)] = [bound1, bound2, bound3, MVAboundaries[np.argmax(_metrics)], np.max(_metrics) / metric_1cat]
                        max_MVAbound = MVAboundaries[np.argmax(_metrics)]

                        # Select arrays for best MVA bound again
                        events_sig_cats = [
                            signal_events[(signal_events[resolution] < bound1) & (signal_events['min_mvaID'] > max_MVAbound)],
                            signal_events[(signal_events[resolution] >= bound1) & (signal_events[resolution] < bound2) & (signal_events['min_mvaID'] > max_MVAbound)],
                            signal_events[(signal_events[resolution] >= bound2) & (signal_events[resolution] < bound3) & (signal_events['min_mvaID'] > max_MVAbound)],
                            signal_events[(signal_events[resolution] >= bound3) & (signal_events['min_mvaID'] > max_MVAbound)]
                        ]

                        events_bkg_cats = [
                            bkg_events[(bkg_events[resolution] < bound1) & (bkg_events['min_mvaID'] > max_MVAbound)],
                            bkg_events[(bkg_events[resolution] >= bound1) & (bkg_events[resolution] < bound2) & (bkg_events['min_mvaID'] > max_MVAbound)],
                            bkg_events[(bkg_events[resolution] >= bound2) & (bkg_events[resolution] < bound3) & (bkg_events['min_mvaID'] > max_MVAbound)],
                            bkg_events[(bkg_events[resolution] >= bound3) & (bkg_events['min_mvaID'] > max_MVAbound)]
                        ]

                        # Histogramming
                        hists_signal = [hist.Hist(hist.axis.Regular(160, 100, 180)) for _ in range(4)]
                        hists_diphoton = [hist.Hist(hist.axis.Regular(160, 100, 180)) for _ in range(4)]
                        hists_GJet = [hist.Hist(hist.axis.Regular(160, 100, 180)) for _ in range(4)]
                        for _i, (event_sig, event_bkg) in enumerate(zip(events_sig_cats, events_bkg_cats)):
                            hists_signal[_i].fill(event_sig.mass.values, weight=event_sig.weight.values)
                            hists_diphoton[_i].fill(event_bkg.mass[event_bkg.process == "Diphoton"].values, weight=event_bkg.weight[event_bkg.process == "Diphoton"].values)
                            hists_GJet[_i].fill(event_bkg.mass[event_bkg.process != "Diphoton"].values, weight=event_bkg.weight[event_bkg.process != "Diphoton"].values)

                        # Extract fitted values
                        params_list = params[np.argmax(_metrics)]
                        y_vals_bkg = [exponential(x_steps, *params_list[i_ * 2 + 1]) for i_ in range(4)]
                        y_vals_sig = [double_gaussian(x_steps, *params_list[i_ * 2]) for i_ in range(4)]

                        # Put to figure
                        fig, axes = plt.subplots(1, 4, figsize=(40, 10))
                        for _i in range(4):
                            plot_category(hists_signal[_i], [hists_GJet[_i], hists_diphoton[_i]], x_steps, y_vals_sig[_i], y_vals_bkg[_i], params_list[_i * 2], ax=axes[_i])

                        fig.suptitle("MVA ID bound: {:.3f}, resolution bounds: [{:.3f},{:.3f},{:.3f}], metric: {:.4f}".format(max_MVAbound, bound1, bound2, bound3, np.max(_metrics) / metric_1cat), fontsize=24)
                        fig.tight_layout()
                        path = "./Plots_4cats/bound1_{:.3f}/".format(bound1)
                        if not os.path.exists(path):
                            os.makedirs(path)
                        fig.savefig(path + "bound2_{}_bound3_{}.pdf".format(j,k))
                        plt.close()

            ### plot slices of 3D tensor as heatmap
            metric_ = metrics[i]
            fig, ax = plt.subplots(figsize=(10, 8))
            width = (resboundaries[-1] - resboundaries[0]) / (len(resboundaries) - 1)
            c = ax.imshow(metric_.T / metric_1cat, cmap='viridis', vmin=1, origin="lower", extent=[resboundaries[0], resboundaries[-1] + width, resboundaries[0], resboundaries[-1] + width])
            ax.set_xlabel('Boundary 2 Value')
            ax.set_ylabel('Boundary 3 Value')
            ax.set_title(f'Sensitivity 4 cats / 1 cat, bound 1: {resboundaries[i]:.3f}')
            cbar = fig.colorbar(c, ax=ax)
            cbar.set_label('Metric Value')

            num_rows, num_cols = metric_.T.shape
            for i_ in range(num_rows):
                for j_ in range(num_cols):
                    value = metric_.T[i_, j_] / metric_1cat
                    x = resboundaries[j_] + width / 2
                    y = resboundaries[i_] + width / 2
                    if value == 0:
                        continue
                    fontsize = 14 if len(resboundaries) < 15 else 10
                    ax.text(x, y, f"{value:.2f}", va='center', ha='center', color='white', fontsize=fontsize)
            plt.tight_layout()
            plt.savefig(path_plots_4_cats + f"sensitivity_{i}.pdf")
            plt.clf()

        relative_metrics.append(np.max(metrics[~np.isnan(metrics)]))
        metrics_df.to_csv("./metrics_4cats.csv", float_format='%.5f')
        t1 = time.time()
        print(f"time for the whole 4 cat scan: {t1-t0:.2f}s.")

if 4 not in cats:
    metric_4cats = metric_3cats  # just for later comparison
    relative_metrics.append(metric_4cats)

relative_metrics = np.array(relative_metrics) / metric_1cat
print(relative_metrics)

labels = ["1 cat.", "2 cat.", "3 cat.", "4 cat."]
categories = [1, 2, 3, 4]
plt.figure(figsize=(10, 6))
plt.scatter(categories, relative_metrics, color='blue', s=100)
for i, label in enumerate(labels):
    plt.annotate(label, (categories[i], relative_metrics[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14)
plt.xlabel("Number of categories")
plt.ylabel("Rel. approx. significance")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.ylim(0.9, 1.3)
plt.tight_layout()
plt.savefig("relative_significance.pdf")
