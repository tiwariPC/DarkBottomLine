# Hist to plot sigma_m
import os 
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt 
import mplhep, hist
plt.style.use([mplhep.style.CMS])
import pandas as pd
import scipy
import mplhep as hep
import glob
import warnings

import json

warnings.filterwarnings("ignore")

def plotter( mc_hist, mc_hist_up, mc_hist_down , data_hist, output_filename, flow_bins='none' , xlabel = 'myVariable', yscale='linear', IspostEE=False, IsBarrel=False):

    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.03}, sharex=True)

    stat_errors     = np.sqrt(mc_hist.variances())
    data_stat_error = np.sqrt(data_hist.variances()) 

    # plotting nominal and data histograms
    hep.histplot(
        mc_hist,
        label = r'$Z \rightarrow ee$ - simulation',
        yerr = stat_errors,
        density = True,
        linewidth=3,
        ax=ax[0],
        flow=flow_bins
    )

    hep.histplot(
        data_hist,
        label = "Data",
        yerr= data_stat_error,
        density = True,
        color="black",
        linewidth=3,
        histtype='errorbar',
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[0],
        flow=flow_bins
    )

    ax[0].set_xlabel('')
    ax[0].margins(y=0.15)
    ax[0].set_ylim(0, 1.1*ax[0].get_ylim()[1])
    ax[0].tick_params(labelsize=22)

    # Ploting the systematic bands around the nominal sigma_m curve
    data_hist_numpy = data_hist.to_numpy()
    mc_hist_numpy = mc_hist.to_numpy()

    # Calculating the integrals to normalize the systematic yields, since it is a density that is being plotted!
    integral_mc_down = mc_hist_down.sum().value * (mc_hist_down.to_numpy()[1][1] - mc_hist_down.to_numpy()[1][0])
    integral_mc_up   = mc_hist_up.sum().value * (mc_hist_up.to_numpy()[1][1] - mc_hist_up.to_numpy()[1][0])

    # Lower and upper bounds of the systematic band - normalized to one
    lower_bound = mc_hist_down.to_numpy()[0]/integral_mc_down 
    upper_bound = mc_hist_up.to_numpy()[0]/integral_mc_up

    # Using the matplotlib .fill_between() to plot the bands around the nominal variation
    ax[0].fill_between(data_hist_numpy[1][:-1],
        lower_bound,
        upper_bound,
        step='post',
        hatch='XXX',
        alpha=0.9,
        facecolor="none",
        edgecolor="tab:purple", 
        linewidth=0)


    # line at 1 in the ratio plot
    ax[1].axhline(1, 0, 1, label=None, linestyle='--', color="black", linewidth=1)

    # Integrals of data and nominal mc for ratio calculation
    integral_data = data_hist.sum().value * (data_hist_numpy[1][1] - data_hist_numpy[1][0])
    integral_mc = mc_hist.sum().value * (mc_hist_numpy[1][1] - mc_hist_numpy[1][0])
    
    # to numpy histos
    mc_hist_numpy      = mc_hist.to_numpy()
    mc_hist_up_numpy   = mc_hist_up.to_numpy()
    mc_hist_down_numpy = mc_hist_down.to_numpy()

    integral_mc_up = mc_hist_up.sum().value * (mc_hist_up_numpy[1][1] - mc_hist_up_numpy[1][0])
    error_up       = np.abs(mc_hist_up_numpy[0]/integral_mc_up - mc_hist_numpy[0]/integral_mc)/(mc_hist_numpy[0]/integral_mc)

    integral_mc_down = mc_hist_down.sum().value * (mc_hist_down_numpy[1][1] - mc_hist_down_numpy[1][0])
    error_down       = np.abs(mc_hist_down_numpy[0]/integral_mc_down - mc_hist_numpy[0]/integral_mc)/(mc_hist_numpy[0]/integral_mc)

    # Plotting the bands around 1
    lower_bound = 1 - error_down
    upper_bound = 1 + error_up

    # Plot the hatched region
    ax[1].fill_between(data_hist_numpy[1][:-1],
        lower_bound,
        upper_bound,
        step='post',
        hatch='XXX',
        alpha=0.9,
        facecolor="none",
        edgecolor="tab:purple", 
        linewidth=0
    )

    # Calculating the ratio betwenn nominal mc and data and the stat error of mc + data
    values1, variances1 = data_hist.values(), data_hist.variances()
    values2, variances2 = mc_hist.values()  , mc_hist.variances()

    # Calculate the ratio 
    ratio = (values1/np.sum(values1)) / (values2/np.sum(values2))

    # Error from both data stat and MC stat
    error_ratio = np.sqrt((np.sqrt(variances2) / values2)**2 + (np.sqrt(variances1) / values1))**2

    error_ratio[error_ratio <= 0] = 0

    hep.histplot(
        ratio[:-1],
        bins=data_hist_numpy[1][:-1],
        label=None,
        color="black",
        histtype='errorbar',
        yerr= error_ratio[:-1],
        markersize=12,
        elinewidth=3,
        alpha=1,
        ax=ax[1]
    )

    # Setting the labels
    ax[0].set_ylabel("Normalised density", fontsize=26)
    ax[1].set_ylabel("Data / MC", fontsize=26)
    ax[1].set_xlabel(xlabel, fontsize=26, usetex=True)
    
    # Some axes configurations
    ax[0].tick_params(labelsize=24)
    ax[0].set_yscale(yscale)
    ax[1].set_ylim(0., 1.1*ax[0].get_ylim()[1])
    ax[1].set_ylim(0.6, 1.4)

    ax[0].legend(loc="upper right", fontsize=20)

    # Text indicating the region
    #if(IsBarrel):
    #    ax[0].text(0.05, 0.95, r'EB-EB', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')
    #else:
    #    ax[0].text(0.05, 0.95, r'Not EB-EB', transform=ax[0].transAxes, fontsize=20, verticalalignment='top')

    # Setting the lumi labels
    if(IspostEE):
        hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=27.0)
    else:
        hep.cms.label(data=True, ax=ax[0], loc=0, label="Private Work", com=13.6, lumi=8.1)

    plt.subplots_adjust(hspace=0.03)
    plt.tight_layout()
    fig.savefig(output_filename, bbox_inches='tight')


def create_mask(arr):

    data_mass = np.array(arr["mass"])

    #mask data interval 
    mask_mass = np.logical_and(data_mass > 80, data_mass < 100)
    mask_mass = np.logical_and(mask_mass, np.abs(np.array(arr["tag_eta"])) < 2.5 )
    mask_mass = np.logical_and(mask_mass, np.abs(np.array(arr["tag_pt"]))  > 42 )
    
    mask_mass = np.logical_and(mask_mass, np.abs(np.array(arr["probe_pt"]))  > 22 )
    mask_mass = np.logical_and(mask_mass, np.abs(np.array(arr["probe_eta"])) < 2.5 ) 

    #lets make the electron veto cut!
    mask_mass = np.logical_and(mask_mass, np.abs(np.array(arr["tag_electronVeto"]) ) == False )
    mask_mass = np.logical_and(mask_mass, np.abs(np.array(arr["probe_electronVeto"]) ) == False )

    # also, the cutBased ID!
    mask_mass = np.logical_and(mask_mass, np.array(arr["tag_cutBased"]) > 0 )
    mask_mass = np.logical_and(mask_mass, np.array(arr["probe_cutBased"]) > 0 )

    return mask_mass


def get_sigmaE_over_E(ak_arr, energyErr_field):
    return np.array(ak_arr[energyErr_field]/(ak_arr.probe_pt * np.cosh(ak_arr.probe_eta)))


def main():

    # Read the plot_settings json to check what we want to plot
    with open('./plot_settings.json', 'r') as f:
        plot_settings = json.load(f)
    
    # We need to read the nominal up and down systs
    energy_Err_systs = ["nominal","energyErrShift_up","energyErrShift_down"]
    store_sigma_over_m_syst = [] 
    store_sigma_over_m_corr_smeared_syst = [] 
    store_sigma_over_m_corr_syst = [] 
    store_sigma_over_m_smeared_syst = [] 
    _weights = [] # for MC
    store_mc_ebeb_masks = []
    store_mc_not_ebeb_masks = []

    for target in plot_settings.keys():
        print('INFO: Making the plot for', target, 'now...')
        target_settings = plot_settings[target] # Just a shortcut to avoid diving into deeply nested dictionaries
        ## Data
        ak_arr = ak.from_parquet(glob.glob(target_settings['files_data']))#, columns=target_settings['variable'])
        ak_arr = ak_arr[create_mask(ak_arr)] # Apply event selection
        if target == 'probe_energyErr_DIVbyE':
            data_arr = get_sigmaE_over_E(ak_arr, 'probe_energyErr')
        elif target == 'probe_corr_energyErr_DIVbyE':
            data_arr = get_sigmaE_over_E(ak_arr, 'probe_energyErr')
        # In Data, we do not have corrected variables
        elif target_settings['variable'] == 'sigma_m_over_m_corr':
            data_arr = ak_arr['sigma_m_over_m']
        elif target_settings['variable'] == 'sigma_m_over_m_Smeared_corrected':
            data_arr = ak_arr['sigma_m_over_m_Smeared']
        elif target_settings['variable'] == 'probe_corr_energyErr':
            data_arr = ak_arr['probe_energyErr']
        else:
            data_arr = ak_arr[target_settings['variable']]
        data_weights = np.ones(len(data_arr))

        MC_arr_list = [] # Needs to be re-initialised in each pass through the loop

        ## MC
        for variation in energy_Err_systs:
            print('Reading variation:', variation) 
            ak_arr = ak.from_parquet(glob.glob(target_settings['files_MC'] + str(variation) + "/*.parquet"))
            ak_arr = ak_arr[create_mask(ak_arr)]
            if target == 'probe_energyErr_DIVbyE':
                MC_arr_list.append(get_sigmaE_over_E(ak_arr, 'probe_energyErr'))
            elif target == 'probe_corr_energyErr_DIVbyE':
                MC_arr_list.append(get_sigmaE_over_E(ak_arr, 'probe_corr_energyErr'))
            else:
                MC_arr_list.append(np.array(ak_arr[target_settings['variable']])) # Maybe a dict would be better suited...
            _weights.append(np.array(ak_arr["weight"]))

        binning = target_settings['binning']
        
        data_hist    = hist.new.Reg(binning[0], binning[1], binning[2]).Weight()
        mc_hist      = hist.new.Reg(binning[0], binning[1], binning[2]).Weight() 
        mc_hist_up   = hist.new.Reg(binning[0], binning[1], binning[2]).Weight() 
        mc_hist_down = hist.new.Reg(binning[0], binning[1], binning[2]).Weight()

        if target_settings['density'] == True:
            data_hist.fill(data_arr, weight = data_weights/np.sum(data_weights))
            mc_hist.fill(MC_arr_list[0], weight = _weights[0]/np.sum(_weights[0]))
            mc_hist_up.fill(MC_arr_list[1], weight = _weights[1]/np.sum(_weights[1]))
            mc_hist_down.fill(MC_arr_list[2], weight = _weights[2]/np.sum(data_weights[2]))
        else:
            data_hist.fill(data_arr, weight = data_weights/np.sum(data_weights))
            mc_hist.fill(MC_arr_list[0], weight = _weights[0])
            mc_hist_up.fill(MC_arr_list[1], weight = _weights[1])
            mc_hist_down.fill(MC_arr_list[2], weight = _weights[2])
        
        plotter(mc_hist, mc_hist_up, mc_hist_down, data_hist, 'plots_smear/test/'+target+'.png', flow_bins = target_settings['flow_bins'], xlabel = target_settings['xlabel'], yscale=target_settings['yscale'])


if __name__ == "__main__":
    main()