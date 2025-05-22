import os
import sys
import awkward as ak
import pandas
import numpy as np
import bottom_line.tools.decorrelator as decorr


def decorrelate_mass_resolution(events: ak.Array, type: str, year, IsSAS_ET_Dependent=False):
    # type = "nominal","smeared","corr","corr_smeared"

    # reading the CDFs files
    var, dVar, ref = "sigma_m_over_m", "mass", 125.0  # Varable to be decorrelated, decorrelated w.r.t and the reference bin
    dummyDf = pandas.DataFrame({'{}'.format(var): [0], '{}'.format(dVar): [0]})
    decl = decorr.decorrelator(dummyDf, var, dVar, np.linspace(100., 180., 161))

    # setting up the decorrelator
    df = pandas.DataFrame()

    if type == "nominal":
        if year == "2022postEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022postEE/nominal_sigma_m_postEE_CDFs.pkl.gz')
        elif year == "2022preEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022preEE/nominal_sigma_m_preEE_CDFs.pkl.gz')
        elif year == "2023postBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023postBPix/nominal_sigma_m_postBPix_CDFs.pkl.gz')
        elif year == "2023preBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023preBPix/nominal_sigma_m_preBPix_CDFs.pkl.gz')
        else:
            print("Specify a valid era: 2022postEE, 2022preEE, 2023postBPix, 2023preBPix")
            sys.exit(1)
        df["sigma_m_over_m"] = events.sigma_m_over_m.to_numpy()

    elif type == "smeared":
        if year == "2022postEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022postEE/sigma_m_smeared_postEE_CDFs.pkl.gz')
        elif year == "2022preEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022preEE/sigma_m_smeared_preEE_CDFs.pkl.gz')
        elif year == "2023postBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023postBPix/sigma_m_smeared_postBPix_CDFs.pkl.gz')
        elif year == "2023preBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023preBPix/sigma_m_smeared_preBPix_CDFs.pkl.gz')
        else:
            print("Specify a valid era: 2022postEE, 2022preEE, 2023postBPix, 2023preBPix")
            sys.exit(1)
        df["sigma_m_over_m"] = events.sigma_m_over_m_Smeared.to_numpy()

    elif type == "corr":
        if year == "2022postEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022postEE/sigma_m_corr_postEE_CDFs.pkl.gz')
        elif year == "2022preEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022preEE/sigma_m_corr_preEE_CDFs.pkl.gz')
        elif year == "2023postBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023postBPix/sigma_m_corr_postBPix_CDFs.pkl.gz')
        elif year == "2023preBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023preBPix/sigma_m_corr_preBPix_CDFs.pkl.gz')
        else:
            print("Specify a valid era: 2022postEE, 2022preEE, 2023postBPix, 2023preBPix")
            sys.exit(1)
        df["sigma_m_over_m"] = events.sigma_m_over_m_corr.to_numpy()

    elif type == "corr_smeared":
        if not IsSAS_ET_Dependent:
            print(
                "Currently, we do not support the decorrelation for the corr_smeared for 2022 and 2023 "
                "if you use the non-SAS ET dependent corrections. Exiting..."
            )

            sys.exit(1)
        if year == "2022postEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022postEE/sigma_m_smeared_corr_postEE_CDFs.pkl.gz')
        elif year == "2022preEE":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2022preEE/sigma_m_smeared_corr_preEE_CDFs.pkl.gz')
        elif year == "2023postBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023postBPix/sigma_m_smeared_corr_postBPix_CDFs.pkl.gz')
        elif year == "2023preBPix":
            decl.loadCdfs(os.path.dirname(__file__) + '/decorrelation_CDFs/2023preBPix/sigma_m_smeared_corr_preBPix_CDFs.pkl.gz')
        else:
            print("Specify a valid era: 2022postEE, 2022preEE, 2023postBPix, 2023preBPix")
            sys.exit(1)
        df["sigma_m_over_m"] = events.sigma_m_over_m_Smeared_corr.to_numpy()

    else:
        print("Specify a valid type: nominal,smeared,corr,corr_smeared")
        sys.exit(1)

    # Reading directly the smeared sigma_m_over_m
    df["mass"] = events.mass.to_numpy()
    df["weight"] = events.weight.to_numpy()

    decl.df = df.loc[:, [var, dVar]]
    decl.df.reset_index(inplace=True)

    # options.ref is the mass bin (125.)
    df['{}_decorr'.format(var)] = decl.doDecorr(ref)

    # performing the decorrelation
    events["sigma_m_over_m_decorr"] = decl.doDecorr(ref)

    # returning the array with the decorrelated mass resolution
    return events["sigma_m_over_m_decorr"]
