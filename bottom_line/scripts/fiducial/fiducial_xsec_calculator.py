### Examples of command line
# python fiducial_xsec_calculator.py PATH_TO_FOLDER_WITH_ParticleLevelProcessor_OUTPUT --process all --bin '|0|0.15|0.3|0.6|0.9|2.5|' --era all --obs YH
## Without NNLOPS reweighing
# python fiducial_xsec_calculator.py /work/atarabin/HiggsDNA/prod/particleLevel/ --process ggH --bin '|0|1|2|3|100|' --era all --obs NJ --weight "genWeight"
## POWHEG
# python fiducial_xsec_calculator.py /work/atarabin/HiggsDNA/prod/particleLevel/ --process ggH --bin '|0|15|30|45|80|120|200|350|1000|' --era all --powheg
## For inclusive XS
# python fiducial_xsec_calculator.py /work/atarabin/HiggsDNA/prod/particleLevel/ --process all --bin '|0|500|' --era all
import argparse
import awkward as ak
import numpy as np
from scipy import interpolate


def compute_fid_xsec(in_frac, mass_points, xs_value, BR, no_interpolation, target_mass=125.38):
    """
    Compute the fiducial cross section for a given observable.

    Parameters:
      in_frac (dict): Dictionary of in-fiducial fractions keyed by mass point.
      mass_points (list): List of mass points (as strings) provided via the command line.
      xs_value (float): Cross section value from the XS map for the process.
      BR (float): Branching ratio.
      no_interpolation (bool): Flag to disable interpolation when only mass point 125 is provided.
      target_mass (float): The mass at which to evaluate the spline (default 125.38).
    
    Returns:
      float: The computed fiducial cross section.
    """
    if no_interpolation:
        value = in_frac["125"]
    else:
        # Convert mass_points to floats and compute spline interpolation.
        masses = [float(p) for p in mass_points]
        points = [in_frac[p] for p in mass_points]
        spline = interpolate.splrep(masses, points, k=2)
        value = float(interpolate.splev(target_mass, spline))
    return value * xs_value * 1000 * BR



available_processes = ['ggH', 'VBFH', 'VH', 'ttH', 'all', 'xH']
available_mass_points = ['120', '125', '130']
available_fid_selections = ['fiducialGeometricFlag', 'fiducialClassicalFlag']
available_years = ['2022']
available_eras = ['preEE', 'postEE', 'all']

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description = "Calculate the inclusive fiducial cross section of pp->H(yy)+X process(es) based on processed samples without detector-level selections. ")
parser.add_argument('path', type = str, help = "Path to the top-level folder containing the different directories. Please only run this on the output of the ParticleLevelProcessor.")
parser.add_argument('--process', type = str, choices = available_processes, default = 'ggH', help = "Please specify the process(es) for which you want to calculate the inclusive fiducial xsec.")
parser.add_argument('--mass-points', nargs='+', choices = available_mass_points, default = available_mass_points, help = "Please specify the mass points to run over. If only one single mass point of 125 is specified: No interpolation is performed.")
parser.add_argument('--fid-selection', type = str, choices = available_fid_selections, default = 'fiducialGeometricFlag', help = "Please specify the fiducial selection flag to use.")
parser.add_argument('--year', type = str, choices = available_years, default = '2022', help = 'Please specify the desired year if you want to combine samples from multiple eras.')
parser.add_argument('--era', type = str, choices = available_eras, default = 'postEE', help = "Please specify the era(s) that you want to run over. If you specify 'all', an inverse variance weighting is performed to increase the precision.")
parser.add_argument('--bin', type = str, default = '|0|5000|', help = "Bin boundaries of the differential XS. The default")
parser.add_argument('--obs', type = str, default = 'PTH', help = "Name of the differential observable: PTH, YH, NJ")
parser.add_argument('--weight', type = str, default = 'weight', help = "Weight to use.")
parser.add_argument('--powheg', action="store_true", help="To process powheg sample.")

args = parser.parse_args()

args = parser.parse_args()

# New check for mass points
if len(args.mass_points) == 1:
    if args.mass_points[0] != '125':
        parser.error("When specifying a single mass point, only '125' is allowed.")
    no_interpolation = True
else:
    no_interpolation = False

if args.fid_selection == 'fiducialGeometricFlag':
    print('INFO: Using the geometric fiducial flag for the selection.')
elif args.fid_selection == 'fiducialClassicalFlag':
    print('INFO: Using the classical fiducial flag for the selection.')

path_folder = args.path # Use the specified folder path
# Pepare the processes array appropriately
if args.process == 'all':
    processes = available_processes
    processes.remove('all')
    processes.remove('xH')
elif args.process == 'xH':
    processes = ['VBFH', 'VH', 'ttH']
else:
    processes = [args.process]
year = args.year
# Pepare the eras array appropriately
if args.era == 'all':
    eras = available_eras
    eras.remove('all')
else:
    eras = [args.era]

# Convert bin boundaries in a list
obs_bins = [float(num) for num in args.bin.strip("|").split("|")]

# See also the following pages (note numbers always in picobarn)
# 13: for 125
# 13p6: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap, for 125.38
# 14: for 125
XS_map = {'13':   {'ggH': 48.58, 'VBFH': 3.782, 'VH': 2.2569, 'ttH': 0.5071},
         '13p6': {'ggH': 51.96, 'VBFH': 4.067, 'VH': 2.3781, 'ttH': 0.5638},
         '14':   {'ggH': 54.67, 'VBFH': 4.278, 'VH': 2.4991, 'ttH': 0.6137},}

XS_map_scale_up = {'13':   {},
                   '13p6': {'ggH': 53.98644, 'VBFH': 4.087335, 'VH': 2.3376723, 'ttH': 0.597628},
                   '14':   {},}

XS_map_scale_dn = {'13':   {},
                   '13p6': {'ggH': 49.93356, 'VBFH': 4.054799, 'VH': 2.4185277, 'ttH': 0.5113666},
                   '14':   {},}

XS_map_pdf_up = {'13':   {},
                   '13p6': {'ggH': 52.94724, 'VBFH': 4.152407, 'VH': 2.4137715, 'ttH': 0.580714},
                   '14':   {},}

XS_map_pdf_dn = {'13':   {},
                   '13p6': {'ggH': 50.97276, 'VBFH': 3.981593, 'VH': 2.3424285, 'ttH': 0.546886},
                   '14':   {},}

XS_map_alphaS_up = {'13':   {},
                   '13p6': {'ggH': 53.31096, 'VBFH': 4.087335, 'VH': 2.3995029, 'ttH': 0.575076},
                   '14':   {},}

XS_map_alphaS_dn = {'13':   {},
                   '13p6': {'ggH': 50.60904, 'VBFH': 4.046665, 'VH': 2.3566971, 'ttH': 0.552524},
                   '14':   {},}


# This depends on how you named your samples in HiggsDNA
processMap = {'ggH':  'GluGluHtoGG',
              'VBFH': 'VBFHtoGG',
              'VH':   'VHtoGG',
              'ttH':   'ttHtoGG',}

BR = 0.2270/100 # SM value for mH close to 125: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
mass_points = args.mass_points # The fiducial acceptance should be extrapolated at 125.38 using a spline between 120, 125, and 130 if we have the mass points
# For powheg only the 125 GeV sample is available
# [FIXME] For the time being, no extrapolation, its effects is in any case small.
# [FIXME] Idea: compute the relative variation between 125 and 125.38 GeV with Madgraph and then apply it to powheg
mass_powheg = {120:125, 125:125, 130:125}

fid_xsecs_per_bin = {}
fid_xsecs_per_bin_scale_up = {}
fid_xsecs_per_bin_scale_dn = {}
fid_xsecs_per_bin_pdf_up = {}
fid_xsecs_per_bin_pdf_dn = {}
fid_xsecs_per_bin_alpha_up = {}
fid_xsecs_per_bin_alpha_dn = {}
for b in range(len(obs_bins)-1):
    fid_xsecs_per_bin_process = {}
    fid_xsecs_per_bin_process_scale_up = {}
    fid_xsecs_per_bin_process_scale_dn = {}
    fid_xsecs_per_bin_process_pdf_up = {}
    fid_xsecs_per_bin_process_pdf_dn = {}
    fid_xsecs_per_bin_process_alpha_up = {}
    fid_xsecs_per_bin_process_alpha_dn = {}
    for process in processes:
        print(f'INFO: Now extracting fraction of in-fiducial events for process {process} ...')
        in_frac_per_mass = {}
        in_frac_per_mass_scale_up = {}
        in_frac_per_mass_scale_dn = {}
        in_frac_per_mass_pdf_up = {}
        in_frac_per_mass_pdf_dn = {}
        in_frac_per_mass_alpha_up = {}
        in_frac_per_mass_alpha_dn = {}
        for mass in mass_points:
            print(f'INFO: Now extracting numbers for mass {mass}...')
            in_frac_per_mass_era = {}
            in_frac_per_mass_era_scale_up = {}
            in_frac_per_mass_era_scale_dn = {}
            in_frac_per_mass_era_pdf_up = {}
            in_frac_per_mass_era_pdf_dn = {}
            in_frac_per_mass_era_alpha_up = {}
            in_frac_per_mass_era_alpha_dn = {}
            sumw2_tmp = []
            for era in eras:
                print(f'INFO: Now extracting numbers for era {era}, process {process}, and bin {b} ...')
                # Extract the events
                process_string = path_folder + processMap[process] + '_M-' + str(mass) + '_' + era
                if args.powheg: process_string = path_folder + processMap[process] + '_M-' + str(mass_powheg[mass]) + '_powheg'
                arr = ak.from_parquet(process_string)
                # Calculating the relevant fractions
                inFiducialFlag = (arr[args.fid_selection] == True) & (abs(arr[args.obs]) >= obs_bins[b]) & (abs(arr[args.obs]) < obs_bins[b+1]) # Only for this type of tagger right now, can be customised in the future

                sumwAll = ak.sum(arr[args.weight])
                sumwIn = ak.sum(arr[args.weight][(inFiducialFlag)])
                in_frac = sumwIn/sumwAll

                in_frac_scale = []
                for i in [0,1,3,5,7,8]:
                    sumwAll = ak.sum(arr[args.weight] * arr["LHEScaleWeight_"+str(i)])
                    sumwIn = ak.sum(arr[args.weight][(inFiducialFlag)] * arr["LHEScaleWeight_"+str(i)][(inFiducialFlag)])
                    in_frac_scale.append(sumwIn/sumwAll)
                in_frac_scale_up = max(in_frac_scale)
                in_frac_scale_dn = min(in_frac_scale)


                in_frac_pdf = []
                for i in range(1,101):
                    sumwAll = ak.sum(arr[args.weight] * arr["LHEPdfWeight_"+str(i)])
                    sumwIn = ak.sum(arr[args.weight][(inFiducialFlag)] * arr["LHEPdfWeight_"+str(i)][(inFiducialFlag)])
                    in_frac_pdf.append(sumwIn/sumwAll)
                in_frac_pdf = np.array(in_frac_pdf)
                ## Formula taken from https://arxiv.org/pdf/1510.03865 (Eq. 20)
                in_frac_pdf = np.sqrt(np.sum(np.square(in_frac_pdf - in_frac)))
                in_frac_pdf_up = in_frac + in_frac_pdf
                in_frac_pdf_dn = in_frac - in_frac_pdf

                sumwAll = ak.sum(arr[args.weight] * arr["LHEPdfWeight_101"])
                sumwIn = ak.sum(arr[args.weight][(inFiducialFlag)] * arr["LHEPdfWeight_101"][(inFiducialFlag)])
                in_frac_alpha_up = sumwIn/sumwAll

                sumwAll = ak.sum(arr[args.weight] * arr["LHEPdfWeight_102"])
                sumwIn = ak.sum(arr[args.weight][(inFiducialFlag)] * arr["LHEPdfWeight_102"][(inFiducialFlag)])
                in_frac_alpha_dn = sumwIn/sumwAll

                print(f"INFO: Fraction of in-fiducial events: {in_frac} ...")
                print(f"INFO: Fraction of in-fiducial events scale up: {in_frac_scale_up} ...")
                print(f"INFO: Fraction of in-fiducial events scale dn: {in_frac_scale_dn} ...")
                print(f"INFO: Fraction of in-fiducial events pdf up: {in_frac_pdf_up} ...")
                print(f"INFO: Fraction of in-fiducial events pdf dn: {in_frac_pdf_dn} ...")
                print(f"INFO: Fraction of in-fiducial events alpha up: {in_frac_alpha_up} ...")
                print(f"INFO: Fraction of in-fiducial events alpha dn: {in_frac_alpha_dn} ...")

                sumw2 = ak.sum(arr[args.weight][(inFiducialFlag)]**2) # This is the MC stat variance
                sumw2_tmp.append(sumw2)

                in_frac_per_mass_era[era] = in_frac * 1/sumw2

                in_frac_per_mass_era_scale_up[era] = in_frac_scale_up * 1/sumw2
                in_frac_per_mass_era_scale_dn[era] = in_frac_scale_dn * 1/sumw2
                in_frac_per_mass_era_pdf_up[era] = in_frac_pdf_up * 1/sumw2
                in_frac_per_mass_era_pdf_dn[era] = in_frac_pdf_dn * 1/sumw2
                in_frac_per_mass_era_alpha_up[era] = in_frac_alpha_up * 1/sumw2
                in_frac_per_mass_era_alpha_dn[era] = in_frac_alpha_dn * 1/sumw2

            sumw2_tmp = np.asarray(sumw2_tmp)
            result = np.sum(np.asarray([in_frac_per_mass_era[era] for era in eras]))
            in_frac_per_mass[mass] = result / np.sum(1/sumw2_tmp)

            result = np.sum(np.asarray([in_frac_per_mass_era_scale_up[era] for era in eras]))
            in_frac_per_mass_scale_up[mass] = result / np.sum(1/sumw2_tmp)

            result = np.sum(np.asarray([in_frac_per_mass_era_scale_dn[era] for era in eras]))
            in_frac_per_mass_scale_dn[mass] = result / np.sum(1/sumw2_tmp)

            result = np.sum(np.asarray([in_frac_per_mass_era_pdf_up[era] for era in eras]))
            in_frac_per_mass_pdf_up[mass] = result / np.sum(1/sumw2_tmp)

            result = np.sum(np.asarray([in_frac_per_mass_era_pdf_dn[era] for era in eras]))
            in_frac_per_mass_pdf_dn[mass] = result / np.sum(1/sumw2_tmp)

            result = np.sum(np.asarray([in_frac_per_mass_era_alpha_up[era] for era in eras]))
            in_frac_per_mass_alpha_up[mass] = result / np.sum(1/sumw2_tmp)

            result = np.sum(np.asarray([in_frac_per_mass_era_alpha_dn[era] for era in eras]))
            in_frac_per_mass_alpha_dn[mass] = result / np.sum(1/sumw2_tmp)
    
        fid_xsecs_per_bin_process[process] = compute_fid_xsec(
            in_frac_per_mass, args.mass_points, XS_map['13p6'][process], BR, no_interpolation
        )
        fid_xsecs_per_bin_process_scale_up[process] = compute_fid_xsec(
            in_frac_per_mass_scale_up, args.mass_points, XS_map_scale_up['13p6'][process], BR, no_interpolation
        )
        fid_xsecs_per_bin_process_scale_dn[process] = compute_fid_xsec(
            in_frac_per_mass_scale_dn, args.mass_points, XS_map_scale_dn['13p6'][process], BR, no_interpolation
        )
        fid_xsecs_per_bin_process_pdf_up[process] = compute_fid_xsec(
            in_frac_per_mass_pdf_up, args.mass_points, XS_map_pdf_up['13p6'][process], BR, no_interpolation
        )
        fid_xsecs_per_bin_process_pdf_dn[process] = compute_fid_xsec(
            in_frac_per_mass_pdf_dn, args.mass_points, XS_map_pdf_dn['13p6'][process], BR, no_interpolation
        )
        fid_xsecs_per_bin_process_alpha_up[process] = compute_fid_xsec(
            in_frac_per_mass_alpha_up, args.mass_points, XS_map_alphaS_up['13p6'][process], BR, no_interpolation
        )
        fid_xsecs_per_bin_process_alpha_dn[process] = compute_fid_xsec(
            in_frac_per_mass_alpha_dn, args.mass_points, XS_map_alphaS_dn['13p6'][process], BR, no_interpolation
        )


        points = [in_frac_per_mass[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process[process] = float(interpolate.splev(125.38, spline)) * XS_map['13p6'][process] * 1000 * BR

        points = [in_frac_per_mass_scale_up[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process_scale_up[process] = float(interpolate.splev(125.38, spline)) * XS_map_scale_up['13p6'][process] * 1000 * BR

        points = [in_frac_per_mass_scale_dn[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process_scale_dn[process] = float(interpolate.splev(125.38, spline)) * XS_map_scale_dn['13p6'][process] * 1000 * BR

        points = [in_frac_per_mass_pdf_up[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process_pdf_up[process] = float(interpolate.splev(125.38, spline)) * XS_map_pdf_up['13p6'][process] * 1000 * BR

        points = [in_frac_per_mass_pdf_dn[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process_pdf_dn[process] = float(interpolate.splev(125.38, spline)) * XS_map_pdf_dn['13p6'][process] * 1000 * BR

        points = [in_frac_per_mass_alpha_up[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process_alpha_up[process] = float(interpolate.splev(125.38, spline)) * XS_map_alphaS_up['13p6'][process] * 1000 * BR

        points = [in_frac_per_mass_alpha_dn[p] for p in mass_points]
        spline = interpolate.splrep(mass_points, points, k=2)
        fid_xsecs_per_bin_process_alpha_dn[process] = float(interpolate.splev(125.38, spline)) * XS_map_alphaS_dn['13p6'][process] * 1000 * BR

    fid_xsecs_per_bin[b] = np.sum(np.asarray([fid_xsecs_per_bin_process[process] for process in processes]))
    print(f"The fiducial cross section for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin[b]} fb")

    fid_xsecs_per_bin_scale_up[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_scale_up[process] for process in processes]))
    print(f"The fiducial cross section (scale_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_scale_up[b]} fb")

    fid_xsecs_per_bin_scale_dn[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_scale_dn[process] for process in processes]))
    print(f"The fiducial cross section (scale_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_scale_dn[b]} fb")

    fid_xsecs_per_bin_pdf_up[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_pdf_up[process] for process in processes]))
    print(f"The fiducial cross section (pdf_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_pdf_up[b]} fb")

    fid_xsecs_per_bin_pdf_dn[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_pdf_dn[process] for process in processes]))
    print(f"The fiducial cross section (pdf_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_pdf_dn[b]} fb")

    fid_xsecs_per_bin_alpha_up[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_alpha_up[process] for process in processes]))
    print(f"The fiducial cross section (alpha_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_alpha_up[b]} fb")

    fid_xsecs_per_bin_alpha_dn[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_alpha_dn[process] for process in processes]))
    print(f"The fiducial cross section (alpha_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_alpha_dn[b]} fb")


final_fid_xsec = np.sum(np.asarray([fid_xsecs_per_bin[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section is given by: {final_fid_xsec} fb")

final_fid_xsec_scale_up = np.sum(np.asarray([fid_xsecs_per_bin_scale_up[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (scale up) is given by: {final_fid_xsec_scale_up} fb")

final_fid_xsec_scale_dn = np.sum(np.asarray([fid_xsecs_per_bin_scale_dn[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (scale dn) is given by: {final_fid_xsec_scale_dn} fb")

final_fid_xsec_pdf_up = np.sum(np.asarray([fid_xsecs_per_bin_pdf_up[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (pdf_up) is given by: {final_fid_xsec_pdf_up} fb")

final_fid_xsec_pdf_dn = np.sum(np.asarray([fid_xsecs_per_bin_pdf_dn[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (pdf_dn) is given by: {final_fid_xsec_pdf_dn} fb")

final_fid_xsec_alpha_up = np.sum(np.asarray([fid_xsecs_per_bin_alpha_up[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (alpha_up) is given by: {final_fid_xsec_alpha_up} fb")

final_fid_xsec_alpha_dn = np.sum(np.asarray([fid_xsecs_per_bin_alpha_dn[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (alpha_dn) is given by: {final_fid_xsec_alpha_dn} fb")

output = 'fidXS_'+args.obs+'_'+args.process
if args.powheg: output += '_powheg'
if args.weight != "weight": output += '_'+args.weight

with open(output+'.py', 'w') as f:
        f.write('Boundaries = '+str(obs_bins)+' \n')
        f.write('fidXS = '+str(list(fid_xsecs_per_bin.values()))+' \n')
        f.write('fidXS_scale_up = '+str(list(fid_xsecs_per_bin_scale_up.values()))+' \n')
        f.write('fidXS_scale_dn = '+str(list(fid_xsecs_per_bin_scale_dn.values()))+' \n')
        f.write('fidXS_pdf_up = '+str(list(fid_xsecs_per_bin_pdf_up.values()))+' \n')
        f.write('fidXS_pdf_dn = '+str(list(fid_xsecs_per_bin_pdf_dn.values()))+' \n')
        f.write('fidXS_alpha_up = '+str(list(fid_xsecs_per_bin_alpha_up.values()))+' \n')
        f.write('fidXS_alpha_dn = '+str(list(fid_xsecs_per_bin_alpha_dn.values()))+' \n')
