# DarkBottomLine Framework

A modular Coffea-based analysis framework for CMS Run 3 bbMET analysis.

## Overview

DarkBottomLine is designed to process NanoAOD datasets using Coffea, producing flat output (ROOT or Parquet) containing analysis-level variables. The framework is generic, configurable for each Run 3 year (2022–2024) via metaconditions, and operates on NanoAOD datasets as input.

## Features

- **Modular Design**: Separate modules for objects, selections, corrections, weights, and histograms
- **Config-Driven**: Year-specific parameters in YAML configuration files
- **Coffea Integration**: Uses Coffea NanoEvents for efficient event processing
- **Correction Support**: Integration with correctionlib for scale factors
- **Multiple Executors**: Support for iterative, futures, and Dask execution backends
- **Flexible Output**: Support for ROOT, Parquet, and pickle output formats
- **Validation Tools**: Jupyter notebook for framework validation and plotting

## Installation

### Locally

#### Prerequisites

- Python 3.9+
- Conda or pip package manager

#### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd DarkBottomLine
   ```

2. Create a conda environment:

   ```bash
   conda create -n darkbottomline python=3.9
   conda activate darkbottomline
   ```

3. Install dependencies and the package:

   ```bash
   source local_setup.sh
   ```

### Lxplus

1. Source the following file:

   ```bash
   source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
   ```

   If the above method does not work, try installing and loading the CMSSW release:

   ```bash
   cmsrel CMSSW_15_0_17
   cd CMSSW_15_0_17/src
   cmsenv
   ```

2. Clone the repository:

   ```bash
   git clone https://github.com/tiwariPC/DarkBottomLine.git
   cd DarkBottomLine
   ```

3. Check the pre-installed packages that come with CMSSW release:

   ```bash
   python3 check_requirements.py

   # To install the missing packages
   python3 check_requirements.py --install --local-dir ./.local

   # Please add the suggested PYTHONPATH in the output of the above installation to your Python path
   ```

4. Run final installation script:

   ```bash
   chmod +x lxplus_setup.sh
   ./lxplus_setup.sh
   ```

#### Environment Setup (Lxplus)

After installation, you need to set up your environment before using DarkBottomLine. The `lxplus_setup.sh` script automatically sets up the environment for the current session, but for future logins you'll need to use the `start.sh` script.

##### First-time Installation (Automatic Setup)

When you run `lxplus_setup.sh`, it automatically:

- Sets up `PYTHONPATH` to include the installed packages
- Adds the `darkbottomline` command to your `PATH`
- Exports these paths for the current session

After running `lxplus_setup.sh`, you can immediately use DarkBottomLine in that session (after sourcing LCG environment).

##### Using DarkBottomLine in Future Sessions

Every time you start a new shell session on lxplus, you need to:

1. **Source LCG environment first** (critical):

   ```bash
   source-lcg
   # Or if you don't have the function:
   source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
   ```

2. Source the start.sh script to set up DarkBottomLine environment:

   ```bash
   cd /path/to/DarkBottomLine
   source start.sh
   # Or:
   . start.sh
   ```

3. Verify the setup:

   ```bash
   darkbottomline --help
   # Or:
   python3 -c "from darkbottomline import DarkBottomLineProcessor; print('✓ Import successful')"
   ```

#### Condor Setup

```bash
cd condorJobs
# Edit submit.sub file, change the user letter and username in line 3
# Change the <full_path> in runanalysis.sh and relevant command as needed
voms-proxy-init --voms cms --valid 192:00 && cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
condor_submit submit.sub
```

## Quick Start

### Full Analysis Workflow

The DarkBottomLine framework supports a complete analysis workflow from NanoAOD processing to plot generation. Here's how to run the entire analysis:

#### Step 1: Run Multi-Region Analysis

Run the analysis on your input files (data, MC backgrounds, signal). You can provide input files one by one, as multiple arguments, or listed in a `.txt` file.

```bash
# Activate virtual environment
source venv/bin/activate

# Run analysis on a single data file
darkbottomline analyze \
    --config configs/2024.yaml \
    --regions-config configs/regions.yaml \
    --input /path/to/data/nano_data.root \
    --output outputs/hists/regions_data.pkl \
    --max-events 10000

# Run analysis on MC backgrounds from a list of files in a .txt file
darkbottomline analyze \
    --config configs/2024.yaml \
    --regions-config configs/regions.yaml \
    --input my_background_files.txt \
    --output outputs/hists/regions_dy.pkl

# Run analysis on multiple signal files directly
darkbottomline analyze \
    --config configs/2024.yaml \
    --regions-config configs/regions.yaml \
    --input /path/to/signal/nano_signal_1.root /path/to/signal/nano_signal_2.root \
    --output outputs/hists/regions_signal.pkl
```

When using a `.txt` file for input, list one file path per line. Empty lines and lines starting with `#` will be ignored.

**Analysis Options:**

- `--config`: Base configuration file (e.g., `configs/2024.yaml`)
- `--regions-config`: Regions configuration file (e.g., `configs/regions.yaml`)
- `--input`: Input NanoAOD ROOT file(s). Can be a single file, multiple files, or a `.txt` file containing a list of file paths.
- `--output`: Output pickle file path
- `--executor`: Execution backend (iterative, futures, dask) - default: iterative
- `--workers`: Number of parallel workers (for futures/dask) - default: 4
- `--chunk-size`: Number of events per chunk for futures/dask executors (default: 50000 for futures, 200000 for dask). Useful for managing memory with large files.
- `--max-events`: Maximum number of events to process (optional, for testing). For futures/dask executors, this is converted to maxchunks based on chunk-size.

#### Step 2: Generate Plots

Generate data/MC plots from the analysis results:

```bash
# Generate plots from analysis results
darkbottomline make-plots \
    --input outputs/hists/regions_data.pkl \
    --save-dir outputs \
    --show-data

# With custom plotting configuration
darkbottomline make-plots \
    --input outputs/hists/regions_data.pkl \
    --save-dir outputs \
    --show-data \
    --plot-config configs/plotting.yaml \
    --version 20251105_1100
```

**Plotting Options:**

- `--input`: Input results pickle file
- `--save-dir`: Base output directory (default: `outputs`)
- `--show-data`: Include data points on plots
- `--plot-config`: Plotting configuration file (default: `configs/plotting.yaml`)
- `--version`: Version string for output directory (default: auto-generate timestamp)
- `--regions`: Specific regions to plot (optional, default: all regions)

#### Complete Workflow Example

```bash
# 1. Setup
source venv/bin/activate
cd /path/to/DarkBottomLine

# 2. Run analysis on all samples (using a .txt file for inputs)
# Create a file, e.g. dy_inputs.txt, with your list of ROOT files.
darkbottomline analyze \
    --config configs/2024.yaml \
    --regions-config configs/regions.yaml \
    --input dy_inputs.txt \
    --output outputs/hists/regions_dy.pkl

# 3. Generate plots
darkbottomline make-plots \
    --input outputs/hists/regions_data.pkl \
    --save-dir outputs \
    --show-data

# 4. Plots are saved in: outputs/plots/{version}/
#    - PNG: outputs/plots/{version}/png/{category}/{region}/
#    - PDF: outputs/plots/{version}/pdf/{category}/{region}/
#    - ROOT: outputs/plots/{version}/root/
#    - Text: outputs/plots/{version}/text/{category}/{region}/
#    - Summary: outputs/plots/{version}/region_summary.{png,pdf}
```

### Basic Usage (Simple Analysis)

For a simple single-region analysis without the multi-region framework:

```bash
darkbottomline run \
    --config configs/2024.yaml \
    --input /path/to/nanoaod_or_file_list.txt \
    --output results.pkl \
    --event-selection-output output/event_selected.pkl  # optional: save events for event-level selection
    --executor iterative
```

### Command Line Options

**Analysis Commands:**

- `analyze`: Multi-region analysis with full region definitions
- `run`: Simple single-region analysis
- `--config`: Path to YAML configuration file
- `--regions-config`: Path to regions configuration file (for `analyze` command)
- `--input`: Path to input NanoAOD file(s). Can be a single file, multiple files, or a `.txt` file containing a list of file paths.
- `--output`: Path to output file (supports .parquet, .root, .pkl)
- `--executor`: Execution backend (iterative, futures, dask)
- `--workers`: Number of parallel workers (for futures/dask)
- `--chunk-size`: Number of events per chunk for futures/dask executors (default: 50000 for futures, 200000 for dask). Helps manage memory with large files.
- `--max-events`: Maximum number of events to process. For futures/dask executors, converted to maxchunks based on chunk-size.
- `--event-selection-output`: Optional path to save events that pass event-level selection (supports `.pkl` and `.root`).
  - If you provide a `.pkl` path, a plain-Python-serializable pickle will be saved and a raw awkward backup `*.awk_raw.pkl` will also be created.
  - If you provide a `.root` path, a small ROOT TTree `Events` will be written containing scalar branches (event identifiers, MET scalars, and object multiplicities).

**Plotting Commands:**

- `make-plots`: Generate individual variable plots and grouped plots
- `make-stacked-plots`: Generate stacked Data/MC plots with ratio
- `--show-data`: Show data points on plots
- `--plot-config`: Plotting configuration file
- `--version`: Version string for output directory

### Example with Different Executors

The input flexibility works with all executors. For example:

```bash
# Iterative execution (single-threaded, good for debugging)
python run_analysis.py --config configs/2023.yaml --input my_files.txt --output results.pkl --executor iterative

# Futures execution (multi-threaded, good for local parallelization)
python run_analysis.py --config configs/2023.yaml --input file1.root file2.root --output results.parquet --executor futures --workers 4

# Futures execution with custom chunk size (for large files)
python run_analysis.py --config configs/2023.yaml --input large_file.root --output results.pkl --executor futures --workers 8 --chunk-size 100000

# Dask execution (distributed, good for production)
python run_analysis.py --config configs/2023.yaml --input nanoaod.root --output results.root --executor dask --workers 8

# Dask execution with custom chunk size (default is 200000 for dask)
python run_analysis.py --config configs/2023.yaml --input large_file.root --output results.root --executor dask --workers 8 --chunk-size 500000
```

**Chunk Size Notes:**

- Chunk size controls how many events are processed per chunk, helping manage memory usage
- Smaller chunks (e.g., 50000) use less memory but may have more overhead
- Larger chunks (e.g., 200000+) are more efficient but require more memory
- Default: 50000 for futures executor, 200000 for dask executor
- Only applies to `futures` and `dask` executors (uses `run_uproot_job` internally)
- The `iterative` executor loads all events at once and doesn't use chunking

## Configuration

The framework uses YAML configuration files for year-specific parameters. Configuration files are located in the `configs/` directory:

- `configs/2022.yaml`: 2022 data-taking parameters
- `configs/2023.yaml`: 2023 data-taking parameters
- `configs/2024.yaml`: 2024 data-taking parameters
- `configs/regions.yaml`: Region definitions with categories and channels
- `configs/plotting.yaml`: Plotting configuration and exclusions

### Region Definitions

Regions are defined in `configs/regions.yaml` with the format: `{category}:{region_type}_{channel}`

**Categories:**

- `1b`: 1 b-tag category (≤2 jets, 1 b-jet)
- `2b`: 2 b-tag category (3 jets, 2 b-jets)

**Region Types:**

- `SR`: Signal region
- `CR_Wlnu`: W+jets control region
- `CR_Top`: Top control region
- `CR_Zll`: Z+jets control region

**Channels:**

- `mu`: Muon channel
- `el`: Electron channel

**Example Regions:**

- `1b:SR` - Signal region, 1 b-tag
- `2b:SR` - Signal region, 2 b-tags
- `1b:CR_Wlnu_mu` - W+jets CR, 1b, muon channel
- `2b:CR_Top_el` - Top CR, 2b, electron channel
- `1b:CR_Zll_mu` - Z+jets CR, 1b, muon channel

### Configuration Structure

```yaml
year: 2023
lumi: 35.9  # fb^-1

# Correction file paths
corrections:
pileup: data/corrections/pileup_2023.json.gz
  btagSF: data/corrections/btagging_2023.json.gz
  muonSF: data/corrections/muonSF_2023.json.gz
  electronSF: data/corrections/electronSF_2023.json.gz

# Electron SF: Use EGM electron.json.gz (e.g. from jsonpog-integration). If loading fails with
# "binning edges are not monotone increasing", the code applies an automatic fix when loading.
# See: https://cms-analysis-corrections.docs.cern.ch/ (Run3 EGM corrections).

# Trigger paths
triggers:
  MET: ["HLT_PFMET120_PFMHT120_IDTight"]
  SingleMuon: ["HLT_IsoMu24", "HLT_IsoMu27"]

# Object selection cuts
objects:
  muons:
    pt_min: 20.0
    eta_max: 2.4
    id: "tight"
    iso: "tight"
  # ... more object configurations

# Event selection
event_selection:
  min_muons: 0
  max_muons: 2
  min_jets: 2
  min_bjets: 1
  met_min: 50.0
```

## Region Structure

The analysis uses a category-based region structure with channel separation:

### Categories

1. **1b Category**: 1 b-tag, ≤2 jets
   - SR: `1b:SR`
   - W CR: `1b:CR_Wlnu_mu`, `1b:CR_Wlnu_el`
   - Z CR: `1b:CR_Zll_mu`, `1b:CR_Zll_el`
   - No Top CR (removed as per requirements)

2. **2b Category**: 2 b-tags, 3 jets (Top CR may have >3 jets)
   - SR: `2b:SR`
   - Top CR: `2b:CR_Top_mu`, `2b:CR_Top_el`
   - Z CR: `2b:CR_Zll_mu`, `2b:CR_Zll_el`
   - No W CR (removed as per requirements)

### Control Region Definitions

**Z CR Separation:**

- **Z_1b**: `(njet <= 2) and (jet1Pt > 100.)`
- **Z_2b**: `(njet <= 3 and njet > 1) and (jet1Pt > 100.)`

**Channel Separation:**

- All CRs (Top, W, Z) have separate muon and electron channels
- Taus are vetoed for the full analysis

## Framework Components

### 1. Object Selection (`darkbottomline/objects.py`)

Physics object selection and cleaning functions:

- `select_muons()`: Muon selection with ID and isolation cuts
- `select_electrons()`: Electron selection with ID and isolation cuts
- `select_taus()`: Tau selection with ID and decay mode cuts
- `select_jets()`: AK4 jet selection with jet ID cuts
- `select_fatjets()`: AK8 fat jet selection
- `clean_jets_from_leptons()`: Delta-R based overlap removal
- `get_bjet_mask()`: B-tagging working point selection

### 2. Region Management (`darkbottomline/regions.py`)

Multi-region analysis with category and channel separation:

- `RegionManager`: Manages multiple analysis regions
- `Region`: Single region with cuts and properties
- `apply_regions()`: Apply region cuts to events
- Supports category-based regions (1b, 2b) with channel separation

### 3. Multi-Region Analyzer (`darkbottomline/analyzer.py`)

Multi-region analysis processor:

- `DarkBottomLineAnalyzer`: Extends base processor for multi-region analysis
- `process()`: Process events through all defined regions
- `_fill_region_histograms()`: Fill histograms for each region
- `_calculate_region_cutflow()`: Calculate cutflow per region
- `save_results()`: Save results with full region names preserved

### 4. Plotting (`darkbottomline/plotting.py`)

Data/MC plotting with CMS styling:

- `PlotManager`: Manages plot creation and styling
- `create_all_plots()`: Generate all plots for all regions
- `_get_excluded_variables_for_region()`: Region-specific plot exclusions
- Supports multiple formats: PNG, PDF, ROOT, TXT
- CMS plotting style with `mplhep`
- Configurable exclusions via `configs/plotting.yaml`

### 5. Histograms (`darkbottomline/histograms.py`)

Histogram definitions and filling:

- `HistogramManager`: Manages histogram creation and filling
- Histogram types: MET, jet kinematics, lepton kinematics, b-tagging, derived variables
- Support for both hist library and fallback implementation
- ~40+ histogram definitions matching StackPlotter variables

### 6. Weights (`darkbottomline/weights.py`)

Weight calculation and combination:

- `WeightCalculator`: Combines all weights using Coffea's Weights class
- `add_generator_weight()`: Generator weight handling
- `add_corrections()`: Correction weight application
- `get_weight()`: Final weight calculation with systematic variations

## Output Formats

### Parquet Output

```python
# Skimmed events with selected objects
skimmed_events = {
    "event": events.event,
    "run": events.run,
    "luminosityBlock": events.luminosityBlock,
    "MET": {"pt": events.MET.pt, "phi": events.MET.phi},
    "weights": event_weights,
    "muons": selected_muons,
    "electrons": selected_electrons,
    "jets": selected_jets,
    "bjets": selected_bjets,
}
```

### ROOT Output

Histograms saved as ROOT histograms with metadata.

### Pickle Output

Complete analysis results including histograms, cutflow, and metadata.

## Output Structure

### Analysis Results

Analysis results are saved as pickle files with the following structure:

```text
outputs/hists/
├── regions_data.pkl
├── regions_dy.pkl
├── regions_signal.pkl
└── ...
```

Each pickle file contains:

- `region_histograms`: Dictionary of histograms per region
- `regions`: Region processing results
- `region_cutflow`: Cutflow statistics per region
- `region_validation`: Region validation results
- `metadata`: Analysis metadata

### Plot Output Structure

Plots are organized in a versioned directory structure:

```text
outputs/plots/{version}/
├── png/
│   ├── 1b/
│   │   ├── SR/
│   │   │   ├── 1b_SR_met.png
│   │   │   ├── 1b_SR_met_log.png
│   │   │   └── ...
│   │   ├── Wlnu_mu/
│   │   │   ├── 1b_Wlnu_mu_lep1_pt.png
│   │   │   └── ...
│   │   ├── Wlnu_el/
│   │   ├── Zll_mu/
│   │   └── Zll_el/
│   └── 2b/
│       ├── SR/
│       ├── Top_mu/
│       ├── Top_el/
│       ├── Zll_mu/
│       └── Zll_el/
├── pdf/ (same structure as png/)
├── text/ (same structure as png/)
├── root/
│   ├── met.root (one file per variable)
│   └── ...
└── region_summary.{png,pdf}
```

**File Naming Convention:**

- Format: `{category}_{region_dir}_{variable_name}.{format}`
- Examples:
  - `1b_SR_met.png`
  - `2b_Top_mu_lep1_pt.png`
  - `1b_Zll_mu_z_mass.png`

**Plot Exclusions:**

- **1b SR**: Excludes jet3 plots and all lepton plots
- **2b SR**: Excludes lepton plots (includes jet3)
- **Top/W CRs**: Exclude `z_mass` and `z_pt` plots
- **Z CRs**: Include `z_mass` and `z_pt` plots

See `configs/plotting.yaml` for configurable exclusions.

## Validation

Use the validation notebooks to test and verify the framework:

```bash
jupyter notebook notebooks/
```

**Available Validation Notebooks:**

1. `01_plot_exclusions_validation.ipynb` - Test plot exclusions
2. `02_region_definitions_validation.ipynb` - Validate region definitions
3. `03_histogram_structure_validation.ipynb` - Check histogram structure
4. `04_plot_output_structure_validation.ipynb` - Verify plot directory structure
5. `05_configuration_validation.ipynb` - Validate configuration files
6. `06_data_mc_comparison_validation.ipynb` - Compare data/MC yields

See `notebooks/README.md` for detailed documentation.

## Extension Guide

### Adding New Years

1. Create new YAML configuration file in `configs/`
2. Update luminosity values and trigger paths
3. Adjust object selection cuts if needed

### Adding New Corrections

1. Add correction file path to configuration
2. Implement correction method in `CorrectionManager`
3. Add correction to weight calculation

### Adding New Histograms

1. Define histogram in `HistogramManager.define_histograms()`
2. Add filling logic in `fill_histograms()`
3. Update validation notebook if needed

## Dependencies

- **Core**: coffea, awkward, uproot, correctionlib
- **Execution**: dask, distributed
- **Output**: pyarrow, pandas
- **Visualization**: matplotlib, jupyter, mplhep
- **Histogramming**: hist
- **ROOT**: pyroot (optional, for ROOT file output)

Install all dependencies:

```bash
source local_setup.sh
```

For ROOT support (optional):

```bash
# Install ROOT via conda or system package manager
conda install -c conda-forge root
```

## Troubleshooting

### Common Issues

1. **Missing correction files**: Ensure correction files are in the correct paths specified in configuration. Warnings are acceptable if corrections are not needed for testing.

2. **Import errors**: Check that all dependencies are installed correctly:

   ```bash
   source local_setup.sh
   ```

3. **Memory issues**:
   - Use `--max-events` to limit events for testing:

     ```bash
     darkbottomline analyze ... --max-events 10000
     ```

   - For futures/dask executors, reduce `--chunk-size` to process smaller chunks:

     ```bash
     darkbottomline analyze ... --executor futures --chunk-size 25000
     ```

4. **Executor issues**: Try different executors (iterative, futures, dask)

5. **ROOT not available**: ROOT files won't be generated if ROOT library is not installed. Other formats (PNG, PDF, TXT) will still be created.

6. **Old region format**: If you see regions like `CR_Zll` instead of `1b:CR_Zll_mu`, the data file was created with an old version. Re-run the analysis with the updated `regions.yaml`.

7. **Plot exclusions not working**: Check that `configs/plotting.yaml` is loaded correctly and exclusion patterns match variable names.

### Debug Mode

Run with debug logging to see detailed information:

```bash
# Set log level
export PYTHONPATH=.
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Run analysis
darkbottomline analyze ... --log-level DEBUG
```

### Run Validation Notebooks

Run validation notebooks to check framework setup:

```bash
jupyter notebook notebooks/01_plot_exclusions_validation.ipynb
```

## Complete Analysis Example

Here's a complete example running the full analysis workflow:

```bash
#!/bin/bash
# Complete analysis workflow

# Setup
source venv/bin/activate
cd /path/to/DarkBottomLine

# Configuration
CONFIG="configs/2024.yaml"
REGIONS_CONFIG="configs/regions.yaml"
INPUT_DIR="/path/to/nanoaod"
OUTPUT_DIR="outputs/hists"

# 1. Run analysis on all samples
echo "Running analysis on data..."
darkbottomline analyze \
    --config $CONFIG \
    --regions-config $REGIONS_CONFIG \
    --input ${INPUT_DIR}/nano_data.root \
    --output ${OUTPUT_DIR}/regions_data.pkl

echo "Running analysis on DY..."
darkbottomline analyze \
    --config $CONFIG \
    --regions-config $REGIONS_CONFIG \
    --input ${INPUT_DIR}/nano_dy.root \
    --output ${OUTPUT_DIR}/regions_dy.pkl

echo "Running analysis on signal..."
darkbottomline analyze \
    --config $CONFIG \
    --regions-config $REGIONS_CONFIG \
    --input ${INPUT_DIR}/nano_signal.root \
    --output ${OUTPUT_DIR}/regions_signal.pkl

# 2. Generate plots
echo "Generating plots..."
darkbottomline make-plots \
    --input ${OUTPUT_DIR}/regions_data.pkl \
    --save-dir outputs \
    --show-data \
    --plot-config configs/plotting.yaml

echo "Analysis complete! Plots saved to outputs/plots/{version}/"
```

## Documentation

- **Analysis Structure**: See `docs/analysis_structure.md` for region naming conventions and structure flow
- **Plotting Configuration**: See `docs/plotting_configuration.md` for plot exclusion configuration
- **Validation Notebooks**: See `notebooks/README.md` for validation notebook documentation
- **Developer Guide**: See `DEVELOPER_GUIDE.md` for a comprehensive guide on where to make changes (plotting, variables, histograms, regions, etc.)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on top of the Coffea framework
- Uses correctionlib for scale factor corrections
- Inspired by CMS analysis workflows
- Plotting style follows CMS figure guidelines using `mplhep`
