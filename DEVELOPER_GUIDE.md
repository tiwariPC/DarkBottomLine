# Developer Guide: Where to Make Changes

This guide maps common modification tasks to specific files and locations in the DarkBottomLine framework. Use this document as a quick reference when you need to modify the analysis code, plotting styles, variable definitions, or configuration files.

---

## Table of Contents

1. [Environment and Installation](#environment-and-installation)
2. [Plotting and Visualization](#plotting-and-visualization)
3. [Variable Definitions](#variable-definitions)
4. [Histogram Definitions](#histogram-definitions)
5. [Region Definitions](#region-definitions)
6. [Physics Object Selection](#physics-object-selection)
7. [Corrections and Weights](#corrections-and-weights)
8. [Configuration Files](#configuration-files)
9. [Command-Line Interface](#command-line-interface)
10. [Analysis Workflow](#analysis-workflow)

---

## Environment and Installation

### Overview of Setup Files

The project uses two separate installation workflows depending on the target environment. `environment.yml` is the single source of truth for all package dependencies.

| File | Purpose | When to use |
| ---- | ------- | ----------- |
| `environment.yml` | Conda environment definition (all deps) | Always — canonical dependency list |
| `local_setup.sh` | Create/activate conda env, `pip install -e .` | Local computer |
| `lxplus_setup.sh` | `pip install -e . --prefix .local` | lxplus only |
| `check_requirements.py` | Check/install missing packages via pip | lxplus only |
| `start.sh` | Set `PYTHONPATH`/`PATH` for `.local` packages | lxplus, every new session |
| `setup.py` | Package metadata, entry points, `install_requires` | Used by both `pip install -e .` calls |

### Local Setup (Normal Computer)

```bash
source local_setup.sh
```

This script uses `mamba`/`conda` to create the `darkbottomline` environment from `environment.yml` (if it does not already exist), activates it, and runs `pip install -e .` to register the package.

### Lxplus Setup

```bash
# 1. Load LCG environment (provides a base Python)
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh

# 2. Check and install missing packages into .local/
python3 check_requirements.py --install --local-dir ./.local

# 3. Install the package itself into .local/
./lxplus_setup.sh

# 4. In future sessions: source start.sh to restore PYTHONPATH/PATH
source start.sh
```

### Updating Dependencies

`environment.yml` is the **only** file you need to edit when adding or changing a dependency. `setup.py`'s `install_requires` must be kept in sync manually — update it at the same time.

**Files to edit when adding a dependency:**

1. `environment.yml` — add to the conda `dependencies:` list (or under `pip:` for pip-only packages)
2. `setup.py` — add to `install_requires` (runtime deps only; omit dev/test tools like `pytest`, `jupyter`)

**Do not** add dependencies to any other file. `requirements.txt` has been removed — `environment.yml` supersedes it.

### Package Entry Point

The `darkbottomline` CLI command is registered via `setup.py`:

```python
entry_points={
    "console_scripts": [
        "darkbottomline=darkbottomline.cli:main",
    ],
}
```

If you add a new top-level command, register it in `darkbottomline/cli.py` and it will be automatically available after `pip install -e .`.

### Checking Requirements (lxplus)

`check_requirements.py` reads package requirements from `environment.yml` (not `requirements.txt`):

```bash
# Check what is installed vs missing
python3 check_requirements.py

# Install missing packages into ./.local
python3 check_requirements.py --install --local-dir ./.local

# Use a different environment file
python3 check_requirements.py --requirements /path/to/environment.yml
```

---

## Plotting and Visualization

### Change Plot Style (Colors, Fonts, Markers)

**File**: `darkbottomline/utils/plot_utils.py`

- **Colors**: Edit `CMSPlotStyle.colors` dictionary (lines 24-33)
- **Font Sizes**: Edit `CMSPlotStyle.font_sizes` dictionary (lines 35-41)
- **Line Styles**: Edit `CMSPlotStyle.line_styles` dictionary (lines 43-48)
- **Markers**: Edit `CMSPlotStyle.markers` dictionary (lines 50-58)
- **Global Style Settings**: Edit `CMSPlotStyle.set_style()` method (lines 60-104)

**Example**:
```python
# In darkbottomline/utils/plot_utils.py
self.colors = {
    'primary': '#1f77b4',  # Change this to modify primary color
    'secondary': '#ff7f0e',
    # ... add more colors
}
```

### Change Process Colors (Data, MC, Signal)

**File**: `darkbottomline/utils/plot_utils.py`

- **Location**: `get_process_colors()` function (lines 122-140)

**Example**:
```python
# In darkbottomline/utils/plot_utils.py
def get_process_colors() -> Dict[str, str]:
    return {
        'data': '#000000',      # Change data color
        'signal': '#ff0000',    # Change signal color
        'ttbar': '#1f77b4',     # Change ttbar color
        # ... add more processes
    }
```

### Change Process Labels (LaTeX Labels)

**File**: `darkbottomline/utils/plot_utils.py`

- **Location**: `get_process_labels()` function (lines 143-161)

**Example**:
```python
# In darkbottomline/utils/plot_utils.py
def get_process_labels() -> Dict[str, str]:
    return {
        'ttbar': 't\\bar{t}',  # Change LaTeX label
        'wjets': 'W+jets',
        # ... add more labels
    }
```

### Modify Plot Appearance (Figure Size, DPI, etc.)

**File**: `darkbottomline/plotting.py`

- **Location**: `PlotManager.__init__()` method (lines 25-80)

**Example**:
```python
# In darkbottomline/plotting.py
def __init__(self, config: Optional[Dict[str, Any]] = None):
    # ...
    self.figsize = self.config.get("figsize", (12, 8))  # Change default figure size
    self.dpi = self.config.get("dpi", 300)              # Change default DPI
```

### Configure Plot Exclusions (Which Plots to Generate)

**File**: `configs/plotting.yaml`

- **Location**: `region_exclusions` section (lines 40-75)

**Example**:
```yaml
# In configs/plotting.yaml
region_exclusions:
  # Exclude z_mass and z_pt from Top CRs
  "Top": ["z_mass", "z_pt"]

  # Exclude specific variables from 1b SR
  "1b:SR": ["jet3_pt", "lep1_pt"]

  # Exclude variables from all 2b regions
  "2b:": ["some_2b_specific_var"]
```

### Disable Log Scale for Specific Variables

**File**: `configs/plotting.yaml`

- **Location**: `no_log_scale_vars` section (lines 7-16)

**Example**:
```yaml
# In configs/plotting.yaml
no_log_scale_vars:
  - n_jets
  - n_bjets
  - n_muons
  # Add more variables that should use linear scale
```

### Modify Plot Output Structure or File Naming

**File**: `darkbottomline/plotting.py`

- **Location**: `save_plot_multi_format()` method (lines ~200-300)
- **Location**: `_parse_region_name()` method (lines ~150-200)

**Example**:
```python
# In darkbottomline/plotting.py
def save_plot_multi_format(self, fig, hist_name, region, version, ...):
    region_info = self._parse_region_name(region)
    category = region_info["category"]
    region_dir = region_info["region_dir"]

    # Modify directory structure here
    png_dir = os.path.join(base_output_dir, "plots", version, "png", category, region_dir)

    # Modify file naming here
    png_path = os.path.join(png_dir, f"{hist_name}.png")
```

### Add Custom Plotting Functions

**File**: `darkbottomline/plotting.py`

- **Location**: Add new methods to `PlotManager` class
- **Reference**: Existing methods like `_create_kinematic_plots()`, `_create_multiplicity_plots()`

---

## Variable Definitions

### Add a New Analysis Variable

**File**: `darkbottomline/regions.py`

- **Location**: `Region._get_variable_value()` method (lines 111-260)

**Steps**:
1. Add the variable calculation logic in `_get_variable_value()`
2. Add the variable to histogram definitions (see [Histogram Definitions](#histogram-definitions))
3. Ensure the variable is computed in the analyzer (see [Analysis Workflow](#analysis-workflow))

**Example**:
```python
# In darkbottomline/regions.py
def _get_variable_value(self, events: ak.Array, objects: Dict[str, Any], var: str):
    # ... existing variable definitions ...

    # Add new variable
    if var == "MyNewVariable":
        # Calculate your variable here
        jets = objects.get("jets", ak.Array([]))
        return ak.max(jets.pt, axis=1)  # Example calculation

    # ... rest of method ...
```

### Modify Existing Variable Calculations

**File**: `darkbottomline/regions.py`

- **Location**: `Region._get_variable_value()` method (lines 111-260)

**Examples of existing variables**:
- `MET` (lines 116-120): MET calculation
- `Recoil` (lines 148-169): Recoil calculation
- `MT` (lines 170-197): Transverse mass calculation
- `Mll` (lines 198-207): Dilepton invariant mass
- `DeltaPhi` (lines 208-226): Delta phi between jets and MET
- `metQuality` (lines 238-246): MET quality variable

### Add Variables Computed in Analyzer

**File**: `darkbottomline/analyzer.py`

- **Location**: `_fill_region_histograms()` method or `_process_region()` method
- **Reference**: Look for where variables are computed from events/objects

**Note**: Most variables are computed in `regions.py`, but some may be computed directly in the analyzer during histogram filling.

---

## Histogram Definitions

### Add a New Histogram

**File**: `darkbottomline/histograms.py`

- **Location**: `HistogramManager._define_hist_histograms()` method (lines 40-400+)

**Steps**:
1. Add histogram definition in `_define_hist_histograms()`
2. Add corresponding fallback definition in `_define_fallback_histograms()` if needed
3. Ensure the variable is computed (see [Variable Definitions](#variable-definitions))
4. Ensure the histogram is filled in the analyzer (see [Analysis Workflow](#analysis-workflow))

**Example**:
```python
# In darkbottomline/histograms.py
def _define_hist_histograms(self) -> Dict[str, Any]:
    histograms = {}

    # ... existing histograms ...

    # Add new histogram
    histograms["my_new_variable"] = hist.Hist(
        hist.axis.Regular(50, 0, 500, name="my_new_variable", label="My New Variable [GeV]"),
        storage=hist.storage.Weight()
    )

    return histograms
```

### Modify Histogram Binning

**File**: `darkbottomline/histograms.py`

- **Location**: `HistogramManager._define_hist_histograms()` method

**Example**:
```python
# In darkbottomline/histograms.py
# Change MET histogram binning from 50 bins (0-500) to 100 bins (0-1000)
histograms["met"] = hist.Hist(
    hist.axis.Regular(100, 0, 1000, name="met", label="MET [GeV]"),  # Modified
    storage=hist.storage.Weight()
)
```

### Add 2D Histograms

**File**: `darkbottomline/histograms.py`

- **Location**: `HistogramManager._define_hist_histograms()` method

**Example**:
```python
# In darkbottomline/histograms.py
histograms["met_vs_njets"] = hist.Hist(
    hist.axis.Regular(50, 0, 500, name="met", label="MET [GeV]"),
    hist.axis.Regular(10, 0, 10, name="n_jets", label="Number of Jets"),
    storage=hist.storage.Weight()
)
```

---

## Region Definitions

### Add a New Region

**File**: `configs/regions.yaml`

- **Location**: `regions` section (lines 4-118)

**Steps**:
1. Add region definition in `regions.yaml`
2. Ensure category and region naming follows convention: `{category}:{region_type}_{channel}`
   - Categories: `1b`, `2b`
   - Region types: `SR`, `CR_Wlnu`, `CR_Top`, `CR_Zll`
   - Channels: `mu`, `el` (for CRs)

**Example**:
```yaml
# In configs/regions.yaml
regions:
  "1b:CR_Wlnu_mu":  # Category:RegionType_Channel
    description: "W+jets control region in 1b category (muon)"
    cuts:
      Nbjets: "==1"
      Nmuons: "==1"
      Recoil: ">250"
      MET: ">100"
      # ... more cuts ...
```

### Modify Region Cuts

**File**: `configs/regions.yaml`

- **Location**: `regions` section, under specific region's `cuts` sub-section

**Example**:
```yaml
# In configs/regions.yaml
regions:
  "1b:SR":
    cuts:
      MET: ">250"        # Change MET cut from 50 to 250
      DeltaPhi: ">0.5"   # Change DeltaPhi cut from 0.1 to 0.5
```

### Add New Cut Variables

**File**: `darkbottomline/regions.py`

- **Location**: `Region._get_variable_value()` method (see [Variable Definitions](#variable-definitions))
- **Note**: After adding a variable, you can use it in `regions.yaml` cuts

### Modify Region Validation Settings

**File**: `configs/regions.yaml`

- **Location**: `settings` and `validation` sections (if present)

**Example**:
```yaml
# In configs/regions.yaml
settings:
  min_events: 10      # Minimum events per region
  max_overlap: 0.1    # Maximum overlap fraction between regions

validation:
  check_orthogonality: true
  check_overlap: true
```

---

## Physics Object Selection

### Modify Muon Selection

**File**: `darkbottomline/objects.py`

- **Location**: `select_muons()` function (lines 10-41)

**Example**:
```python
# In darkbottomline/objects.py
def select_muons(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    muons = ak.zip({
        "pt": events["Muon_pt"],
        # ... other properties ...
    })

    # Modify cuts here
    pt_mask = muons.pt > config["pt_min"]  # Change pt_min in config
    eta_mask = abs(muons.eta) < config["eta_max"]  # Change eta_max in config
    # ... modify ID/iso cuts ...
```

### Modify Electron Selection

**File**: `darkbottomline/objects.py`

- **Location**: `select_electrons()` function (lines 44-75)

**Note**: Similar structure to muon selection

### Modify Tau Selection

**File**: `darkbottomline/objects.py`

- **Location**: `select_taus()` function (lines 78-110+)

### Modify Jet Selection

**File**: `darkbottomline/objects.py`

- **Location**: `select_jets()` function (lines ~120-200)

### Modify B-Jet Selection

**File**: `darkbottomline/objects.py`

- **Location**: `select_bjets()` function (lines ~200-280)

### Modify Object Selection Cuts (pT, η, ID, etc.)

**File**: `configs/{year}.yaml` (e.g., `configs/2024.yaml`)

- **Location**: Object selection sections (e.g., `muons`, `electrons`, `jets`, `bjets`)

**Example**:
```yaml
# In configs/2024.yaml
muons:
  pt_min: 20      # Change minimum pT
  eta_max: 2.4    # Change maximum |η|
  # ... other selection parameters ...
```

---

## Corrections and Weights

### Add New Correction

**File**: `darkbottomline/corrections.py`

- **Location**: `CorrectionManager` class methods

**Steps**:
1. Add correction calculation method in `CorrectionManager`
2. Register it in `get_all_corrections()` method
3. Ensure correctionlib JSON file is available if using correctionlib

### Modify Pileup Reweighting

**File**: `darkbottomline/corrections.py`

- **Location**: `CorrectionManager.get_pileup_weight()` method

### Modify Scale Factors (B-tag, Lepton, etc.)

**File**: `darkbottomline/corrections.py`

- **Location**: Relevant `get_*_sf()` methods (e.g., `get_btag_sf()`, `get_muon_sf()`, `get_electron_sf()`)

**Note**: Scale factors are typically loaded from correctionlib JSON files specified in the config.

### Modify Weight Calculation

**File**: `darkbottomline/weights.py`

- **Location**: `WeightCalculator` class methods

**Example**:
```python
# In darkbottomline/weights.py
def add_weight(self, name: str, weight: Union[ak.Array, np.ndarray, float], ...):
    # Modify how weights are added/combined
    self.weights.add(name, weight)
```

---

## Configuration Files

### Modify Year-Specific Configuration

**File**: `configs/{year}.yaml` (e.g., `configs/2024.yaml`, `configs/2023.yaml`)

- **Location**: Various sections depending on what you want to change

**Common sections**:
- `year`: Data-taking year
- `luminosity`: Integrated luminosity
- `objects`: Object selection cuts (muons, electrons, jets, bjets)
- `corrections`: Correction file paths and settings
- `systematics`: Systematic uncertainty settings

### Modify Plotting Configuration

**File**: `configs/plotting.yaml`

- **Location**: See [Plotting and Visualization](#plotting-and-visualization) section above

### Modify Combine Configuration (for Statistical Analysis)

**File**: `configs/combine.yaml`

- **Location**: Process definitions, systematic uncertainties, etc.

### Modify DNN Configuration

**File**: `configs/dnn.yaml`

- **Location**: DNN model paths, input variables, output settings

---

## Command-Line Interface

### Add New CLI Command

**File**: `darkbottomline/cli.py`

- **Location**: Add new command function and register it in `main()` function

**Example**:
```python
# In darkbottomline/cli.py
@cli.command()
@click.option('--input', required=True, help='Input file')
def my_new_command(input):
    """Description of new command."""
    # Implementation here
    pass
```

### Modify Existing CLI Options

**File**: `darkbottomline/cli.py`

- **Location**: Find the relevant command function and modify its `@click.option()` decorators

**Example**:
```python
# In darkbottomline/cli.py
@cli.command()
@click.option('--input', required=True, help='Input file')
@click.option('--output', default='outputs/', help='Output directory')  # Modify default
@click.option('--max-events', type=int, help='Maximum events to process')  # Add new option
def analyze(input, output, max_events):
    # Implementation
    pass
```

---

## Analysis Workflow

### Modify Event Processing Logic

**File**: `darkbottomline/analyzer.py`

- **Location**: `DarkBottomLineAnalyzer.process()` method (lines 118-180+)

**Key methods**:
- `process()`: Main event processing loop
- `_process_region()`: Process events for a specific region
- `_fill_region_histograms()`: Fill histograms for a region

### Modify Histogram Filling Logic

**File**: `darkbottomline/analyzer.py`

- **Location**: `_fill_region_histograms()` method

**Note**: This method fills histograms from computed variables. Ensure variables are computed in `regions.py` or `analyzer.py`.

### Modify Object Building

**File**: `darkbottomline/objects.py`

- **Location**: `build_objects()` function

**Note**: This function builds physics objects from events using the selection functions.

### Modify Event Selection

**File**: `darkbottomline/selections.py`

- **Location**: `apply_selection()` function

**Note**: This applies global event selection cuts (e.g., MET filters, triggers).

---

## Quick Reference: File Locations Summary

| Task | File | Location |
| ---- | ---- | -------- |
| Plot colors/styles | `darkbottomline/utils/plot_utils.py` | `CMSPlotStyle` class |
| Process colors/labels | `darkbottomline/utils/plot_utils.py` | `get_process_colors()`, `get_process_labels()` |
| Plot exclusions | `configs/plotting.yaml` | `region_exclusions` section |
| Log scale settings | `configs/plotting.yaml` | `no_log_scale_vars` section |
| Add variable | `darkbottomline/regions.py` | `Region._get_variable_value()` |
| Modify variable | `darkbottomline/regions.py` | `Region._get_variable_value()` |
| Add histogram | `darkbottomline/histograms.py` | `_define_hist_histograms()` |
| Modify histogram binning | `darkbottomline/histograms.py` | `_define_hist_histograms()` |
| Add region | `configs/regions.yaml` | `regions` section |
| Modify region cuts | `configs/regions.yaml` | `regions.{region_name}.cuts` |
| Modify muon selection | `darkbottomline/objects.py` | `select_muons()` |
| Modify electron selection | `darkbottomline/objects.py` | `select_electrons()` |
| Modify jet selection | `darkbottomline/objects.py` | `select_jets()` |
| Modify object cuts (pT, η) | `configs/{year}.yaml` | Object sections |
| Add correction | `darkbottomline/corrections.py` | `CorrectionManager` class |
| Modify weights | `darkbottomline/weights.py` | `WeightCalculator` class |
| Modify CLI | `darkbottomline/cli.py` | Command functions |
| Modify processing | `darkbottomline/analyzer.py` | `process()` method |

---

## Tips for Making Changes

1. **Always test changes**: Run a small subset of events first to verify your changes work correctly.

2. **Check dependencies**: When adding a variable, ensure:
   - Variable is defined in `regions.py`
   - Histogram is defined in `histograms.py`
   - Histogram is filled in `analyzer.py`
   - Variable is included in plot generation (if needed)

3. **Follow naming conventions**:
   - Regions: `{category}:{region_type}_{channel}` (e.g., `1b:CR_Wlnu_mu`)
   - Variables: lowercase with underscores (e.g., `jet_pt`, `met_phi`)
   - Histograms: match variable names

4. **Use configuration files**: Prefer YAML configs over hardcoding values when possible.

5. **Check validation notebooks**: Use `notebooks/*.ipynb` to validate your changes.

6. **Update documentation**: If you add significant features, update `README.md` or this guide.

---

## Getting Help

- **See README.md**: Main project documentation with workflow examples
- **See notebooks/README.md**: Validation notebook documentation
- **Check code comments**: Many functions have detailed docstrings
- **Review existing code**: Similar functionality likely exists as a template

---

Last updated: April 2026


