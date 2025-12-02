# Ophys - 2-Photon Imaging Analysis

Comprehensive Python toolkit for 2-photon calcium imaging data analysis, including FOV configuration, Suite2P trace extraction, orientation/direction tuning analysis, and visualization.

## Overview

This repository contains a complete Python-based pipeline for analyzing 2-photon imaging experiments. It replaces MATLAB-based workflows with a modern, type-safe, and extensible Python implementation.

### Key Features

- **FOV Configuration**: Auto-extract parameters from PsychoPy stimulus files
- **Suite2P Integration**: Extract and organize calcium traces from Suite2P output
- **Trial Organization**: Organize continuous traces into stimulus-aligned trial structure
- **Tuning Analysis**: Calculate orientation and direction selectivity indices
- **Comprehensive Plotting**: Publication-ready visualizations
- **Batch Processing**: Analyze multiple FOVs efficiently
- **HDF5 Storage**: Save/load extraction results in portable format

## Structure

```
Ophys/
├── fov_config_suite2p.py         # FOV configuration dataclass
├── fov_utils.py                  # FOV utility functions
├── ophys_analysis/               # Main analysis package
│   ├── __init__.py
│   ├── cell_data.py              # Cell and CellExtraction classes
│   ├── trace_extraction.py       # Suite2P extraction
│   ├── tuning_analysis.py        # Orientation/direction analysis
│   ├── plotting.py               # Visualization functions
│   └── io_utils.py               # HDF5 I/O
├── notebooks/                    # Jupyter analysis notebooks
│   └── orientation_tuning_analysis.ipynb
├── example_extraction.py         # Usage examples
├── batch_process_fovs.py         # Batch processing script
└── README.md                     # This file
```

## Installation

### Requirements

```bash
pip install numpy scipy matplotlib h5py pandas
```

Optional for Jupyter notebooks:
```bash
pip install jupyter ipykernel
```

### Setup

```bash
git clone https://github.com/msarvestani/Ophys.git
cd Ophys
```

No installation needed - just import modules directly or add to your Python path.

## Complete Analysis Workflow

### 1. Configure FOV

Create FOV configuration from your stimulus file:

```python
from fov_config_suite2p import FOV
from fov_utils import create_fov_from_stimfile

# Auto-create from stimulus file
fov = create_fov_from_stimfile(
    stimfile='X:/Madineh/Calcium_Imaging/20251113_Derrick/visual_stim.py',
    TifStack_path='X:/Madineh/Calcium_Imaging/20251113_Derrick',
    ImagingFile=[0],   # Which imaging file(s) to use
    Spk2File=[0],      # Which Spike2 file(s) to use
)

# Set additional parameters
fov.factor = 1
fov.brain_region = 'V1'
fov.layer = 'L2/3'
```

This automatically extracts:
- `animal_name` from path (e.g., '20251113_Derrick' → 'Derrick')
- `recording_date` from folder name
- `stim_dur`, `postPeriod` from stimulus file
- `stim_type`, `have_blank` from stimulus parameters

### 2. Extract Suite2P Traces

Process Suite2P output and organize into trial structure:

```python
from ophys_analysis import extract_suite2p_traces

# Run extraction
ce = extract_suite2p_traces(fov, fnum=0)

# Print summary
ce.print_summary()
# Output:
#   Total cells: 245
#   Responsive cells: 128 (52.2%)
#   Non-responsive: 117
```

### 3. Analyze Tuning

Calculate orientation and direction selectivity:

```python
from ophys_analysis import get_tuning_madineh
import numpy as np

# Get tuning for a responsive cell
cell = ce.cells[0]
n_dirs = len(cell.uniqStims) - 1  # Exclude blank
stimInfo = np.arange(0, 360, 360/n_dirs)

tuning, _, fitdata = get_tuning_madineh(
    cell.condition_response[:n_dirs],
    stimInfo
)

print(f"Preferred orientation: {tuning['pref_ort_fit']:.1f}°")
print(f"OTI: {tuning['oti_fit']:.3f}")
print(f"DTI: {tuning['dti_fit']:.3f}")
```

### 4. Visualize Results

Create comprehensive analysis plots:

```python
from ophys_analysis import (
    plot_population_summary,
    plot_tuning_distributions,
    plot_orientation_map,
    create_full_analysis_report,
)

# Individual plots
plot_population_summary(ce)
plot_tuning_distributions(ce)
plot_orientation_map(ce)

# Or create complete report
create_full_analysis_report(ce, output_dir='analysis_results')
```

### 5. Save/Load Results

Store extraction results in HDF5 format:

```python
from ophys_analysis import save_extraction_hdf5, load_extraction_hdf5

# Save
save_extraction_hdf5(ce, 'extraction_results.h5')

# Load later
ce_loaded = load_extraction_hdf5('extraction_results.h5')
```

### 6. Batch Processing

Process multiple FOVs at once:

```bash
python batch_process_fovs.py \
    --input_dir X:/Madineh/Calcium_Imaging \
    --output_dir batch_results \
    --brain_region V1 \
    --layer L2/3
```

This will:
- Find all FOVs in the input directory
- Extract traces for each
- Calculate tuning metrics
- Generate analysis plots
- Create summary CSV with all results

## API Documentation

### FOV Configuration

#### FOV Parameters

**Required (manual entry):**
- `TifStack_path`: Path to imaging data directory
- `ImagingFile`: List of imaging file indices (e.g., `[0, 1, 2]`)
- `Spk2File`: List of Spike2 file indices (e.g., `[0]`)

**Optional (with defaults):**
- `factor`: Frame averaging factor (default: 1)
- `brain_region`: Brain region (default: 'v1')
- `layer`: Cortical layer (default: None)
- `zoom`: Microscope zoom (default: 3)
- `piezo_planes`: Number of z-planes (default: 0)
- `prePeriod`: Pre-stimulus baseline (default: 0.5s)
- `extraction`: Method ('suite2p' or 'cnmf', default: 'suite2p')
- `EPI_data`: EPI data flag (default: 0)

**Auto-populated from stimulus file:**
- `animal_name`: Extracted from folder (e.g., '20251113_Derrick' → 'Derrick')
- `recording_date`: From folder name (YYYYMMDD pattern)
- `stim_dur`: Stimulus duration (seconds)
- `postPeriod`: Inter-stimulus interval (seconds)
- `stim_type`: Stimulus type ('grating', 'image', etc.)
- `have_blank`: Whether blank trials included (boolean)
- `sampRate`: Calculated as 30/factor (Hz)

### Data Structures

#### Cell Class
Individual neuron with traces and tuning properties:

```python
from ophys_analysis import Cell

cell = Cell()
cell.raw              # Raw fluorescence trace
cell.dff              # ΔF/F trace
cell.cyc              # Trial-structured data [n_stim, n_trials, n_timepoints]
cell.xPos, cell.yPos  # Spatial position
cell.mask             # ROI mask coordinates
cell.ROI_responsiveness  # Boolean: visually responsive?
cell.condition_response  # Mean response per stimulus
```

#### CellExtraction Class
Collection of cells with metadata:

```python
from ophys_analysis import CellExtraction

ce = CellExtraction()
ce.cells              # List of Cell objects
ce.twophotontimes     # Frame timestamps
ce.stimOn, ce.stimID  # Stimulus timing and identity
ce.uniqStims          # Unique stimulus values

# Helper methods
ce.to_array('xPos')   # Get array of attribute across cells
ce.get_responsive_cells()  # Filter for responsive cells
ce.print_summary()    # Print statistics
```

### Core Functions

#### extract_suite2p_traces()
Main extraction function:

```python
extract_suite2p_traces(fov, fnum=0)
```
- Loads Suite2P F.npy, Fneu.npy, iscell.npy, stat.npy
- Loads Spike2 timing data
- Organizes traces into trial structure
- Calculates ΔF/F with baseline correction
- Detects visually responsive cells (2σ above baseline)
- Returns: `CellExtraction` object

#### get_tuning_madineh()
Calculate tuning metrics:

```python
tuning, meanResponse_fit, fitdata = get_tuning_madineh(meanResponse, stimInfo)
```
- Fits double Gaussian to direction tuning
- Calculates orientation/direction selectivity
- Returns dictionary with:
  - `pref_ort_fit`, `pref_dir_fit`: Preferred orientation/direction (degrees)
  - `oti_fit`, `dti_fit`: Orientation/direction tuning indices (0-1)
  - `oti_vec`, `dti_vec`: Vector-based selectivity
  - `fit_bandwidth`: Tuning width (degrees)
  - `fit_r`: Goodness of fit (correlation)

### Plotting Functions

#### plot_cell_tuning_curve()
Comprehensive single-cell analysis:
- Time series traces for each orientation
- Polar tuning plot
- Fitted tuning curve with statistics

#### plot_orientation_map()
Spatial maps of tuning preferences:
- HSV-colored orientation map (0-180°)
- Direction map (0-360°)
- Overlaid on FOV image (optional)

#### plot_tuning_distributions()
Population-level statistics:
- OTI/DTI histograms
- Preferred orientation/direction distributions
- Tuning bandwidth distribution
- DTI vs OTI scatter

#### plot_population_summary()
Overall dataset summary:
- Cell responsiveness statistics
- Spatial distribution
- Response amplitude distribution

#### create_full_analysis_report()
Generates all plots and saves to directory:
```python
create_full_analysis_report(ce, fov_image=None, output_dir='results')
```

### Utility Functions

#### FOV utilities (`fov_utils.py`)

```python
from fov_utils import create_fov_from_stimfile, export_fov_to_dict

# Create FOV from stimulus file
fov = create_fov_from_stimfile(stimfile, TifStack_path, ImagingFile, Spk2File)

# Export to dictionary
fov_dict = export_fov_to_dict(fov)
```

#### I/O functions

```python
from ophys_analysis import save_extraction_hdf5, load_extraction_hdf5

# Save to HDF5
save_extraction_hdf5(ce, 'results.h5')

# Load from HDF5
ce = load_extraction_hdf5('results.h5')
```

HDF5 structure:
```
/fov_metadata/          - FOV parameters (attributes)
/acquisition/
    twophotontimes      - Frame timestamps
    stimOn              - Stimulus onset times
    stimID              - Stimulus identities
    uniqStims           - Unique stimulus values
/cells/
    cell_0/             - Cell 0 data
        raw             - Raw fluorescence
        dff             - ΔF/F
        cyc             - Trial structure
        ...             - Other attributes
    cell_1/
    ...
```

## Jupyter Notebook Tutorial

See `notebooks/orientation_tuning_analysis.ipynb` for a complete interactive tutorial covering:
- FOV setup and configuration
- Suite2P trace extraction
- Trial organization
- Tuning analysis
- Visualization
- Advanced filtering and analysis
- Spatial clustering analysis
- Exporting results to CSV

Run it with:
```bash
jupyter notebook notebooks/orientation_tuning_analysis.ipynb
```

## Batch Processing

The `batch_process_fovs.py` script processes multiple FOVs efficiently:

```bash
# Basic usage
python batch_process_fovs.py --input_dir /path/to/data --output_dir results

# With parameters
python batch_process_fovs.py \
    --input_dir X:/Madineh/Calcium_Imaging \
    --output_dir batch_results \
    --pattern "*/visual_stim.py" \
    --brain_region V1 \
    --layer L2/3 \
    --factor 1
```

Output structure:
```
batch_results/
├── batch_summary.csv              # Summary statistics for all FOVs
├── 20251113_Derrick/
│   ├── fov_config.json
│   ├── extraction_results.h5
│   ├── tuning_metrics.csv
│   └── plots/
│       ├── population_summary.png
│       ├── orientation_maps.png
│       ├── tuning_distributions.png
│       └── cell_*_tuning.png
└── 20251114_Mouse2/
    └── ...
```

## Data Analysis Examples

### Filter cells by tuning properties

```python
# Find orientation-selective cells
oti_threshold = 0.3
dti_threshold = 0.3

ori_selective = []
for i, cell in enumerate(ce.cells):
    if cell.ROI_responsiveness:
        n_dirs = len(cell.uniqStims) - 1
        stimInfo = np.arange(0, 360, 360/n_dirs)
        tuning, _, _ = get_tuning_madineh(cell.condition_response[:n_dirs], stimInfo)

        if tuning['oti_fit'] > oti_threshold and tuning['dti_fit'] < dti_threshold:
            ori_selective.append(i)

print(f"Found {len(ori_selective)} orientation-selective cells")
```

### Export to pandas DataFrame

```python
import pandas as pd

results = []
for i, cell in enumerate(ce.cells):
    if cell.ROI_responsiveness:
        n_dirs = len(cell.uniqStims) - 1
        stimInfo = np.arange(0, 360, 360/n_dirs)
        tuning, _, _ = get_tuning_madineh(cell.condition_response[:n_dirs], stimInfo)

        results.append({
            'cell_id': i,
            'x': cell.xPos,
            'y': cell.yPos,
            'pref_ori': tuning['pref_ort_fit'],
            'pref_dir': tuning['pref_dir_fit'],
            'oti': tuning['oti_fit'],
            'dti': tuning['dti_fit'],
            'bandwidth': tuning['fit_bandwidth'],
        })

df = pd.DataFrame(results)
df.to_csv('tuning_summary.csv', index=False)
```

### Analyze spatial clustering

```python
from scipy.spatial.distance import pdist, squareform

# Get positions and orientations
positions = ce.to_array('xPos', 'yPos')  # Shape: (n_cells, 2)
orientations = []  # Extract from tuning analysis

# Calculate distance matrices
spatial_dist = squareform(pdist(positions))
ori_diff = ...  # Calculate orientation differences

# Correlation analysis
from scipy.stats import pearsonr
r, p = pearsonr(spatial_dist.flatten(), ori_diff.flatten())
```

## Migration from MATLAB

Key changes from MATLAB workflow:

| MATLAB | Python |
|--------|--------|
| `FOV(num).fileImaging` | `fov.ImagingFile` |
| `FOV(num).fileSpk` | `fov.Spk2File` |
| `extraction_2025_suite2p.m` | `extract_suite2p_traces()` |
| `getTuningMadineh.m` | `get_tuning_madineh()` |
| `.mat` files | HDF5 (`.h5`) files |
| Manual parameter entry | Auto-extraction from stimulus files |

## Troubleshooting

### Suite2P files not found
Ensure your data directory has the structure:
```
TifStack_path/
├── suite2p/
│   └── plane0/
│       ├── F.npy
│       ├── Fneu.npy
│       ├── iscell.npy
│       └── stat.npy
└── imaging_processed/
    └── *.smr
```

### No cells detected
- Check `iscell.npy` from Suite2P
- Verify Suite2P classification threshold
- Check that imaging and timing files are synchronized

### Tuning fit fails
- Requires at least 8 orientation conditions
- Check that stimulus information is correct
- Verify response data is not all zeros/NaN

### Memory issues with large datasets
- Process FOVs one at a time
- Use batch processing script
- Save intermediate results to HDF5

## Tips

1. **Use raw strings** for Windows paths: `r'X:\Path\To\Data'`
2. **Check extraction** by calling `ce.print_summary()`
3. **Save often** to HDF5 to avoid re-processing
4. **Visualize early** to catch data issues
5. **Use notebooks** for interactive exploration
6. **Batch process** for analyzing multiple experiments

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## License

See LICENSE file for details.

## Citation

If you use this code, please cite:
```
[Add citation information]
```
