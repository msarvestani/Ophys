# Ophys - 2-Photon Imaging Analysis

Various analysis scripts for 2P (optical physiology) data, including configuration management and Suite2P integration.

## Overview

This repository contains Python-based tools for managing and analyzing 2-photon imaging experiments. The system replaces MATLAB-based FOV structure files with a more flexible Python implementation that auto-extracts parameters from stimulus files.

## Structure

```
Ophys/
├── fov_config_suite2p.py   # Main configuration file (define FOVs here)
├── fov_utils.py             # Utility functions for working with FOVs
└── README.md                # This file
```

## Features

- **Auto-population**: Extracts parameters from PsychoPy stimulus files automatically
- **Type-safe**: Uses Python dataclasses for clear structure and validation
- **Cross-platform**: Works on Windows, Mac, and Linux
- **Extensible**: Easy to add new FOVs and parameters
- **Utilities**: Helper functions for filtering, validating, and analyzing FOV datasets

## Quick Start

### 1. Define FOVs

Edit `fov_config_suite2p.py` and add your FOV entries:

```python
fov1 = FOV(
    TifStack_path=r'X:\Experimental_Data\BrainImaging\20251113_Derrick',
    animal_name='Derrick',
    factor=4,
    fileImaging=[0],
    fileSpk=[16],
    brain_region='v1',
)
fovs.append(fov1)
```

### 2. Auto-populate from Stimulus Files

Run the configuration script:

```bash
python fov_config_suite2p.py
```

This will:
- Extract recording dates from folder names (e.g., `20251113_Derrick` → 2025-11-13)
- Find PsychoPy stimulus files in each directory
- Extract parameters: `stimDuration`, `isi`, `doBlank`, stimulus type
- Print a summary of all FOVs

### 3. Use in Analysis Scripts

```python
from fov_config_suite2p import fovs, FOV
from fov_utils import filter_fovs, print_fov_statistics

# Get all grating experiments
grating_fovs = filter_fovs(fovs, stim_type='grating')

# Get specific animal
derrick_fovs = filter_fovs(fovs, animal_name='Derrick')

# Print statistics
print_fov_statistics(fovs)
```

## FOV Parameters

### Manual Entry (Required)
- `TifStack_path`: Path to imaging data directory
- `animal_name`: Animal identifier
- `fileImaging`: List of imaging file indices (e.g., `[0, 1, 2]`)
- `fileSpk`: List of spike2 file indices (e.g., `[16, 17]`)

### Manual Entry (Optional with Defaults)
- `factor`: Frame averaging factor (default: 1)
- `brain_region`: Brain region recorded (default: 'v1')
- `zoom`: Microscope zoom level (default: 3)
- `piezo_planes`: Number of piezo planes (default: 0)
- `thresh`: Threshold value (default: 0)
- `prePeriod`: Pre-stimulus period in seconds (default: 0.5)
- `extraction`: Extraction method (default: 'suite2p')
- `registration_type`: Registration type (default: 'Smooth')
- `badTrials`: List of bad trial indices (default: [0])

### Auto-populated
These are extracted automatically when you run the script:
- `recording_date`: Extracted from folder name (YYYYMMDD pattern)
- `stim_dur`: Stimulus duration from stimulus file
- `postPeriod`: Post-stimulus period (matches `isi` in stimulus file)
- `stim_type`: Type of stimulus ('grating', 'image', 'dots', etc.)
- `have_blank`: Whether blank trials are included (from `doBlank`)
- `sampRate`: Calculated as 30/factor

## Utility Functions

The `fov_utils.py` module provides helpful functions:

### Filtering
```python
from fov_utils import filter_fovs

# Filter by multiple criteria
filtered = filter_fovs(fovs,
                       animal_name='Derrick',
                       brain_region='v1',
                       stim_type='grating')
```

### Statistics
```python
from fov_utils import print_fov_statistics, get_unique_animals

# Print dataset summary
print_fov_statistics(fovs)

# Get unique values
animals = get_unique_animals(fovs)
```

### Validation
```python
from fov_utils import validate_all_fovs

# Check all FOVs for issues
validate_all_fovs(fovs, verbose=True)
```

### Export
```python
from fov_utils import export_fovs_to_json

# Export to JSON for sharing/archiving
export_fovs_to_json(fovs, 'fov_dataset.json')
```

## Stimulus File Parsing

The system automatically extracts parameters from PsychoPy stimulus files:

**Extracted Parameters:**
- `stimDuration` → `fov.stim_dur`
- `isi` → `fov.postPeriod`
- `doBlank` → `fov.have_blank`
- Stimulus type detected from: `GratingStim`, `ImageStim`, `DotStim`, etc.

**File Discovery:**
The system searches for stimulus files using these patterns:
1. `*stim*.py`
2. `*grating*.py`
3. `*visual*.py`
4. `*ori*.py`
5. Any `.py` file (fallback)

## Example Output

```
======================================================================
FOV Configuration - Auto-Population from Stimulus Files
======================================================================
Processing 1 FOV(s)...

[1/1] Processing: Derrick
----------------------------------------------------------------------
  ✓ Extracted date: 2025-11-13
  → Parsing stimulus file: grating_stimulus.py
  ✓ Populated: stim_dur=4.0s, postPeriod=2.0s, stim_type='grating', have_blank=1

======================================================================
SUMMARY
======================================================================

======================================================================
FOV 1: Derrick
======================================================================
  Path:             X:\Experimental_Data\BrainImaging\20251113_Derrick
  Recording Date:   2025-11-13
  File Imaging:     [0]
  File Spk:         [16]
  Brain Region:     v1
  Factor:           4
  Sample Rate:      7.5 Hz
  Stim Type:        grating
  Stim Duration:    4.0s
  Pre Period:       0.5s
  Post Period:      2.0s
  Have Blank:       1
  Extraction:       suite2p
  Registration:     Smooth
```

## Migration from MATLAB

### Old (MATLAB)
```matlab
num=1;
FOV(num).TifStack_path ='X:\Experimental_Data\BrainImaging\20251113_Derrick\';
FOV(num).animal_name = 'Derrick'
FOV(num).factor = 4;
FOV(num).fileImaging=[0];
FOV(num).fileSpk=[16];
FOV(num).stim_dur = 4;
% ... many more manual entries
```

### New (Python)
```python
fov1 = FOV(
    TifStack_path=r'X:\Experimental_Data\BrainImaging\20251113_Derrick',
    animal_name='Derrick',
    factor=4,
    fileImaging=[0],
    fileSpk=[16],
    # stim_dur, postPeriod, stim_type, etc. auto-populated!
)
fovs.append(fov1)
```

## Requirements

```
Python 3.7+
No external dependencies for configuration (uses only standard library)
```

## Tips

1. **Use raw strings** for Windows paths: `r'X:\Path\To\Data'`
2. **Run validation** after adding new FOVs: `validate_all_fovs(fovs)`
3. **Check extraction** by running the script and reviewing the summary
4. **Filter efficiently** using the utility functions instead of manual loops
5. **Export to JSON** for sharing datasets with collaborators
