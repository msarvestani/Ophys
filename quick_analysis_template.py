#!/usr/bin/env python
"""
Quick Analysis Template

Simple template for analyzing a single FOV. Just edit the paths below and run!

Usage:
    python quick_analysis_template.py
"""

from pathlib import Path
from fov_utils import create_fov_from_stimfile
from ophys_analysis import (
    extract_suite2p_traces,
    save_extraction_hdf5,
    create_full_analysis_report,
)

# ============================================================================
# EDIT THESE PATHS FOR YOUR ANALYSIS
# ============================================================================

# Path to your data directory (contains suite2p/, t00016/, etc.)
DATA_DIR = r'X:/Experimental_Data/BrainImaging/20251113_Derrick'

# Path to your stimulus file (or set to None to auto-detect)
# When None, it will search in Spk2 subdirectories (t00016/, t00017/, etc.)
STIM_FILE = None  # Auto-detect .py file in t00016/ etc.
# Or specify exact path:
# STIM_FILE = r'X:/Experimental_Data/BrainImaging/20251113_Derrick/t00016/driftinggrating_orientation.py'

# Output directory for results
OUTPUT_DIR = r'X:/Experimental_Data/BrainImaging/20251113_Derrick/analysis_results'

# Experimental parameters
IMAGING_FILES = [0]       # Which imaging file(s) to use
SPK2_FILES = [16]         # ACTUAL Spk2 directory number (for t00016/, use [16] not [0]!)
BRAIN_REGION = 'V1'       # Brain region
FACTOR = 1                # Downsampling factor

# ============================================================================
# ANALYSIS CODE (no need to edit below)
# ============================================================================

def main():
    print("="*70)
    print("QUICK FOV ANALYSIS")
    print("="*70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stimulus file will be auto-detected by create_fov_from_stimfile
    # It searches in Spk2 subdirectories (t00016/, etc.) automatically

    # Step 1: Configure FOV
    print("\n[1/4] Configuring FOV...")
    fov = create_fov_from_stimfile(
        stimfile=STIM_FILE,  # None = auto-detect in Spk2 subdirectories
        TifStack_path=DATA_DIR,
        ImagingFile=IMAGING_FILES,
        Spk2File=SPK2_FILES,
        factor=FACTOR,
        brain_region=BRAIN_REGION,
    )

    print(f"  Animal: {fov.animal_name}")
    print(f"  Recording date: {fov.recording_date}")
    print(f"  Stim type: {fov.stim_type}")

    # Step 2: Extract traces
    print("\n[2/4] Extracting Suite2P traces...")
    ce = extract_suite2p_traces(fov, fnum=0)
    print()
    ce.print_summary()

    # Step 3: Save to HDF5
    print("\n[3/4] Saving extraction results...")
    h5_file = output_path / 'extraction_results.h5'
    save_extraction_hdf5(ce, str(h5_file))

    # Step 4: Generate full analysis report
    print("\n[4/4] Generating analysis report...")
    create_full_analysis_report(ce, output_dir=str(output_path))

    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"  - extraction_results.h5")
    print(f"  - population_summary.png")
    print(f"  - orientation_maps.png")
    print(f"  - tuning_distributions.png")
    print(f"  - cell_*_tuning.png (first 10 responsive cells)")


if __name__ == '__main__':
    main()
