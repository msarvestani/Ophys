#!/usr/bin/env python
"""
Batch processing script for multiple FOVs.

This script processes multiple field-of-views (FOVs) in batch mode,
performing extraction, tuning analysis, and visualization for each.

Usage:
    python batch_process_fovs.py --config batch_config.py
    python batch_process_fovs.py --input_dir /path/to/data --pattern "*/visual_stim.py"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional
import json
import time
from datetime import datetime

import numpy as np

from fov_config_suite2p import FOV
from fov_utils import create_fov_from_stimfile, export_fov_to_dict
from ophys_analysis import (
    extract_suite2p_traces,
    save_extraction_hdf5,
    create_full_analysis_report,
    get_tuning_madineh,
)


def find_stimulus_files(base_dir: Path, pattern: str = "*/visual_stim.py") -> List[Path]:
    """
    Find all stimulus files matching pattern.

    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for stimulus files

    Returns:
        List of stimulus file paths
    """
    stim_files = list(base_dir.glob(pattern))
    return sorted(stim_files)


def create_fov_config(stim_file: Path,
                       imaging_indices: List[int] = [0],
                       spk2_indices: List[int] = [0],
                       factor: int = 1,
                       brain_region: str = 'V1',
                       layer: Optional[str] = None) -> FOV:
    """
    Create FOV configuration from stimulus file.

    Args:
        stim_file: Path to stimulus file
        imaging_indices: Imaging file indices
        spk2_indices: Spike2 file indices
        factor: Downsampling factor
        brain_region: Brain region
        layer: Cortical layer

    Returns:
        FOV object
    """
    data_dir = stim_file.parent

    fov = create_fov_from_stimfile(
        stimfile=str(stim_file),
        TifStack_path=str(data_dir),
        ImagingFile=imaging_indices,
        Spk2File=spk2_indices,
    )

    fov.factor = factor
    fov.brain_region = brain_region
    if layer:
        fov.layer = layer

    return fov


def process_single_fov(stim_file: Path,
                        output_dir: Path,
                        fov_params: Optional[Dict] = None,
                        save_plots: bool = True) -> Dict:
    """
    Process a single FOV.

    Args:
        stim_file: Path to stimulus file
        output_dir: Output directory for results
        fov_params: Optional FOV parameters
        save_plots: Whether to save analysis plots

    Returns:
        Dictionary with processing results and statistics
    """
    start_time = time.time()

    # Create output directory
    fov_output_dir = output_dir / stim_file.parent.name
    fov_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {stim_file.parent.name}")
    print("=" * 60)

    try:
        # Create FOV configuration
        if fov_params is None:
            fov_params = {}

        fov = create_fov_config(stim_file, **fov_params)

        # Save FOV configuration
        fov_dict = export_fov_to_dict(fov)
        with open(fov_output_dir / 'fov_config.json', 'w') as f:
            json.dump(fov_dict, f, indent=2, default=str)

        # Extract traces
        print("  Extracting Suite2P traces...")
        ce = extract_suite2p_traces(fov, fnum=0)

        # Save extraction results
        print("  Saving extraction to HDF5...")
        h5_file = fov_output_dir / 'extraction_results.h5'
        save_extraction_hdf5(ce, str(h5_file))

        # Calculate statistics
        n_cells = len(ce.cells)
        n_responsive = sum(c.ROI_responsiveness for c in ce.cells)

        # Calculate tuning metrics for responsive cells
        tuning_metrics = []
        for i, cell in enumerate(ce.cells):
            if cell.ROI_responsiveness:
                n_dirs = len(cell.uniqStims) - 1
                if n_dirs > 0:
                    stimInfo = np.arange(0, 360, 360/n_dirs)
                    try:
                        tuning, _, _ = get_tuning_madineh(cell.condition_response[:n_dirs], stimInfo)
                        tuning_metrics.append({
                            'cell_id': i,
                            'pref_ort': tuning['pref_ort_fit'],
                            'pref_dir': tuning['pref_dir_fit'],
                            'oti': tuning['oti_fit'],
                            'dti': tuning['dti_fit'],
                            'bandwidth': tuning['fit_bandwidth'],
                            'fit_r': tuning['fit_r'],
                        })
                    except Exception as e:
                        print(f"    Warning: Could not analyze cell {i}: {e}")

        # Save tuning metrics
        import pandas as pd
        if tuning_metrics:
            df = pd.DataFrame(tuning_metrics)
            df.to_csv(fov_output_dir / 'tuning_metrics.csv', index=False)

        # Generate plots
        if save_plots:
            print("  Generating analysis plots...")
            plots_dir = fov_output_dir / 'plots'
            create_full_analysis_report(ce, output_dir=str(plots_dir))

        # Calculate summary statistics
        stats = {
            'fov_name': stim_file.parent.name,
            'animal_name': fov.animal_name,
            'recording_date': fov.recording_date,
            'brain_region': fov.brain_region,
            'layer': fov.layer,
            'n_cells': n_cells,
            'n_responsive': n_responsive,
            'pct_responsive': 100 * n_responsive / n_cells if n_cells > 0 else 0,
            'n_tuned': len(tuning_metrics),
            'mean_oti': np.mean([m['oti'] for m in tuning_metrics]) if tuning_metrics else None,
            'mean_dti': np.mean([m['dti'] for m in tuning_metrics]) if tuning_metrics else None,
            'processing_time': time.time() - start_time,
            'status': 'success',
        }

        print(f"  ✓ Completed: {n_responsive}/{n_cells} responsive cells")
        print(f"  Processing time: {stats['processing_time']:.1f}s")

        return stats

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return {
            'fov_name': stim_file.parent.name,
            'status': 'failed',
            'error': str(e),
            'processing_time': time.time() - start_time,
        }


def batch_process(input_dir: Path,
                   output_dir: Path,
                   pattern: str = "*/visual_stim.py",
                   fov_params: Optional[Dict] = None,
                   save_plots: bool = True) -> List[Dict]:
    """
    Batch process multiple FOVs.

    Args:
        input_dir: Input directory containing FOVs
        output_dir: Output directory for results
        pattern: Glob pattern for stimulus files
        fov_params: Optional FOV parameters
        save_plots: Whether to save analysis plots

    Returns:
        List of processing statistics for each FOV
    """
    # Find all stimulus files
    stim_files = find_stimulus_files(input_dir, pattern)

    if not stim_files:
        print(f"No stimulus files found matching pattern: {pattern}")
        return []

    print(f"Found {len(stim_files)} FOVs to process")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Process each FOV
    all_stats = []
    for i, stim_file in enumerate(stim_files, 1):
        print(f"\nFOV {i}/{len(stim_files)}")
        stats = process_single_fov(stim_file, output_dir, fov_params, save_plots)
        all_stats.append(stats)

    # Save summary
    import pandas as pd
    summary_df = pd.DataFrame(all_stats)
    summary_file = output_dir / 'batch_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 60)

    n_success = sum(s['status'] == 'success' for s in all_stats)
    n_failed = len(all_stats) - n_success

    print(f"Total FOVs: {len(all_stats)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")

    if n_success > 0:
        successful_stats = [s for s in all_stats if s['status'] == 'success']
        print(f"\nTotal cells: {sum(s['n_cells'] for s in successful_stats)}")
        print(f"Total responsive: {sum(s['n_responsive'] for s in successful_stats)}")
        print(f"Mean responsiveness: {np.mean([s['pct_responsive'] for s in successful_stats]):.1f}%")
        print(f"Total processing time: {sum(s['processing_time'] for s in all_stats):.1f}s")

    print(f"\nSummary saved to: {summary_file}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description='Batch process 2-photon imaging FOVs')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing FOVs')
    parser.add_argument('--output_dir', type=str, default='batch_results',
                        help='Output directory for results')
    parser.add_argument('--pattern', type=str, default='*/visual_stim.py',
                        help='Glob pattern for stimulus files')
    parser.add_argument('--brain_region', type=str, default='V1',
                        help='Brain region')
    parser.add_argument('--layer', type=str, default=None,
                        help='Cortical layer')
    parser.add_argument('--factor', type=int, default=1,
                        help='Downsampling factor')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip plot generation')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # FOV parameters
    fov_params = {
        'brain_region': args.brain_region,
        'layer': args.layer,
        'factor': args.factor,
    }

    # Run batch processing
    batch_process(
        input_dir=input_dir,
        output_dir=output_dir,
        pattern=args.pattern,
        fov_params=fov_params,
        save_plots=not args.no_plots,
    )


if __name__ == '__main__':
    main()
