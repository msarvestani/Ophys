#!/usr/bin/env python
"""
Single FOV Analysis Script

Analyzes a single field-of-view from start to finish:
1. Configure FOV from stimulus file
2. Extract Suite2P traces
3. Generate full analysis report

Usage:
    python analyze_single_fov.py --data_dir /path/to/data --output_dir results
    python analyze_single_fov.py --data_dir X:/Madineh/Calcium_Imaging/20251113_Derrick \
                                  --output_dir analysis_results \
                                  --brain_region V1 \
                                  --factor 1
"""

import argparse
import sys
from pathlib import Path

from fov_utils import create_fov_from_stimfile
from ophys_analysis import (
    extract_suite2p_traces,
    save_extraction_hdf5,
    create_full_analysis_report,
)


def analyze_fov(data_dir: Path,
                output_dir: Path,
                stim_file: str = None,
                imaging_files: list = [0],
                spk2_files: list = [0],
                brain_region: str = 'V1',
                factor: int = 1,
                save_h5: bool = True,
                fit_r_threshold: float = None):
    """
    Analyze a single FOV from start to finish.

    Args:
        data_dir: Path to data directory
        output_dir: Path to output directory
        stim_file: Path to stimulus file (auto-detected if None)
        imaging_files: List of imaging file indices
        spk2_files: List of Spike2 file indices
        brain_region: Brain region
        factor: Downsampling factor
        save_h5: Whether to save HDF5 file
        fit_r_threshold: If specified, only include cells with Gaussian fit r >= this value
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SINGLE FOV ANALYSIS")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Step 1: Stimulus file detection
    if stim_file is None:
        print("Step 1: Auto-detecting stimulus file in Spk2 subdirectories...")
    else:
        stim_file = Path(stim_file)
        print(f"Step 1: Using stimulus file: {stim_file}")

    # Step 2: Configure FOV
    print("\nStep 2: Configuring FOV...")
    try:
        fov = create_fov_from_stimfile(
            stimfile=str(stim_file) if stim_file else None,
            TifStack_path=str(data_dir),
            ImagingFile=imaging_files,
            Spk2File=spk2_files,
            factor=factor,
            brain_region=brain_region,
        )

        print(f"  Animal: {fov.animal_name}")
        print(f"  Recording date: {fov.recording_date}")
        print(f"  Brain region: {fov.brain_region}")
        print(f"  Stim type: {fov.stim_type}")
        print(f"  Stim duration: {fov.stim_dur}s")
    except Exception as e:
        print(f"  Error configuring FOV: {e}")
        sys.exit(1)

    # Step 3: Extract traces
    print("\nStep 3: Extracting Suite2P traces...")
    try:
        ce = extract_suite2p_traces(fov, fnum=0)
        print()
        ce.print_summary()
    except Exception as e:
        print(f"  Error during extraction: {e}")
        sys.exit(1)

    # Step 4: Save HDF5
    if save_h5:
        print("\nStep 4: Saving extraction results...")
        h5_file = output_dir / 'extraction_results.h5'
        save_extraction_hdf5(ce, str(h5_file))

    # Step 5: Generate full analysis report
    print("\nStep 5: Generating analysis report...")
    try:
        # Save in subfolder named after ImagingFile (e.g., t0, t1, etc.)
        imaging_file_num = fov.ImagingFile[0] if isinstance(fov.ImagingFile, list) else fov.ImagingFile
        report_dir = output_dir / f"t{imaging_file_num}"
        report_dir.mkdir(parents=True, exist_ok=True)
        create_full_analysis_report(ce, output_dir=str(report_dir), fit_r_threshold=fit_r_threshold)
        print(f"\n{'='*70}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*70}")
        print(f"Extraction saved to: {output_dir}")
        print(f"  - extraction_results.h5")
        print(f"Analysis report saved to: {report_dir}")
        print(f"  - population_summary.png")
        print(f"  - orientation_maps.png")
        print(f"  - tuning_distributions.png")
        print(f"  - cell_*_tuning.png (for responsive cells)")
    except Exception as e:
        print(f"  Error generating report: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a single FOV from Suite2P output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (auto-detect stimulus file)
  python analyze_single_fov.py --data_dir X:/Madineh/Calcium_Imaging/20251113_Derrick

  # With custom parameters
  python analyze_single_fov.py \\
      --data_dir X:/Madineh/Calcium_Imaging/20251113_Derrick \\
      --output_dir my_results \\
      --brain_region V1 \\
      --layer L2/3 \\
      --factor 1

  # Specify stimulus file explicitly
  python analyze_single_fov.py \\
      --data_dir X:/Data/20251113_Derrick \\
      --stim_file X:/Data/20251113_Derrick/visual_stim.py
        """
    )

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory (contains suite2p/ and imaging_processed/)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir/analysis_results)')
    parser.add_argument('--stim_file', type=str, default=None,
                        help='Path to stimulus file (auto-detected if not provided)')
    parser.add_argument('--imaging_files', type=int, nargs='+', default=[0],
                        help='Imaging file indices (default: [0])')
    parser.add_argument('--spk2_files', type=int, nargs='+', default=[0],
                        help='Spike2 file indices (default: [0])')
    parser.add_argument('--brain_region', type=str, default='V1',
                        help='Brain region (default: V1)')
    parser.add_argument('--factor', type=int, default=1,
                        help='Downsampling factor (default: 1)')
    parser.add_argument('--no_save_h5', action='store_true',
                        help='Skip saving HDF5 file')
    parser.add_argument('--fit_r_threshold', type=float, default=None,
                        help='Only include cells with Gaussian fit r >= this value (e.g., 0.9)')

    args = parser.parse_args()

    # Set default output directory
    if args.output_dir is None:
        args.output_dir = Path(args.data_dir) / 'analysis_results'

    # Run analysis
    analyze_fov(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        stim_file=args.stim_file,
        imaging_files=args.imaging_files,
        spk2_files=args.spk2_files,
        brain_region=args.brain_region,
        factor=args.factor,
        save_h5=not args.no_save_h5,
        fit_r_threshold=args.fit_r_threshold,
    )


if __name__ == '__main__':
    main()
