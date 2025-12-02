"""
Example Usage of FOV Configuration System

This script demonstrates how to use the FOV configuration and utility functions
in your analysis workflow.
"""

from fov_config_suite2p import fovs, FOV, populate_fov_from_stimulus
from fov_utils import (
    filter_fovs,
    print_fov_statistics,
    validate_all_fovs,
    get_unique_animals,
    get_unique_stim_types,
    export_fovs_to_json
)
from datetime import datetime


def example_1_basic_filtering():
    """Example 1: Basic filtering operations"""
    print("\n" + "="*70)
    print("Example 1: Basic Filtering")
    print("="*70)

    # Filter by stimulus type
    grating_fovs = filter_fovs(fovs, stim_type='grating')
    print(f"Found {len(grating_fovs)} grating experiments")

    # Filter by animal
    derrick_fovs = filter_fovs(fovs, animal_name='Derrick')
    print(f"Found {len(derrick_fovs)} recordings from Derrick")

    # Filter by brain region
    v1_fovs = filter_fovs(fovs, brain_region='v1')
    print(f"Found {len(v1_fovs)} V1 recordings")

    # Combined filtering
    v1_grating = filter_fovs(fovs, brain_region='v1', stim_type='grating')
    print(f"Found {len(v1_grating)} V1 grating recordings")


def example_2_date_filtering():
    """Example 2: Filter by date range"""
    print("\n" + "="*70)
    print("Example 2: Date Range Filtering")
    print("="*70)

    # Filter recordings from November 2025
    start_date = datetime(2025, 11, 1)
    end_date = datetime(2025, 11, 30)

    nov_fovs = filter_fovs(fovs, date_range=(start_date, end_date))
    print(f"Found {len(nov_fovs)} recordings from November 2025")

    for fov in nov_fovs:
        date_str = fov.recording_date.strftime('%Y-%m-%d') if fov.recording_date else 'N/A'
        print(f"  - {fov.animal_name} on {date_str}")


def example_3_dataset_statistics():
    """Example 3: Print dataset statistics"""
    print("\n" + "="*70)
    print("Example 3: Dataset Statistics")
    print("="*70)

    print_fov_statistics(fovs)

    # Get unique values
    animals = get_unique_animals(fovs)
    stim_types = get_unique_stim_types(fovs)

    print(f"Animals in dataset: {animals}")
    print(f"Stimulus types: {stim_types}")


def example_4_validation():
    """Example 4: Validate FOVs"""
    print("\n" + "="*70)
    print("Example 4: FOV Validation")
    print("="*70)

    validate_all_fovs(fovs, verbose=True)


def example_5_export():
    """Example 5: Export to JSON"""
    print("\n" + "="*70)
    print("Example 5: Export to JSON")
    print("="*70)

    # Export all FOVs
    output_file = 'fov_dataset.json'
    export_fovs_to_json(fovs, output_file)

    # Export only V1 recordings
    v1_fovs = filter_fovs(fovs, brain_region='v1')
    export_fovs_to_json(v1_fovs, 'v1_recordings.json')


def example_6_custom_filtering():
    """Example 6: Custom filtering with lambda functions"""
    print("\n" + "="*70)
    print("Example 6: Custom Filtering")
    print("="*70)

    # Filter high sample rate recordings (>10 Hz)
    high_rate = [f for f in fovs if f.sampRate and f.sampRate > 10]
    print(f"Found {len(high_rate)} high sample rate recordings (>10 Hz)")

    # Filter long stimulus presentations (>3 seconds)
    long_stim = [f for f in fovs if f.stim_dur and f.stim_dur > 3]
    print(f"Found {len(long_stim)} long stimulus presentations (>3s)")

    # Filter recordings with blank trials
    with_blank = [f for f in fovs if f.have_blank == 1]
    print(f"Found {len(with_blank)} recordings with blank trials")


def example_7_accessing_parameters():
    """Example 7: Accessing FOV parameters"""
    print("\n" + "="*70)
    print("Example 7: Accessing Parameters")
    print("="*70)

    if len(fovs) > 0:
        fov = fovs[0]
        print(f"Animal: {fov.animal_name}")
        print(f"Path: {fov.TifStack_path}")
        print(f"Recording date: {fov.recording_date}")
        print(f"Stimulus type: {fov.stim_type}")
        print(f"Sample rate: {fov.sampRate} Hz")
        print(f"Imaging files: {fov.ImagingFile}")
        print(f"Spike2 files: {fov.Spk2File}")
        print(f"EPI data: {fov.EPI_data}")


def example_8_manual_population():
    """Example 8: Manually populate a single FOV"""
    print("\n" + "="*70)
    print("Example 8: Manual Population")
    print("="*70)

    # Create a new FOV without adding it to the main list
    # Note: animal_name is auto-extracted from the path
    test_fov = FOV(
        TifStack_path=r'X:\Experimental_Data\BrainImaging\20251114_TestAnimal',
        ImagingFile=[0],
        Spk2File=[20],
        factor=2,
    )

    # Manually populate from stimulus file
    print("Before population:")
    print(f"  animal_name: {test_fov.animal_name}")
    print(f"  stim_dur: {test_fov.stim_dur}")
    print(f"  stim_type: {test_fov.stim_type}")

    # This would populate if the path existed
    # populate_fov_from_stimulus(test_fov)

    print("\nNote: Path doesn't exist, so population would fail gracefully")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("FOV Configuration System - Usage Examples")
    print("="*70)
    print(f"\nTotal FOVs loaded: {len(fovs)}")

    # Run examples
    example_1_basic_filtering()
    example_2_date_filtering()
    example_3_dataset_statistics()
    example_4_validation()
    # example_5_export()  # Commented out to avoid creating files
    example_6_custom_filtering()
    example_7_accessing_parameters()
    example_8_manual_population()

    print("\n" + "="*70)
    print("Examples completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
