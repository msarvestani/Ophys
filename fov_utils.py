"""
Utility functions for working with FOV configurations.

This module provides helper functions for loading, filtering, and
working with FOV (Field of View) objects.
"""

from pathlib import Path
from typing import List, Optional, Callable
from datetime import datetime, timedelta
import json

# Import FOV class and functions (assumes fov_config_suite2p is in same directory)
try:
    from fov_config_suite2p import FOV, populate_fov_from_stimulus
except ImportError:
    print("Warning: Could not import FOV class. Make sure fov_config_suite2p.py is in the same directory.")


def find_stim_file_in_spk2_dirs(data_dir: Path, spk2_indices: List[int]) -> Optional[Path]:
    """
    Find stimulus file in Spk2 subdirectories.

    The typical structure is:
    data_dir/
        t00016/  (Spk2File[0] might be file #16)
            driftinggrating_orientation.py
            spike2data files...
        t00017/  (Spk2File[1] might be file #17)
            ...

    Args:
        data_dir: Path to main data directory
        spk2_indices: List of Spk2 file indices (e.g., [0] or [16])
                     If indices are small (< 100), will also search all t* directories

    Returns:
        Path to stimulus file or None if not found
    """
    data_path = Path(data_dir)

    # Search in each specified Spk2 directory
    for spk2_idx in spk2_indices:
        # Try common naming patterns for Spk2 directories
        possible_dirs = [
            data_path / f't{spk2_idx:05d}',  # t00016
            data_path / f't{spk2_idx:04d}',  # t0016
            data_path / f'spk2_{spk2_idx}',
            data_path / str(spk2_idx),
        ]

        for spk2_dir in possible_dirs:
            if spk2_dir.exists():
                # Search for .py files in this directory
                py_files = list(spk2_dir.glob('*.py'))
                if py_files:
                    # Prefer files with common stimulus keywords
                    for pattern in ['*grating*.py', '*orientation*.py', '*stim*.py', '*visual*.py']:
                        matches = list(spk2_dir.glob(pattern))
                        if matches:
                            return matches[0]
                    # Otherwise return first .py file
                    return py_files[0]

    # If indices are small (likely list indices, not file numbers), search all t* directories
    if max(spk2_indices) < 100:
        # Find all directories matching t[0-9]+ pattern
        for item in data_path.iterdir():
            if item.is_dir() and item.name.startswith('t') and item.name[1:].isdigit():
                py_files = list(item.glob('*.py'))
                if py_files:
                    # Prefer files with common stimulus keywords
                    for pattern in ['*grating*.py', '*orientation*.py', '*stim*.py', '*visual*.py']:
                        matches = list(item.glob(pattern))
                        if matches:
                            return matches[0]
                    # Otherwise return first .py file
                    return py_files[0]

    # If not found in Spk2 dirs, try main directory
    py_files = list(data_path.glob('*.py'))
    if py_files:
        for pattern in ['*grating*.py', '*orientation*.py', '*stim*.py', '*visual*.py']:
            matches = list(data_path.glob(pattern))
            if matches:
                return matches[0]
        return py_files[0]

    return None


def create_fov_from_stimfile(stimfile: Optional[str],
                              TifStack_path: str,
                              ImagingFile: List[int],
                              Spk2File: List[int],
                              **kwargs) -> FOV:
    """
    Create and populate a FOV from stimulus file in one step.

    This is a convenience function that creates a FOV object and
    automatically populates it from the stimulus file and path.

    If stimfile is None, it will automatically search for the stimulus file
    in the Spk2File subdirectories (e.g., t00016/) and then the main directory.

    Args:
        stimfile: Path to stimulus file (or None to auto-detect)
                 Can be absolute, relative to TifStack_path, or just filename
        TifStack_path: Path to imaging data directory
        ImagingFile: List of imaging file indices
        Spk2File: List of Spike2 file indices
        **kwargs: Additional FOV parameters (factor, brain_region, etc.)

    Returns:
        FOV object with auto-populated fields

    Example:
        >>> # Auto-detect stimulus file
        >>> fov = create_fov_from_stimfile(
        ...     stimfile=None,  # Will search in t00016/ etc.
        ...     TifStack_path='X:/Data/20251113_Derrick',
        ...     ImagingFile=[0],
        ...     Spk2File=[0],
        ...     factor=1,
        ...     brain_region='V1'
        ... )

        >>> # Or specify exact path
        >>> fov = create_fov_from_stimfile(
        ...     stimfile='X:/Data/20251113_Derrick/t00016/driftinggrating_orientation.py',
        ...     TifStack_path='X:/Data/20251113_Derrick',
        ...     ImagingFile=[0],
        ...     Spk2File=[0]
        ... )
    """
    # Create FOV with required and optional parameters
    fov = FOV(
        TifStack_path=TifStack_path,
        ImagingFile=ImagingFile,
        Spk2File=Spk2File,
        **kwargs
    )

    # Determine stimulus filename
    stim_filename = None

    if stimfile is None:
        # Auto-detect in Spk2 subdirectories
        stim_path = find_stim_file_in_spk2_dirs(Path(TifStack_path), Spk2File)
        if stim_path is None:
            raise FileNotFoundError(
                f"Could not find stimulus file (.py) in Spk2 subdirectories.\n"
                f"  Searched in: {TifStack_path}\n"
                f"  Looking for subdirectories: t{Spk2File[0]:05d}, t{Spk2File[0]:04d}, etc.\n"
                f"  Please either:\n"
                f"    1. Specify stimfile parameter explicitly, or\n"
                f"    2. Ensure .py file exists in the Spk2 subdirectory"
            )
        # Get relative path from TifStack_path
        try:
            stim_filename = str(stim_path.relative_to(Path(TifStack_path)))
        except ValueError:
            # If not relative, use absolute path
            stim_filename = str(stim_path)
        print(f"  Auto-detected stimulus file: {stim_filename}")
    else:
        stim_path = Path(stimfile)
        if stim_path.is_absolute():
            # Get relative path from TifStack_path or use filename
            try:
                stim_filename = str(stim_path.relative_to(Path(TifStack_path)))
            except ValueError:
                stim_filename = stim_path.name
        else:
            # Use as-is if relative or just filename
            stim_filename = str(stim_path)

    # Populate from stimulus file
    fov = populate_fov_from_stimulus(fov, stim_filename=stim_filename)

    return fov


def filter_fovs(fovs: List[FOV],
                animal_name: Optional[str] = None,
                brain_region: Optional[str] = None,
                stim_type: Optional[str] = None,
                date_range: Optional[tuple] = None) -> List[FOV]:
    """
    Filter FOVs based on criteria.

    Args:
        fovs: List of FOV objects
        animal_name: Filter by animal name (case-insensitive partial match)
        brain_region: Filter by brain region (case-insensitive)
        stim_type: Filter by stimulus type (case-insensitive)
        date_range: Tuple of (start_date, end_date) as datetime objects

    Returns:
        List of filtered FOV objects

    Example:
        >>> filtered = filter_fovs(fovs, animal_name='Derrick', stim_type='grating')
    """
    filtered = fovs

    if animal_name:
        filtered = [f for f in filtered
                   if f.animal_name and animal_name.lower() in f.animal_name.lower()]

    if brain_region:
        filtered = [f for f in filtered
                   if f.brain_region.lower() == brain_region.lower()]

    if stim_type:
        filtered = [f for f in filtered
                   if f.stim_type and f.stim_type.lower() == stim_type.lower()]

    if date_range:
        start_date, end_date = date_range
        filtered = [f for f in filtered
                   if f.recording_date and start_date <= f.recording_date <= end_date]

    return filtered


def get_fov_by_animal(fovs: List[FOV], animal_name: str) -> List[FOV]:
    """
    Get all FOVs for a specific animal.

    Args:
        fovs: List of FOV objects
        animal_name: Animal name to search for

    Returns:
        List of FOV objects for that animal
    """
    return [f for f in fovs if f.animal_name and f.animal_name == animal_name]


def get_fov_by_date(fovs: List[FOV], date: datetime) -> List[FOV]:
    """
    Get all FOVs recorded on a specific date.

    Args:
        fovs: List of FOV objects
        date: Date to search for

    Returns:
        List of FOV objects from that date
    """
    return [f for f in fovs
            if f.recording_date and f.recording_date.date() == date.date()]


def get_unique_animals(fovs: List[FOV]) -> List[str]:
    """Get list of unique animal names in FOV list"""
    return sorted(list(set(f.animal_name for f in fovs if f.animal_name)))


def get_unique_brain_regions(fovs: List[FOV]) -> List[str]:
    """Get list of unique brain regions in FOV list"""
    return sorted(list(set(f.brain_region for f in fovs)))


def get_unique_stim_types(fovs: List[FOV]) -> List[str]:
    """Get list of unique stimulus types in FOV list"""
    stim_types = [f.stim_type for f in fovs if f.stim_type]
    return sorted(list(set(stim_types)))


def get_date_range(fovs: List[FOV]) -> tuple:
    """
    Get the date range of recordings in FOV list.

    Returns:
        Tuple of (earliest_date, latest_date) or (None, None) if no dates
    """
    dates = [f.recording_date for f in fovs if f.recording_date]
    if not dates:
        return (None, None)
    return (min(dates), max(dates))


def print_fov_statistics(fovs: List[FOV]):
    """
    Print summary statistics about a list of FOVs.

    Args:
        fovs: List of FOV objects
    """
    if not fovs:
        print("No FOVs to analyze")
        return

    print("\n" + "="*70)
    print("FOV Dataset Statistics")
    print("="*70)
    print(f"Total FOVs:        {len(fovs)}")
    print(f"Unique Animals:    {len(get_unique_animals(fovs))}")
    print(f"  Animals:         {', '.join(get_unique_animals(fovs))}")
    print(f"Brain Regions:     {', '.join(get_unique_brain_regions(fovs))}")
    print(f"Stimulus Types:    {', '.join(get_unique_stim_types(fovs))}")

    date_range = get_date_range(fovs)
    if date_range[0]:
        print(f"Date Range:        {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    else:
        print(f"Date Range:        N/A")

    # Sample rate distribution
    sample_rates = [f.sampRate for f in fovs if f.sampRate]
    if sample_rates:
        print(f"Sample Rates:      {set(sample_rates)} Hz")

    # Extraction methods
    extraction_methods = [f.extraction for f in fovs]
    print(f"Extraction:        {set(extraction_methods)}")

    print("="*70 + "\n")


def validate_fov(fov: FOV) -> tuple:
    """
    Validate FOV object and return status.

    Args:
        fov: FOV object to validate

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []

    # Check if path exists
    path = Path(fov.TifStack_path)
    if not path.exists():
        warnings.append(f"Path does not exist: {fov.TifStack_path}")

    # Check if critical fields are populated
    if fov.animal_name is None:
        warnings.append("animal_name not populated")
    if fov.stim_dur is None:
        warnings.append("stim_dur not populated")
    if fov.postPeriod is None:
        warnings.append("postPeriod not populated")
    if fov.stim_type is None:
        warnings.append("stim_type not populated")
    if fov.recording_date is None:
        warnings.append("recording_date not populated")

    # Check for reasonable values
    if fov.sampRate and (fov.sampRate < 1 or fov.sampRate > 100):
        warnings.append(f"Unusual sample rate: {fov.sampRate} Hz")
    if fov.stim_dur and (fov.stim_dur < 0.1 or fov.stim_dur > 60):
        warnings.append(f"Unusual stimulus duration: {fov.stim_dur}s")

    is_valid = len(warnings) == 0
    return (is_valid, warnings)


def validate_all_fovs(fovs: List[FOV], verbose: bool = True):
    """
    Validate all FOVs in list and print report.

    Args:
        fovs: List of FOV objects
        verbose: If True, print detailed warnings
    """
    print("\n" + "="*70)
    print("FOV Validation Report")
    print("="*70)

    valid_count = 0
    for i, fov in enumerate(fovs, 1):
        is_valid, warnings = validate_fov(fov)
        animal_display = fov.animal_name if fov.animal_name else 'Unknown'
        if is_valid:
            valid_count += 1
            if verbose:
                print(f"✓ FOV {i} ({animal_display}): Valid")
        else:
            print(f"✗ FOV {i} ({animal_display}): {len(warnings)} warning(s)")
            if verbose:
                for warning in warnings:
                    print(f"    - {warning}")

    print("-"*70)
    print(f"Valid: {valid_count}/{len(fovs)}")
    print("="*70 + "\n")


def export_fov_to_dict(fov: FOV) -> dict:
    """
    Convert FOV object to dictionary for serialization.

    Args:
        fov: FOV object

    Returns:
        Dictionary representation of FOV
    """
    fov_dict = {
        'animal_name': fov.animal_name,
        'TifStack_path': fov.TifStack_path,
        'ImagingFile': fov.ImagingFile,
        'Spk2File': fov.Spk2File,
        'factor': fov.factor,
        'sampRate': fov.sampRate,
        'fileImagingind': fov.fileImagingind,
        'stim_dur': fov.stim_dur,
        'prePeriod': fov.prePeriod,
        'postPeriod': fov.postPeriod,
        'stim_type': fov.stim_type,
        'fly_back': fov.fly_back,
        'brain_region': fov.brain_region,
        'recording_date': fov.recording_date.isoformat() if fov.recording_date else None,
        'zoom': fov.zoom,
        'piezo_planes': fov.piezo_planes,
        'current_plane': fov.current_plane,
        'two_chan': fov.two_chan,
        'use_registered': fov.use_registered,
        'registration_type': fov.registration_type,
        'extraction': fov.extraction,
        'badTrials': fov.badTrials,
        'thresh': fov.thresh,
        'have_blank': fov.have_blank,
        'EPI_data': fov.EPI_data,
    }
    return fov_dict


def export_fovs_to_json(fovs: List[FOV], output_path: str):
    """
    Export FOVs to JSON file.

    Args:
        fovs: List of FOV objects
        output_path: Path to output JSON file
    """
    fov_dicts = [export_fov_to_dict(fov) for fov in fovs]

    with open(output_path, 'w') as f:
        json.dump(fov_dicts, f, indent=2)

    print(f"✓ Exported {len(fovs)} FOVs to {output_path}")


def create_analysis_summary(fovs: List[FOV]) -> str:
    """
    Create a formatted summary string for analysis notebooks/reports.

    Args:
        fovs: List of FOV objects

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("="*70)
    lines.append("Dataset Summary")
    lines.append("="*70)
    lines.append(f"Total recordings: {len(fovs)}")
    lines.append(f"Animals: {', '.join(get_unique_animals(fovs))}")
    lines.append(f"Brain regions: {', '.join(get_unique_brain_regions(fovs))}")
    lines.append(f"Stimulus types: {', '.join(get_unique_stim_types(fovs))}")

    date_range = get_date_range(fovs)
    if date_range[0]:
        lines.append(f"Recording period: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")

    lines.append("="*70)

    return "\n".join(lines)


# Example usage
if __name__ == '__main__':
    print("FOV Utilities Module")
    print("This module provides utility functions for working with FOV objects.")
    print("\nExample usage:")
    print("  from fov_utils import filter_fovs, print_fov_statistics")
    print("  from fov_config_suite2p import fovs")
    print("  ")
    print("  # Filter FOVs")
    print("  grating_fovs = filter_fovs(fovs, stim_type='grating')")
    print("  ")
    print("  # Print statistics")
    print("  print_fov_statistics(fovs)")
    print("  ")
    print("  # Validate FOVs")
    print("  validate_all_fovs(fovs)")
