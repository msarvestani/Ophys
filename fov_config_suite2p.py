"""
FOV Configuration for Suite2P Analysis

This module defines FOV (Field of View) parameters for 2-photon imaging experiments.
Parameters are partially entered manually and partially auto-populated from stimulus files.

Usage:
    1. Add FOV entries below with manual parameters
    2. Run this script: python fov_config_suite2p.py
    3. Auto-populated fields will be extracted from stimulus files
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Union
import re
from datetime import datetime


@dataclass
class FOV:
    """
    Field of View parameters for 2P imaging experiments.

    Manual Entry Required:
        - TifStack_path: Path to imaging data directory
        - ImagingFile: List of imaging file indices
        - Spk2File: List of spike2 file indices

    Optional Manual Entries (have defaults):
        - factor: Frame averaging factor in ThorImage (default: 1)
        - brain_region: Brain region recorded (default: 'v1')
        - zoom: Microscope zoom level (default: 3)
        - EPI_data: EPI data flag (default: 0)
        - And others...

    Auto-populated from path and stimulus file:
        - animal_name: Extracted from folder name (e.g., '20251113_Derrick' -> 'Derrick')
        - recording_date: Extracted from folder name (YYYYMMDD pattern)
        - stim_dur: Stimulus duration
        - postPeriod: Post-stimulus period (matches ISI from stim file)
        - stim_type: Type of stimulus ('grating', 'image', etc.)
        - have_blank: Whether blank trials are included
    """

    # ========== MANUAL ENTRY REQUIRED ==========
    TifStack_path: str
    ImagingFile: List[int]
    Spk2File: List[int]

    # ========== OPTIONAL MANUAL ENTRIES (with defaults) ==========
    factor: int = 1  # Frame averaging factor in ThorImage
    fileImagingind: int = 1
    fly_back: int = 0
    brain_region: str = 'v1'
    zoom: int = 3
    piezo_planes: int = 0
    current_plane: int = 1
    two_chan: int = 0
    use_registered: int = 0  # Was the data externally registered?
    registration_type: str = 'Smooth'  # 'Smooth' or 'Ben'
    extraction: str = 'suite2p'  # 'miji' or 'suite2p'
    badTrials: List[int] = field(default_factory=lambda: [0])
    thresh: float = 0
    prePeriod: float = 0.5  # Fixed value
    EPI_data: int = 0  # EPI data flag

    # ========== AUTO-POPULATED FROM PATH AND STIMULUS FILE ==========
    animal_name: Optional[str] = None  # Extracted from folder name
    sampRate: Optional[float] = None  # Calculated as 30/factor
    stim_dur: Optional[float] = None
    postPeriod: Optional[float] = None  # Matches isi from stimulus file
    stim_type: Optional[str] = None
    have_blank: Optional[int] = None
    recording_date: Optional[datetime] = None
    stim_values: Optional[List[float]] = None  # Actual stimulus values (e.g., orientations)

    def __post_init__(self):
        """Calculate dependent fields after initialization"""
        # Calculate sampling rate from factor
        self.sampRate = 30 / self.factor


def read_stim_orientations(directory: Path) -> Optional[List[float]]:
    """
    Read stimulus orientation values from stimorientations.txt file.

    Args:
        directory: Directory containing the stimorientations.txt file

    Returns:
        List of orientation values or None if file not found
    """
    stim_file = directory / 'stimorientations.txt'
    if not stim_file.exists():
        return None

    try:
        with open(stim_file, 'r') as f:
            content = f.read().strip()

        # Parse values - could be space, comma, or newline separated
        # Handle formats like "0.  45.  90. 135." or "0, 45, 90, 135" or "0\n45\n90"
        import re
        # Split by any whitespace or comma
        values_str = re.split(r'[,\s]+', content)
        values = []
        for v in values_str:
            v = v.strip()
            if v:
                try:
                    values.append(float(v))
                except ValueError:
                    pass

        if values:
            return values
    except Exception as e:
        print(f"  Warning: Error reading stimorientations.txt: {e}")

    return None


def extract_params_from_stimulus_file(stim_file_path: Path) -> dict:
    """
    Extract parameters from a PsychoPy stimulus file.

    Extracts:
        - stimDuration -> stim_dur
        - isi -> postPeriod
        - doBlank -> have_blank
        - stim_type (detected from stimulus class usage)

    Args:
        stim_file_path: Path to the Python stimulus file

    Returns:
        Dictionary with extracted parameters
    """
    params = {}

    try:
        with open(stim_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return params

    # Extract stimDuration
    match = re.search(r'stimDuration\s*=\s*(\d+\.?\d*)', content)
    if match:
        params['stim_dur'] = float(match.group(1))

    # Extract isi (inter-stimulus interval) -> postPeriod
    match = re.search(r'isi\s*=\s*(\d+\.?\d*)', content)
    if match:
        params['postPeriod'] = float(match.group(1))

    # Extract doBlank -> have_blank
    match = re.search(r'doBlank\s*=\s*(\d+)', content)
    if match:
        params['have_blank'] = int(match.group(1))

    # Detect stimulus type from content
    # Look for stimulus description in comments or class instantiation
    if 'GratingStim' in content:
        # Check for more specific description in comments
        desc_match = re.search(r'[\'"].*?(grating|orientation|ori|tuning).*?[\'"]',
                               content, re.IGNORECASE)
        if desc_match:
            params['stim_type'] = 'grating'
        else:
            params['stim_type'] = 'grating'
    elif 'ImageStim' in content and 'GratingStim' not in content:
        params['stim_type'] = 'image'
    elif 'DotStim' in content or 'ElementArrayStim' in content:
        params['stim_type'] = 'dots'
    elif 'MovieStim' in content:
        params['stim_type'] = 'movie'
    else:
        params['stim_type'] = 'unknown'

    return params


def extract_date_from_path(path_str: str) -> Optional[datetime]:
    """
    Extract recording date from folder path containing YYYYMMDD pattern.

    Args:
        path_str: Path string like 'X:\\Experimental_Data\\20251113_Derrick\\'

    Returns:
        datetime object or None if no date found
    """
    # Look for YYYYMMDD pattern in path
    match = re.search(r'(\d{8})', path_str)
    if match:
        date_str = match.group(1)
        try:
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            print(f"  Warning: Found date string '{date_str}' but couldn't parse it")
            return None
    return None


def extract_animal_name_from_path(path_str: str) -> Optional[str]:
    """
    Extract animal name from folder path.

    Expects pattern like: '20251113_Derrick' or '20251113_Derrick_session1'
    Returns the part after the date and underscore.

    Args:
        path_str: Path string like 'X:\\Experimental_Data\\20251113_Derrick\\'

    Returns:
        Animal name string or None if no match found
    """
    # Look for pattern: YYYYMMDD_AnimalName
    # The animal name is everything after the date until the next path separator or end
    match = re.search(r'\d{8}_([^/\\]+)', path_str)
    if match:
        animal_name = match.group(1)
        # Remove any trailing path separators or whitespace
        animal_name = animal_name.rstrip('/\\').strip()
        return animal_name
    return None


def find_stimulus_file(directory: Path, stim_filename: Optional[str] = None) -> Optional[Path]:
    """
    Find stimulus file in directory.

    Args:
        directory: Directory to search
        stim_filename: Optional specific filename to look for

    Returns:
        Path to stimulus file or None
    """
    if stim_filename:
        stim_path = directory / stim_filename
        if stim_path.exists():
            return stim_path
        else:
            print(f"  Warning: Specified stimulus file not found: {stim_filename}")
            return None

    # Search for Python files that look like stimulus scripts
    # Common patterns: *stim*.py, *grating*.py, *visual*.py
    patterns = ['*stim*.py', '*grating*.py', '*visual*.py', '*ori*.py']

    for pattern in patterns:
        py_files = list(directory.glob(pattern))
        if py_files:
            return py_files[0]  # Return first match

    # If no pattern match, try any .py file
    py_files = list(directory.glob('*.py'))
    if py_files:
        print(f"  Warning: No obvious stimulus file found, using: {py_files[0].name}")
        return py_files[0]

    return None


def find_spk2_directory(base_path: Path, spk2_indices: List[int]) -> Optional[Path]:
    """
    Find the Spk2 subdirectory based on indices.

    Args:
        base_path: Base data directory
        spk2_indices: List of Spk2 file indices

    Returns:
        Path to Spk2 directory or None
    """
    for spk2_idx in spk2_indices:
        # Try common naming patterns
        possible_dirs = [
            base_path / f't{spk2_idx:05d}',  # t00016
            base_path / f't{spk2_idx:04d}',  # t0016
            base_path / f'spk2_{spk2_idx}',
            base_path / str(spk2_idx),
        ]
        for spk2_dir in possible_dirs:
            if spk2_dir.exists():
                return spk2_dir
    return None


def populate_fov_from_stimulus(fov: FOV, stim_filename: Optional[str] = None) -> FOV:
    """
    Auto-populate FOV parameters from stimulus file and path.

    Args:
        fov: FOV object with manual entries filled
        stim_filename: Optional specific stimulus filename to use

    Returns:
        FOV object with auto-populated fields
    """
    path = Path(fov.TifStack_path)

    # Extract animal name from path
    fov.animal_name = extract_animal_name_from_path(str(path))
    if fov.animal_name:
        print(f"  ✓ Extracted animal name: {fov.animal_name}")
    else:
        print(f"  ⚠ Could not extract animal name from path")

    # Extract recording date from path
    fov.recording_date = extract_date_from_path(str(path))
    if fov.recording_date:
        print(f"  ✓ Extracted date: {fov.recording_date.strftime('%Y-%m-%d')}")
    else:
        print(f"  ⚠ Could not extract date from path")

    # Check if directory exists
    if not path.exists():
        print(f"  ⚠ Warning: Directory does not exist: {path}")
        return fov

    # Find Spk2 directory for stimulus files
    spk2_dir = find_spk2_directory(path, fov.Spk2File)

    # Find stimulus file (check Spk2 dir first, then main path)
    stim_file = None
    search_dir = spk2_dir if spk2_dir else path

    if stim_filename:
        # If filename provided, search for it
        stim_file = find_stimulus_file(search_dir, stim_filename)
        if stim_file is None and spk2_dir:
            stim_file = find_stimulus_file(path, stim_filename)
    else:
        stim_file = find_stimulus_file(search_dir)
        if stim_file is None and spk2_dir:
            stim_file = find_stimulus_file(path)

    if stim_file is None:
        print(f"  ⚠ Warning: No stimulus file found in {path}")
        return fov

    print(f"  → Parsing stimulus file: {stim_file.name}")

    # Extract parameters from stimulus file
    try:
        params = extract_params_from_stimulus_file(stim_file)

        # Update FOV with extracted parameters
        updates = []
        if 'stim_dur' in params:
            fov.stim_dur = params['stim_dur']
            updates.append(f"stim_dur={params['stim_dur']}s")
        if 'postPeriod' in params:
            fov.postPeriod = params['postPeriod']
            updates.append(f"postPeriod={params['postPeriod']}s")
        if 'stim_type' in params:
            fov.stim_type = params['stim_type']
            updates.append(f"stim_type='{params['stim_type']}'")
        if 'have_blank' in params:
            fov.have_blank = params['have_blank']
            updates.append(f"have_blank={params['have_blank']}")

        if updates:
            print(f"  ✓ Populated: {', '.join(updates)}")
        else:
            print(f"  ⚠ No parameters extracted from stimulus file")

    except Exception as e:
        print(f"  ✗ Error parsing stimulus file: {e}")

    # Read stimulus orientation values from stimorientations.txt
    # Check in same directory as stimulus file first, then Spk2 dir
    stim_values = read_stim_orientations(stim_file.parent)
    if stim_values is None and spk2_dir and stim_file.parent != spk2_dir:
        stim_values = read_stim_orientations(spk2_dir)

    if stim_values:
        fov.stim_values = stim_values
        print(f"  ✓ Loaded stim_values: {stim_values}")
    else:
        print(f"  ⚠ No stimorientations.txt found - stim_values not set")

    return fov


def export_fov_to_dict(fov: FOV) -> dict:
    """
    Export FOV object to dictionary for saving to HDF5.

    Args:
        fov: FOV object

    Returns:
        Dictionary with FOV parameters as JSON-serializable types
    """
    fov_dict = {}

    # Get all fields from the dataclass
    for field_info in fields(fov):
        field_name = field_info.name
        value = getattr(fov, field_name)

        # Convert to JSON-serializable types
        if value is None:
            continue
        elif isinstance(value, (str, int, float, bool)):
            fov_dict[field_name] = value
        elif isinstance(value, list):
            # Convert lists to comma-separated string
            fov_dict[field_name] = str(value)
        elif isinstance(value, datetime):
            # Convert datetime to ISO string
            fov_dict[field_name] = value.isoformat()
        else:
            # Convert anything else to string
            fov_dict[field_name] = str(value)

    return fov_dict


def print_fov_summary(fov: FOV, index: int):
    """Print a summary of FOV parameters"""
    print(f"\n{'='*70}")
    print(f"FOV {index}: {fov.animal_name or 'Unknown'}")
    print(f"{'='*70}")
    print(f"  Path:             {fov.TifStack_path}")
    print(f"  Animal Name:      {fov.animal_name or 'N/A'}")
    print(f"  Recording Date:   {fov.recording_date.strftime('%Y-%m-%d') if fov.recording_date else 'N/A'}")
    print(f"  Imaging File:     {fov.ImagingFile}")
    print(f"  Spk2 File:        {fov.Spk2File}")
    print(f"  Brain Region:     {fov.brain_region}")
    print(f"  Factor:           {fov.factor}")
    print(f"  Sample Rate:      {fov.sampRate} Hz")
    print(f"  EPI Data:         {fov.EPI_data}")
    print(f"  Stim Type:        {fov.stim_type or 'N/A'}")
    print(f"  Stim Duration:    {fov.stim_dur}s" if fov.stim_dur else "  Stim Duration:    N/A")
    print(f"  Pre Period:       {fov.prePeriod}s")
    print(f"  Post Period:      {fov.postPeriod}s" if fov.postPeriod else "  Post Period:      N/A")
    print(f"  Have Blank:       {fov.have_blank if fov.have_blank is not None else 'N/A'}")
    if fov.stim_values:
        print(f"  Stim Values:      {fov.stim_values}")
    else:
        print(f"  Stim Values:      N/A")
    print(f"  Extraction:       {fov.extraction}")
    print(f"  Registration:     {fov.registration_type}")


# ============================================================================
# DEFINE FOVs BELOW
# ============================================================================
# Add your FOV entries here. Only manual fields need to be specified.
# Auto-populated fields will be filled when you run this script.

fovs: List[FOV] = []

# Example FOV 1
fov1 = FOV(
    TifStack_path=r'X:\Experimental_Data\BrainImaging\20251113_Derrick',
    ImagingFile=[0],
    Spk2File=[16],
    factor=4,
    fileImagingind=1,
    brain_region='v1',
    zoom=3,
    piezo_planes=0,
    current_plane=1,
    two_chan=0,
    use_registered=0,
    registration_type='Smooth',
    extraction='suite2p',
    badTrials=[0],
    thresh=0,
    EPI_data=0,
)
fovs.append(fov1)

# Add more FOVs here...
# fov2 = FOV(
#     TifStack_path=r'X:\Experimental_Data\BrainImaging\20251114_Animal2',
#     ImagingFile=[0, 1],
#     Spk2File=[10, 11],
#     factor=2,
#     brain_region='v1',
# )
# fovs.append(fov2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution: auto-populate FOVs and print summary"""
    print("\n" + "="*70)
    print("FOV Configuration - Auto-Population from Stimulus Files")
    print("="*70)
    print(f"\nProcessing {len(fovs)} FOV(s)...\n")

    # Auto-populate each FOV
    for i, fov in enumerate(fovs, 1):
        print(f"[{i}/{len(fovs)}] Processing FOV from: {fov.TifStack_path}")
        print("-" * 70)
        populate_fov_from_stimulus(fov)
        print()

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for i, fov in enumerate(fovs, 1):
        print_fov_summary(fov, i)

    print("\n" + "="*70)
    print(f"✓ Successfully processed {len(fovs)} FOV(s)")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
