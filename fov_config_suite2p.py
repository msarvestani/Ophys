"""
FOV Configuration for Suite2P Analysis

This module defines FOV (Field of View) parameters for 2-photon imaging experiments.
Parameters are partially entered manually and partially auto-populated from stimulus files.

Usage:
    1. Add FOV entries below with manual parameters
    2. Run this script: python fov_config_suite2p.py
    3. Auto-populated fields will be extracted from stimulus files
"""

from dataclasses import dataclass, field
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
        - animal_name: Animal identifier
        - fileImaging: List of imaging file indices
        - fileSpk: List of spike file indices

    Optional Manual Entries (have defaults):
        - factor: Frame averaging factor in ThorImage (default: 1)
        - brain_region: Brain region recorded (default: 'v1')
        - zoom: Microscope zoom level (default: 3)
        - And others...

    Auto-populated from stimulus file:
        - stim_dur: Stimulus duration
        - postPeriod: Post-stimulus period (matches ISI from stim file)
        - stim_type: Type of stimulus ('grating', 'image', etc.)
        - have_blank: Whether blank trials are included
        - recording_date: Extracted from folder name
    """

    # ========== MANUAL ENTRY REQUIRED ==========
    TifStack_path: str
    animal_name: str
    fileImaging: List[int]
    fileSpk: List[int]

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

    # ========== AUTO-POPULATED FROM STIMULUS FILE ==========
    sampRate: Optional[float] = None  # Calculated as 30/factor
    stim_dur: Optional[float] = None
    postPeriod: Optional[float] = None  # Matches isi from stimulus file
    stim_type: Optional[str] = None
    have_blank: Optional[int] = None
    recording_date: Optional[datetime] = None

    def __post_init__(self):
        """Calculate dependent fields after initialization"""
        # Calculate sampling rate from factor
        self.sampRate = 30 / self.factor


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

    # Find stimulus file
    stim_file = find_stimulus_file(path, stim_filename)

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

    return fov


def print_fov_summary(fov: FOV, index: int):
    """Print a summary of FOV parameters"""
    print(f"\n{'='*70}")
    print(f"FOV {index}: {fov.animal_name}")
    print(f"{'='*70}")
    print(f"  Path:             {fov.TifStack_path}")
    print(f"  Recording Date:   {fov.recording_date.strftime('%Y-%m-%d') if fov.recording_date else 'N/A'}")
    print(f"  File Imaging:     {fov.fileImaging}")
    print(f"  File Spk:         {fov.fileSpk}")
    print(f"  Brain Region:     {fov.brain_region}")
    print(f"  Factor:           {fov.factor}")
    print(f"  Sample Rate:      {fov.sampRate} Hz")
    print(f"  Stim Type:        {fov.stim_type or 'N/A'}")
    print(f"  Stim Duration:    {fov.stim_dur}s" if fov.stim_dur else "  Stim Duration:    N/A")
    print(f"  Pre Period:       {fov.prePeriod}s")
    print(f"  Post Period:      {fov.postPeriod}s" if fov.postPeriod else "  Post Period:      N/A")
    print(f"  Have Blank:       {fov.have_blank if fov.have_blank is not None else 'N/A'}")
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
    animal_name='Derrick',
    factor=4,
    fileImaging=[0],
    fileSpk=[16],
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
)
fovs.append(fov1)

# Add more FOVs here...
# fov2 = FOV(
#     TifStack_path=r'X:\Experimental_Data\BrainImaging\20251114_Animal2',
#     animal_name='Animal2',
#     factor=2,
#     fileImaging=[0, 1],
#     fileSpk=[10, 11],
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
        print(f"[{i}/{len(fovs)}] Processing: {fov.animal_name}")
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
