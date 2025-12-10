"""
Trace extraction from Suite2P output.

This module converts Suite2P ROI traces into trial-structured data
organized by stimulus conditions.
"""

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import medfilt
from typing import Optional, Tuple
import warnings

from .cell_data import Cell, CellExtraction


def filter_baseline_dff(raw: np.ndarray, pt: int = 99) -> np.ndarray:
    """
    Remove slow baseline from raw F traces using median filtering.

    Process:
    1. Apply median filter for low-pass trace
    2. Calculate dF/F: (raw - baseline) / baseline

    Args:
        raw: Raw fluorescence trace
        pt: Median filter window size (default: 99)

    Returns:
        Baseline-corrected dF/F trace
    """
    raw_new = raw.copy()

    # Apply median filter to get baseline
    raw_new = medfilt(raw_new, kernel_size=pt)

    # Calculate dF/F
    raw_new = (raw - raw_new) / raw_new

    return raw_new


def format_spike2_dir(file_num: int) -> str:
    """
    Format Spike2 directory name with proper zero padding.

    Args:
        file_num: File number

    Returns:
        Formatted directory name (e.g., 't00016' for file_num=16)
    """
    return f't{file_num:05d}'


def find_spike2_dir(base_path: Path, file_num: int) -> Path:
    """
    Get Spike2 directory path from file number.

    Args:
        base_path: Base directory containing Spk2 subdirectories
        file_num: Actual Spk2 file/directory number (e.g., 16 for t00016/)

    Returns:
        Path to Spike2 directory

    Raises:
        FileNotFoundError: If the specified directory doesn't exist
    """
    spk2_dir = base_path / format_spike2_dir(file_num)

    if not spk2_dir.exists():
        raise FileNotFoundError(
            f"Spike2 directory not found: {spk2_dir}\n"
            f"  Spk2File parameter: [{file_num}]\n"
            f"  Expected directory: {format_spike2_dir(file_num)}\n"
            f"\n"
            f"  Make sure Spk2File contains the ACTUAL directory number.\n"
            f"  For directory 't00016/', use Spk2File=[16]\n"
            f"  For directory 't00145/', use Spk2File=[145]\n"
            f"  Check your data directory for the correct t* folder name."
        )

    return spk2_dir


def load_spike2_data(spk2_dir: Path, factor: int, n_frames: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load timing and stimulus information from Spike2 files.

    Args:
        spk2_dir: Path to Spike2 directory
        factor: Frame averaging factor
        n_frames: Number of frames in Suite2P output (for validation/alignment)

    Returns:
        Tuple of (twophotontimes, stimOn, stimID)
    """
    # Load 2P frame times
    twophotontimes_path = spk2_dir / 'twophotontimes.txt'
    twophotontimes = np.loadtxt(twophotontimes_path)

    # Only downsample if we need to match Suite2P frame count
    # Suite2P outputs at the same rate as its input (after any frame averaging in ThorImage)
    # If n_frames is provided, use it to determine the correct downsampling
    if n_frames is not None:
        # Calculate what factor would give us the right number of frames
        actual_factor = len(twophotontimes) // n_frames
        if actual_factor > 1 and abs(len(twophotontimes) // actual_factor - n_frames) <= 2:
            # Use the calculated factor that matches Suite2P output
            twophotontimes = twophotontimes[::actual_factor]
            if actual_factor != factor:
                print(f"  Note: Adjusted frame alignment factor from {factor} to {actual_factor} "
                      f"to match Suite2P frames ({n_frames})")
        elif factor > 1:
            # Fall back to specified factor if it gives reasonable match
            downsampled_len = len(twophotontimes) // factor
            if abs(downsampled_len - n_frames) <= n_frames * 0.1:  # Within 10%
                twophotontimes = twophotontimes[::factor]
            else:
                # Lengths don't match - don't downsample
                print(f"  Warning: Frame count mismatch - twophotontimes has {len(twophotontimes)} entries, "
                      f"Suite2P has {n_frames} frames. Not downsampling twophotontimes.")
    elif factor > 1:
        # No n_frames provided, use specified factor (legacy behavior)
        twophotontimes = twophotontimes[::factor]

    # Load stimulus timing
    stim_path = spk2_dir / 'stimontimes.txt'
    S = np.loadtxt(stim_path)

    # Parse stimulus data: alternating stimID and stimOn times
    # File format: [stimID_0, stimOn_0, stimID_1, stimOn_1, ...]
    stimID = S[0::2]  # Indices 0, 2, 4, ... (stimulus IDs)
    stimOn = S[1::2]  # Indices 1, 3, 5, ... (stimulus onset times)

    # Remove first stimID if it's 0 (initialization artifact)
    if len(stimID) > 0 and stimID[0] == 0:
        stimOn = np.delete(stimOn, 0)
        stimID = np.delete(stimID, 0)

    # Ensure stimIDs are 1-indexed for proper trial organization
    # If min stimID is 0, shift all IDs up by 1
    if len(stimID) > 0 and np.min(stimID) == 0:
        stimID = stimID + 1

    return twophotontimes, stimOn, stimID


def load_suite2p_data(suite2p_path: Path) -> dict:
    """
    Load Suite2P output from Fall.mat file.

    Args:
        suite2p_path: Path to Fall.mat file

    Returns:
        Dictionary with Suite2P data (iscell, F, Fneu, spks, stat)
    """
    print(f"  Loading Suite2P data from: {suite2p_path}")

    # Load the .mat file
    data = loadmat(str(suite2p_path))

    # Extract key variables
    suite2p_data = {
        'iscell': data['iscell'],
        'F': data['F'],
        'Fneu': data['Fneu'],
        'spks': data['spks'],
        'stat': data['stat'].flatten(),  # MATLAB cell array becomes object array
    }

    return suite2p_data


def load_registration_offsets(reg_path: Path, n_frames: int) -> np.ndarray:
    """
    Load registration offset information.

    Args:
        reg_path: Path to regOffsets.mat file
        n_frames: Number of frames (for creating zeros if file doesn't exist)

    Returns:
        Registration offsets array [n_frames, 2]
    """
    try:
        data = loadmat(str(reg_path))
        regOffsets = data['regOffsets']
    except (FileNotFoundError, KeyError):
        # If file doesn't exist, return zeros
        regOffsets = np.zeros((n_frames, 2))

    return regOffsets


def convert_stim_times_to_frames(stimOn: np.ndarray,
                                   twophotontimes: np.ndarray) -> np.ndarray:
    """
    Convert stimulus onset times to 2P frame indices.

    Args:
        stimOn: Stimulus onset times (seconds)
        twophotontimes: 2P frame timestamps (seconds)

    Returns:
        Frame indices for each stimulus onset
    """
    numStims = len(stimOn)
    stimOn2pFrame = np.zeros(numStims, dtype=int)

    for ii in range(numStims):
        # Find frames before and after stimulus onset
        id1_mask = stimOn[ii] < twophotontimes
        id2_mask = stimOn[ii] > twophotontimes

        if np.any(id1_mask):
            id1 = np.where(id1_mask)[0][0]  # First frame after stim
        else:
            id1 = None

        if np.any(id2_mask):
            id2 = np.where(id2_mask)[0][-1]  # Last frame before stim
        else:
            id2 = None

        # Use the frame before stim onset
        if id2 is not None:
            stimOn2pFrame[ii] = id2
        elif id1 is not None and len(np.diff(id1)) > 0:
            stimOn2pFrame[ii] = np.where(np.diff(id1) == 1)[0][0]
        else:
            stimOn2pFrame[ii] = 0

    return stimOn2pFrame


def organize_into_trials(cell: Cell,
                          stimID: np.ndarray,
                          stimOn2pFrame: np.ndarray,
                          uniqStims: np.ndarray,
                          prePeriod2: int,
                          stimDur2: int,
                          postPeriod2: int,
                          regOffsets: np.ndarray,
                          ntrials: int) -> None:
    """
    Organize continuous traces into trial structure.

    Modifies cell in place, adding:
    - cyc: Trial-organized dF/F [n_stim, n_trials, n_timepoints]
    - Fotrace: Pre-stimulus baseline traces
    - regoffsets_trialx/y: Motion offsets per trial

    Args:
        cell: Cell object to populate
        stimID: Stimulus ID for each presentation
        stimOn2pFrame: Frame index of each stimulus onset
        uniqStims: Unique stimulus IDs
        prePeriod2: Pre-stimulus frames
        stimDur2: Stimulus duration frames
        postPeriod2: Post-stimulus frames
        regOffsets: Registration offsets [n_frames, 2]
        ntrials: Number of trials per condition
    """
    n_stims = len(uniqStims)
    n_timepoints = stimDur2 + postPeriod2

    # Initialize arrays
    # Note: prePeriod2 + 1 because arange includes endpoint (prestimTime goes to stimOn + 1)
    cell.cyc = np.zeros((n_stims, ntrials, n_timepoints))
    cell.Fotrace = np.zeros((n_stims, ntrials, prePeriod2 + 1))
    cell.regoffsets_trialx = np.zeros((n_stims, ntrials, n_timepoints))
    cell.regoffsets_trialy = np.zeros((n_stims, ntrials, n_timepoints))

    # Track trials per condition
    trialList = np.zeros(n_stims, dtype=int)

    numStims = ntrials * n_stims

    for ii in range(numStims):  # Process ALL stimulus presentations, not numStims-1
        # Define time windows
        prestimTime = np.arange(stimOn2pFrame[ii] - prePeriod2, stimOn2pFrame[ii] + 1)
        stimTime = np.arange(stimOn2pFrame[ii] + 1,
                             stimOn2pFrame[ii] + 1 + stimDur2 + postPeriod2)

        # Check bounds
        prestimTime = prestimTime[prestimTime > 0]
        if len(stimTime) > 0 and np.max(stimTime) >= len(cell.raw):
            # Truncate if exceeds trace length
            valid_idx = stimTime < len(cell.raw)
            stimTime = stimTime[valid_idx]

        if len(stimTime) == 0:
            continue

        # Extract traces
        Ftrace = cell.raw[stimTime]
        Fo = np.nanmedian(cell.raw[prestimTime])
        Fo_trace = cell.raw[prestimTime]

        # Calculate dF/F
        if Fo == 0:
            dFtrace = np.full(Ftrace.shape, np.nan)
            dFo_trace = np.full(Fo_trace.shape, np.nan)
        else:
            dFtrace = (Ftrace - Fo) / Fo
            dFo_trace = (Fo_trace - Fo) / Fo

        # Determine condition index
        if n_stims == 1:
            ind = 0
        else:
            ind = int(stimID[ii]) - 1  # Convert to 0-indexed

        # Store in trial structure
        if ind < n_stims:
            trial_num = trialList[ind]
            if trial_num < ntrials:
                # Store dF/F trace
                cell.cyc[ind, trial_num, :len(dFtrace)] = dFtrace

                # Store baseline trace
                cell.Fotrace[ind, trial_num, :len(dFo_trace)] = dFo_trace

                # Store motion offsets
                if len(stimTime) <= n_timepoints:
                    cell.regoffsets_trialx[ind, trial_num, :len(stimTime)] = regOffsets[stimTime, 0]
                    cell.regoffsets_trialy[ind, trial_num, :len(stimTime)] = regOffsets[stimTime, 1]

                trialList[ind] += 1

    # Calculate response metrics
    cell.trial_response = np.nanmean(cell.cyc, axis=2)
    cell.trial_pre_stim_response = np.nanmean(cell.Fotrace, axis=2)
    cell.trial_pre_stim_response_std = np.nanstd(cell.Fotrace, axis=2)

    # Condition-averaged responses
    cell.condition_response = np.nanmean(cell.trial_response, axis=1)
    cell.condition_pre_stim_response = np.nanmean(cell.trial_pre_stim_response, axis=1)
    cell.condition_response_std = np.nanstd(cell.trial_response, axis=1)
    cell.condition_pre_stim_response_std = np.nanstd(cell.trial_pre_stim_response, axis=1)

    # Response significance threshold (2 SD above baseline)
    cell.condition_baseline_plus_2SD = (
        cell.condition_pre_stim_response + 2 * cell.condition_pre_stim_response_std
    )

    # Determine if responsive for each condition
    cell.condition_response_significance = (
        cell.condition_response > cell.condition_baseline_plus_2SD
    ).astype(int)

    # Overall responsiveness: significant response to any condition
    cell.ROI_responsiveness = bool(np.any(cell.condition_response_significance))


def extract_suite2p_traces(fov, fnum: int = 0, save_dir: Optional[Path] = None) -> CellExtraction:
    """
    Extract and organize Suite2P traces into trial structure.

    This is the main function that converts the MATLAB extraction_2025_suite2p.m

    Args:
        fov: FOV object with experiment parameters
        fnum: Index of which imaging/spike file pair to use (default: 0)
        save_dir: Optional directory to save output

    Returns:
        CellExtraction object containing all cells and metadata

    Example:
        >>> from fov_config_suite2p import fovs
        >>> from ophys_analysis import extract_suite2p_traces
        >>> ce = extract_suite2p_traces(fovs[0])
        >>> ce.print_summary()
    """
    print("\n" + "="*70)
    print("Extracting Suite2P Traces")
    print("="*70)

    # Extract FOV parameters
    TwoPhoton_path = Path(fov.TifStack_path)
    file2p = fov.ImagingFile[fnum]
    fileSpk = fov.Spk2File[fnum]
    factor = fov.factor
    fileImagingind = fov.fileImagingind

    prePeriod = fov.prePeriod
    stimDur = fov.stim_dur
    postPeriod = fov.postPeriod

    print(f"  Animal: {fov.animal_name}")
    print(f"  Path: {TwoPhoton_path}")
    print(f"  Imaging file: {file2p}, Spike2 file: {fileSpk}")

    # Set up directory paths
    spk2_dir = find_spike2_dir(TwoPhoton_path, fileSpk)
    folder_dir = TwoPhoton_path
    name = str(file2p)

    # Load Suite2P data FIRST to get frame count for alignment
    suite2p_path = TwoPhoton_path / f't{file2p}' / 'suite2p' / 'plane0' / 'Fall.mat'
    suite2p_data = load_suite2p_data(suite2p_path)
    n_frames = suite2p_data['F'].shape[1]  # Number of frames in Suite2P output
    print(f"  Suite2P frame count: {n_frames}")

    # Load Spike2 timing data (with frame count for proper alignment)
    print("\n  Loading Spike2 timing data...")
    twophotontimes, stimOn, stimID = load_spike2_data(spk2_dir, factor, n_frames=n_frames)

    uniqStims = np.unique(stimID)
    print(f"  Loaded {len(uniqStims)} unique stimulus codes")
    print(f"  Loaded {len(stimOn)} stimulus presentations")
    print(f"  Aligned twophotontimes: {len(twophotontimes)} entries")

    # Validate frame alignment
    if abs(len(twophotontimes) - n_frames) > 2:
        warnings.warn(
            f"Frame count mismatch after alignment: twophotontimes has {len(twophotontimes)} entries "
            f"but Suite2P has {n_frames} frames. Trial cutting may be inaccurate."
        )

    # Load registration offsets
    reg_path = TwoPhoton_path / f't{file2p}' / 'Registered_TempMod' / 'regOffsets.mat'
    regOffsets = load_registration_offsets(reg_path, len(twophotontimes))

    # Initialize CellExtraction object
    ce = CellExtraction(fov=fov, fov_index=fnum)
    ce.twophotontimes = twophotontimes
    ce.stimOn = stimOn
    ce.stimID = stimID
    ce.copyStimID = stimID.copy()
    ce.copyStimOn = stimOn.copy()
    ce.uniqStims = uniqStims
    ce.regOffsets = regOffsets

    # Get cell indices (iscell[:, 0] == 1)
    cell_inds = np.where(suite2p_data['iscell'][:, 0] == 1)[0]
    numROIs = len(cell_inds)
    print(f"\n  Processing {numROIs} ROIs...")

    # Process each ROI
    for i, ind in enumerate(cell_inds):
        if (i + 1) % 50 == 0:
            print(f"    Processing ROI {i+1}/{numROIs}")

        cell = Cell()

        # Extract ROI position and mask
        stat = suite2p_data['stat'][ind]

        try:
            # Properly flatten nested MATLAB arrays - handle multiple nesting levels
            ypix_raw = stat['ypix'][0]
            xpix_raw = stat['xpix'][0]
            lam_raw = stat['lam'][0]

            # Recursively flatten until we get 1D arrays
            while isinstance(ypix_raw, np.ndarray) and ypix_raw.dtype == object:
                ypix_raw = ypix_raw[0]
            while isinstance(xpix_raw, np.ndarray) and xpix_raw.dtype == object:
                xpix_raw = xpix_raw[0]
            while isinstance(lam_raw, np.ndarray) and lam_raw.dtype == object:
                lam_raw = lam_raw[0]

            ypix = np.asarray(ypix_raw).flatten().astype(int)
            xpix = np.asarray(xpix_raw).flatten().astype(int)
            lam = np.asarray(lam_raw).flatten()

            # Set positions as scalars from unwrapped data
            cell.yPos = float(np.median(ypix))
            cell.xPos = float(np.median(xpix))

            # Create mask - note: numpy indexing is [row, col] = [y, x]
            mask = np.zeros((512, 512))
            valid_idx = (xpix < 512) & (ypix < 512) & (xpix >= 0) & (ypix >= 0)
            mask[ypix[valid_idx], xpix[valid_idx]] = lam[valid_idx]

            mask_coords = np.argwhere(mask > 0)
            cell.mask = mask_coords
            cell.mask_2d = mask  # Store full 2D weighted mask
        except Exception as e:
            print(f"    Warning: Could not extract mask for ROI {i}: {e}")
            # Fallback: set default position and mask
            cell.yPos = 256.0
            cell.xPos = 256.0
            cell.mask = np.array([[cell.xPos, cell.yPos]])

        # Extract traces
        cell.file = name
        cell.raw = suite2p_data['F'][ind, :].flatten()
        cell.neu = suite2p_data['Fneu'][ind, :].flatten()
        cell.spks = suite2p_data['spks'][ind, :].flatten()

        # Calculate timing
        cell.scanPeriod = np.mean(np.diff(twophotontimes))
        cell.rate = 1 / cell.scanPeriod
        cell.scans = np.arange(len(cell.raw))
        cell.scanTimes = cell.scans * cell.scanPeriod + np.min(twophotontimes)

        # Calculate dF/F using baseline filter
        cell.dff = filter_baseline_dff(cell.raw)

        ce.cells.append(cell)

    # Convert timing parameters to frames
    scanPeriod = ce.cells[0].scanPeriod
    prePeriod2 = int(np.ceil(prePeriod / scanPeriod))
    stimDur2 = int(np.ceil(stimDur / scanPeriod))
    postPeriod2 = int(np.ceil(postPeriod / scanPeriod))

    # Remove blank stimulus if present (stimID == 0 or max stimID)
    if np.sum(uniqStims == 0) == 1:
        uniqStims = uniqStims[uniqStims != 0]

    ntrials = int(np.floor(len(stimOn) / len(uniqStims)))

    # Convert stimulus times to frame indices
    numStims = ntrials * len(uniqStims)
    stimOn2pFrame = convert_stim_times_to_frames(stimOn, twophotontimes)

    print(f"\n  Organizing into trials...")
    print(f"    Stimuli: {len(uniqStims)}, Trials per stimulus: {ntrials}")
    print(f"    Pre-period: {prePeriod2} frames, Stim: {stimDur2} frames, Post: {postPeriod2} frames")

    # Organize each cell into trial structure
    for i, cell in enumerate(ce.cells):
        cell.stimOn2pFrame = stimOn2pFrame
        cell.uniqStims = uniqStims

        organize_into_trials(
            cell, stimID, stimOn2pFrame, uniqStims,
            prePeriod2, stimDur2, postPeriod2,
            regOffsets, ntrials
        )

    # Create save directory
    if save_dir is None:
        save_dir = folder_dir / f'{file2p}_suite2p'
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n  ✓ Extraction complete!")
    ce.print_summary()

    return ce
