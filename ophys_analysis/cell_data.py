"""
Data structures for neural activity analysis.

This module defines classes to hold cell/ROI data from 2-photon imaging experiments.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path


class Cell:
    """
    Individual cell/ROI data from 2-photon imaging.

    Attributes:
        # Raw traces
        raw: Raw fluorescence trace
        dff: Delta F/F trace (baseline corrected)
        neu: Neuropil fluorescence
        spks: Spike inference from Suite2P

        # Trial-structured data
        cyc: Trial-organized data [n_stim, n_trials, n_timepoints]
        Fotrace: Baseline fluorescence traces [n_stim, n_trials, n_pre_timepoints]

        # Spatial information
        xPos: X position of ROI centroid (pixels)
        yPos: Y position of ROI centroid (pixels)
        mask: ROI mask coordinates [[x1, y1], [x2, y2], ...]

        # Timing information
        scanPeriod: Time between frames (seconds)
        rate: Sampling rate (Hz)
        scans: Frame indices
        scanTimes: Absolute time for each frame (seconds)

        # Registration/motion correction
        regoffsets_trialx: X motion offsets [n_stim, n_trials, n_timepoints]
        regoffsets_trialy: Y motion offsets [n_stim, n_trials, n_timepoints]

        # Response metrics (computed during extraction)
        trial_response: Mean response per trial [n_stim, n_trials]
        trial_pre_stim_response: Pre-stimulus baseline [n_stim, n_trials]
        trial_pre_stim_response_std: Std of pre-stim baseline
        condition_response: Mean response per condition [n_stim]
        condition_pre_stim_response: Mean pre-stim per condition [n_stim]
        condition_response_std: Std of condition response
        condition_pre_stim_response_std: Std of pre-stim response
        condition_baseline_plus_2SD: Significance threshold (baseline + 2*SD)
        condition_response_significance: Binary significance per condition [n_stim]
        ROI_responsiveness: Overall visual responsiveness flag

        # Metadata
        file: Source file name
        stimOn2pFrame: Stimulus onset frames
        uniqStims: Unique stimulus IDs
    """

    def __init__(self):
        # Raw traces
        self.raw: Optional[np.ndarray] = None
        self.dff: Optional[np.ndarray] = None
        self.neu: Optional[np.ndarray] = None
        self.spks: Optional[np.ndarray] = None

        # Trial-structured data
        self.cyc: Optional[np.ndarray] = None  # [n_stim, n_trials, n_timepoints]
        self.Fotrace: Optional[np.ndarray] = None

        # Spatial
        self.xPos: Optional[float] = None
        self.yPos: Optional[float] = None
        self.mask: Optional[np.ndarray] = None  # ROI mask coordinates [[y, x], ...]
        self.mask_2d: Optional[np.ndarray] = None  # Full 2D weighted mask array

        # Timing
        self.scanPeriod: Optional[float] = None
        self.rate: Optional[float] = None
        self.scans: Optional[np.ndarray] = None
        self.scanTimes: Optional[np.ndarray] = None

        # Registration
        self.regoffsets_trialx: Optional[np.ndarray] = None
        self.regoffsets_trialy: Optional[np.ndarray] = None

        # Response metrics
        self.trial_response: Optional[np.ndarray] = None
        self.trial_pre_stim_response: Optional[np.ndarray] = None
        self.trial_pre_stim_response_std: Optional[np.ndarray] = None
        self.condition_response: Optional[np.ndarray] = None
        self.condition_pre_stim_response: Optional[np.ndarray] = None
        self.condition_response_std: Optional[np.ndarray] = None
        self.condition_pre_stim_response_std: Optional[np.ndarray] = None
        self.condition_baseline_plus_2SD: Optional[np.ndarray] = None
        self.condition_response_significance: Optional[np.ndarray] = None
        self.ROI_responsiveness: bool = False

        # Metadata
        self.file: Optional[str] = None
        self.stimOn2pFrame: Optional[np.ndarray] = None
        self.uniqStims: Optional[np.ndarray] = None


class CellExtraction:
    """
    Collection of cells with acquisition metadata.

    This class holds all extracted cells from a single FOV along with
    timing, stimulus, and registration information.

    Attributes:
        cells: List of Cell objects
        twophotontimes: 2P frame timestamps from Spike2
        stimOn: Stimulus onset times from Spike2
        stimID: Stimulus ID for each presentation
        copyStimID: Copy of original stimID
        copyStimOn: Copy of original stimOn
        uniqStims: Unique stimulus IDs
        regOffsets: Global registration offsets [n_frames, 2]
        fov: Reference to FOV configuration object
        fov_index: Index of this FOV in the FOV list
    """

    def __init__(self, fov=None, fov_index: int = 0):
        self.cells: List[Cell] = []

        # Acquisition metadata
        self.twophotontimes: Optional[np.ndarray] = None
        self.stimOn: Optional[np.ndarray] = None
        self.stimID: Optional[np.ndarray] = None
        self.copyStimID: Optional[np.ndarray] = None
        self.copyStimOn: Optional[np.ndarray] = None
        self.uniqStims: Optional[np.ndarray] = None
        self.regOffsets: Optional[np.ndarray] = None

        # FOV reference
        self.fov = fov
        self.fov_index = fov_index

    def __len__(self) -> int:
        """Return number of cells"""
        return len(self.cells)

    def __getitem__(self, index: int) -> Cell:
        """Access cells by index"""
        return self.cells[index]

    def to_array(self, attr: str) -> np.ndarray:
        """
        Convert a cell attribute to numpy array across all cells.

        Args:
            attr: Attribute name (e.g., 'xPos', 'raw', 'ROI_responsiveness')

        Returns:
            Numpy array of attribute values for all cells

        Example:
            >>> ce = CellExtraction()
            >>> x_positions = ce.to_array('xPos')
            >>> responsive_flags = ce.to_array('ROI_responsiveness')
        """
        values = [getattr(cell, attr) for cell in self.cells]
        return np.array(values)

    def filter_cells(self, **kwargs) -> List[Cell]:
        """
        Filter cells based on attribute values.

        Args:
            **kwargs: Attribute name and value pairs for filtering

        Returns:
            List of Cell objects matching the criteria

        Example:
            >>> ce = CellExtraction()
            >>> responsive_cells = ce.filter_cells(ROI_responsiveness=True)
            >>> v1_cells = ce.filter_cells(brain_region='v1')  # if added later
        """
        filtered = self.cells
        for attr, value in kwargs.items():
            filtered = [c for c in filtered if getattr(c, attr, None) == value]
        return filtered

    def get_responsive_cells(self) -> List[Cell]:
        """
        Get all visually responsive cells.

        Returns:
            List of cells where ROI_responsiveness is True
        """
        return [cell for cell in self.cells if cell.ROI_responsiveness]

    def get_cell_indices(self, condition: callable) -> List[int]:
        """
        Get indices of cells matching a condition.

        Args:
            condition: Function that takes a Cell and returns bool

        Returns:
            List of cell indices

        Example:
            >>> ce = CellExtraction()
            >>> high_activity = ce.get_cell_indices(lambda c: np.mean(c.raw) > 100)
        """
        return [i for i, cell in enumerate(self.cells) if condition(cell)]

    def summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the extraction.

        Returns:
            Dictionary with summary information
        """
        n_cells = len(self.cells)
        n_responsive = sum(c.ROI_responsiveness for c in self.cells)

        summary_dict = {
            'n_cells': n_cells,
            'n_responsive': n_responsive,
            'fraction_responsive': n_responsive / n_cells if n_cells > 0 else 0,
        }

        if self.fov is not None:
            summary_dict['animal_name'] = self.fov.animal_name
            summary_dict['brain_region'] = self.fov.brain_region
            summary_dict['recording_date'] = self.fov.recording_date
            summary_dict['stim_type'] = self.fov.stim_type

        return summary_dict

    def print_summary(self):
        """Print a formatted summary of the extraction"""
        summary = self.summary()
        print("\n" + "="*70)
        print("Cell Extraction Summary")
        print("="*70)
        print(f"Total cells:          {summary['n_cells']}")
        print(f"Responsive cells:     {summary['n_responsive']}")
        print(f"Fraction responsive:  {summary['fraction_responsive']:.2%}")

        if 'animal_name' in summary:
            print(f"Animal:               {summary.get('animal_name', 'N/A')}")
            print(f"Brain region:         {summary.get('brain_region', 'N/A')}")
            print(f"Recording date:       {summary.get('recording_date', 'N/A')}")
            print(f"Stimulus type:        {summary.get('stim_type', 'N/A')}")
        print("="*70)
