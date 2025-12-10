"""
Ophys Analysis Package

Python tools for 2-photon imaging data analysis, including Suite2P trace
extraction, trial organization, and orientation tuning analysis.
"""

from .cell_data import Cell, CellExtraction
from .trace_extraction import extract_suite2p_traces
from .tuning_analysis import get_tuning_madineh, double_gauss, fit_tuning_direction
from .io_utils import save_extraction_hdf5, load_extraction_hdf5
from .plotting import (
    plot_cell_tuning_curve,
    plot_orientation_map,
    plot_tuning_distributions,
    plot_population_summary,
    create_full_analysis_report,
    get_well_fit_cells,
)

__version__ = '0.1.0'
__all__ = [
    'Cell',
    'CellExtraction',
    'extract_suite2p_traces',
    'get_tuning_madineh',
    'double_gauss',
    'fit_tuning_direction',
    'save_extraction_hdf5',
    'load_extraction_hdf5',
    'plot_cell_tuning_curve',
    'plot_orientation_map',
    'plot_tuning_distributions',
    'plot_population_summary',
    'create_full_analysis_report',
    'get_well_fit_cells',
]
