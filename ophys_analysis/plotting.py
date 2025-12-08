"""
Visualization functions for 2-photon imaging analysis.

This module provides plotting functions for:
- Individual cell tuning curves
- Orientation and direction preference maps
- Population statistics
- Trial-averaged traces
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from typing import Optional, Tuple, List
import warnings
from scipy.spatial import ConvexHull

from .cell_data import Cell, CellExtraction
from .tuning_analysis import get_tuning_madineh, double_gauss


def get_stim_info(ce: CellExtraction, n_dirs: int) -> np.ndarray:
    """
    Get stimulus info (orientations/directions) from FOV or generate default.

    Args:
        ce: CellExtraction object with optional fov reference
        n_dirs: Number of directions (excluding blank)

    Returns:
        Array of stimulus values in degrees
    """
    # Try to get actual stim values from FOV
    if ce.fov is not None and hasattr(ce.fov, 'stim_values') and ce.fov.stim_values is not None:
        stim_values = np.array(ce.fov.stim_values)
        # Return only the number of directions needed (in case blank is included)
        if len(stim_values) >= n_dirs:
            return stim_values[:n_dirs]
        else:
            return stim_values

    # Fallback: generate evenly spaced values
    return np.arange(0, 360, 360/n_dirs)


def plot_cell_tuning_curve(cell: Cell,
                             stimInfo: np.ndarray,
                             tuning: dict,
                             fitdata: np.ndarray,
                             cell_index: int,
                             stim_dur: float,
                             save_path: Optional[str] = None):
    """
    Plot comprehensive tuning analysis for a single cell.

    Creates a figure with:
    - Time series traces for each orientation
    - Polar tuning curve
    - Fitted tuning curve
    - ROI mask overlay

    Args:
        cell: Cell object
        stimInfo: Stimulus orientations/directions (degrees)
        tuning: Tuning metrics from get_tuning_madineh
        fitdata: Fitted tuning curve values
        cell_index: Cell number for title
        stim_dur: Stimulus duration (seconds)
        save_path: Optional path to save figure
    """
    n_dirs = len(stimInfo)
    row_width = int(np.ceil(n_dirs / 2))

    fig = plt.figure(figsize=(14, 10))

    # Get response data
    response = cell.condition_response[:n_dirs]
    response_sem = cell.condition_response_std[:n_dirs]

    # Sort by angle
    sort_idx = np.argsort(stimInfo)
    stimInfo_sorted = stimInfo[sort_idx]
    response_sorted = response[sort_idx]
    response_sem_sorted = response_sem[sort_idx]

    # Ensure non-negative for plotting
    if np.min(response_sorted) < 0:
        response_sorted = response_sorted - np.min(response_sorted)

    n_points = cell.cyc.shape[2]
    time_trace = np.linspace(0, n_points * cell.scanPeriod, n_points)

    # Plot time series for each orientation
    for i, sort_i in enumerate(sort_idx):
        ax = plt.subplot(3, row_width, i + 1)

        # Highlight stimulus period
        ymax = max(np.nanmax(cell.cyc[sort_i, :, :]) * 1.2, 0.5)
        ymin = min(np.nanmin(cell.cyc[sort_i, :, :]) * 1.2, -0.2)
        ax.axvspan(0, stim_dur, alpha=0.3, color='yellow')

        # Plot individual trials
        for trial in range(cell.cyc.shape[1]):
            ax.plot(time_trace, cell.cyc[sort_i, trial, :],
                   alpha=0.3, linewidth=0.5, color='gray')

        # Plot mean trace
        mean_trace = np.nanmean(cell.cyc[sort_i, :, :], axis=0)
        ax.plot(time_trace, mean_trace, 'k', linewidth=2)

        ax.set_ylim([ymin, ymax])
        ax.set_xlim([0, np.max(time_trace)])

        if i == 0:
            ax.set_title(f'ROI {cell_index}', fontweight='bold')
        else:
            ax.set_title(f'{stimInfo_sorted[i]:.0f}°')

        if i == 0 or i == row_width:
            ax.set_ylabel('ΔF/F')
        else:
            ax.set_yticks([])

        if i >= row_width:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticks([])

    # Plot polar tuning curve
    ax_polar = plt.subplot(3, 3, 7, projection='polar')
    theta_rad = np.deg2rad(np.append(stimInfo_sorted, stimInfo_sorted[0]))
    r_values = np.append(response_sorted, response_sorted[0])
    ax_polar.plot(theta_rad, r_values, 'o-', linewidth=2)
    ax_polar.set_theta_zero_location('E')
    ax_polar.set_theta_direction(1)

    # Plot Cartesian tuning curve with fit
    ax_tuning = plt.subplot(3, 3, 8)
    ax_tuning.errorbar(stimInfo_sorted, response_sorted, yerr=response_sem_sorted,
                       fmt='o', capsize=3, label='Data')
    fit_x = np.arange(1, 361, 5)
    ax_tuning.plot(fit_x, fitdata, 'r--', linewidth=2, label='Fit')
    ax_tuning.set_xlabel('Direction (°)')
    ax_tuning.set_ylabel('ΔF/F')
    ax_tuning.set_xlim([0, 360])
    ax_tuning.set_xticks(tuning['xlabel_vals'])
    ax_tuning.legend()
    ax_tuning.set_title(f"Visual Resp: {cell.ROI_responsiveness}")

    # Add statistics text
    stats_text = (
        f"Pref Ort: {tuning['pref_ort_fit']:.0f}°, OTI: {tuning['oti_vec']:.2f}\n"
        f"Pref Dir: {tuning['pref_dir_fit']:.0f}°, DTI: {tuning['dti_vec']:.2f}\n"
        f"Fit r: {tuning['fit_r']:.2f}, BW: {tuning['fit_bandwidth']:.0f}°"
    )
    ax_tuning.text(0.02, 0.98, stats_text, transform=ax_tuning.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot ROI mask (if available and if you have the FOV image)
    # ax_mask = plt.subplot(3, 3, 9)
    # # This would show the ROI overlay on the FOV image
    # # For now, just show the mask coordinates
    # if cell.mask is not None:
    #     ax_mask.scatter(cell.mask[:, 0], cell.mask[:, 1], s=1)
    #     ax_mask.set_aspect('equal')
    #     ax_mask.set_title('ROI Mask')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_orientation_map(ce: CellExtraction,
                          cell_indices: Optional[List[int]] = None,
                          fov_image: Optional[np.ndarray] = None,
                          save_path: Optional[str] = None):
    """
    Plot orientation preference map.

    Args:
        ce: CellExtraction object
        cell_indices: Indices of cells to plot (default: all responsive)
        fov_image: Optional FOV image for background
        save_path: Optional path to save figure
    """
    if cell_indices is None:
        cell_indices = [i for i, c in enumerate(ce.cells) if c.ROI_responsiveness]

    # Get tuning for all cells
    tuning_data = []
    for idx in cell_indices:
        cell = ce.cells[idx]
        n_dirs = len(cell.uniqStims) - 1
        if n_dirs > 0:
            stimInfo = get_stim_info(ce, n_dirs)
            try:
                tuning, _, _ = get_tuning_madineh(cell.condition_response[:n_dirs], stimInfo)
                tuning_data.append((idx, tuning))
            except:
                pass

    # Create color map (HSV for orientation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Orientation map (0-180°)
    if fov_image is not None:
        ax1.imshow(fov_image, cmap='gray', alpha=0.5)
        ax2.imshow(fov_image, cmap='gray', alpha=0.5)

    # Track bounds for axis limits
    all_x = []
    all_y = []

    for idx, tuning in tuning_data:
        cell = ce.cells[idx]
        pref_ort = tuning['pref_ort_fit']
        pref_dir = tuning['pref_dir_fit']

        # Color by orientation (0-180)
        ort_color = plt.cm.hsv(pref_ort / 180.0)
        # Color by direction (0-360)
        dir_color = plt.cm.hsv(pref_dir / 360.0)

        if cell.mask is not None and len(cell.mask) > 3:
            # mask from np.argwhere is [row, col] = [y, x], need to swap for Polygon [x, y]
            mask_xy = cell.mask[:, ::-1] if cell.mask.shape[1] == 2 else cell.mask

            # Track bounds
            all_x.extend(mask_xy[:, 0])
            all_y.extend(mask_xy[:, 1])

            # Get convex hull for clean ROI boundary (like actual neuron shape)
            try:
                hull = ConvexHull(mask_xy)
                hull_points = mask_xy[hull.vertices]

                # Plot orientation map
                poly = Polygon(hull_points, facecolor=ort_color, edgecolor='none', alpha=0.6)
                ax1.add_patch(poly)

                # Plot direction map
                poly = Polygon(hull_points, facecolor=dir_color, edgecolor='none', alpha=0.6)
                ax2.add_patch(poly)
            except Exception:
                # Fallback: plot as scatter if convex hull fails
                ax1.scatter(mask_xy[:, 0], mask_xy[:, 1], c=[ort_color], s=1, alpha=0.6)
                ax2.scatter(mask_xy[:, 0], mask_xy[:, 1], c=[dir_color], s=1, alpha=0.6)

    # Set axis limits based on data if no fov_image provided
    if fov_image is None and len(all_x) > 0:
        margin = 20  # pixels
        x_min, x_max = min(all_x) - margin, max(all_x) + margin
        y_min, y_max = min(all_y) - margin, max(all_y) + margin
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_max, y_min)  # Invert y-axis for image coordinates
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_max, y_min)
    elif fov_image is None:
        # Default to typical FOV size if no data
        ax1.set_xlim(0, 512)
        ax1.set_ylim(512, 0)
        ax2.set_xlim(0, 512)
        ax2.set_ylim(512, 0)

    ax1.set_title('Orientation Preference Map', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.set_aspect('equal')

    ax2.set_title('Direction Preference Map', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.set_aspect('equal')

    # Add color bars
    sm1 = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=0, vmax=180))
    sm1.set_array([])
    cbar1 = plt.colorbar(sm1, ax=ax1, label='Preferred Orientation (°)')
    cbar1.set_ticks([0, 45, 90, 135, 180])

    sm2 = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=0, vmax=360))
    sm2.set_array([])
    cbar2 = plt.colorbar(sm2, ax=ax2, label='Preferred Direction (°)')
    cbar2.set_ticks([0, 90, 180, 270, 360])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_tuning_distributions(ce: CellExtraction,
                                cell_indices: Optional[List[int]] = None,
                                save_path: Optional[str] = None):
    """
    Plot distributions of tuning metrics.

    Args:
        ce: CellExtraction object
        cell_indices: Indices of cells to analyze (default: all responsive)
        save_path: Optional path to save figure
    """
    if cell_indices is None:
        cell_indices = [i for i, c in enumerate(ce.cells) if c.ROI_responsiveness]

    # Collect tuning metrics
    oti_values = []
    dti_values = []
    pref_ort = []
    pref_dir = []
    bandwidths = []

    for idx in cell_indices:
        cell = ce.cells[idx]
        n_dirs = len(cell.uniqStims) - 1
        if n_dirs > 0:
            stimInfo = get_stim_info(ce, n_dirs)
            try:
                tuning, _, _ = get_tuning_madineh(cell.condition_response[:n_dirs], stimInfo)
                oti_values.append(tuning['oti_fit'])
                dti_values.append(tuning['dti_fit'])
                pref_ort.append(tuning['pref_ort_fit'])
                pref_dir.append(tuning['pref_dir_fit'])
                bandwidths.append(tuning['fit_bandwidth'])
            except:
                pass

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # OTI distribution
    axes[0, 0].hist(oti_values, bins=np.linspace(0, 1, 11), edgecolor='black')
    axes[0, 0].set_xlabel('OTI')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(f'Orientation Tuning Index\nMean: {np.mean(oti_values):.2f} ± {np.std(oti_values):.2f}')

    # DTI distribution
    axes[0, 1].hist(dti_values, bins=np.linspace(0, 1, 11), edgecolor='black')
    axes[0, 1].set_xlabel('DTI')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Direction Tuning Index\nMean: {np.mean(dti_values):.2f} ± {np.std(dti_values):.2f}')

    # Bandwidth distribution
    axes[0, 2].hist(bandwidths, bins=20, edgecolor='black')
    axes[0, 2].set_xlabel('Tuning Width (°)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title(f'Tuning Bandwidth\nMean: {np.mean(bandwidths):.1f}°')

    # Preferred orientation distribution
    axes[1, 0].hist(pref_ort, bins=np.linspace(0, 180, 19), edgecolor='black')
    axes[1, 0].set_xlabel('Preferred Orientation (°)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Preferred Orientation Distribution')
    axes[1, 0].set_xticks([0, 45, 90, 135, 180])

    # Preferred direction distribution
    axes[1, 1].hist(pref_dir, bins=np.linspace(0, 360, 37), edgecolor='black')
    axes[1, 1].set_xlabel('Preferred Direction (°)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Preferred Direction Distribution')
    axes[1, 1].set_xticks([0, 90, 180, 270, 360])

    # DTI vs OTI
    axes[1, 2].scatter(dti_values, oti_values, alpha=0.5)
    axes[1, 2].plot([0, 1], [0, 1], 'r--', label='Unity')
    axes[1, 2].set_xlabel('DTI')
    axes[1, 2].set_ylabel('OTI')
    axes[1, 2].set_title('Direction vs Orientation Selectivity')
    axes[1, 2].legend()
    axes[1, 2].set_xlim([0, 1])
    axes[1, 2].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_population_summary(ce: CellExtraction, save_path: Optional[str] = None):
    """
    Plot summary statistics for the entire population.

    Args:
        ce: CellExtraction object
        save_path: Optional path to save figure
    """
    n_cells = len(ce.cells)
    n_responsive = sum(c.ROI_responsiveness for c in ce.cells)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Cell responsiveness
    axes[0, 0].bar(['Responsive', 'Non-responsive'],
                   [n_responsive, n_cells - n_responsive],
                   color=['green', 'gray'])
    axes[0, 0].set_ylabel('Number of cells')
    axes[0, 0].set_title(f'Cell Responsiveness\n{n_responsive}/{n_cells} ({100*n_responsive/n_cells:.1f}%)')

    # Spatial distribution
    x_pos = ce.to_array('xPos')
    y_pos = ce.to_array('yPos')
    responsive_mask = ce.to_array('ROI_responsiveness').astype(bool)

    axes[0, 1].scatter(x_pos[~responsive_mask], y_pos[~responsive_mask],
                       c='gray', s=10, alpha=0.3, label='Non-responsive')
    axes[0, 1].scatter(x_pos[responsive_mask], y_pos[responsive_mask],
                       c='green', s=10, alpha=0.6, label='Responsive')
    axes[0, 1].set_xlabel('X position (pixels)')
    axes[0, 1].set_ylabel('Y position (pixels)')
    axes[0, 1].set_title('Spatial Distribution')
    axes[0, 1].legend()
    axes[0, 1].set_aspect('equal')

    # Response amplitude distribution
    max_responses = []
    for cell in ce.cells:
        if cell.condition_response is not None:
            max_responses.append(np.max(cell.condition_response))

    axes[1, 0].hist(max_responses, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Max ΔF/F')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Response Amplitude Distribution\nMean: {np.mean(max_responses):.2f}')

    # Summary text
    summary_text = (
        f"Dataset Summary\n"
        f"{'='*40}\n"
        f"Total cells: {n_cells}\n"
        f"Responsive: {n_responsive} ({100*n_responsive/n_cells:.1f}%)\n"
        f"Non-responsive: {n_cells - n_responsive}\n"
        f"\n"
        f"Animal: {ce.fov.animal_name if ce.fov else 'N/A'}\n"
        f"Brain region: {ce.fov.brain_region if ce.fov else 'N/A'}\n"
        f"Stim type: {ce.fov.stim_type if ce.fov else 'N/A'}\n"
        f"Recording date: {ce.fov.recording_date if ce.fov else 'N/A'}\n"
    )
    axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                    verticalalignment='center', fontsize=12,
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_full_analysis_report(ce: CellExtraction,
                                  fov_image: Optional[np.ndarray] = None,
                                  output_dir: Optional[str] = None):
    """
    Create a complete analysis report with all plots.

    Args:
        ce: CellExtraction object
        fov_image: Optional FOV image
        output_dir: Directory to save plots (if None, displays instead)
    """
    from pathlib import Path

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

    # 1. Population summary
    print("Creating population summary...")
    plot_population_summary(ce,
                             save_path=str(output_path / 'population_summary.png') if output_dir else None)

    # 2. Orientation maps
    print("Creating orientation/direction maps...")
    plot_orientation_map(ce, fov_image=fov_image,
                          save_path=str(output_path / 'orientation_maps.png') if output_dir else None)

    # 3. Tuning distributions
    print("Creating tuning distributions...")
    plot_tuning_distributions(ce,
                               save_path=str(output_path / 'tuning_distributions.png') if output_dir else None)

    # 4. Individual cell tuning curves (for responsive cells)
    responsive_indices = [i for i, c in enumerate(ce.cells) if c.ROI_responsiveness]
    print(f"Creating tuning curves for {len(responsive_indices)} responsive cells...")

    for idx in responsive_indices[:10]:  # Limit to first 10 for now
        cell = ce.cells[idx]
        n_dirs = len(cell.uniqStims) - 1
        if n_dirs > 0:
            stimInfo = get_stim_info(ce, n_dirs)
            try:
                tuning, _, fitdata = get_tuning_madineh(cell.condition_response[:n_dirs], stimInfo)
                stim_dur = ce.fov.stim_dur if ce.fov else 4.0
                plot_cell_tuning_curve(
                    cell, stimInfo, tuning, fitdata, idx, stim_dur,
                    save_path=str(output_path / f'cell_{idx}_tuning.png') if output_dir else None
                )
            except Exception as e:
                print(f"  Warning: Could not plot cell {idx}: {e}")

    print("✓ Analysis report complete!")
