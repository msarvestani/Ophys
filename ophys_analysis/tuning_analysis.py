"""
Orientation and direction tuning analysis.

This module provides functions to fit tuning curves and calculate
orientation/direction selectivity indices.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Tuple, Dict
import warnings


# ============================================================================
# Circular Statistics Helper Functions
# ============================================================================

def circ_ang2rad(angles_deg: np.ndarray) -> np.ndarray:
    """Convert angles from degrees to radians"""
    return np.deg2rad(angles_deg)


def circ_axial(angles_rad: np.ndarray, axial_correction: int = 1) -> np.ndarray:
    """
    Apply axial correction to circular data (for orientation doubling).

    Args:
        angles_rad: Angles in radians
        axial_correction: Axial correction factor (1 for orientation doubling)

    Returns:
        Corrected angles in radians
    """
    return angles_rad * 2 * axial_correction


def wrapTo360(angles_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 360) range"""
    return np.mod(angles_deg, 360)


def wrapTo180(angles_deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [0, 180) range"""
    return np.mod(angles_deg, 180)


# ============================================================================
# Gaussian Fitting Functions
# ============================================================================

def double_gauss(params: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Double Gaussian function for direction tuning.

    Model: Y = R_offset + R_pref * exp(-(theta - theta_pref)^2 / (2*sigma^2))
              + R_opp * exp(-(theta + 180 - theta_pref)^2 / (2*sigma^2))

    Args:
        params: [R_offset, R_pref, theta_pref, sigma, R_opp]
            R_offset: Baseline response
            R_pref: Response at preferred direction
            theta_pref: Preferred direction (degrees)
            sigma: Tuning width (degrees)
            R_opp: Response at opposite direction
        X: Direction values (degrees)

    Returns:
        Fitted response values
    """
    R_offset, R_pref, theta_pref, sigma, R_opp = params

    # Calculate angular differences (with wrapping)
    D = X - theta_pref
    D = np.array([min(abs(d), abs(d + 360), abs(d - 360)) for d in D])

    D2 = X + 180 - theta_pref
    D2 = np.array([min(abs(d), abs(d + 360), abs(d - 360)) for d in D2])

    # Double Gaussian
    Y = (R_offset +
         R_pref * np.exp(-(D ** 2) / (2 * sigma ** 2)) +
         R_opp * np.exp(-(D2 ** 2) / (2 * sigma ** 2)))

    return Y


def circular_gauss(params: np.ndarray, theta: float) -> float:
    """
    Single Gaussian function for initial parameter estimation.

    Args:
        params: [R_offset, R_pref, theta_pref, sigma]
        theta: Direction value (degrees)

    Returns:
        Response value
    """
    R_offset, R_pref, theta_pref, sigma = params[:4]

    D = theta - theta_pref
    D = min(abs(D), abs(D + 360), abs(D - 360))

    Y = R_offset + R_pref * np.exp(-(D ** 2) / (2 * sigma ** 2))

    return Y


def fit_tuning_direction(meanResponse: np.ndarray,
                          X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit double Gaussian to direction tuning curve.

    Args:
        meanResponse: Mean response at each direction
        X: Direction values (degrees)

    Returns:
        Tuple of (params, r_matrix, p_matrix)
        params: Fitted parameters [R_offset, R_pref, theta_pref, sigma, R_opp]
        r_matrix: Correlation matrix (r[0,1] is the correlation coefficient)
        p_matrix: P-value matrix
    """
    minVal = np.min(meanResponse)
    maxVal = np.max(meanResponse)
    Resp = meanResponse[:len(X)]

    # Handle edge cases for bounds calculation
    if minVal <= 0:
        minVal_bound = 0.0001
    else:
        minVal_bound = minVal
    if maxVal <= minVal_bound:
        maxVal_bound = minVal_bound + 1
    else:
        maxVal_bound = maxVal

    # Initialize parameters: [R_offset, R_pref, theta_pref, sigma, R_opp]
    G0 = np.zeros(5)
    G0[0] = max(minVal, 0.0001)  # R_offset
    G0[1] = max(maxVal - minVal, 0.0001)  # R_pref
    G0[3] = 20  # sigma (initial guess)

    # Find initial estimate for preferred direction using brute force
    minError = np.inf
    bestPeak = 0

    for i in range(0, 360, 5):
        G0[2] = i
        error = 0
        for j, x_val in enumerate(X):
            pred = circular_gauss(G0, x_val)
            error += (pred - Resp[j]) ** 2

        if error < minError:
            minError = error
            bestPeak = i

    G0[2] = bestPeak

    # Alternative: use peak of data
    max_idx = np.argmax(Resp)
    G0[2] = X[max_idx]

    # Estimate opposite direction response
    no = len(X)
    prefIndex = np.argmin(np.abs(bestPeak - X))
    orthoIndex = int((prefIndex + no / 2) % no)
    G0[4] = max(Resp[orthoIndex], 0.0001)  # R_opp

    # Set sigma bounds
    if len(X) <= 12:
        sigma_min = 150 / len(X)
    else:
        sigma_min = 2.813

    # Bounds: [R_offset, R_pref, theta_pref, sigma, R_opp]
    LB = np.array([minVal_bound * 0.1, (maxVal_bound - minVal_bound) * 0.5, 0, sigma_min, 0.0001])
    UB = np.array([max(minVal_bound * 1.2, maxVal_bound), (maxVal_bound - minVal_bound) * 1.5, 360, 50, (maxVal_bound - minVal_bound) * 1.5])

    # Ensure G0 is within bounds
    G0 = np.clip(G0, LB, UB)

    # Fit using least squares - use Resp (truncated to match X), not full meanResponse
    def residuals(params):
        return Resp - double_gauss(params, X)

    result = least_squares(residuals, G0, bounds=(LB, UB))
    G = result.x

    # Generate fit data at the stimulus positions
    FITDATA = double_gauss(G, X)

    # Calculate correlation
    r_matrix = np.corrcoef(Resp, FITDATA)
    # For p-value, we'll use a simple approximation
    n = len(Resp)
    r = r_matrix[0, 1]
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    from scipy.stats import t as t_dist
    p_val = 2 * (1 - t_dist.cdf(np.abs(t_stat), n - 2))
    p_matrix = np.array([[1, p_val], [p_val, 1]])

    return G, r_matrix, p_matrix


def get_tuning_madineh(meanResponse: np.ndarray,
                        stimInfo: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Calculate direction and orientation tuning metrics.

    This is the Python conversion of getTuningMadineh.m

    Args:
        meanResponse: Trial-averaged response at each stimulus condition
        stimInfo: Vector of directions (degrees, not including blank)

    Returns:
        Tuple of (DIR2, meanResponse_fit, FITDATA)
        DIR2: Dictionary with tuning metrics
        meanResponse_fit: Response data used for fitting (may be circularly shifted)
        FITDATA: Fitted tuning curve values
    """
    DIR2 = {}

    n_dirs = len(stimInfo)
    div = 360 / n_dirs
    n_orts = n_dirs // 2

    # Convert to radians for circular stats
    oriRad = circ_axial(circ_ang2rad(stimInfo), 1)

    # ========================================================================
    # Calculate orientation tuning using vector method
    # ========================================================================
    rk2_ort = np.zeros(n_orts)
    exponent_ort = np.zeros(n_orts, dtype=complex)

    for k in range(n_orts):
        # Average response at this orientation (both directions)
        rk2_ort[k] = np.nanmean([meanResponse[k], meanResponse[k + n_orts]])
        # Angle doubling for orientation
        exponent_ort[k] = rk2_ort[k] * np.exp(1j * 2 * oriRad[k])

    biasvector_ort = np.sum(exponent_ort) / np.sum(rk2_ort)
    DIR2['oti_vec'] = np.abs(biasvector_ort)

    # Preferred orientation from vector method
    pref_angle2_ort = np.rad2deg(np.angle(biasvector_ort) / 2)
    DIR2['pref_ort_vec'] = wrapTo360(pref_angle2_ort)
    if DIR2['pref_ort_vec'] > 180:
        DIR2['pref_ort_vec'] = DIR2['pref_ort_vec'] - 180

    # ========================================================================
    # Calculate direction tuning using vector method
    # ========================================================================
    rk2_dir = np.zeros(2 * n_orts)
    exponent_dir = np.zeros(2 * n_orts, dtype=complex)

    for k in range(2 * n_orts):
        rk2_dir[k] = np.nanmean([meanResponse[k]])
        exponent_dir[k] = rk2_dir[k] * np.exp(1j * oriRad[k])

    biasvector_dir = np.sum(exponent_dir) / np.sum(rk2_dir)
    DIR2['dti_vec'] = np.abs(biasvector_dir)

    # ========================================================================
    # Fit double Gaussian to get fit-based metrics
    # ========================================================================

    # Fit directly without rolling - the double Gaussian handles circularity
    # via its angular difference calculation
    meanResponse_fit = meanResponse
    G, r, p = fit_tuning_direction(meanResponse_fit, stimInfo)
    DIR2['fit_r'] = r[0, 1]
    DIR2['xlabel_vals'] = np.arange(0, 361, div)

    # Generate smooth fit curve for plotting
    fit_x_smooth = np.arange(0, 360, 1)  # 1-degree resolution
    FITDATA = double_gauss(G, fit_x_smooth)

    DIR2['fit_bandwidth'] = G[3]  # sigma

    # ========================================================================
    # Calculate fit-based orientation and direction selectivity
    # ========================================================================

    # Unpack fit parameters
    R_offset, R_pref, theta_pref, sigma, R_opp = G

    # Find orthogonal direction
    theta_ortho = theta_pref + 90
    if theta_ortho >= 360:
        theta_ortho = theta_ortho - 360

    # Get responses at preferred and orthogonal from the fit curve
    # fit_x_smooth is 0, 1, 2, ..., 359 so index = angle
    ind_pref = int(round(theta_pref)) % 360
    ind_ortho = int(round(theta_ortho)) % 360
    R_pref_fit = FITDATA[ind_pref]
    R_ortho = FITDATA[ind_ortho]

    # Orientation tuning index (fit-based)
    DIR2['oti_fit'] = (R_pref - R_ortho) / R_pref if R_pref != 0 else 0

    # Direction tuning index (fit-based)
    DIR2['dti_fit'] = min((R_pref - R_opp) / R_pref if R_pref != 0 else 0, 1)

    # Preferred direction and orientation
    DIR2['pref_dir_fit'] = theta_pref
    DIR2['pref_ort_fit'] = theta_pref
    if DIR2['pref_ort_fit'] > 180:
        DIR2['pref_ort_fit'] = min(abs(theta_pref + 180), abs(theta_pref - 180))

    # Bound checking
    DIR2['dti_fit'] = max(0, min(1, DIR2['dti_fit']))
    DIR2['oti_fit'] = max(0, min(1, DIR2['oti_fit']))

    # If negative, use vector-based values
    if DIR2['dti_fit'] < 0:
        DIR2['dti_fit'] = DIR2['dti_vec']
    if DIR2['oti_fit'] < 0:
        DIR2['oti_fit'] = DIR2['oti_vec']

    return DIR2, meanResponse_fit, FITDATA
