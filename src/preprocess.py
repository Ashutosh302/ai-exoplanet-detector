"""
src/preprocess.py
Detrending, outlier removal, BLS period searching, phase folding, and array binning.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares


def clean_and_detrend(lc, window_length: int = 1001, sigma: float = 3.0):
    """
    Removes NaN values, clips positive outliers (flares/cosmic rays), 
    and flattens stellar variability using a Savitzky-Golay filter.
    """
    # 1. Drop NaNs
    lc = lc.remove_nans()
    
    # 2. Sigma clipping: Remove upward spikes only (preserve downward transits)
    lc_clean = lc.remove_outliers(sigma_upper=sigma, sigma_lower=float("inf"))
    
    # 3. Flatten long-term trends
    flat_lc = lc_clean.flatten(window_length=window_length)
    
    return flat_lc


def run_bls(flat_lc, min_period: float = 0.5, max_period: float = 15.0, duration_hours: float = 2.0):
    """
    Computes the Box Least Squares (BLS) periodogram to identify candidate periodic transit signals.
    """
    time = flat_lc.time.value
    flux = flat_lc.flux.value
    
    # Define search grid
    durations = np.linspace(0.05, duration_hours / 24.0, 10)
    periods = np.linspace(min_period, max_period, 5000)
    
    bls = BoxLeastSquares(time, flux)
    periodogram = bls.power(periods, durations)
    
    best_idx = np.argmax(periodogram.power)
    best_period = periodogram.period[best_idx]
    best_t0 = periodogram.transit_time[best_idx]
    best_duration = periodogram.duration[best_idx]
    best_depth = periodogram.depth[best_idx]
    max_power = periodogram.power[best_idx]
    
    stats = {
        "period": float(best_period),
        "t0": float(best_t0),
        "duration": float(best_duration),
        "depth": float(best_depth),
        "power": float(max_power)
    }
    return stats


def generate_folded_vector(flat_lc, period: float, t0: float, num_bins: int = 500):
    """
    Folds the light curve and bins it into a fixed-length 1D vector of shape (num_bins,).
    This fixed shape is required to feed into a 1D CNN.
    """
    time = flat_lc.time.value
    flux = flat_lc.flux.value
    
    # Compute orbital phase (-0.5 to +0.5) centered at the transit midpoint (phase 0)
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    
    # Uniform binning across [-0.5, 0.5]
    bin_edges = np.linspace(-0.5, 0.5, num_bins + 1)
    bin_means = np.zeros(num_bins)
    
    for i in range(num_bins):
        mask = (phase_sorted >= bin_edges[i]) & (phase_sorted < bin_edges[i+1])
        if np.any(mask):
            bin_means[i] = np.nanmedian(flux_sorted[mask])
        else:
            bin_means[i] = 1.0  # Normalized baseline flux
            
    # Normalize baseline to 0 for standard CNN input stability (flux - 1.0)
    normalized_input = bin_means - 1.0
    return normalized_input.astype(np.float32)
