"""
src/fit_parameters.py
Non-linear transit curve fitting and Signal-to-Noise Ratio (SNR) calculations.
"""

import numpy as np
from scipy.optimize import curve_fit


def box_transit_model(phase, depth, duration, baseline=1.0):
    """
    Analytical box model for parameter fitting.
    phase: orbital phase [-0.5, 0.5]
    depth: fractional drop in flux (e.g. 0.01 = 1%)
    duration: transit duration in phase units
    """
    flux = np.ones_like(phase) * baseline
    in_transit = np.abs(phase) < (duration / 2.0)
    flux[in_transit] = baseline - depth
    return flux


def fit_transit_parameters(flat_lc, bls_stats: dict):
    """
    Fits the folded light curve to determine transit depth, duration, and uncertainty.
    """
    time = flat_lc.time.value
    flux = flat_lc.flux.value
    period = bls_stats["period"]
    t0 = bls_stats["t0"]
    
    # Phase calculation
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    
    # Initial parameter guesses
    p0 = [bls_stats["depth"], bls_stats["duration"] / period, 1.0]
    bounds = ([0.0, 0.0001, 0.95], [0.5, 0.5, 1.05])
    
    try:
        popt, pcov = curve_fit(box_transit_model, phase, flux, p0=p0, bounds=bounds)
        depth_fit, duration_phase_fit, baseline_fit = popt
        perr = np.sqrt(np.diag(pcov))
        
        duration_days_fit = duration_phase_fit * period
        duration_err_days = perr[1] * period
        
        results = {
            "period_days": period,
            "transit_depth": depth_fit,
            "transit_depth_err": perr[0],
            "duration_hours": duration_days_fit * 24.0,
            "duration_err_hours": duration_err_days * 24.0,
        }
    except Exception:
        # Fallback to BLS estimates if non-linear fit does not converge
        results = {
            "period_days": period,
            "transit_depth": bls_stats["depth"],
            "transit_depth_err": 0.0,
            "duration_hours": bls_stats["duration"] * 24.0,
            "duration_err_hours": 0.0,
        }
        
    return results


def calculate_snr(flat_lc, period: float, t0: float, duration_days: float, depth: float):
    """
    Calculates Signal-to-Noise Ratio (SNR):
    SNR = (Depth / Out-of-transit Scatter) * sqrt(Number of in-transit points)
    """
    time = flat_lc.time.value
    flux = flat_lc.flux.value
    
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    phase_in_days = phase * period
    
    in_transit_mask = np.abs(phase_in_days) <= (duration_days / 2.0)
    out_of_transit_mask = ~in_transit_mask
    
    n_in_transit = np.sum(in_transit_mask)
    if n_in_transit < 2 or np.sum(out_of_transit_mask) < 2:
        return 0.0
        
    # Standard deviation of baseline noise
    sigma_out = np.std(flux[out_of_transit_mask])
    if sigma_out == 0:
        return 0.0
        
    snr = (depth / sigma_out) * np.sqrt(n_in_transit)
    return float(snr)
