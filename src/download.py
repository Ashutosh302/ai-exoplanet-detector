"""
src/download.py
Module to parse TESS CTL catalogs and download high-cadence light curves.
"""

import os
import pandas as pd
import lightkurve as lk


def load_ctl_targets(csv_path: str, max_targets: int = 100, max_tmag: float = 12.0):
    """
    Reads the xCTL CSV catalog and returns a list of target TIC IDs.
    Filters for brighter stars (Tmag < max_tmag) for better SNR.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Catalog file not found at: {csv_path}")

    print(f"Reading catalog: {csv_path}...")
    # Read relevant columns from the CTL catalog
    df = pd.read_csv(csv_path, usecols=["TIC_ID", "Tmag"])
    
    # Filter by brightness and take the top targets
    filtered_df = df[df["Tmag"] <= max_tmag].head(max_targets)
    tic_ids = [f"TIC {tic}" for tic in filtered_df["TIC_ID"].tolist()]
    
    print(f"Selected {len(tic_ids)} targets with Tmag <= {max_tmag}.")
    return tic_ids


def download_target_lightcurve(target_name: str, save_dir: str = "./data/raw", sector: int = None):
    """
    Downloads a single target's SPOC 2-minute cadence light curve.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    try:
       # 1. Try for the gold-standard 2-minute SPOC data first
        search = lk.search_lightcurve(target_name, author="SPOC", exptime=120)
        
        if sector is not None:
            search = search[search.table['sequence_number'] == sector]
            
        # 2. FALLBACK: If no 2-minute data exists, grab whatever the QLP or TESS-SPOC pipeline has
        if len(search) == 0:
            search = lk.search_lightcurve(target_name, author=["QLP", "TESS-SPOC", "SPOC"])
            
        # 3. If STILL nothing is found, then the star just wasn't observed
        if len(search) == 0:
            return None
            
        # Download the first available sector for this star
        lc = search[0].download(quality_bitmask="hard")
        
        # Save as FITS for offline processing
        safe_filename = target_name.replace(" ", "_") + f"_sec{search[0].table['sequence_number'][0]}.fits"
        file_path = os.path.join(save_dir, safe_filename)
        lc.to_fits(file_path, overwrite=True)
        
        return lc
        
    except Exception as e:
        print(f"Error downloading {target_name}: {e}")
        return None
