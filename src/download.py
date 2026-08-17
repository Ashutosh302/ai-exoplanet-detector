
# """
# src/download.py
# Module to parse TESS CTL catalogs and download high-cadence light curves.
# """

# import glob
# import os
# import lightkurve as lk
# import pandas as pd


# def load_ctl_targets(csv_path: str, max_targets: int = 100, max_tmag: float = 12.0):
#     """
#     Reads the xCTL CSV catalog and returns a list of target TIC IDs.
#     Filters for brighter stars (Tmag < max_tmag) for better SNR.
#     """
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"Catalog file not found at: {csv_path}")

#     print(f"Reading catalog: {csv_path}...")
#     df = pd.read_csv(csv_path, usecols=["TIC_ID", "Tmag"])
    
#     filtered_df = df[df["Tmag"] <= max_tmag].head(max_targets)
#     tic_ids = [f"TIC {tic}" for tic in filtered_df["TIC_ID"].tolist()]
    
#     print(f"Selected {len(tic_ids)} targets with Tmag <= {max_tmag}.")
#     return tic_ids


# def download_target_lightcurve(target_name: str, save_dir: str = "./data/raw", sector: int = None):
#     """
#     Downloads a single target's high-cadence light curve (one sector only).
#     Checks local cache first before querying NASA MAST.
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     target_clean = target_name.strip().replace(" ", "_")

#     # 1. Check if a local FITS file already exists
#     if sector is not None:
#         expected_file = os.path.join(save_dir, f"{target_clean}_sec{sector}.fits")
#         if os.path.exists(expected_file):
#             print(f"Loading cached light curve: {expected_file}")
#             return lk.read(expected_file)
#     else:
#         cached_matches = glob.glob(os.path.join(save_dir, f"{target_clean}_sec*.fits"))
#         if cached_matches:
#             print(f"Loading cached light curve: {cached_matches[0]}")
#             return lk.read(cached_matches[0])

#     try:
#         # 2. Fast search: TESS SPOC 2-minute cadence
#         search = lk.search_lightcurve(
#             target_name,
#             mission="TESS",
#             author="SPOC",
#             exptime=120
#         )

#         # 3. Targeted Fallback: TESS-SPOC or QLP if SPOC 2-min is unavailable
#         if len(search) == 0:
#             search = lk.search_lightcurve(
#                 target_name,
#                 mission="TESS",
#                 author=["TESS-SPOC", "QLP"]
#             )

#         if len(search) == 0:
#             print(f"No suitable light curve found for {target_name}")
#             return None

#         # Filter by sector if specified
#         if sector is not None:
#             sector_matches = search[search.table["sequence_number"] == sector]
#             if len(sector_matches) > 0:
#                 search = sector_matches

#         # Download ONLY the first available sector to prevent timeouts
#         lc = search[0].download(quality_bitmask="hard")
#         if lc_collection is None or len(lc_collection) == 0:
#             return None

#         lc = lc_collection.stitch()

#         # Clean NaNs and infinite values
#         lc = lc.remove_nans()

#         # Extract sequence number safely
#         try:
#             sec_num = int(search[0].table["sequence_number"][0])
#         except Exception:
#             sec_num = 0

#         # Save to disk for fast reuse
#         safe_filename = f"{target_clean}_sec{sec_num}.fits"
#         file_path = os.path.join(save_dir, safe_filename)
#         lc.to_fits(file_path, overwrite=True)

#         return lc

#     except Exception as e:
#         print(f"Error downloading {target_name}: {e}")
#         return None


"""
src/download.py
Module to parse TESS CTL catalogs and download high-cadence light curves.
"""

import glob
import os
import lightkurve as lk
import pandas as pd


def load_ctl_targets(csv_path: str, max_targets: int = 100, max_tmag: float = 12.0):
    """
    Reads the xCTL CSV catalog and returns a list of target TIC IDs.
    Filters for brighter stars (Tmag < max_tmag) for better SNR.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Catalog file not found at: {csv_path}")

    print(f"Reading catalog: {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["TIC_ID", "Tmag"])
    
    filtered_df = df[df["Tmag"] <= max_tmag].head(max_targets)
    tic_ids = [f"TIC {tic}" for tic in filtered_df["TIC_ID"].tolist()]
    
    print(f"Selected {len(tic_ids)} targets with Tmag <= {max_tmag}.")
    return tic_ids


def download_target_lightcurve(target_name: str, save_dir: str = "./data/raw", sector: int = None):
    """
    Downloads a single target's high-cadence light curve (one sector only).
    Checks local cache first before querying NASA MAST.
    """
    os.makedirs(save_dir, exist_ok=True)
    target_clean = target_name.strip().replace(" ", "_")

    # 1. Check if a local FITS file already exists
    if sector is not None:
        expected_file = os.path.join(save_dir, f"{target_clean}_sec{sector}.fits")
        if os.path.exists(expected_file):
            print(f"Loading cached light curve: {expected_file}")
            return lk.read(expected_file)
    else:
        cached_matches = glob.glob(os.path.join(save_dir, f"{target_clean}_sec*.fits"))
        if cached_matches:
            print(f"Loading cached light curve: {cached_matches[0]}")
            return lk.read(cached_matches[0])

    try:
        # 2. Fast search: TESS SPOC 2-minute cadence
        search = lk.search_lightcurve(
            target_name,
            mission="TESS",
            author="SPOC",
            exptime=120
        )

        # 3. Targeted Fallback: TESS-SPOC or QLP if SPOC 2-min is unavailable
        if len(search) == 0:
            search = lk.search_lightcurve(
                target_name,
                mission="TESS",
                author=["TESS-SPOC", "QLP"]
            )

        if len(search) == 0:
            print(f"No suitable light curve found for {target_name}")
            return None

        # Filter by sector if specified
        if sector is not None:
            sector_matches = search[search.table["sequence_number"] == sector]
            if len(sector_matches) > 0:
                search = sector_matches

        # Download ONLY the first available sector to prevent timeouts
        lc = search[0].download(quality_bitmask="hard")
        if lc is None:
            return None

        # Clean NaNs and infinite values
        lc = lc.remove_nans()

        # Extract sequence number safely
        try:
            sec_num = int(search[0].table["sequence_number"][0])
        except Exception:
            sec_num = 0

        # Save to disk for fast reuse
        safe_filename = f"{target_clean}_sec{sec_num}.fits"
        file_path = os.path.join(save_dir, safe_filename)
        lc.to_fits(file_path, overwrite=True)

        return lc

    except Exception as e:
        print(f"Error downloading {target_name}: {e}")
        return None
