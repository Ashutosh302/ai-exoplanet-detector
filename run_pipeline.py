# """
# run_pipeline.py
# Main pipeline orchestrator.
# """

# import os
# import torch
# import matplotlib.pyplot as plt
# from src.download import download_target_lightcurve
# from src.preprocess import clean_and_detrend, run_bls, generate_folded_vector
# from src.classifier import ExoplanetCNN1D, predict_lightcurve
# from src.fit_parameters import fit_transit_parameters, calculate_snr


# def run_pipeline_for_target(target_name: str, model, device: str = "cpu"):
#     print(f"\n==================================================")
#     print(f"Processing: {target_name}")
#     print(f"==================================================")
    
#     # 1. Download / Load raw data
#     raw_lc = download_target_lightcurve(target_name)
#     if raw_lc is None:
#         print(f"Skipping {target_name} due to missing data.")
#         return
        
#     # 2. Preprocess: Detrend & Outlier Removal
#     clean_lc = clean_and_detrend(raw_lc)
    
#     # 3. Box Least Squares Signal Detection
#     bls_stats = run_bls(clean_lc, min_period=0.8, max_period=10.0)
#     print(f"BLS Peak: Period = {bls_stats['period']:.4f} days, BLS Power = {bls_stats['power']:.2f}")
    
#     # 4. Prepare Folded 1D Array for CNN
#     folded_vector = generate_folded_vector(clean_lc, bls_stats["period"], bls_stats["t0"], num_bins=500)
    
#     # 5. AI Classification
#     label, confidence, all_probs = predict_lightcurve(model, folded_vector, device=device)
#     print(f"Classification Result: '{label}' (Confidence: {confidence*100:.2f}%)")
    
#     # 6. Parameter Fitting & SNR
#     fit_res = fit_transit_parameters(clean_lc, bls_stats)
#     snr = calculate_snr(
#         clean_lc, 
#         period=fit_res["period_days"], 
#         t0=bls_stats["t0"], 
#         duration_days=fit_res["duration_hours"] / 24.0, 
#         depth=fit_res["transit_depth"]
#     )
    
#     print("\n--- Estimated Parameters ---")
#     print(f"Orbital Period : {fit_res['period_days']:.4f} days")
#     print(f"Transit Depth  : {fit_res['transit_depth']*100:.4f}% (+/- {fit_res['transit_depth_err']*100:.4f}%)")
#     print(f"Transit Duration: {fit_res['duration_hours']:.2f} hours")
#     print(f"Signal-to-Noise: {snr:.2f} sigma")
    
#     # 7. Visualization Plot
#     fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
#     # Subplot 1: Flattened Light Curve
#     axes[0].scatter(clean_lc.time.value, clean_lc.flux.value, s=1, color="black", alpha=0.5)
#     axes[0].set_title(f"{target_name} - Cleaned Light Curve")
#     axes[0].set_xlabel("Time (BJD - 2457000)")
#     axes[0].set_ylabel("Normalized Flux")
    
#     # Subplot 2: Folded Transit with AI Label
#     phase_folded = clean_lc.fold(period=fit_res["period_days"], epoch_time=bls_stats["t0"])
#     axes[1].scatter(phase_folded.time.value, phase_folded.flux.value, s=2, color="teal", alpha=0.6)
#     axes[1].set_title(f"Classification: {label} ({confidence*100:.1f}%) | SNR: {snr:.1f}")
#     axes[1].set_xlabel("Phase (Days)")
#     axes[1].set_ylabel("Normalized Flux")
    
#     os.makedirs("./outputs", exist_ok=True)
#     out_file = f"./outputs/{target_name.replace(' ', '_')}_result.png"
#     plt.tight_layout()
#     plt.savefig(out_file, dpi=150)
#     plt.close()
#     print(f"Visualization saved to: {out_file}")


# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
    
#     # Load Model
#     model = ExoplanetCNN1D()
#     model_path = "./models/exoplanet_cnn.pth"
    
#     if not os.path.exists(model_path):
#         print("Model file not found. Running training first...")
#         import trainmode
#         trainmode.train()
        
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # Test on a famous verified exoplanet (WASP-126 b / TOI-114)
#     sample_targets = ["TIC 25155310"]
#     for target in sample_targets:
#         run_pipeline_for_target(target, model, device=device)


# if __name__ == "__main__":
#     main()


"""
run_pipeline.py
Batch processing pipeline for multiple exoplanet test targets.
"""

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.download import download_target_lightcurve, load_ctl_targets
from src.preprocess import clean_and_detrend, run_bls, generate_folded_vector
from src.classifier import ExoplanetCNN1D, predict_lightcurve
from src.fit_parameters import fit_transit_parameters, calculate_snr


def process_single_target(target_name: str, model, device: str = "cpu", save_plot: bool = True):
    """
    Processes one star. Returns a dictionary of results or None if data retrieval fails.
    """
    try:
        # 1. Download / Load data
        raw_lc = download_target_lightcurve(target_name)
        if raw_lc is None:
            return None
            
        # 2. Preprocess
        clean_lc = clean_and_detrend(raw_lc)
        
        # 3. BLS Period Search
        bls_stats = run_bls(clean_lc, min_period=0.8, max_period=15.0)
        
        # 4. Prepare Folded 1D Array for CNN
        folded_vector = generate_folded_vector(clean_lc, bls_stats["period"], bls_stats["t0"], num_bins=500)
        
        # 5. AI Inference
        label, confidence, _ = predict_lightcurve(model, folded_vector, device=device)
        
        # 6. Fit Parameters & SNR
        fit_res = fit_transit_parameters(clean_lc, bls_stats)
        snr = calculate_snr(
            clean_lc,
            period=fit_res["period_days"],
            t0=bls_stats["t0"],
            duration_days=fit_res["duration_hours"] / 24.0,
            depth=fit_res["transit_depth"]
        )
        
        result_dict = {
            "TIC_ID": target_name,
            "Classification": label,
            "Confidence": round(confidence, 4),
            "Period_Days": round(fit_res["period_days"], 4),
            "Depth_Percent": round(fit_res["transit_depth"] * 100, 4),
            "Duration_Hours": round(fit_res["duration_hours"], 2),
            "SNR": round(snr, 2),
            "Status": "Candidate" if (label == "Exoplanet Candidate" and confidence >= 0.70 and snr >= 7.0) else "Rejected/Other"
        }
        
        # 7. Generate diagnostic plot for Candidates and Binaries
        if save_plot and label in ["Exoplanet Candidate", "Eclipsing Binary"]:
            os.makedirs("./outputs", exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            
            # Subplot 1: Cleaned Light Curve
            axes[0].scatter(clean_lc.time.value, clean_lc.flux.value, s=1, color="black", alpha=0.4)
            axes[0].set_title(f"{target_name} - Detrended")
            axes[0].set_xlabel("Time (BJD - 2457000)")
            axes[0].set_ylabel("Normalized Flux")
            
            # Subplot 2: Folded Transit
            phase_folded = clean_lc.fold(period=fit_res["period_days"], epoch_time=bls_stats["t0"])
            axes[1].scatter(phase_folded.time.value, phase_folded.flux.value, s=2, color="teal", alpha=0.6)
            axes[1].set_title(f"{label} ({confidence*100:.1f}%) | SNR: {snr:.1f}")
            axes[1].set_xlabel("Phase (Days)")
            axes[1].set_ylabel("Normalized Flux")
            
            out_file = f"./outputs/{target_name.replace(' ', '_')}_result.png"
            plt.tight_layout()
            plt.savefig(out_file, dpi=120)
            plt.close()
            
        return result_dict
        
    except Exception as e:
        print(f"\n[Warning] Failed processing {target_name}: {e}")
        return None


def run_batch(target_list: list, model, device: str = "cpu"):
    """
    Iterates over all targets and writes out a CSV summary report.
    """
    print(f"\n========================================================")
    print(f"Starting batch analysis for {len(target_list)} targets...")
    print(f"========================================================")
    
    results = []
    for target in tqdm(target_list, desc="Processing Light Curves"):
        res = process_single_target(target, model, device=device)
        if res is not None:
            results.append(res)
            
    # Save results to a CSV report
    if results:
        df_results = pd.DataFrame(results)
        report_path = "./detection_report.csv"
        df_results.to_csv(report_path, index=False)
        print(f"\n--- Batch Run Completed ---")
        print(f"Report saved to: {report_path}")
        print("\nSummary Table:")
        print(df_results[["TIC_ID", "Classification", "Confidence", "SNR", "Status"]].to_string(index=False))
    else:
        print("No targets were successfully processed.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Trained CNN Model
    model = ExoplanetCNN1D()
    model_path = "./models/exoplanet_cnn.pth"
    if not os.path.exists(model_path):
        import trainmode
        trainmode.train()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ========================================================
    # CHOOSE YOUR INPUT METHOD BELOW:
    # ========================================================
    
    # OPTION A: Curated Benchmark Test Suite (Great for Demos / Testing)
    # Includes confirmed exoplanets, an eclipsing binary, and stellar noise
    benchmark_targets = [
        "TIC 25155310",   # WASP-126 b (Confirmed Hot Jupiter Exoplanet)
        "TIC 149603524",  # TOI-175 / L 98-59 (Multi-planet system)
        "TIC 307210830",  # HD 21749 b (Sub-Neptune Exoplanet)
        "TIC 272074677",  # Known Eclipsing Binary
    ]
    
    # OPTION B: Load N targets directly from the downloaded xCTL CSV catalog
    # ctl_path = "./xCTL_v08.01.csv"
    # ctl_targets = load_ctl_targets(ctl_path, max_targets=20, max_tmag=11.0)
    
    # Run the batch
    run_batch(benchmark_targets, model, device=device)


if __name__ == "__main__":
    main()