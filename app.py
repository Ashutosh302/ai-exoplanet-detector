# """
# app.py
# Interactive Streamlit Web App for AI Exoplanet Detection with Astrophysical Insights.
# Run with: python -m streamlit run app.py
# """

# import os
# import streamlit as st
# import torch
# import matplotlib.pyplot as plt

# # Import from your pipeline modules
# from src.download import download_target_lightcurve
# from src.preprocess import clean_and_detrend, run_bls, generate_folded_vector
# from src.classifier import ExoplanetCNN1D, predict_lightcurve
# from src.fit_parameters import fit_transit_parameters, calculate_snr

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="AI Exoplanet Detector",
#     page_icon="🪐",
#     layout="wide"
# )

# # --- Astrophysical Explanations Knowledge Base ---
# ASTRO_EXPLANATIONS = {
#     "Exoplanet Candidate": {
#         "title": "🪐 Exoplanet Transit Signal",
#         "badge": "Planetary Candidate",
#         "description": (
#             "An **exoplanet** is a planet orbiting a distant star. "
#             "When the planet crosses directly in front of the host star along our line of sight, "
#             "it temporarily blocks a tiny fraction of the star's photons."
#         ),
#         "key_characteristics": [
#             "**U-Shaped Transit:** A distinct flat bottom appears while the entire planetary disk traverses across the star.",
#             "**Shallow Depth:** Generally between 0.01% (Earth-sized) to ~1–2% (Hot Jupiters).",
#             "**Strict Periodicity:** Repeats at constant intervals defined by Kepler's Third Law."
#         ],
#         "physics_insight": (
#             "The fractional transit depth $\\delta$ directly estimates the planet-to-star radius ratio: "
#             "$$\\delta \\approx \\left(\\frac{R_{p}}{R_{*}}\\right)^2$$"
#         )
#     },
#     "Eclipsing Binary": {
#         "title": "⭐ Eclipsing Binary System",
#         "badge": "Stellar Companion",
#         "description": (
#             "An **Eclipsing Binary** consists of two gravitationally bound stars orbiting each other. "
#             "Because both bodies are luminous stars, their mutual transits and occultations cause massive periodic drops in total flux."
#         ),
#         "key_characteristics": [
#             "**V-Shaped Profile:** Grazing geometry and spherical stellar limbs yield steep, pointed dips.",
#             "**Substantial Flux Loss:** Dips commonly range from 5% to over 50% light reduction.",
#             "**Primary & Secondary Eclipses:** Often produces alternating deep and shallow dips as the hotter and cooler stars take turns in front."
#         ],
#         "physics_insight": (
#             "The relative depth ratio between the two eclipses scales with the fourth power of their effective temperatures: "
#             "$$\\frac{\\delta_{secondary}}{\\delta_{primary}} \\approx \\left(\\frac{T_{2}}{T_{1}}\\right)^4$$"
#         )
#     },
#     "Noise / False Positive": {
#         "title": "📉 Stellar Activity / Instrumental Artifact",
#         "badge": "Non-Planetary",
#         "description": (
#             "Stars are dynamic, turbulent bodies, and orbital space telescopes experience mechanical vibrations. "
#             "These factors frequently mimic transit-like dips without any orbiting companion."
#         ),
#         "key_characteristics": [
#             "**Stellar Flares & Sunspots:** Magnetic reconnection triggers sudden flux bursts, while rotating starspots produce sinusoidal modulation.",
#             "**Aperture Contamination:** Background eclipsing binaries bleeding photons into the target pixel aperture.",
#             "**Momentum Dumps:** Spacecraft reaction wheel desaturations that cause tracking jitters."
#         ],
#         "physics_insight": (
#             "Genuine planetary transits maintain an invariant transit duration and depth across all observation sectors."
#         )
#     }
# }

# # --- Model Loading (Cached) ---
# @st.cache_resource
# def load_trained_model():
#     model = ExoplanetCNN1D()
#     model_path = "./models/exoplanet_cnn.pth"
#     if not os.path.exists(model_path):
#         import trainmode
#         trainmode.train()
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model

# # --- Data Fetching (Cached) ---
# @st.cache_data(show_spinner=False)
# def fetch_and_process(target_name):
#     raw_lc = download_target_lightcurve(target_name)
#     if raw_lc is None:
#         return None, None, None
#     clean_lc = clean_and_detrend(raw_lc)
#     bls_stats = run_bls(clean_lc, min_period=0.8, max_period=15.0)
#     return raw_lc, clean_lc, bls_stats

# # --- Header ---
# st.title("🪐 AI Exoplanet Transit Detector")
# st.markdown(
#     "Detect and classify exoplanetary candidates from **NASA TESS Light Curves** using deep learning and Box Least Squares periodograms."
# )

# # --- Sidebar ---
# st.sidebar.header("🎯 Target Selection")

# preset_options = {
#     "WASP-126 b (Confirmed Hot Jupiter)": "TIC 25155310",
#     "TOI-175 / L 98-59 (Multi-planet System)": "TIC 149603524",
#     "HD 21749 b (Sub-Neptune Exoplanet)": "TIC 307210830",
#     "TOI-700 d (Habitable Zone Earth-Size)": "TIC 150428135",
#     "Pi Mensae c (First TESS Discovery)": "TIC 261136679",
#     "LHS 3844 b (11-Hour Orbit Planet)": "TIC 410153553",
#     "Known Eclipsing Binary System": "TIC 272074677",
#     "V0757 Pup (Deep Binary System)": "TIC 6939791",
#     "Stellar Flare / Noise Artifact": "TIC 107150013",
#     "Custom TIC ID": "custom"
# }

# selection = st.sidebar.selectbox("Choose a target star:", list(preset_options.keys()))

# if preset_options[selection] == "custom":
#     target_id = st.sidebar.text_input("Enter TIC ID (e.g. TIC 25155310):", value="TIC 25155310")
# else:
#     target_id = preset_options[selection]
#     st.sidebar.info(f"Target: **{target_id}**")

# run_button = st.sidebar.button("🚀 Analyze Light Curve", type="primary", use_container_width=True)

# # --- Execution ---
# if run_button or target_id:
#     model = load_trained_model()
    
#     with st.spinner(f"Querying NASA MAST Archive for {target_id}..."):
#         raw_lc, clean_lc, bls_stats = fetch_and_process(target_id)
        
#     if raw_lc is None:
#         st.error(f"❌ Could not retrieve high-cadence SPOC data for **{target_id}**. Ensure the ID is valid and observed in TESS 2-min cadence mode.")
#     else:
#         # 1. AI Inference
#         folded_vector = generate_folded_vector(clean_lc, bls_stats["period"], bls_stats["t0"], num_bins=500)
#         label, confidence, all_probs = predict_lightcurve(model, folded_vector, device="cpu")
        
#         # 2. Parameter Fitting & SNR
#         fit_res = fit_transit_parameters(clean_lc, bls_stats)
#         snr = calculate_snr(
#             clean_lc,
#             period=fit_res["period_days"],
#             t0=bls_stats["t0"],
#             duration_days=fit_res["duration_hours"] / 24.0,
#             depth=fit_res["transit_depth"]
#         )
        
#         # --- ASTROPHYSICAL SANITY CHECK ---
#         # 1. If the transit depth is > 3%, it's physically too large to be a planet.
#         if fit_res["transit_depth"] > 0.03 and label == "Exoplanet Candidate":
#             label = "Eclipsing Binary"
#             st.warning("⚠️ **Physics Override:** The AI initially guessed 'Exoplanet', but a transit depth of over 3% is physically impossible for a planet. Reclassified as an Eclipsing Binary.")

#         # 2. Catch erratic noise, flares, or flatlines that lack a true transit profile
#         elif label == "Exoplanet Candidate" and (bls_stats['power'] < 5.0 or fit_res['transit_depth'] < 0.0005):
#             label = "Noise / False Positive"
#             st.warning("⚠️ **Physics Override:** Reclassified as Noise / False Positive because the signal lacks a consistent planetary transit profile.")

#         # 3. Top Metrics Banner
#         st.subheader("🔍 Analysis Verdict")
#         col1, col2, col3, col4 = st.columns(4)
        
#         if label == "Exoplanet Candidate":
#             col1.metric("Classification", "🪐 Exoplanet", delta=f"{confidence*100:.1f}% Confidence")
#         elif label == "Eclipsing Binary":
#             col1.metric("Classification", "⭐ Binary Star", delta="Non-Planetary", delta_color="inverse")
#         else:
#             col1.metric("Classification", "📉 Noise / Artifact", delta="Low Signal", delta_color="off")
            
#         col2.metric("Signal-to-Noise (SNR)", f"{snr:.2f} σ")
#         col3.metric("Orbital Period", f"{fit_res['period_days']:.4f} days")
#         col4.metric("Transit Depth", f"{fit_res['transit_depth']*100:.3f} %")

#         # 4. Diagnostic Charts
#         st.subheader("📈 Light Curve Visualizations")
#         fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        
#         # Left: Full Detrended Light Curve
#         axes[0].scatter(clean_lc.time.value, clean_lc.flux.value, s=1, color="black", alpha=0.4)
#         axes[0].set_title(f"{target_id} — Detrended Light Curve", fontsize=11)
#         axes[0].set_xlabel("Time (BJD - 2457000)")
#         axes[0].set_ylabel("Normalized Flux")
#         axes[0].grid(True, linestyle="--", alpha=0.3)
        
#         # Right: Folded Transit
#         phase_folded = clean_lc.fold(period=fit_res["period_days"], epoch_time=bls_stats["t0"])
#         axes[1].scatter(phase_folded.time.value, phase_folded.flux.value, s=2, color="teal", alpha=0.6)
#         axes[1].set_title(f"Phase Folded | {label} ({confidence*100:.1f}%)", fontsize=11)
#         axes[1].set_xlabel("Phase (Days)")
#         axes[1].set_ylabel("Normalized Flux")
#         axes[1].grid(True, linestyle="--", alpha=0.3)
        
#         plt.tight_layout()
#         st.pyplot(fig)
        
#         # 5. Phenomenon Explanation Card
#         st.subheader("📚 Astrophysical Breakdown")
#         info = ASTRO_EXPLANATIONS.get(label, ASTRO_EXPLANATIONS["Noise / False Positive"])
        
#         with st.container(border=True):
#             c_top1, c_top2 = st.columns([3, 1])
#             with c_top1:
#                 st.markdown(f"### {info['title']}")
#             with c_top2:
#                 st.caption(f"**Classification Category:** `{info['badge']}`")
                
#             st.write(info["description"])
            
#             st.markdown("**Key Light Curve Signatures:**")
#             for item in info["key_characteristics"]:
#                 st.markdown(f"- {item}")
                
#             st.info(f"💡 **Physical Insight:**\n\n{info['physics_insight']}")
        
#         # 6. Technical Breakdown Expander
#         with st.expander("📊 View Model Confidence Distribution & Parameters"):
#             p_col1, p_col2 = st.columns(2)
#             with p_col1:
#                 st.write("**Model Probabilities:**")
#                 st.progress(float(all_probs[0]), text=f"Exoplanet Candidate: {all_probs[0]*100:.2f}%")
#                 st.progress(float(all_probs[1]), text=f"Eclipsing Binary: {all_probs[1]*100:.2f}%")
#                 st.progress(float(all_probs[2]), text=f"Noise / False Positive: {all_probs[2]*100:.2f}%")
#             with p_col2:
#                 st.write("**Fitted Transit Metrics:**")
#                 st.write(f"- **Transit Duration:** {fit_res['duration_hours']:.2f} hours (± {fit_res['duration_err_hours']:.2f} hrs)")
#                 st.write(f"- **Transit Midpoint ($T_0$):** {bls_stats['t0']:.4f} BJD")
#                 st.write(f"- **BLS Max Power:** {bls_stats['power']:.2f}")

"""
app.py
Interactive Streamlit Web App for AI Exoplanet Detection with Astrophysical Insights.
Run with: python -m streamlit run app.py
"""

import os
import streamlit as st
import torch
import matplotlib.pyplot as plt

from src.download import download_target_lightcurve
from src.preprocess import clean_and_detrend, run_bls, generate_folded_vector
from src.classifier import ExoplanetCNN1D, predict_lightcurve
from src.fit_parameters import fit_transit_parameters, calculate_snr

# --- Page Configuration ---
st.set_page_config(page_title="AI Exoplanet Detector", page_icon="🪐", layout="wide")

# --- Astrophysical Explanations Knowledge Base ---
ASTRO_EXPLANATIONS = {
    "Exoplanet Candidate": {
        "title": "🪐 Exoplanet Transit Signal",
        "badge": "Planetary Candidate",
        "description": "An exoplanet is a planet orbiting a distant star. When the planet crosses in front of the star, it blocks a tiny fraction of photons.",
        "key_characteristics": ["U-Shaped Transit", "Shallow Depth", "Strict Periodicity"],
        "physics_insight": "Depth delta ~ (Rp / R*)^2"
    },
    "Eclipsing Binary": {
        "title": "⭐ Eclipsing Binary System",
        "badge": "Stellar Companion",
        "description": "Two luminous stars orbiting each other causing periodic drops in total flux.",
        "key_characteristics": ["V-Shaped Profile", "Substantial Flux Loss", "Primary & Secondary Eclipses"],
        "physics_insight": "Depth ratio scales with temperature ratio squared."
    },
    "Noise / False Positive": {
        "title": "📉 Stellar Activity / Artifact",
        "badge": "Non-Planetary",
        "description": "Stellar flares or telescope vibrations mimicking transit dips.",
        "key_characteristics": ["Stellar Flares", "Aperture Contamination", "Momentum Dumps"],
        "physics_insight": "Transits must remain invariant across sectors."
    }
}

# --- Model Loading (Cached) ---
@st.cache_resource
def load_trained_model():
    model = ExoplanetCNN1D()
    model_path = "./models/exoplanet_cnn.pth"
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

# --- Data Fetching (Cached) ---
@st.cache_data(show_spinner=False)
def fetch_and_process(target_name):
    raw_lc = download_target_lightcurve(target_name)
    if raw_lc is None:
        return None, None, None
    clean_lc = clean_and_detrend(raw_lc)
    bls_stats = run_bls(clean_lc, min_period=0.8, max_period=15.0)
    return raw_lc, clean_lc, bls_stats

# --- Header ---
st.title("🪐 AI Exoplanet Transit Detector")
st.markdown(
    "Detect and classify exoplanetary candidates from **NASA TESS Light Curves** using deep learning and Box Least Squares periodograms."
)

# --- Sidebar ---
st.sidebar.header("🎯 Target Selection")

with st.sidebar.expander("ℹ️ Guide for Custom TIC IDs"):
    st.write(
        "To ensure successful processing, custom TIC IDs must:\n"
        "* Have **2-minute SPOC cadence data** available in the TESS archive.\n"
        "* Not be limited strictly to FFI (Full Frame Image) data.\n\n"
        "*If an ID fails, it likely lacks the required high-cadence data format.*"
    )

preset_options = {
    "WASP-126 b (Confirmed Hot Jupiter)": "TIC 25155310",
    "TOI-175 / L 98-59 (Multi-planet System)": "TIC 149603524",
    "HD 21749 b (Sub-Neptune Exoplanet)": "TIC 307210830",
    "Known Eclipsing Binary System": "TIC 272074677",
    "Stellar Flare / Noise Artifact": "TIC 107150013",
    "Custom TIC ID": "custom"
}

selection = st.sidebar.selectbox("Choose a target star:", list(preset_options.keys()))

if preset_options[selection] == "custom":
    target_id = st.sidebar.text_input("Enter TIC ID (e.g. TIC 25155310):", value="TIC 25155310")
else:
    target_id = preset_options[selection]
    st.sidebar.info(f"Target: **{target_id}**")

run_button = st.sidebar.button("🚀 Analyze Light Curve", type="primary", use_container_width=True)

# --- Execution ---
if run_button or target_id:
    model = load_trained_model()
    
    with st.spinner(f"Querying NASA MAST Archive for {target_id}..."):
        raw_lc, clean_lc, bls_stats = fetch_and_process(target_id)
        
    if raw_lc is None:
        st.error(
            f"❌ Could not retrieve high-cadence data for **{target_id}**.\n\n"
            "**Possible reasons:**\n"
            "- The star was only observed in Full Frame Images (FFIs), not 2-minute cadence.\n"
            "- The TIC ID is invalid or has no Lightkurve SPOC records.\n"
            "- The NASA MAST server timed out.\n\n"
            "**Tip:** Try a verified preset from the dropdown menu!"
        )
    else:
        # 1. AI Inference
        folded_vector = generate_folded_vector(clean_lc, bls_stats["period"], bls_stats["t0"], num_bins=500)
        label, confidence, all_probs = predict_lightcurve(model, folded_vector, device="cpu")
        
        # 2. Parameter Fitting & SNR
        fit_res = fit_transit_parameters(clean_lc, bls_stats)
        snr = calculate_snr(
            clean_lc,
            period=fit_res["period_days"],
            t0=bls_stats["t0"],
            duration_days=fit_res["duration_hours"] / 24.0,
            depth=fit_res["transit_depth"]
        )
        
        # --- ASTROPHYSICAL SANITY CHECK ---
        if fit_res["transit_depth"] > 0.03 and label == "Exoplanet Candidate":
            label = "Eclipsing Binary"
            st.warning("⚠️ **Physics Override:** Reclassified as an Eclipsing Binary due to a transit depth greater than 3%.")

        elif label == "Exoplanet Candidate" and (bls_stats['power'] < 5.0 or fit_res['transit_depth'] < 0.0005):
            label = "Noise / False Positive"
            st.warning("⚠️ **Physics Override:** Reclassified as Noise/False Positive because the signal lacks a consistent planetary transit profile.")

        # 3. Top Metrics Banner
        st.subheader("🔍 Analysis Verdict")
        col1, col2, col3, col4 = st.columns(4)
        
        if label == "Exoplanet Candidate":
            col1.metric("Classification", "🪐 Exoplanet", delta=f"{confidence*100:.1f}% Confidence")
        elif label == "Eclipsing Binary":
            col1.metric("Classification", "⭐ Binary Star", delta="Non-Planetary", delta_color="inverse")
        else:
            col1.metric("Classification", "📉 Noise / Artifact", delta="Low Signal", delta_color="off")
            
        col2.metric("Signal-to-Noise (SNR)", f"{snr:.2f} σ")
        col3.metric("Orbital Period", f"{fit_res['period_days']:.4f} days")
        col4.metric("Transit Depth", f"{fit_res['transit_depth']*100:.3f} %")

        # 4. Diagnostic Charts
        st.subheader("📈 Light Curve Visualizations")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        
        axes[0].scatter(clean_lc.time.value, clean_lc.flux.value, s=1, color="black", alpha=0.4)
        axes[0].set_title(f"{target_id} — Detrended Light Curve", fontsize=11)
        axes[0].set_xlabel("Time (BJD - 2457000)")
        axes[0].set_ylabel("Normalized Flux")
        axes[0].grid(True, linestyle="--", alpha=0.3)
        
        phase_folded = clean_lc.fold(period=fit_res["period_days"], epoch_time=bls_stats["t0"])
        axes[1].scatter(phase_folded.time.value, phase_folded.flux.value, s=2, color="teal", alpha=0.6)
        axes[1].set_title(f"Phase Folded | {label} ({confidence*100:.1f}%)", fontsize=11)
        axes[1].set_xlabel("Phase (Days)")
        axes[1].set_ylabel("Normalized Flux")
        axes[1].grid(True, linestyle="--", alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # 5. Phenomenon Explanation Card
        st.subheader("📚 Astrophysical Breakdown")
        info = ASTRO_EXPLANATIONS.get(label, ASTRO_EXPLANATIONS["Noise / False Positive"])
        
        with st.container(border=True):
            c_top1, c_top2 = st.columns([3, 1])
            with c_top1:
                st.markdown(f"### {info['title']}")
            with c_top2:
                st.caption(f"**Classification Category:** `{info['badge']}`")
                
            st.write(info["description"])
            
            st.markdown("**Key Light Curve Signatures:**")
            for item in info["key_characteristics"]:
                st.markdown(f"- {item}")
                
            st.info(f"💡 **Physical Insight:**\n\n{info['physics_insight']}")
        
        # 6. Technical Breakdown Expander
        with st.expander("📊 View Model Confidence Distribution & Parameters"):
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.write("**Model Probabilities:**")
                st.progress(float(all_probs[0]), text=f"Exoplanet Candidate: {all_probs[0]*100:.2f}%")
                st.progress(float(all_probs[1]), text=f"Eclipsing Binary: {all_probs[1]*100:.2f}%")
                st.progress(float(all_probs[2]), text=f"Noise / False Positive: {all_probs[2]*100:.2f}%")
            with p_col2:
                st.write("**Fitted Transit Metrics:**")
                st.write(f"- **Transit Duration:** {fit_res['duration_hours']:.2f} hours (± {fit_res['duration_err_hours']:.2f} hrs)")
                st.write(f"- **Transit Midpoint ($T_0$):** {bls_stats['t0']:.4f} BJD")
                st.write(f"- **BLS Max Power:** {bls_stats['power']:.2f}")
