import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(
    page_title="Solar IMS — Anomaly Detection",
    page_icon="☀️",
    layout="wide"
)

@st.cache_resource
def load_model_and_scaler():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam

    model_path = None
    scaler_path = None

    # Search in current directory and all subdirectories
    for root, dirs, files in os.walk('.'):
        for f in files:
            full_path = os.path.join(root, f)
            if f.endswith('.keras'):
                model_path = full_path
            if f.endswith('.pkl'):
                scaler_path = full_path

    if model_path is None:
        raise FileNotFoundError("solar_ims_model.keras not found")
    if scaler_path is None:
        raise FileNotFoundError("solar_ims_scaler.pkl not found")

    # Load keras format model — no compile=False needed
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# ── Sidebar ──
st.sidebar.title("☀️ Solar IMS")
st.sidebar.markdown("**Use Case 1**")
st.sidebar.markdown("Anomaly Detection &\nPredictive Maintenance")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📋 Project Overview",
    "📊 Model Results",
    "🕐 Anomaly Timestamps",
    "⚡ Live Demo"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset**")
st.sidebar.markdown("NREL PVDAQ System 9068")
st.sidebar.markdown("4.7 MW | Colorado | 2017–2023")
st.sidebar.markdown("---")
st.sidebar.markdown("**Bandhan Technologies Inc.**")
st.sidebar.markdown("June 2026")

# ════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ════════════════════════════════════════════════════════
if page == "📋 Project Overview":
    st.title("☀️ Solar Intelligent Monitoring System")
    st.subheader("Use Case 1 — Anomaly Detection & Predictive Maintenance")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Plant Capacity", "4.7 MW", "Single-axis tracker")
    col2.metric("Data Range", "6 Years", "2017 — 2023")
    col3.metric("Model Parameters", "53,195", "LSTM Autoencoder")
    col4.metric("Anomaly Threshold", "0.0047", "95th percentile MSE")

    st.markdown("---")
    st.subheader("What Problem Are We Solving?")
    col1, col2 = st.columns(2)
    with col1:
        st.error("**Without Solar IMS**")
        st.markdown("""
        - Plant faults go undetected for days or weeks
        - Manual inspection is error-prone and unscalable
        - Every undetected fault day = lost revenue
        - Impossible to monitor 100s of plants manually
        """)
    with col2:
        st.success("**With Solar IMS**")
        st.markdown("""
        - Anomalies detected within 5 minutes of occurrence
        - LSTM model learns normal pattern — flags deviations
        - Works on real sensor data automatically
        - Same model scales across 1 to 1000 plants
        """)

    st.markdown("---")
    st.subheader("How It Works — 7 Steps")
    steps = [
        ("1", "Download", "4 CSV files from NREL S3", "#0D9488"),
        ("2", "Clean & Merge", "Fill missing values, join on timestamp", "#0D9488"),
        ("3", "Select Features", "12 key columns from 162", "#F59E0B"),
        ("4", "Normalize", "Scale all values to 0–1", "#F59E0B"),
        ("5", "Sequences", "96-step sliding windows (8 hrs)", "#7C3AED"),
        ("6", "Train LSTM", "Model learns normal plant behaviour", "#7C3AED"),
        ("7", "Flag Anomalies", "High reconstruction error = anomaly", "#EA580C"),
    ]
    cols = st.columns(7)
    for col, (n, label, desc, color) in zip(cols, steps):
        col.markdown(f"""
        <div style='text-align:center; padding:10px; background:{color};
        border-radius:50%; width:40px; height:40px; margin:auto;
        color:white; font-weight:bold; line-height:20px;'>{n}</div>
        <div style='text-align:center; font-weight:bold; margin-top:8px; font-size:12px;'>{label}</div>
        <div style='text-align:center; color:gray; font-size:11px;'>{desc}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Dataset Files")
    files_df = pd.DataFrame({
        "File": ["9068_ac_power_data.csv", "9068_irradiance_data.csv",
                 "9068_environment_data.csv", "9068_acvolt_curr_data.csv"],
        "What It Contains": [
            "Power output per inverter module and plant meter",
            "Sunlight intensity at Pad 1 and Pad 2 (POA)",
            "Temperature, wind speed, inverter thermal readings",
            "AC current and voltage across three electrical phases"
        ],
        "Columns": ["17", "5", "100", "43"]
    })
    st.dataframe(files_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Numbers")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows Merged", "640,289")
    col2.metric("Features Selected", "12")
    col3.metric("Sequences Created", "640,193")
    col4.metric("Model Parameters", "53,195")

# ════════════════════════════════════════════════════════
# PAGE 2 — MODEL RESULTS
# ════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("📊 Model Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Data", "6 Years", "2017 — 2023")
    col2.metric("Best Val Loss", "0.0011", "After 50 epochs")
    col3.metric("Anomaly Threshold", "0.004733", "95th percentile MSE")
    col4.metric("Anomalies Detected", "6,402", "Out of 128,039 sequences")

    st.markdown("---")
    st.subheader("Anomaly Detection — System 9068 (July 2022 to November 2023)")

    if os.path.exists('anomaly_detection_6yr_model.png'):
        st.image('anomaly_detection_6yr_model.png', use_container_width=True)
    else:
        st.warning("anomaly_detection_6yr_model.png not found in repository.")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Reading the Graph")
        st.markdown("**Blue line** — Reconstruction error over time")
        st.markdown("Low and flat = plant behaving normally")
        st.markdown("Spike upward = model struggled to reconstruct = anomaly")
        st.markdown("**Red dashed line** — Anomaly threshold at 0.0047")
        st.markdown("**Red shaded areas** — Confirmed anomaly periods")
    with col2:
        st.subheader("Key Anomaly Periods")
        periods = pd.DataFrame({
            "Period": ["Aug–Sep 2022", "Mar–May 2023", "Jul 2023", "Aug 2023"],
            "Observation": [
                "Early anomaly spikes at start of test period",
                "Most active anomaly period — multiple large spikes",
                "Single highest spike at 0.021 — worst anomaly detected",
                "Sustained red region — prolonged fault period"
            ]
        })
        st.dataframe(periods, use_container_width=True)

    st.markdown("---")
    st.subheader("LSTM Autoencoder Architecture")
    arch_df = pd.DataFrame({
        "Layer": ["Input", "Encoder LSTM", "RepeatVector", "Decoder LSTM", "Output Dense"],
        "Output Shape": ["(None, 96, 11)", "(None, 64)", "(None, 96, 64)", "(None, 96, 64)", "(None, 96, 11)"],
        "Parameters": ["0", "19,456", "0", "33,024", "715"],
        "Role": [
            "Accepts 96 timesteps × 11 features",
            "Compresses sequence to 64 numbers",
            "Copies 64 numbers 96 times for decoder",
            "Reconstructs original sequence",
            "Maps back to 11 original features"
        ]
    })
    st.dataframe(arch_df, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY TIMESTAMPS
# ════════════════════════════════════════════════════════
elif page == "🕐 Anomaly Timestamps":
    st.title("🕐 Anomaly Timestamps")
    st.markdown("---")

    csv_path = None
    for f in os.listdir('.'):
        if f.endswith('.csv'):
            csv_path = f

    if csv_path and os.path.exists(csv_path):
        df_anomalies = pd.read_csv(csv_path)
        df_anomalies.columns = ['Timestamp']
        df_anomalies['Timestamp'] = pd.to_datetime(df_anomalies['Timestamp'])
        df_anomalies['Date'] = df_anomalies['Timestamp'].dt.date
        df_anomalies['Time'] = df_anomalies['Timestamp'].dt.time
        df_anomalies['Month'] = df_anomalies['Timestamp'].dt.strftime('%B %Y')

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Anomaly Readings", f"{len(df_anomalies):,}")
        col2.metric("First Anomaly", str(df_anomalies['Timestamp'].min().date()))
        col3.metric("Last Anomaly", str(df_anomalies['Timestamp'].max().date()))

        st.markdown("---")
        st.subheader("Anomalies by Month")
        monthly = df_anomalies.groupby('Month').size().reset_index(name='Count')
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(monthly['Month'], monthly['Count'], color='steelblue', alpha=0.8)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Anomaly Readings')
        ax.set_title('Anomaly Count by Month')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("Full Anomaly Timestamp List")
        st.dataframe(df_anomalies[['Timestamp', 'Date', 'Time']], use_container_width=True)

        csv_download = df_anomalies.to_csv(index=False)
        st.download_button(
            label="Download Anomaly Timestamps CSV",
            data=csv_download,
            file_name="anomaly_timestamps.csv",
            mime="text/csv"
        )
    else:
        st.warning("No CSV file found. Please add anomaly_timestamps_6yr.csv to the repository.")

# ════════════════════════════════════════════════════════
# PAGE 4 — LIVE DEMO
# ════════════════════════════════════════════════════════
elif page == "⚡ Live Demo":
    st.title("⚡ Live Anomaly Detection Demo")
    st.markdown("---")

    st.info("""
    **How this demo works:**
    The LSTM Autoencoder was trained on 6 years of real plant data from NREL PVDAQ System 9068.
    It tries to reconstruct the input sequence. If reconstruction error exceeds the threshold
    of **0.004733**, the sequence is flagged as an anomaly.
    The model has never seen this synthetic data before.
    """)

    st.markdown("---")
    st.subheader("Synthetic Plant Data — Simulated Inverter Fault")

    col1, col2, col3 = st.columns(3)
    col1.success("Readings 1–14: Normal operation")
    col2.error("Readings 15–17: Inverter 1 fault")
    col3.success("Readings 18–20: Recovery")

    np.random.seed(42)
    timestamps = pd.date_range('2023-06-01 09:00', periods=20, freq='5min')

    df_demo = pd.DataFrame({
        'Timestamp': timestamps,
        'Total Power':      [0.85,0.86,0.84,0.85,0.87,0.85,0.84,0.86,0.85,0.84,0.86,0.85,0.84,0.85, 0.12,0.10,0.11, 0.80,0.83,0.85],
        'Irradiance':       [0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81, 0.80,0.81,0.80, 0.80,0.80,0.81],
        'Ambient Temp':     [0.55,0.55,0.56,0.55,0.55,0.56,0.55,0.55,0.56,0.55,0.55,0.56,0.55,0.55, 0.56,0.56,0.57, 0.55,0.55,0.56],
        'Panel Temp':       [0.60,0.60,0.61,0.60,0.60,0.61,0.60,0.60,0.61,0.60,0.60,0.61,0.60,0.60, 0.61,0.61,0.62, 0.60,0.61,0.60],
        'Inverter 1 Power': [0.43,0.43,0.44,0.43,0.43,0.44,0.43,0.43,0.44,0.43,0.43,0.44,0.43,0.43, 0.05,0.04,0.05, 0.40,0.42,0.43],
        'Inverter 2 Power': [0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43, 0.42,0.43,0.42, 0.42,0.42,0.43],
        'IGBT Temp Inv1':   [0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50, 0.90,0.92,0.91, 0.55,0.52,0.51],
        'IGBT Temp Inv2':   [0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50, 0.50,0.51,0.50, 0.50,0.50,0.51],
        'Phase 1 Current':  [0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84, 0.20,0.18,0.19, 0.80,0.82,0.84],
        'Phase 2 Current':  [0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84, 0.84,0.85,0.84, 0.84,0.84,0.85],
        'Phase 3 Current':  [0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84, 0.84,0.85,0.84, 0.84,0.84,0.85],
    })

    def highlight_fault(row):
        if row.name in [14, 15, 16]:
            return ['background-color: #FFF3CD; color: black'] * len(row)
        return [''] * len(row)

    st.dataframe(df_demo.style.apply(highlight_fault, axis=1), use_container_width=True)

    st.markdown("---")

    if st.button("▶ Run Anomaly Detection", type="primary", use_container_width=True):
        with st.spinner("Loading model and running anomaly detection..."):
            try:
                model, scaler = load_model_and_scaler()

                feature_cols = [
                    'Total Power', 'Irradiance', 'Ambient Temp', 'Panel Temp',
                    'Inverter 1 Power', 'Inverter 2 Power',
                    'IGBT Temp Inv1', 'IGBT Temp Inv2',
                    'Phase 1 Current', 'Phase 2 Current', 'Phase 3 Current'
                ]

                demo_data = df_demo[feature_cols].values
                demo_padded = np.tile(demo_data, (5, 1))[:96]
                demo_input = demo_padded.reshape(1, 96, 11)

                demo_pred = model.predict(demo_input, verbose=0)
                demo_error = float(np.mean(np.power(demo_input - demo_pred, 2)))

                THRESHOLD = 0.004733

                st.markdown("---")
                st.subheader("Detection Result")

                col1, col2, col3 = st.columns(3)
                col1.metric("Reconstruction Error", f"{demo_error:.6f}")
                col2.metric("Anomaly Threshold", f"{THRESHOLD:.6f}")
                col3.metric("Error vs Threshold", f"{demo_error/THRESHOLD:.1f}x higher")

                if demo_error > THRESHOLD:
                    st.error(f"""
                    ### ⚠️ ANOMALY DETECTED

                    Reconstruction error **{demo_error:.6f}** exceeds threshold **{THRESHOLD:.6f}**

                    **Root cause indicators from sensor data:**
                    - Total plant power collapsed to ~10% while irradiance remained high
                    - Inverter 1 IGBT temperature spiked to 90% of maximum — thermal stress
                    - Phase 1 current dropped sharply while Phases 2 and 3 remained normal

                    **Recommended action:** Inspect Inverter 1 immediately — likely thermal fault
                    """)
                else:
                    st.success(f"### ✅ Normal Operation — Error {demo_error:.6f} is within normal range")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                colors_power = ['red' if i in [14,15,16] else 'steelblue' for i in range(20)]
                ax1.bar(range(20), df_demo['Total Power'], color=colors_power, alpha=0.8)
                ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Expected ~0.85')
                ax1.set_title('Plant Power Output — Red = Fault Period')
                ax1.set_xlabel('Reading Number')
                ax1.set_ylabel('Normalized Power')
                ax1.legend()

                bar_colors = ['red' if demo_error > THRESHOLD else 'steelblue', 'orange']
                ax2.bar(['Reconstruction\nError', 'Anomaly\nThreshold'],
                        [demo_error, THRESHOLD],
                        color=bar_colors, alpha=0.85, width=0.4)
                ax2.set_title('Error vs Threshold')
                ax2.set_ylabel('MSE Value')
                for i, v in enumerate([demo_error, THRESHOLD]):
                    ax2.text(i, v + 0.0001, f'{v:.6f}', ha='center', fontsize=9)

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure solar_ims_model.keras and solar_ims_scaler.pkl are in the repository.")
