import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Solar IMS — Anomaly Detection",
    page_icon="☀️",
    layout="wide"
)

# ── Load model and scaler ────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    from tensorflow.keras.models import load_model
    from tensorflow.keras.optimizers import Adam
    model = load_model('solar_ims_lstm_model.h5', compile=False)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    scaler = joblib.load('solar_ims_scaler.pkl')
    return model, scaler

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Sun_appearance.jpg/240px-Sun_appearance.jpg", width=80)
st.sidebar.title("Solar IMS")
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
st.sidebar.markdown("4.7 MW | Colorado")
st.sidebar.markdown("2017 – 2023")
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
    col4.metric("Anomaly Threshold", "0.0047", "95th percentile")

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
    cols = st.columns(7)
    steps = [
        ("1", "Download", "4 CSV files from NREL S3"),
        ("2", "Clean & Merge", "Fill missing values, join on timestamp"),
        ("3", "Select Features", "12 key columns from 162"),
        ("4", "Normalize", "Scale all values to 0–1"),
        ("5", "Sequences", "96-step sliding windows"),
        ("6", "Train LSTM", "Model learns normal pattern"),
        ("7", "Flag Anomalies", "High error = anomaly"),
    ]
    colors = ["#0D9488", "#0D9488", "#F59E0B", "#F59E0B", "#7C3AED", "#7C3AED", "#EA580C"]
    for col, (n, label, desc), color in zip(cols, steps, colors):
        col.markdown(f"""
        <div style='text-align:center; padding:10px; background:{color}; border-radius:50%; width:40px; height:40px; margin:auto; color:white; font-weight:bold; line-height:20px;'>{n}</div>
        <div style='text-align:center; font-weight:bold; margin-top:8px; font-size:12px;'>{label}</div>
        <div style='text-align:center; color:gray; font-size:11px;'>{desc}</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Dataset Files")
    files_df = pd.DataFrame({
        "File": ["9068_ac_power_data.csv", "9068_irradiance_data.csv", "9068_environment_data.csv", "9068_acvolt_curr_data.csv"],
        "What It Contains": ["Power output per inverter module and plant meter", "Sunlight intensity at Pad 1 and Pad 2", "Temperature, wind speed, inverter thermal readings", "AC current and voltage across three phases"],
        "Columns": ["17", "5", "100", "43"]
    })
    st.dataframe(files_df, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 2 — MODEL RESULTS
# ════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("📊 Model Results")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Data", "6 Years", "2017 — 2023")
    col2.metric("Val Loss", "0.0011", "Final best epoch")
    col3.metric("Anomaly Threshold", "0.004733", "95th percentile MSE")
    col4.metric("Anomalies Detected", "6,402", "Out of 128,039 sequences")

    st.markdown("---")
    st.subheader("Anomaly Detection — System 9068 (July 2022 to November 2023)")

    if os.path.exists('anomaly_detection_6yr_model.png'):
        st.image('anomaly_detection_6yr_model.png', use_container_width=True)
    else:
        st.warning("anomaly_detection_6yr_model.png not found. Please add it to the repository.")

    st.markdown("---")
    st.subheader("What the Graph Shows")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Blue line** — Reconstruction error over time")
        st.markdown("When low and flat → plant is behaving normally")
        st.markdown("When spiking → model struggled to reconstruct → anomaly")
    with col2:
        st.markdown("**Red dashed line** — Anomaly threshold at 0.0047")
        st.markdown("**Red shaded areas** — Confirmed anomaly periods")
        st.markdown("Most active anomaly period: March to May 2023")

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

    if os.path.exists('anomaly_timestamps_6yr.csv'):
        df_anomalies = pd.read_csv('anomaly_timestamps_6yr.csv')
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

        # Monthly breakdown
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

        # Download button
        csv = df_anomalies.to_csv(index=False)
        st.download_button(
            label="Download Anomaly Timestamps CSV",
            data=csv,
            file_name="anomaly_timestamps.csv",
            mime="text/csv"
        )
    else:
        st.warning("anomaly_timestamps_6yr.csv not found. Please add it to the repository.")

# ════════════════════════════════════════════════════════
# PAGE 4 — LIVE DEMO
# ════════════════════════════════════════════════════════
elif page == "⚡ Live Demo":
    st.title("⚡ Live Anomaly Detection Demo")
    st.markdown("---")

    st.info("""
    **How this demo works:**
    The LSTM Autoencoder was trained on 6 years of normal plant behaviour.
    It tries to reconstruct the input data. If reconstruction error exceeds
    the threshold of **0.004733**, the sequence is flagged as an anomaly.
    """)

    st.markdown("---")
    st.subheader("Synthetic Plant Data — Simulated Inverter Fault")
    st.markdown("""
    The data below simulates a real inverter fault scenario:
    - **Readings 1–14** → Normal healthy operation
    - **Readings 15–17** → Inverter 1 fault (IGBT overheating, power collapse)
    - **Readings 18–20** → Recovery after fault cleared
    """)

    # Generate synthetic data
    np.random.seed(42)
    timestamps = pd.date_range('2023-06-01 09:00', periods=20, freq='5min')

    df_demo = pd.DataFrame({
        'Timestamp': timestamps,
        'Total Power (norm)':    [0.85,0.86,0.84,0.85,0.87,0.85,0.84,0.86,0.85,0.84,0.86,0.85,0.84,0.85, 0.12,0.10,0.11, 0.80,0.83,0.85],
        'Irradiance (norm)':     [0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81,0.80,0.81, 0.80,0.81,0.80, 0.80,0.80,0.81],
        'Ambient Temp (norm)':   [0.55,0.55,0.56,0.55,0.55,0.56,0.55,0.55,0.56,0.55,0.55,0.56,0.55,0.55, 0.56,0.56,0.57, 0.55,0.55,0.56],
        'Panel Temp (norm)':     [0.60,0.60,0.61,0.60,0.60,0.61,0.60,0.60,0.61,0.60,0.60,0.61,0.60,0.60, 0.61,0.61,0.62, 0.60,0.61,0.60],
        'Inverter 1 Power':      [0.43,0.43,0.44,0.43,0.43,0.44,0.43,0.43,0.44,0.43,0.43,0.44,0.43,0.43, 0.05,0.04,0.05, 0.40,0.42,0.43],
        'Inverter 2 Power':      [0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43,0.42,0.43, 0.42,0.43,0.42, 0.42,0.42,0.43],
        'IGBT Temp Inv1 (norm)': [0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50, 0.90,0.92,0.91, 0.55,0.52,0.51],
        'IGBT Temp Inv2 (norm)': [0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50,0.51,0.50,0.50, 0.50,0.51,0.50, 0.50,0.50,0.51],
        'Phase 1 Current':       [0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84, 0.20,0.18,0.19, 0.80,0.82,0.84],
        'Phase 2 Current':       [0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84, 0.84,0.85,0.84, 0.84,0.84,0.85],
        'Phase 3 Current':       [0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84,0.85,0.84,0.84, 0.84,0.85,0.84, 0.84,0.84,0.85],
    })

    # Highlight fault rows
    def highlight_fault(row):
        if row.name in [14, 15, 16]:
            return ['background-color: #FFF3CD'] * len(row)
        return [''] * len(row)

    st.dataframe(df_demo.style.apply(highlight_fault, axis=1), use_container_width=True)

    st.markdown("---")

    if st.button("▶ Run Anomaly Detection on This Data", type="primary"):
        with st.spinner("Loading model and running detection..."):
            try:
                model, scaler = load_model_and_scaler()

                feature_cols = [
                    'Total Power (norm)', 'Irradiance (norm)', 'Ambient Temp (norm)',
                    'Panel Temp (norm)', 'Inverter 1 Power', 'Inverter 2 Power',
                    'IGBT Temp Inv1 (norm)', 'IGBT Temp Inv2 (norm)',
                    'Phase 1 Current', 'Phase 2 Current', 'Phase 3 Current'
                ]

                demo_data = df_demo[feature_cols].values
                demo_padded = np.tile(demo_data, (5, 1))[:96]
                demo_input = demo_padded.reshape(1, 96, 11)

                demo_pred = model.predict(demo_input, verbose=0)
                demo_error = np.mean(np.power(demo_input - demo_pred, 2))

                THRESHOLD = 0.004733

                st.markdown("---")
                st.subheader("Detection Result")

                col1, col2, col3 = st.columns(3)
                col1.metric("Reconstruction Error", f"{demo_error:.6f}")
                col2.metric("Anomaly Threshold", f"{THRESHOLD:.6f}")
                col3.metric("Error vs Threshold", f"{demo_error/THRESHOLD:.1f}x")

                if demo_error > THRESHOLD:
                    st.error(f"""
                    ### ⚠️ ANOMALY DETECTED

                    Reconstruction error **{demo_error:.6f}** exceeds threshold **{THRESHOLD:.6f}**

                    **Root cause indicators:**
                    - Total plant power dropped to ~10% while irradiance remained high
                    - Inverter 1 IGBT temperature spiked to 90% of maximum
                    - Phase 1 current dropped sharply while Phases 2 and 3 remained normal

                    **Recommended action:** Inspect Inverter 1 immediately — likely thermal fault
                    """)
                else:
                    st.success(f"""
                    ### ✅ Normal Operation

                    Reconstruction error **{demo_error:.6f}** is within normal range (threshold: {THRESHOLD:.6f})
                    """)

                # Plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                ax1.plot(df_demo['Timestamp'], df_demo['Total Power (norm)'],
                         color='steelblue', marker='o', linewidth=1.5, markersize=4)
                ax1.axvspan(df_demo['Timestamp'].iloc[14], df_demo['Timestamp'].iloc[16],
                            color='red', alpha=0.2, label='Fault period')
                ax1.set_title('Plant Power Output')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Normalized Power')
                ax1.tick_params(axis='x', rotation=30)
                ax1.legend()

                ax2.bar(['Reconstruction\nError', 'Anomaly\nThreshold'],
                        [demo_error, THRESHOLD],
                        color=['red' if demo_error > THRESHOLD else 'steelblue', 'orange'],
                        alpha=0.8, width=0.4)
                ax2.set_title('Error vs Threshold')
                ax2.set_ylabel('MSE')

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.info("Make sure solar_ims_lstm_model.h5 and solar_ims_scaler.pkl are in the repository.")
