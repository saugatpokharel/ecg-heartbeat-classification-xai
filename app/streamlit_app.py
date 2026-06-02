# ── Imports ───────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import wfdb
import os
import tempfile
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Page configuration ────────────────────────────────────
st.set_page_config(
    page_title="ECG Heartbeat Classification",
    page_icon="🫀",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────
st.title("🫀 ECG Heartbeat Classification System")
st.markdown("""
This system analyses ECG recordings from the MIT-BIH Arrhythmia
Database and classifies each heartbeat as Normal or Abnormal
using a trained Random Forest model. SHAP explanations show
exactly why each prediction was made.
""")
st.divider()

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    return model, scaler, feature_names

@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)

rf_model, scaler, feature_names = load_model()
explainer = load_shap_explainer(rf_model)

data_path = 'data/mit-bih-arrhythmia-database-1.0.0'

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("ECG Analysis System")
st.sidebar.success("Model loaded successfully!")
st.sidebar.divider()
st.sidebar.markdown("""
**Beat types:**
- 🟢 Normal (N)
- 🔴 Abnormal (L, R, V, A...)
""")
st.sidebar.divider()
st.sidebar.markdown("**Model:** Random Forest")
st.sidebar.markdown("**Accuracy:** 97.26%")
st.sidebar.markdown("**Dataset:** MIT-BIH Arrhythmia Database")

# ── Feature extraction function ───────────────────────────
def extract_features_single(beat, rr_interval):
    features = {}
    features['mean']             = np.mean(beat)
    features['std']              = np.std(beat)
    features['max']              = np.max(beat)
    features['min']              = np.min(beat)
    features['range']            = np.max(beat) - np.min(beat)
    features['r_peak_amplitude'] = beat[180]
    features['skewness']         = stats.skew(beat)
    features['kurtosis']         = stats.kurtosis(beat)
    threshold  = 0.3 * beat[180]
    qrs_region = beat[150:210]
    if beat[180] > 0:
        qrs_samples = np.sum(qrs_region > threshold)
    else:
        qrs_samples = np.sum(qrs_region < threshold)
    features['qrs_width']        = qrs_samples / 360
    features['rr_interval']      = rr_interval
    features['mean_first_half']  = np.mean(beat[:180])
    features['mean_second_half'] = np.mean(beat[180:])
    features['half_difference']  = (features['mean_second_half']
                                   - features['mean_first_half'])
    return features

# ── Process record function ───────────────────────────────
def process_record_path(record_path):
    valid_beats = ['N', 'L', 'R', 'V', 'A', 'F', 'f',
                   'j', 'a', 'J', 'S', 'E', 'e']
    before = 180
    after  = 180

    record     = wfdb.rdrecord(record_path)
    signal     = record.p_signal[:, 0]
    annotation = wfdb.rdann(record_path, 'atr')

    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    beats          = []
    beat_features  = []
    beat_labels    = []
    beat_positions = []

    for i, (sample, symbol) in enumerate(
            zip(ann_samples, ann_symbols)):
        if symbol not in valid_beats:
            continue
        if sample - before < 0 or sample + after >= len(signal):
            continue

        beat = signal[sample - before: sample + after]

        if i < len(ann_samples) - 1:
            rr = (ann_samples[i + 1] - sample) / 360
        elif i > 0:
            rr = (sample - ann_samples[i - 1]) / 360
        else:
            rr = 0.0

        features = extract_features_single(beat, rr)
        beats.append(beat)
        beat_features.append(features)
        beat_labels.append(symbol)
        beat_positions.append(sample)

    return (signal, beats, beat_features,
            beat_labels, beat_positions, record.fs)

# ── Analysis display function ─────────────────────────────
def show_analysis(signal, beats, beat_features,
                  beat_labels, beat_positions, fs, title):

    features_df   = pd.DataFrame(beat_features)
    predictions   = rf_model.predict(features_df)
    probabilities = rf_model.predict_proba(features_df)

    total_beats    = len(predictions)
    normal_count   = int(np.sum(predictions == 0))
    abnormal_count = int(np.sum(predictions == 1))

    # Summary metrics
    st.subheader(f"{title} — Analysis Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Beats", total_beats)
    col2.metric("Normal Beats",
                f"{normal_count} ({normal_count/total_beats*100:.1f}%)")
    col3.metric("Abnormal Beats",
                f"{abnormal_count} ({abnormal_count/total_beats*100:.1f}%)")

    st.divider()

    # ECG plot
    st.subheader("Raw ECG Signal — First 10 Seconds")
    fig, ax = plt.subplots(figsize=(14, 3))
    plot_samples = min(10 * fs, len(signal))
    time         = np.arange(plot_samples) / fs
    ax.plot(time, signal[:plot_samples],
            color='steelblue', linewidth=0.8)
    for pos, pred in zip(beat_positions, predictions):
        if pos < plot_samples:
            color = 'green' if pred == 0 else 'red'
            ax.axvline(x=pos/fs, color=color,
                       alpha=0.4, linewidth=1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # Beat table
    st.subheader("Beat Classification Results")
    results_data = []
    for i, (pred, prob, label, pos) in enumerate(
            zip(predictions, probabilities,
                beat_labels, beat_positions)):
        results_data.append({
            'Beat #':       i + 1,
            'Time (sec)':   round(pos / fs, 2),
            'Actual Label': label,
            'Prediction':   '🟢 Normal' if pred == 0
                            else '🔴 Abnormal',
            'Confidence':   f"{max(prob)*100:.1f}%"
        })

    results_df   = pd.DataFrame(results_data)
    show_abnormal = st.checkbox(
        "Show abnormal beats only",
        value=False,
        key=f"checkbox_{title}")
    if show_abnormal:
        display_df = results_df[
            results_df['Prediction'] == '🔴 Abnormal']
        st.write(f"Showing {len(display_df)} abnormal beats")
    else:
        display_df = results_df
        st.write(f"Showing all {len(display_df)} beats")
    st.dataframe(display_df,
                 use_container_width=True, height=300)

    st.divider()

    # Beat inspector
    st.subheader("Individual Beat Inspector")
    beat_number = st.number_input(
        "Enter beat number:",
        min_value=1,
        max_value=total_beats,
        value=1,
        step=1,
        key=f"beat_input_{title}")

    beat_idx            = beat_number - 1
    selected_beat       = np.array(beats[beat_idx])
    selected_prediction = predictions[beat_idx]
    selected_prob       = probabilities[beat_idx]
    selected_label      = beat_labels[beat_idx]
    selected_features   = features_df.iloc[beat_idx]

    pred_text  = ("🟢 Normal" if selected_prediction == 0
                  else "🔴 Abnormal")
    confidence = max(selected_prob) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Beat Number", beat_number)
    col2.metric("Prediction",  pred_text)
    col3.metric("Confidence",  f"{confidence:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(selected_beat, color='steelblue', linewidth=1.5)
    ax.axvline(x=180, color='red', linestyle='--',
               alpha=0.7, label='R-peak')
    ax.set_title(
        f'Beat {beat_number} — Predicted: {pred_text}'
        f'  |  Actual: {selected_label}',
        fontsize=11)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.divider()

    # SHAP explanation
    st.subheader("SHAP Explanation")
    st.markdown("Why did the model make this prediction?")

    shap_values = explainer.shap_values(
        features_df.iloc[[beat_idx]])
    shap_array  = np.array(shap_values)

    if len(shap_array.shape) == 3:
        shap_vals = shap_array[0, :, 1]
    else:
        shap_vals = shap_array[1][0]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors     = ['steelblue' if v < 0 else 'tomato'
                  for v in shap_vals]
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

    ax.barh(range(13),
            [shap_vals[i] for i in sorted_idx],
            color=[colors[i] for i in sorted_idx])
    ax.set_yticks(range(13))
    ax.set_yticklabels(
        [feature_names[i] for i in sorted_idx], fontsize=9)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title(
        f'SHAP Explanation — Beat {beat_number}\n'
        f'Blue = towards Normal | Red = towards Abnormal',
        fontsize=11, fontweight='bold')
    ax.set_xlabel('SHAP Value')
    ax.grid(True, alpha=0.2, linestyle='--', axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # Feature values
    st.subheader("Feature Values for this Beat")
    feat_display = pd.DataFrame({
        'Feature': feature_names,
        'Value':   [round(selected_features[f], 4)
                    for f in feature_names]
    })
    st.dataframe(feat_display, use_container_width=True)

# ── Two tabs ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂 Select from Database",
                       "📤 Upload ECG Files"])

# ── Tab 1 — Select from database ─────────────────────────
with tab1:
    st.markdown("Select any of the 48 patients from the MIT-BIH database.")

    all_records = sorted([
        f.replace('.hea', '')
        for f in os.listdir(data_path)
        if f.endswith('.hea')
    ])

    selected_record = st.selectbox(
        "Select patient record:",
        all_records,
        index=0
    )

    with st.spinner(f"Analysing record {selected_record}..."):
        record_path = os.path.join(data_path, selected_record)
        (signal, beats, beat_features,
         beat_labels, beat_positions, fs) = process_record_path(
             record_path)

    show_analysis(signal, beats, beat_features,
                  beat_labels, beat_positions, fs,
                  f"Record {selected_record}")

# ── Tab 2 — Upload ECG files ──────────────────────────────
with tab2:
    st.markdown("""
    Upload your own MIT-BIH format ECG files.
    You need to upload all three files together:
    - `.hea` — header file
    - `.dat` — signal data file
    - `.atr` — annotation file
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        hea_file = st.file_uploader(
            "Upload .hea file", type=['hea'])
    with col2:
        dat_file = st.file_uploader(
            "Upload .dat file", type=['dat'])
    with col3:
        atr_file = st.file_uploader(
            "Upload .atr file", type=['atr'])

    if hea_file and dat_file and atr_file:
        # Check all files have same base name
        hea_name = hea_file.name.replace('.hea', '')
        dat_name = dat_file.name.replace('.dat', '')
        atr_name = atr_file.name.replace('.atr', '')

        if hea_name == dat_name == atr_name:
            st.success(f"All 3 files uploaded — Record {hea_name}")
        else:
            st.error("Files do not match! Please upload files from the same record.")
            st.markdown(f"""
            - HEA file: **{hea_name}**
            - DAT file: **{dat_name}**
            - ATR file: **{atr_name}**

            All three files must have the same name.
            For example: 100.hea + 100.dat + 100.atr
            """)
            st.stop()

        # Save files temporarily
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Get record name from hea filename
            record_name = hea_file.name.replace('.hea', '')

            # Save all three files to temp directory
            for uploaded, ext in [(hea_file, '.hea'),
                                   (dat_file, '.dat'),
                                   (atr_file, '.atr')]:
                file_path = os.path.join(
                    tmp_dir, record_name + ext)
                with open(file_path, 'wb') as f:
                    f.write(uploaded.getvalue())

            # Process the uploaded record
            with st.spinner("Analysing uploaded ECG..."):
                try:
                    record_path = os.path.join(
                        tmp_dir, record_name)
                    (signal, beats, beat_features,
                     beat_labels, beat_positions,
                     fs) = process_record_path(record_path)

                    show_analysis(
                        signal, beats, beat_features,
                        beat_labels, beat_positions, fs,
                        f"Uploaded Record — {record_name}")

                except Exception as e:
                    st.error(f"Error processing files: {e}")
                    st.markdown("""
                    Please make sure:
                    - All three files are from the same record
                    - Files are in valid MIT-BIH format
                    - The .hea, .dat and .atr files have the same name
                    """)
    else:
        st.info("Please upload all three files to begin analysis.")
        st.markdown("""
        **How to get MIT-BIH format files:**
        You can download individual records from
        [PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
        """)