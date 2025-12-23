import streamlit as st
import numpy as np
import librosa
import joblib

# =========================
# Load model & scaler
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("randomforest_best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, scaler, encoder


model, scaler, encoder = load_artifacts()

# =========================
# Feature Extraction (16 fitur)
# =========================
def extract_features(y, sr):
    # MFCC (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # ZCR (1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS (1)
    rms = np.mean(librosa.feature.rms(y=y))

    # Spectral Centroid (1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Gabung → TOTAL 16
    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        centroid
    ])

    return features


# =========================
# UI
# =========================
st.title("🦆🪿 Duck vs Goose Audio Classifier")

uploaded_file = st.file_uploader(
    "Upload audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    try:
        y, sr = librosa.load(uploaded_file, sr=16000)

        features = extract_features(y, sr)
        features = features.reshape(1, -1)

        # Scaling
        features_scaled = scaler.transform(features)

        # Prediction
        pred = model.predict(features_scaled)
        label = encoder.inverse_transform(pred)[0]

        st.success(f"🎯 Prediksi: **{label}**")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses audio.")
        st.exception(e)
