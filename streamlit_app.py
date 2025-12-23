import streamlit as st
import joblib
import numpy as np

from modules.feature_extraction import extract_features
from modules.config import (
    MODEL_PATH,
    SCALER_PATH,
    ENCODER_PATH,
    FEATURE_INDEX_PATH
)

st.set_page_config(
    page_title="Duck–Goose Audio Classification",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    selected_idx = np.load(FEATURE_INDEX_PATH)
    return model, scaler, encoder, selected_idx

model, scaler, encoder, selected_idx = load_artifacts()

st.title("🦆🪿 Duck–Goose Audio Classification")

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    try:
        # 1️⃣ Extract 16 fitur
        features = extract_features(uploaded_file)

        # 2️⃣ APPLY FEATURE SELECTION (INI KUNCI!)
        features_selected = features[selected_idx]

        # 3️⃣ Scaling
        features_scaled = scaler.transform(features_selected.reshape(1, -1))

        # 4️⃣ Predict
        prediction = model.predict(features_scaled)
        label = encoder.inverse_transform(prediction)

        st.success(f"🎯 Hasil Prediksi: **{label[0]}**")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses audio.")
        st.exception(e)
