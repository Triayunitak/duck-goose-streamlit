import streamlit as st
import joblib
import numpy as np

from modules.feature_extraction import extract_features
from modules.config import (
    MODEL_PATH,
    ENCODER_PATH,
    SCALER_PATH,
    FEATURE_INDEX_PATH
)

st.set_page_config(
    page_title="Duck–Goose Audio Classification",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    selected_idx = np.load(FEATURE_INDEX_PATH)
    return model, encoder, scaler, selected_idx

model, encoder, scaler, selected_idx = load_artifacts()

st.title("🦆🪿 Duck–Goose Audio Classification")

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file:
    try:
        # 1️⃣ Extract 16 features
        features = extract_features(uploaded_file)

        # 2️⃣ Scale 16 features
        features_scaled = scaler.transform(
            features.reshape(1, -1)
        )

        # 3️⃣ Select 13 features (SAMA DENGAN TRAINING)
        features_selected = features_scaled[:, selected_idx]

        # 4️⃣ Predict
        prediction = model.predict(features_selected)
        label = encoder.inverse_transform(prediction)

        st.success(f"✅ Hasil Prediksi: **{label[0]}**")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses audio.")
        st.exception(e)
