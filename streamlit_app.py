import streamlit as st
import joblib
import numpy as np

from modules.feature_extraction import extract_features
from modules.config import MODEL_PATH, SCALER_PATH, ENCODER_PATH

st.set_page_config(
    page_title="Duck–Goose Audio Classification",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

model, scaler, encoder = load_artifacts()

st.title("🦆🪿 Duck–Goose Audio Classification")

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    try:
        features = extract_features(uploaded_file)

        # VALIDASI DIMENSI (ANTI ERROR)
        if features.shape[0] != scaler.n_features_in_:
            st.error(
                f"❌ Jumlah fitur tidak sesuai. "
                f"Model expects {scaler.n_features_in_}, "
                f"got {features.shape[0]}"
            )
        else:
            features_scaled = scaler.transform(features.reshape(1, -1))
            prediction = model.predict(features_scaled)
            label = encoder.inverse_transform(prediction)

            st.success(f"🎯 Hasil Prediksi: **{label[0]}**")

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses audio.")
        st.exception(e)
