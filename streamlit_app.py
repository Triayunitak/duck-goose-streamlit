import streamlit as st
import joblib
import numpy as np

from modules.feature_extraction import extract_features
from modules.config import MODEL_PATH, ENCODER_PATH, SCALER_PATH

st.set_page_config(
    page_title="Duck–Goose Audio Classification",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoder, scaler

model, encoder, scaler = load_artifacts()

st.title("🦆🪿 Duck–Goose Audio Classification")

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

if uploaded_file is not None:
    try:
        # Extract MFCC (13)
        features = extract_features(uploaded_file)

        # Safety check
        if features.shape[0] != model.n_features_in_:
            st.error(
                f"❌ Jumlah fitur tidak sesuai. "
                f"Model butuh {model.n_features_in_}, "
                f"input punya {features.shape[0]}"
            )
        else:
            # Scale
            features_scaled = scaler.transform(
                features.reshape(1, -1)
            )

            # Predict
            prediction = model.predict(features_scaled)
            label = encoder.inverse_transform(prediction)

            st.success(f"✅ Hasil Prediksi: **{label[0]}**")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses audio.")
        st.exception(e)
