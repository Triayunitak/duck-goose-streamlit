import streamlit as st
import joblib
import numpy as np
from modules.feature_extraction import extract_features
from modules.config import MODEL_PATH, ENCODER_PATH, SCALER_PATH

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Duck–Goose Audio Classification",
    layout="centered"
)

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, encoder, scaler

model, encoder, scaler = load_artifacts()

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("🦆🪿 Duck–Goose Audio Classification")
st.caption(
    "Sistem identifikasi spesies bebek dan angsa "
    "berdasarkan sinyal audio (bioakustik)."
)

uploaded_file = st.file_uploader(
    "Upload file audio (.wav)",
    type=["wav"]
)

# --------------------------------------------------
# Inference
# --------------------------------------------------
if uploaded_file is not None:
    try:
        # ---- Feature extraction ----
        features = extract_features(uploaded_file)

        # ---- Convert to numpy & flatten ----
        features = np.asarray(features, dtype=np.float64).flatten()

        # ---- HARD VALIDATION ----
        expected = model.n_features_in_

        if features.shape[0] != expected:
            st.error(
                f"❌ Dimensi fitur tidak sesuai\n\n"
                f"Model mengharapkan **{expected} fitur**, "
                f"tetapi input memiliki **{features.shape[0]} fitur**.\n\n"
                "Pastikan fungsi ekstraksi fitur IDENTIK dengan tahap training."
            )
            st.stop()

        if not np.isfinite(features).all():
            st.error(
                "❌ Ditemukan nilai NaN atau Inf pada fitur.\n\n"
                "Audio kemungkinan rusak atau proses ekstraksi tidak stabil."
            )
            st.stop()

        # ---- Scaling ----
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)

        # ---- Prediction ----
        prediction = model.predict(features_scaled)
        label = encoder.inverse_transform(prediction)

        st.success(f"🎯 Hasil Prediksi: **{label[0]}**")

    except Exception as e:
        st.error("Terjadi kesalahan saat proses klasifikasi audio.")
        st.exception(e)
