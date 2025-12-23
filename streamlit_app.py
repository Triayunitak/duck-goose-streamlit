import streamlit as st
import joblib
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
    "Upload file audio (.wav)", type=["wav"]
)

if uploaded_file:
    features = extract_features(uploaded_file)
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    label = encoder.inverse_transform(prediction)

    st.success(f"Hasil Prediksi: **{label[0]}**")
