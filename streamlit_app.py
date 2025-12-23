import streamlit as st
import joblib
import numpy as np
import io
import librosa

# Import fungsi kustom Anda
# Pastikan folder 'modules' ada di direktori yang sama
try:
    from modules.feature_extraction import extract_features
    from modules.config import (
        MODEL_PATH,
        SCALER_PATH,
        ENCODER_PATH,
        FEATURE_INDEX_PATH
    )
except ImportError:
    st.error("Gagal memuat modul kustom. Pastikan struktur folder 'modules' sudah benar.")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Duck–Goose Audio Classification",
    page_icon="🦆",
    layout="centered"
)

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    """Memuat semua model dan scaler ke dalam cache."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        encoder = joblib.load(ENCODER_PATH)
        selected_idx = np.load(FEATURE_INDEX_PATH)
        return model, scaler, encoder, selected_idx
    except Exception as e:
        st.error(f"Gagal memuat file model: {e}")
        return None, None, None, None

model, scaler, encoder, selected_idx = load_artifacts()

# -----------------------------
# UI Header
# -----------------------------
st.title("🦆🪿 Duck–Goose Audio Classification")
st.markdown("""
Aplikasi ini mengklasifikasikan suara bangsa angsa dan bebek (Ducks & Geese) 
menggunakan Model Machine Learning.
""")

st.divider()

# -----------------------------
# File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Unggah file audio suara unggas (.wav)",
    type=["wav"]
)

# -----------------------------
# Inference Logic
# -----------------------------
if uploaded_file is not None:
    # Tampilkan audio player agar user bisa mendengar apa yang diunggah
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner("Menganalisis karakteristik audio..."):
        try:
            # 1️⃣ Ekstraksi Fitur
            # Catatan: Pastikan extract_features bisa menerima objek BytesIO
            # Jika extract_features butuh path, gunakan tempfile.
            features = extract_features(uploaded_file) 

            # 2️⃣ Scaling (Menggunakan 16 fitur awal)
            # Reshape ke (1, 16) karena scaler mengharap array 2D
            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)

            # 3️⃣ Feature Selection (Sesuai index hasil training, misal jadi 13 fitur)
            features_selected = features_scaled[:, selected_idx]

            # 4️⃣ Predict
            prediction = model.predict(features_selected)
            label = encoder.inverse_transform(prediction)
            
            # 5️⃣ Ambil Probabilitas (Jika model mendukung)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features_selected)
                confidence = np.max(probs) * 100
                st.metric("🎯 Hasil Prediksi", f"{label[0]}")
                st.progress(int(confidence), text=f"Tingkat Keyakinan: {confidence:.2f}%")
            else:
                st.success(f"🎯 Hasil Prediksi: **{label[0]}**")

            # Expander untuk melihat detail fitur (Opsional)
            with st.expander("Lihat Detail Fitur"):
                st.write("Fitur Terpilih (Scaled):", features_selected)

        except Exception as e:
            st.error("❌ Terjadi kesalahan saat memproses audio.")
            st.info("Pastikan file .wav Anda tidak korup dan memiliki durasi yang cukup.")
            with st.expander("Detail Error"):
                st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption("Aplikasi Klasifikasi Audio Suara Bebek & Angsa - Dataset: DucksAndGeese (Xeno-canto)")