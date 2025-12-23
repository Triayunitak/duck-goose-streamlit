import librosa
import numpy as np
import io

def extract_features(uploaded_file):
    """
    Extract 13 MFCC features (mean over time)
    Must match training configuration
    """

    # Streamlit uploader → bytes → buffer
    audio_bytes = uploaded_file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    # Load audio
    signal, sr = librosa.load(
        audio_buffer,
        sr=44100,
        mono=True
    )

    # Extract MFCC (13)
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    # Temporal aggregation (mean)
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean  # shape: (13,)
