import librosa
import numpy as np
import io

def extract_features(uploaded_file):
    audio_bytes = uploaded_file.read()
    audio_buffer = io.BytesIO(audio_bytes)

    signal, sr = librosa.load(
        audio_buffer,
        sr=44100,
        mono=True
    )

    # MFCC (13)
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )
    mfcc_mean = np.mean(mfcc, axis=1)

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))

    # RMS
    rms = np.mean(librosa.feature.rms(y=signal))

    # Spectral Centroid
    centroid = np.mean(
        librosa.feature.spectral_centroid(y=signal, sr=sr)
    )

    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        centroid
    ])

    return features  # shape (16,)
