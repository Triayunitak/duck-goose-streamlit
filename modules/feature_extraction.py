import librosa
import numpy as np
from .config import SAMPLE_RATE, N_MFCC

def extract_features(file):
    y, sr = librosa.load(file, sr=SAMPLE_RATE)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = mfcc.mean(axis=1)

    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rms = librosa.feature.rms(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        centroid
    ])

    return features
