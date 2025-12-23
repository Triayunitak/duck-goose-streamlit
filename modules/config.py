from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "randomforest_best_model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.joblib"

SAMPLE_RATE = 22050
N_MFCC = 13
