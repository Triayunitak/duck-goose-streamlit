from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models/randomforest_best_model.pkl"
ENCODER_PATH = BASE_DIR / "models/label_encoder.pkl"
SCALER_PATH = BASE_DIR / "models/scaler.joblib"
FEATURE_INDEX_PATH = BASE_DIR / "models/selected_feature_indices.npy"
