import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "randomforest_best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
FEATURE_INDEX_PATH = os.path.join(BASE_DIR, "models", "selected_feature_indices.npy")
