import pickle
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "pipeline.pkl"


def load_pipeline():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline