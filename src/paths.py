"""Defines commonly used paths. Resolves which dataset is used via "DATASET" environment."""
import os
from pathlib import Path

if os.environ["DATASET"] == "scifact":
    DATASET = "scifact"
elif os.environ["DATASET"] == "pubmed":
    DATASET = "pubmed"
elif os.environ["DATASET"] == "kgat_fever":
    DATASET = "kgat_fever"
else:
    DATASET = "fever"


ROOT_DIR = Path(__file__).resolve().parent.parent / DATASET
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
FEATURES_DIR = ROOT_DIR / "features"
OUTPUTS_DIR = ROOT_DIR / "outputs"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DATA_DIR / "data.db"
TRAIN_DATA_DIR = PROCESSED_DATA_DIR / "train"
VALID_DATA_DIR = PROCESSED_DATA_DIR / "valid"
DEV_DATA_DIR = PROCESSED_DATA_DIR / "dev"
TEST_DATA_DIR = PROCESSED_DATA_DIR / "test"


if __name__ == "__main__":
    dirs = [
        DATA_DIR,
        MODELS_DIR,
        FEATURES_DIR,
        OUTPUTS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        TRAIN_DATA_DIR,
        VALID_DATA_DIR,
        DEV_DATA_DIR,
        TEST_DATA_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
