import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

MAX_LEN = 20
CONFIDENCE_THRESHOLD = 0.6
MODEL_PATH = DATA_DIR / "model.h5"
TOKENIZER_PATH = DATA_DIR / "tokenizer.pickle"
LABEL_ENCODER_PATH = DATA_DIR / "label_encoder.pickle"
USER_PROFILE_DIR = DATA_DIR / "user_profiles"

# Create directories if they don't exist
os.makedirs(USER_PROFILE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)