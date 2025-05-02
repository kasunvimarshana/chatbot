import os
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # Model paths
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "models/intent_model.pkl")) # Trained model file path
    VECTORIZER_PATH: Path = Path(os.getenv("VECTORIZER_PATH", "models/tfidf_vectorizer.pkl")) # Vectorizer file path
    
    # Chatbot parameters
    DEFAULT_RESPONSE: str = os.getenv("DEFAULT_RESPONSE", "I didn't understand that. Could you rephrase?") # Fallback response
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.05)) # Minimum prediction confidence (0-1)
    CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", 1024)) # LRU cache size for predictions
    
    # File paths
    INTENT_FILE: Path = Path(os.getenv("INTENT_FILE", "data/intents.json")) # Intent definitions JSON path
    TRAINING_DATA: Path = Path(os.getenv("TRAINING_DATA", "data/training_data.csv")) # Training data CSV path
    REPORTS_DIR: Path = Path(os.getenv("REPORTS_DIR", "reports")) # Output directory for reports
    
    # Performance
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4)) # Max parallel worker threads

    NGROK_AUTH_TOKEN: str = os.getenv("NGROK_AUTH_TOKEN", "") # Ngrok authentication token

    class Config:
        env_file = ".env"
        case_sensitive = True

# Initialize settings instance
settings = Settings()
