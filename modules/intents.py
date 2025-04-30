import pickle
import numpy as np
from typing import Dict, Any
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recognize_intent(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Recognize intent from processed text"""
    try:
        # Load models
        with open(settings.VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(settings.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        # Vectorize input
        text_vector = vectorizer.transform([text])
        
        # Predict intent
        probas = model.predict_proba(text_vector)[0]
        max_idx = np.argmax(probas)
        max_proba = probas[max_idx]
        
        # Apply confidence threshold
        intent = model.classes_[max_idx] if max_proba >= settings.CONFIDENCE_THRESHOLD else 'unknown'
        
        return {
            "intent": intent,
            "confidence": float(max_proba),
            "alternatives": [
                {"intent": cls, "confidence": float(conf)}
                for cls, conf in zip(model.classes_, probas)
                if conf > 0.1 and cls != intent
            ]
        }
    
    except Exception as e:
        logger.error(f"Error in recognize_intent: {str(e)}")
        return {
            "intent": "error",
            "confidence": 0.0,
            "error": str(e)
        }