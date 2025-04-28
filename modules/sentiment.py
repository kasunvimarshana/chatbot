from transformers import pipeline
import numpy as np

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    """Analyze text sentiment using Hugging Face pipeline"""
    try:
        result = sentiment_analyzer(text)[0]
        if result['label'] == 'POSITIVE' and result['score'] > 0.8:
            return "positive"
        elif result['label'] == 'NEGATIVE' and result['score'] > 0.8:
            return "negative"
        return "neutral"
    except:
        return "neutral"