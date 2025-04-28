from transformers import pipeline
from config import DATA_DIR
import json

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = pipeline("sentiment-analysis")
        self.sentiment_map = self._load_sentiment_map()
    
    def _load_sentiment_map(self):
        try:
            with open(DATA_DIR / "sentiment_map.json") as f:
                return json.load(f)
        except:
            return {
                "positive": ["great", "wonderful", "happy"],
                "negative": ["sad", "angry", "frustrated"],
                "neutral": ["okay", "fine", "alright"]
            }
    
    def analyze_sentiment(self, text):
        """Enhanced sentiment analysis with custom mapping"""
        try:
            # First check for keywords
            text_lower = text.lower()
            for sentiment, keywords in self.sentiment_map.items():
                if any(keyword in text_lower for keyword in keywords):
                    return sentiment
            
            # Fall back to model analysis
            result = self.analyzer(text)[0]
            if result['label'] == 'POSITIVE' and result['score'] > 0.7:
                return "positive"
            elif result['label'] == 'NEGATIVE' and result['score'] > 0.7:
                return "negative"
            return "neutral"
        except Exception as e:
            print(f"Sentiment analysis error: {str(e)}")
            return "neutral"
        
