import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import random

# ======================
# Base Model Interface
# ======================
class AIModel(ABC):
    @abstractmethod
    def predict(self, input_data: Any) -> Dict:
        """Base prediction method all models must implement"""
        pass
    
    @abstractmethod
    def is_appropriate(self, input_data: Any) -> bool:
        """Determine if this model should handle the input"""
        pass


# ======================
# Concrete Model Implementations
# ======================
class GPTConversationModel(AIModel):
    def __init__(self):
        # In a real implementation, this would connect to an actual GPT API
        self.responses = {
            "greeting": ["Hello! How can I help you today?", "Hi there! What can I do for you?"],
            "question": {
                "reset password": "You can reset your password by visiting our website and clicking 'Forgot Password'.",
                "account issues": "For account issues, please contact our support team at support@example.com.",
                "default": "I'd be happy to help with that. Could you provide more details?"
            },
            "farewell": ["Goodbye! Have a great day!", "See you later!"]
        }
    
    def predict(self, input_data: str) -> Dict:
        intent = self._detect_intent(input_data.lower())
        
        if intent == "greeting":
            return {
                "response": random.choice(self.responses["greeting"]),
                "confidence": 0.9,
                "intent": "greeting"
            }
        elif intent == "farewell":
            return {
                "response": random.choice(self.responses["farewell"]),
                "confidence": 0.85,
                "intent": "farewell"
            }
        else:  # question
            topic = self._detect_topic(input_data.lower())
            response = self.responses["question"].get(topic, self.responses["question"]["default"])
            return {
                "response": response,
                "confidence": 0.8 if topic != "default" else 0.6,
                "intent": "question",
                "topic": topic
            }
    
    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data.strip()) > 0
    
    def _detect_intent(self, text: str) -> str:
        text = text.lower()
        if any(word in text for word in ["hi", "hello", "hey"]):
            return "greeting"
        elif any(word in text for word in ["bye", "goodbye", "see you"]):
            return "farewell"
        else:
            return "question"
    
    def _detect_topic(self, text: str) -> str:
        if "password" in text or "login" in text:
            return "reset password"
        elif "account" in text or "profile" in text:
            return "account issues"
        return "default"


class SentimentAnalysisModel(AIModel):
    def predict(self, input_text: str) -> Dict:
        positive_words = ["love", "great", "happy", "awesome", "thanks"]
        negative_words = ["hate", "angry", "bad", "terrible", "awful"]
        
        text_lower = input_text.lower()
        positive_score = sum(word in text_lower for word in positive_words)
        negative_score = sum(word in text_lower for word in negative_words)
        
        if positive_score > negative_score:
            sentiment = "positive"
            score = min(1.0, positive_score * 0.3)
        elif negative_score > positive_score:
            sentiment = "negative"
            score = min(1.0, negative_score * 0.3)
        else:
            sentiment = "neutral"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "score": round(score, 2),
            "keywords": {
                "positive": [w for w in positive_words if w in text_lower],
                "negative": [w for w in negative_words if w in text_lower]
            }
        }
    
    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data.split()) >= 2


class IntentClassifierModel(AIModel):
    def __init__(self):
        # In a real implementation, this would be a trained ML model
        self.intents = {
            "customer_support": ["help", "support", "problem", "issue"],
            "sales": ["buy", "purchase", "price", "cost"],
            "technical": ["error", "bug", "crash", "not working"],
            "general": ["what", "how", "when", "where"]
        }
    
    def predict(self, input_text: str) -> Dict:
        text_lower = input_text.lower()
        scores = {}
        
        for intent, keywords in self.intents.items():
            scores[intent] = sum(keyword in text_lower for keyword in keywords)
        
        best_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, scores[best_intent] * 0.33)  # Scale to 0-1 range
        
        return {
            "intent": best_intent,
            "confidence": round(confidence, 2),
            "alternative_intents": scores
        }
    
    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data.split()) >= 2


# ======================
# Chatbot Core
# ======================
class ChatBot:
    def __init__(self):
        self.orchestrator = ModelOrchestrator()
        self.conversation_history = []
        self.user_profile = {}
        
        # Initialize with default models
        self.orchestrator.add_model(SentimentAnalysisModel())
        self.orchestrator.add_model(IntentClassifierModel())
        self.orchestrator.add_model(GPTConversationModel())
    
    def respond(self, user_input: str) -> str:
        # Get raw prediction from orchestrator
        prediction = self.orchestrator.predict(user_input)
        
        # Add context from other models
        enriched_response = self._enrich_response(user_input, prediction)
        
        # Store conversation history
        self._update_history(user_input, enriched_response)
        
        return enriched_response["response"]
    
    def _enrich_response(self, user_input: str, prediction: Dict) -> Dict:
        # Get sentiment analysis
        sentiment_model = next(
            (m for m in self.orchestrator.models if isinstance(m, SentimentAnalysisModel)), 
            None
        )
        
        if sentiment_model:
            sentiment = sentiment_model.predict(user_input)
            prediction.update({"sentiment": sentiment})
            
            # Adjust response based on sentiment
            if sentiment["sentiment"] == "negative":
                if "sorry" not in prediction["response"].lower():
                    prediction["response"] = "I'm sorry to hear you're having trouble. " + prediction["response"]
        
        # Get intent classification
        intent_model = next(
            (m for m in self.orchestrator.models if isinstance(m, IntentClassifierModel)), 
            None
        )
        
        if intent_model:
            intent = intent_model.predict(user_input)
            prediction.update({"intent_classification": intent})
            
            # Route to appropriate response based on intent
            if intent["intent"] == "technical" and "support" not in prediction["response"].lower():
                prediction["response"] += " For technical issues, our support team can help at support@example.com."
        
        return prediction
    
    def _update_history(self, user_input: str, bot_response: Dict):
        self.conversation_history.append({
            "user": user_input,
            "bot": bot_response["response"],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sentiment": bot_response.get("sentiment"),
                "intent": bot_response.get("intent_classification"),
                "model_used": bot_response.get("model_used", "unknown")
            }
        })
        
        # Keep only last 20 messages
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)


# ======================
# Model Orchestrator
# ======================
class ModelOrchestrator:
    def __init__(self):
        self.models = []
        self.fallback_model = GPTConversationModel()
    
    def add_model(self, model: AIModel):
        self.models.append(model)
    
    def predict(self, input_data: Any) -> Dict:
        # Try specialized models first
        for model in self.models:
            if model.is_appropriate(input_data):
                try:
                    result = model.predict(input_data)
                    result["model_used"] = model.__class__.__name__
                    return result
                except Exception as e:
                    continue
        
        # Fallback to default model
        result = self.fallback_model.predict(input_data)
        result["model_used"] = "fallback"
        return result


# ======================
# Main Application
# ======================
if __name__ == "__main__":
    from datetime import datetime
    
    print("ChatBot Initializing...")
    bot = ChatBot()
    
    print("\nWelcome to the AI ChatBot! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Bot: Goodbye! Have a great day!")
            break
        
        response = bot.respond(user_input)
        print(f"Bot: {response}")
        
        # Optional: Show debug info
        if len(bot.conversation_history) > 0:
            last_conv = bot.conversation_history[-1]
            print(f"\n[Debug Info]")
            print(f"Sentiment: {last_conv['metadata']['sentiment']['sentiment']} ({last_conv['metadata']['sentiment']['score']})")
            print(f"Intent: {last_conv['metadata']['intent']['intent']}")
            print(f"Model Used: {last_conv['metadata']['model_used']}")