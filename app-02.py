import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
from datetime import datetime
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ai_bot')

# ======================
# Base Model Interface
# ======================
class AIModel(ABC):
    """Base interface that all AI models must implement"""
    
    @abstractmethod
    def predict(self, input_data: Any) -> Dict:
        """Base prediction method all models must implement"""
        pass
    
    @abstractmethod
    def is_appropriate(self, input_data: Any) -> bool:
        """Determine if this model should handle the input"""
        pass
    
    @property
    def name(self) -> str:
        """Return the name of the model"""
        return self.__class__.__name__


# ======================
# Model Registry - NEW
# ======================
class ModelRegistry:
    """Singleton registry for all available model types"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance.models = {}
        return cls._instance
    
    def register(self, model_type: str, model_class: Type[AIModel]):
        """Register a model class with a type identifier"""
        self.models[model_type] = model_class
        logger.info(f"Registered model type: {model_type} with class {model_class.__name__}")
    
    def get_model_class(self, model_type: str) -> Type[AIModel]:
        """Get a model class by its type identifier"""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.models[model_type]
    
    def create_model(self, model_type: str, **kwargs) -> AIModel:
        """Create a model instance by its type identifier"""
        model_class = self.get_model_class(model_type)
        return model_class(**kwargs)


# ======================
# Concrete Model Implementations
# ======================
class GPTConversationModel(AIModel):
    def __init__(self, model_version: str = "default"):
        # In a real implementation, this would connect to an actual GPT API
        self.model_version = model_version
        logger.info(f"Initializing GPT model with version: {model_version}")
        
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
    def __init__(self, custom_lexicon: Dict[str, List[str]] = None):
        self.positive_words = ["love", "great", "happy", "awesome", "thanks"]
        self.negative_words = ["hate", "angry", "bad", "terrible", "awful"]
        
        # Allow custom lexicon
        if custom_lexicon:
            if 'positive' in custom_lexicon:
                self.positive_words.extend(custom_lexicon['positive'])
            if 'negative' in custom_lexicon:
                self.negative_words.extend(custom_lexicon['negative'])
    
    def predict(self, input_text: str) -> Dict:
        text_lower = input_text.lower()
        positive_score = sum(word in text_lower for word in self.positive_words)
        negative_score = sum(word in text_lower for word in self.negative_words)
        
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
                "positive": [w for w in self.positive_words if w in text_lower],
                "negative": [w for w in self.negative_words if w in text_lower]
            }
        }
    
    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data.split()) >= 2


class IntentClassifierModel(AIModel):
    def __init__(self, custom_intents: Dict[str, List[str]] = None):
        # In a real implementation, this would be a trained ML model
        self.intents = {
            "customer_support": ["help", "support", "problem", "issue"],
            "sales": ["buy", "purchase", "price", "cost"],
            "technical": ["error", "bug", "crash", "not working"],
            "general": ["what", "how", "when", "where"]
        }
        
        # Allow custom intents
        if custom_intents:
            for intent, keywords in custom_intents.items():
                if intent in self.intents:
                    self.intents[intent].extend(keywords)
                else:
                    self.intents[intent] = keywords
    
    def predict(self, input_text: str) -> Dict:
        text_lower = input_text.lower()
        scores = {}
        
        for intent, keywords in self.intents.items():
            scores[intent] = sum(keyword in text_lower for keyword in keywords)
        
        best_intent = max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else "unknown"
        confidence = min(1.0, scores.get(best_intent, 0) * 0.33)  # Scale to 0-1 range
        
        return {
            "intent": best_intent,
            "confidence": round(confidence, 2),
            "alternative_intents": scores
        }
    
    def is_appropriate(self, input_data: Any) -> bool:
        return isinstance(input_data, str) and len(input_data.split()) >= 2


# ======================
# NEW: Response Enhancer Pipeline
# ======================
class ResponseEnhancer(ABC):
    """Base class for response enhancers that modify bot responses"""
    
    @abstractmethod
    def enhance(self, user_input: str, prediction: Dict) -> Dict:
        """Enhance a prediction based on user input and current prediction"""
        pass


class SentimentResponseEnhancer(ResponseEnhancer):
    """Enhances responses based on sentiment analysis"""
    
    def __init__(self, sentiment_model: SentimentAnalysisModel):
        self.sentiment_model = sentiment_model
    
    def enhance(self, user_input: str, prediction: Dict) -> Dict:
        sentiment = self.sentiment_model.predict(user_input)
        prediction.update({"sentiment": sentiment})
        
        # Adjust response based on sentiment
        if sentiment["sentiment"] == "negative" and sentiment["score"] > 0.3:
            if "sorry" not in prediction["response"].lower():
                prediction["response"] = "I'm sorry to hear you're having trouble. " + prediction["response"]
        elif sentiment["sentiment"] == "positive" and sentiment["score"] > 0.6:
            if "glad" not in prediction["response"].lower():
                prediction["response"] = "I'm glad to hear that! " + prediction["response"]
                
        return prediction


class IntentResponseEnhancer(ResponseEnhancer):
    """Enhances responses based on intent classification"""
    
    def __init__(self, intent_model: IntentClassifierModel):
        self.intent_model = intent_model
    
    def enhance(self, user_input: str, prediction: Dict) -> Dict:
        intent = self.intent_model.predict(user_input)
        prediction.update({"intent_classification": intent})
        
        # Route to appropriate response based on intent
        if intent["intent"] == "technical" and intent["confidence"] > 0.4:
            if "support" not in prediction["response"].lower():
                prediction["response"] += " For technical issues, our support team can help at support@example.com."
        elif intent["intent"] == "sales" and intent["confidence"] > 0.4:
            if "sales" not in prediction["response"].lower():
                prediction["response"] += " You can contact our sales team at sales@example.com for pricing information."
                
        return prediction


# ======================
# Enhanced Model Orchestrator
# ======================
class ModelOrchestrator:
    def __init__(self):
        self.models = []
        self.fallback_model = None
        self.enhancers = []
    
    def add_model(self, model: AIModel):
        """Add a model to the orchestrator"""
        self.models.append(model)
        logger.info(f"Added model: {model.name}")
        return self  # For method chaining
    
    def set_fallback_model(self, model: AIModel):
        """Set the fallback model"""
        self.fallback_model = model
        logger.info(f"Set fallback model: {model.name}")
        return self  # For method chaining
    
    def add_enhancer(self, enhancer: ResponseEnhancer):
        """Add a response enhancer to the pipeline"""
        self.enhancers.append(enhancer)
        logger.info(f"Added enhancer: {enhancer.__class__.__name__}")
        return self  # For method chaining
    
    def predict(self, input_data: Any) -> Dict:
        """Get predictions from appropriate models"""
        # Try specialized models first
        for model in self.models:
            if model.is_appropriate(input_data):
                try:
                    logger.debug(f"Trying model: {model.name}")
                    result = model.predict(input_data)
                    result["model_used"] = model.name
                    
                    # Apply enhancers
                    for enhancer in self.enhancers:
                        result = enhancer.enhance(input_data, result)
                    
                    return result
                except Exception as e:
                    logger.warning(f"Error in model {model.name}: {str(e)}")
                    continue
        
        # Fallback to default model
        if self.fallback_model:
            logger.debug("Using fallback model")
            result = self.fallback_model.predict(input_data)
            result["model_used"] = f"fallback-{self.fallback_model.name}"
            
            # Apply enhancers
            for enhancer in self.enhancers:
                result = enhancer.enhance(input_data, result)
                
            return result
        else:
            logger.error("No appropriate model found and no fallback configured")
            return {
                "response": "I'm sorry, I couldn't process that request.",
                "model_used": "none",
                "error": "No appropriate model found"
            }


# ======================
# Chatbot Core
# ======================
class ChatBot:
    def __init__(self, config: Dict = None):
        self.orchestrator = ModelOrchestrator()
        self.conversation_history = []
        self.user_profile = {}
        self.config = config or {}
        
        # Initialize from config if provided
        self._initialize_from_config()
    
    def _initialize_from_config(self):
        """Initialize models from configuration"""
        if not self.config:
            # Initialize with default models
            gpt_model = GPTConversationModel()
            sentiment_model = SentimentAnalysisModel()
            intent_model = IntentClassifierModel()
            
            # Setup orchestrator
            self.orchestrator.add_model(GPTConversationModel())
            self.orchestrator.set_fallback_model(gpt_model)
            
            # Setup enhancers
            self.orchestrator.add_enhancer(SentimentResponseEnhancer(sentiment_model))
            self.orchestrator.add_enhancer(IntentResponseEnhancer(intent_model))
        else:
            # Initialize from config
            registry = ModelRegistry()
            
            # Register standard models
            registry.register("gpt", GPTConversationModel)
            registry.register("sentiment", SentimentAnalysisModel)
            registry.register("intent", IntentClassifierModel)
            
            # Create models from config
            for model_config in self.config.get("models", []):
                model_type = model_config.get("type")
                model_params = model_config.get("params", {})
                model = registry.create_model(model_type, **model_params)
                
                if model_config.get("is_fallback", False):
                    self.orchestrator.set_fallback_model(model)
                else:
                    self.orchestrator.add_model(model)
            
            # Create enhancers
            for enhancer_config in self.config.get("enhancers", []):
                enhancer_type = enhancer_config.get("type")
                model_type = enhancer_config.get("model_type")
                model_params = enhancer_config.get("model_params", {})
                
                model = registry.create_model(model_type, **model_params)
                
                if enhancer_type == "sentiment":
                    self.orchestrator.add_enhancer(SentimentResponseEnhancer(model))
                elif enhancer_type == "intent":
                    self.orchestrator.add_enhancer(IntentResponseEnhancer(model))
    
    def respond(self, user_input: str) -> str:
        """Generate a response to user input"""
        # Get prediction from orchestrator (includes enhancement)
        prediction = self.orchestrator.predict(user_input)
        
        # Store conversation history
        self._update_history(user_input, prediction)
        
        return prediction["response"]
    
    def _update_history(self, user_input: str, bot_response: Dict):
        """Update the conversation history"""
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
        
        # Keep only last N messages (configurable)
        max_history = self.config.get("max_history", 20)
        if len(self.conversation_history) > max_history:
            self.conversation_history.pop(0)
    
    def export_history(self, format: str = "json") -> str:
        """Export conversation history in the specified format"""
        if format.lower() == "json":
            return json.dumps(self.conversation_history, indent=2)
        else:
            # Simple text format
            history_text = ""
            for entry in self.conversation_history:
                history_text += f"User: {entry['user']}\n"
                history_text += f"Bot: {entry['bot']}\n\n"
            return history_text


# ======================
# Configuration Helper - NEW
# ======================
def create_bot_config(models: List[Dict], enhancers: List[Dict] = None, max_history: int = 20) -> Dict:
    """Create a configuration dictionary for the ChatBot"""
    config = {
        "models": models,
        "enhancers": enhancers or [],
        "max_history": max_history
    }
    return config


# ======================
# Main Application
# ======================
if __name__ == "__main__":
    # Example 1: Simple initialization
    print("ChatBot Initializing (Default Configuration)...")
    bot = ChatBot()
    
    # Example 2: Advanced configuration
    print("\nChatBot Initializing (Custom Configuration)...")
    custom_config = create_bot_config(
        models=[
            {"type": "gpt", "params": {"model_version": "gpt-4"}, "is_fallback": True},
            {"type": "intent", "params": {"custom_intents": {"product": ["features", "specs", "details"]}}}
        ],
        enhancers=[
            {"type": "sentiment", "model_type": "sentiment", "model_params": {
                "custom_lexicon": {"positive": ["excellent", "fantastic"], "negative": ["poor", "disappointing"]}
            }},
            {"type": "intent", "model_type": "intent"}
        ],
        max_history=30
    )
    advanced_bot = ChatBot(config=custom_config)
    
    # Choose which bot to use
    bot = advanced_bot  # or bot for the default one
    
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
            if "sentiment" in last_conv["metadata"]:
                sentiment = last_conv["metadata"]["sentiment"]
                print(f"Sentiment: {sentiment['sentiment']} ({sentiment['score']})")
            if "intent" in last_conv["metadata"]:
                intent = last_conv["metadata"]["intent"]
                print(f"Intent: {intent['intent']} (confidence: {intent['confidence']})")
            print(f"Model Used: {last_conv['metadata']['model_used']}")
