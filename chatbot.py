import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from typing import Optional
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules.sentiment_module import SentimentAnalyzer
from modules.time_module import TimeModule
from modules.date_module import DateModule
from config import (
    MAX_LEN, CONFIDENCE_THRESHOLD, MODEL_PATH, 
    TOKENIZER_PATH, LABEL_ENCODER_PATH, USER_PROFILE_DIR, DATA_DIR
)

class Chatbot:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.context = {}
        self.current_user = None
        self.conversation_history = []
        self.sentiment_analyzer = SentimentAnalyzer()
        self.time_module = TimeModule()
        self.date_module = DateModule()
        self.load_components()
        
    def load_components(self):
        """Load all necessary components"""
        try:
            # Load ML model and preprocessing
            self.model = load_model(self.model_path)
            with open(TOKENIZER_PATH, "rb") as handle:
                self.tokenizer = pickle.load(handle)
            with open(LABEL_ENCODER_PATH, "rb") as handle:
                self.label_encoder = pickle.load(handle)
            
            # Load knowledge bases
            self.intents = self.load_json(DATA_DIR / "intents.json")
            self.small_talk = self.load_json(DATA_DIR / "small_talk.json")
            self.faqs = self.load_json(DATA_DIR / "faqs.json")
            
        except Exception as e:
            raise RuntimeError(f"Initialization error: {str(e)}")

    def load_json(self, path):
        """Helper to load JSON files"""
        try:
            with open(path) as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return []

    def preprocess_input(self, text):
        """Prepare user input for prediction"""
        sequence = self.tokenizer.texts_to_sequences([text])
        return pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    def predict_intent(self, text):
        """Predict intent with confidence score"""
        preprocessed = self.preprocess_input(text)
        prediction = self.model.predict(preprocessed, verbose=0)
        predicted_idx = np.argmax(prediction)
        intent = self.label_encoder.classes_[predicted_idx]
        confidence = float(np.max(prediction))
        return intent, confidence

    def handle_small_talk(self, user_input):
        """Handle casual conversation with sentiment"""
        sentiment = self.sentiment_analyzer.analyze_sentiment(user_input)
        
        for item in self.small_talk.get("patterns", []):
            for pattern in item.get("triggers", []):
                if pattern.lower() in user_input.lower():
                    responses = item["responses"].get(sentiment, item["responses"].get("neutral", []))
                    if responses:
                        return random.choice(responses)
        return None

    def handle_faq(self, user_input):
        """Answer from knowledge base"""
        for item in self.faqs.get("questions", []):
            for question in item.get("variations", []):
                if question.lower() in user_input.lower():
                    return item["answer"]
        return None

    def get_intent_response(self, intent):
        """Get response for specific intent"""
        for intent_data in self.intents:
            if intent_data.get("tag") == intent:
                return random.choice(intent_data.get("responses", []))
        return None
    
    def handle_special_modules(self, user_input):
        """Handle date and time queries using dedicated modules"""
        # Try time module first
        time_response = self.time_module.process(user_input)
        if time_response:
            return time_response
            
        # Try date module if time module didn't handle it
        date_response = self.date_module.process(user_input)
        if date_response:
            return date_response
            
        return None

    def log_conversation(self, user_input, response):
        """Record conversation history"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": response,
            "user_id": self.current_user or "anonymous"
        })

    def load_user_profile(self, user_id):
        """Load user profile from storage"""
        try:
            path = USER_PROFILE_DIR / f"{user_id}.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading profile: {str(e)}")
        return None

    def save_user_profile(self, user_id, data):
        """Save user profile"""
        try:
            path = USER_PROFILE_DIR / f"{user_id}.json"
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving profile: {str(e)}")
            return False

    def personalize_response(self, response):
        """Add personalization if user is known"""
        if not self.current_user:
            return response
        
        profile = self.load_user_profile(self.current_user)
        if profile and profile.get("name"):
            return response + f", {profile['name']}"
        return response

    def process_user_input(self, user_input):
        """Main processing pipeline"""
        if not user_input.strip():
            return "Please type something so I can respond."
        
        # Track conversation
        self.log_conversation(user_input, "")

        # Try special modules (date/time) first
        module_response = self.handle_special_modules(user_input)
        if module_response:
            response = self.personalize_response(module_response)
            self.conversation_history[-1]["bot"] = response
            return response
        
        # Try small talk first
        small_talk_response = self.handle_small_talk(user_input)
        if small_talk_response:
            response = self.personalize_response(small_talk_response)
            self.conversation_history[-1]["bot"] = response
            return response
        
        # Check FAQ knowledge base
        faq_response = self.handle_faq(user_input)
        if faq_response:
            response = self.personalize_response(faq_response)
            self.conversation_history[-1]["bot"] = response
            return response
        
        # Use ML model as fallback
        intent, confidence = self.predict_intent(user_input)
        if confidence > CONFIDENCE_THRESHOLD:
            intent_response = self.get_intent_response(intent)
            if intent_response:
                response = self.personalize_response(intent_response)
                self.conversation_history[-1]["bot"] = response
                return response
        
        # Final fallback with sentiment consideration
        sentiment = self.sentiment_analyzer.analyze_sentiment(user_input)
        fallbacks = {
            "positive": [
                "That sounds great! Can you tell me more?",
                "Wonderful! What else would you like to discuss?"
            ],
            "negative": [
                "I'm sorry you're feeling this way. How can I help?",
                "That sounds difficult. Would you like to talk about it?"
            ],
            "neutral": [
                "I'm not sure I understand. Could you rephrase?",
                "That's interesting. Tell me more.",
                "I'm still learning. Could you ask something else?"
            ]
        }
        response = random.choice(fallbacks.get(sentiment, fallbacks["neutral"]))
        self.conversation_history[-1]["bot"] = response
        return response

    def chat_loop(self):
        """Interactive chat interface"""
        print("Enhanced Chatbot initialized. Type 'quit' to exit.")
        
        # User identification
        while not self.current_user:
            user_id = input("Enter your name or ID to continue: ").strip()
            if user_id.lower() == 'quit':
                return
            if user_id:
                self.current_user = user_id
                profile = self.load_user_profile(user_id)
                if not profile:
                    print("Welcome new user! Let me get to know you.")
                    name = input("What should I call you? ").strip()
                    if name.lower() == 'quit':
                        return
                    self.save_user_profile(user_id, {
                        "name": name, 
                        "created": datetime.now().isoformat(),
                        "preferences": {}
                    })
        
        # Main chat loop
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    print("Goodbye! It was nice chatting with you.")
                    break
                
                response = self.process_user_input(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue