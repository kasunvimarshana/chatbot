import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime
import pickle
import os
from modules.sentiment import analyze_sentiment
from config import MAX_LEN, CONFIDENCE_THRESHOLD, MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, USER_PROFILE_DIR

class Chatbot:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.context = {}
        self.current_user = None
        self.load_components()
        
    def load_components(self):
        """Load all necessary components"""
        try:
            # Load ML model
            self.model = load_model(self.model_path)
            with open(TOKENIZER_PATH, "rb") as handle:
                self.tokenizer = pickle.load(handle)
            with open(LABEL_ENCODER_PATH, "rb") as handle:
                self.label_encoder = pickle.load(handle)
            
            # Load knowledge bases
            self.intents = self.load_json("data/intents.json")
            self.small_talk = self.load_json("data/small_talk.json")
            self.faqs = self.load_json("data/faqs.json")
            
            # Load user profiles directory
            os.makedirs(USER_PROFILE_DIR, exist_ok=True)
            
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def load_json(self, path):
        """Helper to load JSON files"""
        with open(path) as file:
            return json.load(file)

    def preprocess_input(self, text):
        """Prepare user input for prediction"""
        sequence = self.tokenizer.texts_to_sequences([text])
        return pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    def predict_intent(self, text):
        """Predict intent with confidence score"""
        preprocessed = self.preprocess_input(text)
        prediction = self.model.predict(preprocessed)
        predicted_idx = np.argmax(prediction)
        intent = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = np.max(prediction)
        return intent, confidence

    def handle_small_talk(self, user_input):
        """Handle casual conversation"""
        sentiment = analyze_sentiment(user_input)
        
        for item in self.small_talk["patterns"]:
            for pattern in item["triggers"]:
                if pattern.lower() in user_input.lower():
                    # Adjust response based on sentiment
                    if sentiment == "positive":
                        return random.choice(item["responses"]["positive"])
                    elif sentiment == "negative":
                        return random.choice(item["responses"]["negative"])
                    return random.choice(item["responses"]["neutral"])
        return None

    def handle_faq(self, user_input):
        """Answer from knowledge base"""
        for item in self.faqs["questions"]:
            for question in item["variations"]:
                if question.lower() in user_input.lower():
                    return item["answer"]
        return None

    def personalize_response(self, response):
        """Add personalization if user is known"""
        if self.current_user:
            profile = self.load_user_profile(self.current_user)
            if profile:
                name = profile.get("name", "")
                if name:
                    return response.replace("{name}", name)
        return response.replace("{name}", "")

    def load_user_profile(self, user_id):
        """Load user profile from storage"""
        try:
            path = f"{USER_PROFILE_DIR}/{user_id}.json"
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
        except:
            return None

    def save_user_profile(self, user_id, data):
        """Save user profile"""
        try:
            path = f"{USER_PROFILE_DIR}/{user_id}.json"
            with open(path, 'w') as f:
                json.dump(data, f)
            return True
        except:
            return False

    def process_user_input(self, user_input):
        """Main processing pipeline"""
        try:
            # Try small talk
            small_talk_response = self.handle_small_talk(user_input)
            if small_talk_response:
                return self.personalize_response(small_talk_response)
            
            # Check FAQ knowledge base
            faq_response = self.handle_faq(user_input)
            if faq_response:
                return self.personalize_response(faq_response)
            
            # Use ML model as fallback
            intent, confidence = self.predict_intent(user_input)
            if confidence > CONFIDENCE_THRESHOLD:  # Confidence threshold
                for intent_data in self.intents["intents"]:
                    if intent_data["tag"] == intent:
                        return self.personalize_response(random.choice(intent_data["responses"]))
            
            # Final fallback
            fallbacks = [
                "I'm not sure I understand. Could you rephrase?",
                "That's interesting. Tell me more.",
                "I'm still learning. Could you ask something else?"
            ]
            return random.choice(fallbacks)
            
        except Exception as e:
            print(f"Error processing input: {str(e)}")
            return "I encountered an error. Please try again."

    def chat_loop(self):
        """Interactive chat interface"""
        print("Enhanced Chatbot initialized. Type 'quit' to exit.")
        
        # Simple user identification
        user_id = input("Enter your name or ID to continue: ").strip()
        if user_id:
            self.current_user = user_id
            profile = self.load_user_profile(user_id)
            if not profile:
                print("Welcome new user! Let me get to know you.")
                name = input("What should I call you? ").strip()
                if name:
                    self.save_user_profile(user_id, {"name": name, "created": str(datetime.now())})
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                
                response = self.process_user_input(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue