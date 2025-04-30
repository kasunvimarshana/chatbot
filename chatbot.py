from modules.nlp import process_input
from modules.intents import recognize_intent
from modules.responses import generate_response
from config import settings
from functools import lru_cache
import time
import logging
from dataclasses import dataclass
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatContext:
    last_input: str = ""
    last_intent: str = ""
    last_response: str = ""
    timestamp: float = 0.0

class ChatBot:
    def __init__(self):
        self.context = ChatContext() # {}
        logger.info("Chatbot initialized with cache size: %d", settings.CACHE_SIZE)

    @lru_cache(maxsize=settings.CACHE_SIZE)
    def get_response(self, user_input: str) -> str:
        """Process user input and generate response"""
        try:
            # Step 1: Process input text
            processed_text = process_input(user_input)
            
            # Step 2: Recognize intent
            intent = recognize_intent(processed_text, self.context.__dict__)
            logger.info(f"Recognized intent: {intent}")
            
            # Step 3: Generate response
            response = generate_response(intent, self.context.__dict__)
            
            # Update context
            self._update_context(user_input, intent, response)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            return settings.DEFAULT_RESPONSE

    def _update_context(self, user_input: str, intent: dict, response: str):
        """Update conversation context"""
        # self.context.update({
        #     "last_input": user_input,
        #     "last_intent": intent.get("intent"),
        #     "last_response": response,
        #     "timestamp": time.time()
        # })
        # Update context
        self.context.last_input = user_input
        self.context.last_intent = intent.get("intent", "")
        self.context.last_response = response
        self.context.timestamp = time.time()

    def clear_cache(self):
        """Clear the response cache"""
        self.get_response.cache_clear()
        logger.info("Response cache cleared")
