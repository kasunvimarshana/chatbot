from modules.nlp import process_input
from modules.intents import recognize_intent
from modules.responses import generate_response
from config import settings
from functools import lru_cache
import time
import logging
from dataclasses import dataclass, asdict
from typing import Optional, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatContext:
    last_input: str = ""
    last_intent: str = ""
    last_response: str = ""
    timestamp: float = 0.0
    custom_data: Optional[dict] = None

class ChatBot:
    def __init__(self):
        self.context = ChatContext() # {}
        logger.info("Chatbot initialized with cache size: %d", settings.CACHE_SIZE)

    def get_response(self, message: str, context: Optional[dict] = None) -> str:
        """
        Process user input and generate response with context support
        
        Args:
            message: User input text
            context: Optional conversation context dictionary
            
        Returns:
            Generated response text
        """
        try:
            # Update internal context with provided context
            if context:
                self._update_from_external_context(context)
            
            # Process the message
            start_time = time.time()
            response = self._process_message(message)
            processing_time = time.time() - start_time
            
            logger.info(f"Processed message in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            return settings.DEFAULT_RESPONSE
        
    def _process_message(self, user_input: str) -> str:
        """Internal message processing pipeline"""
        # Step 1: Process input text
        processed_text = process_input(user_input)
        
        # Step 2: Recognize intent | recognize_intent(processed_text, self.context.__dict__)
        intent = recognize_intent(processed_text, asdict(self.context))
        logger.info(f"Recognized intent: {intent}")
        
        # Step 3: Generate response | generate_response(intent, self.context.__dict__)
        response = generate_response(intent, asdict(self.context))
        
        # Update context
        self._update_context(user_input, intent, response)
        
        return response
    
    @lru_cache(maxsize=settings.CACHE_SIZE)
    def _cached_response(self, user_input: str) -> str:
        """Cached version of response generation"""
        return self._process_message(user_input)

    def _update_context(self, user_input: str, intent: dict, response: str):
        """Update conversation context"""
        # self.context.update({
        #     "last_input": user_input,
        #     "last_intent": intent.get("intent"),
        #     "last_response": response,
        #     "timestamp": time.time()
        # })
        self.context.last_input = user_input
        self.context.last_intent = intent.get("intent", "")
        self.context.last_response = response
        self.context.timestamp = time.time()

    def _update_from_external_context(self, external_context: dict):
        """Update internal context from external context dictionary"""
        if not external_context:
            return
            
        # Update standard context fields
        if "last_input" in external_context:
            self.context.last_input = external_context["last_input"]
        if "last_intent" in external_context:
            self.context.last_intent = external_context["last_intent"]
        if "last_response" in external_context:
            self.context.last_response = external_context["last_response"]
        if "timestamp" in external_context:
            self.context.timestamp = external_context["timestamp"]
            
        # Handle custom context data
        if "custom_data" in external_context:
            if self.context.custom_data is None:
                self.context.custom_data = {}
            self.context.custom_data.update(external_context["custom_data"])

    def get_current_context(self) -> dict:
        """Get current context as a dictionary"""
        context_dict = asdict(self.context)
        # Remove None values
        return {k: v for k, v in context_dict.items() if v is not None}

    def clear_cache(self):
        """Clear the response cache"""
        self._cached_response.cache_clear()
        logger.info("Response cache cleared")
