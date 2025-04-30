import json
import random
from typing import Dict, Any
from config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load intents
try:
    with open(settings.INTENT_FILE) as f:
        INTENTS = json.load(f)
except Exception as e:
    logging.error(f"Error loading intents: {str(e)}")
    INTENTS = {"intents": []}

def generate_response(intent_data: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Generate appropriate response based on intent"""
    try:
        intent = intent_data.get("intent", "unknown")
        
        if intent == "unknown":
            return _handle_unknown(context)
        
        if intent == "error":
            return settings.DEFAULT_RESPONSE
        
        # Find matching intent
        for intent_obj in INTENTS["intents"]:
            if intent_obj["tag"] == intent:
                return _select_response(intent_obj, context)
        
        return settings.DEFAULT_RESPONSE
    
    except Exception as e:
        logger.error(f"Error in generate_response: {str(e)}")
        return settings.DEFAULT_RESPONSE

def _select_response(intent_obj: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Select the most appropriate response"""
    responses = intent_obj.get("responses", [])
    
    # Context-aware response selection
    if "last_intent" in context and context["last_intent"] == intent_obj["tag"]:
        contextual_responses = [
            r for r in responses 
            if any(word in r.lower() for word in ["continue", "more", "again"])
        ]
        if contextual_responses:
            return random.choice(contextual_responses)
    
    return random.choice(responses) if responses else settings.DEFAULT_RESPONSE

def _handle_unknown(context: Dict[str, Any]) -> str:
    """Handle unknown intents"""
    if "last_intent" in context:
        return f"I'm not sure about that. Were we discussing {context['last_intent']}?"
    return settings.DEFAULT_RESPONSE