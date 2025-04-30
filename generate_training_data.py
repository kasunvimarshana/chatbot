import json
import pandas as pd
import logging
from pathlib import Path
from config import settings
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_intent_data(intents: Dict[str, Any]) -> bool:
    """Validate the structure and content of the intents JSON data."""
    required_keys = {"intents"}
    if not all(key in intents for key in required_keys):
        logger.error(f"Intent file missing required keys: {required_keys}")
        return False
    
    for intent in intents["intents"]:
        if not isinstance(intent.get("tag", ""), str):
            logger.error(f"Invalid tag type in intent: {intent}")
            return False
        if not isinstance(intent.get("patterns", []), list):
            logger.error(f"Patterns should be a list in intent: {intent['tag']}")
            return False
        if not all(isinstance(p, str) for p in intent["patterns"]):
            logger.error(f"Non-string pattern found in intent: {intent['tag']}")
            return False
    
    return True

def generate_training_data(intents: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate training data from intents JSON structure."""
    training_data = []
    for intent in intents["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            if pattern.strip():  # Skip empty patterns
                training_data.append({
                    "text": pattern.strip(),
                    "intent": tag
                })
    return training_data

def analyze_training_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze and log statistics about the generated training data."""
    analysis = {
        "total_examples": len(df),
        "unique_intents": df["intent"].nunique(),
        "intent_distribution": df["intent"].value_counts().to_dict(),
        "examples_per_intent": df.groupby("intent").size().describe().to_dict()
    }
    
    logger.info(f"Total training examples: {analysis['total_examples']}")
    logger.info(f"Unique intents: {analysis['unique_intents']}")
    logger.info("Examples per intent statistics:")
    for stat, value in analysis["examples_per_intent"].items():
        logger.info(f"  {stat}: {value:.1f}")
    
    return analysis

def main():
    """Main function to generate and save training data."""
    try:
        logger.info(f"Loading intents from {settings.INTENT_FILE}")
        
        # Load and validate intents file
        with open(settings.INTENT_FILE, 'r', encoding='utf-8') as f:
            intents = json.load(f)
        
        if not validate_intent_data(intents):
            raise ValueError("Invalid intent data structure")
        
        # Generate training data
        training_data = generate_training_data(intents)
        df = pd.DataFrame(training_data)
        
        # Analyze and log data statistics
        analysis = analyze_training_data(df)
        
        # Save to CSV
        settings.TRAINING_DATA.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(settings.TRAINING_DATA, index=False, encoding='utf-8')
        
        logger.info(f"Successfully saved training data to {settings.TRAINING_DATA}")
        logger.info(f"Intent distribution: {analysis['intent_distribution']}")
        
        return True
    
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file: {e}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    
    return False

if __name__ == "__main__":
    main()