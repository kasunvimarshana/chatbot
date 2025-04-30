import re
import spacy
import logging
from typing import Optional, List

# Constants
SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLED_COMPONENTS = ["parser", "ner"]
MIN_TOKEN_LENGTH = 2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model globally (loaded once)
try:
    nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLED_COMPONENTS)
except Exception as e:
    logger.error(f"Failed to load spaCy model '{SPACY_MODEL}': {e}")
    raise


def update_stopwords(
    words_to_add: Optional[List[str]] = None,
    words_to_remove: Optional[List[str]] = None,
) -> None:
    """
    Add or remove words from spaCy's default stop words list.
    """
    if words_to_add:
        nlp.Defaults.stop_words.update(word.lower() for word in words_to_add)
        logger.info(f"Added stopwords: {words_to_add}")

    if words_to_remove:
        nlp.Defaults.stop_words.difference_update(
            word.lower() for word in words_to_remove
        )
        logger.info(f"Removed stopwords: {words_to_remove}")


def process_input(text: str) -> str:
    """
    Clean, normalize, and lemmatize input text for intent classification.

    Args:
        text (str): Raw input text.

    Returns:
        str: Processed and cleaned text.
    """
    try:
        cleaned_text = preprocess_text(text)
        logger.info(f"Preprocessed text: {cleaned_text}")

        doc = nlp(cleaned_text)
        logger.info(f"spaCy processed document: {doc.text}")

        tokens = extract_meaningful_tokens(doc)
        logger.info(f"Final tokens after processing: {tokens}")

        processed_text = " ".join(tokens)

        return processed_text

    except Exception as e:
        logger.error(f"Error during input processing: {e}")
        return ""


def preprocess_text(text: str) -> str:
    """
    Perform basic text normalization and character cleanup.

    Args:
        text (str): Input raw string.

    Returns:
        str: Cleaned string.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove punctuation/symbols
    return re.sub(r"\s+", " ", text)


def extract_meaningful_tokens(doc: spacy.tokens.Doc) -> List[str]:
    """
    Extract lemmatized tokens, removing stopwords and irrelevant tokens.

    Args:
        doc (spacy.tokens.Doc): spaCy processed document.

    Returns:
        List[str]: List of cleaned tokens.
    """

    # tokens = []
    # for token in doc:
    #     logger.info(f"Token: {token.text}\tStopword: {token.is_stop}\tLemmatized: {token.lemma_}")
    #     if token.is_alpha and not token.is_stop and len(token.lemma_) >= MIN_TOKEN_LENGTH:
    #         tokens.append(token.lemma_.lower())

    tokens = [
        token.lemma_.lower()  # Convert the lemma (base form) of the token to lowercase
        for token in doc  # Iterate over each token in the processed spaCy document
        if token.is_alpha  # Include only alphabetic tokens (excluding numbers, punctuation, etc.)
        # and not token.is_stop  # Exclude stopwords (e.g., "the", "and", "is", etc.)
        and len(token.lemma_)
        >= MIN_TOKEN_LENGTH  # Ignore tokens that have a length less than the minimum threshold (e.g., single characters)
    ]

    return tokens
