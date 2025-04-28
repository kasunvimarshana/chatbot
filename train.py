# train.py
import json
import pickle
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from config import MAX_LEN, MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH, DATA_DIR

class ChatTrainer:
    def __init__(self):
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.max_len = MAX_LEN
        
    def load_all_data(self):
        """Combine data from all sources for training"""
        with open(DATA_DIR / "intents.json") as f:
            intents = json.load(f)
        
        with open(DATA_DIR / "small_talk.json") as f:
            small_talk = json.load(f)
        
        training_sentences = []
        training_labels = []
        
        # Add main intents
        for intent in intents:
            for pattern in intent["patterns"]:
                training_sentences.append(pattern)
                training_labels.append(intent["tag"])
        
        # Add small talk patterns
        for item in small_talk["patterns"]:
            for trigger in item["triggers"]:
                training_sentences.append(trigger)
                training_labels.append("small_talk")
        
        return training_sentences, training_labels
    
    def train_model(self, epochs=200):
        sentences, labels = self.load_all_data()
        
        # Encode labels
        self.label_encoder.fit(labels)
        encoded_labels = self.label_encoder.transform(labels)
        
        # Tokenize text
        self.tokenizer.fit_on_texts(sentences)
        sequences = self.tokenizer.texts_to_sequences(sentences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Model architecture
        vocab_size = len(self.tokenizer.word_index) + 1
        num_classes = len(self.label_encoder.classes_)
        
        model = Sequential([
            Embedding(vocab_size, 16, input_length=self.max_len),
            GlobalAveragePooling1D(),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.01),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            padded_sequences,
            encoded_labels,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Save artifacts
        model.save(MODEL_PATH)
        with open(TOKENIZER_PATH, "wb") as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(LABEL_ENCODER_PATH, "wb") as handle:
            pickle.dump(self.label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return history

if __name__ == "__main__":
    trainer = ChatTrainer()
    history = trainer.train_model(epochs=200)
    print("Training completed successfully!")
    print(f"Model saved to {MODEL_PATH}")