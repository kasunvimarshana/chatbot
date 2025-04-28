chatbot/
│
├── data/
│   ├── intents.json          # Main training data
│   ├── small_talk.json       # Casual conversation patterns
│   ├── faqs.json             # Knowledge base
│   └── model.h5              # Saved model
│
├── modules/
│   └── sentiment.py          # Sentiment analysis
│
├── chatbot.py                # Enhanced main chatbot class
├── train.py                  # Training script
└── app.py                    # Application interface