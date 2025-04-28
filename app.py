from chatbot import Chatbot
import argparse

def main():
    """Entry point for chatbot interaction."""
    parser = argparse.ArgumentParser(description="Enhanced Chatbot")
    parser.add_argument('--train', action='store_true', help='Train the model before chatting')
    args = parser.parse_args()

    if args.train:
        from train import ChatTrainer
        print("Training model...")
        trainer = ChatTrainer()
        trainer.train_model(epochs=200)
        print("Training completed!")

    print("Starting chatbot...")
    bot = Chatbot()
    bot.chat_loop()

if __name__ == "__main__":
    main()