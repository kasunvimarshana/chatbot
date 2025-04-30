import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime
from pathlib import Path
import logging
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create reports directory path
REPORTS_DIR = settings.REPORTS_DIR
REPORTS_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't exist

def plot_class_distribution(y):
    """Plot the distribution of classes with count annotations"""
    plt.figure(figsize=(12, 6))
    ax = y.value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Intent Classes")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    # Add count labels to each bar
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot normalized confusion matrix with class labels"""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",  # Show 2 decimal places
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()


def plot_metrics(report_df):
    """Plot precision, recall, and f1-scores in separate subplots"""
    report_df = report_df.sort_values(by="support", ascending=False)
    plt.figure(figsize=(12, 8))

    # Precision subplot
    plt.subplot(311)
    report_df["precision"].plot(kind="bar", title="Precision", color="skyblue")
    plt.xticks([])  # Hide x-axis labels for cleaner look

    # Recall subplot
    plt.subplot(312)
    report_df["recall"].plot(kind="bar", title="Recall", color="lightgreen")
    plt.xticks([])

    # F1-score subplot
    plt.subplot(313)
    report_df["f1-score"].plot(kind="bar", title="F1-Score", color="salmon")
    plt.xticks(rotation=90)  # Show class names only on bottom plot

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "metrics_comparison.png")
    plt.close()

def plot_feature_importance(model, vectorizer, top_n=30):
    """Plot top feature importances from the Random Forest model"""
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_ attribute")
        return
    
    feature_names = vectorizer.get_feature_names_out()
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.title(f"Top {top_n} Important Features")
    plt.barh(range(top_n), importances[indices], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png")
    plt.close()

def save_classification_report(report_str, report_df):
    """Save classification reports to text and CSV files"""
    # Save text report
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report_str)
    
    # Save metrics dataframe
    report_df.to_csv(REPORTS_DIR / "classification_metrics.csv")


def train_model():
    """Main function to train and save intent classification model"""
    try:
        logger.info("Starting model training...")

        # Load and validate training data
        logger.info(f"Loading data from {settings.TRAINING_DATA}")
        data = pd.read_csv(settings.TRAINING_DATA)
        logger.info(
            f"Loaded {len(data)} samples with {data['intent'].nunique()} unique intents"
        )

        # Visualize class distribution before training
        plot_class_distribution(data['intent'])

        # Log class distribution statistics
        class_dist = data["intent"].value_counts(normalize=True)
        logger.info(f"Class distribution:\n{class_dist.to_string()}")

        # Create TF-IDF vectorizer with n-grams
        vectorizer = TfidfVectorizer(
            max_features=15000,   # Limit vocabulary size
            ngram_range=(1, 3),   # Include unigrams, bigrams and trigrams
            stop_words="english", # Remove common English words
            min_df=3,             # Ignore terms that appear in fewer than 5 docs
            max_df=0.7,           # Ignore terms that appear in more than 80% of docs
            sublinear_tf=True     # Use sublinear tf scaling
        )

        X = vectorizer.fit_transform(data["text"])
        y = data["intent"]

        # Split data with stratification to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(
            f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}"
        )

        # Initialize Random Forest classifier with balanced class weights
        model = RandomForestClassifier(
            n_estimators=500,                   # Number of trees in forest
            max_depth=30,                       # Maximum tree depth
            random_state=42,                    # For reproducibility
            class_weight="balanced_subsample",  # Handle class imbalance
            min_samples_split=5,               # Minimum samples to split node
            min_samples_leaf=2,                 # Minimum samples at leaf node
            max_features='log2',                # Number of features to consider at each split
            n_jobs=-1,                          # Use all available cores
            verbose=1                           # Show training progress
        )

        logger.info("Training model...")
        model.fit(X_train, y_train)

        # Evaluate model on test set
        y_pred = model.predict(X_test)

        # Generate classification reports
        report_str = classification_report(y_test, y_pred, zero_division=0)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        logger.info(f"Classification Report:\n{report_str}")

        # Prepare report DataFrame for visualization
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.iloc[:-3, :]  # Remove average rows
        
        # Generate evaluation visualizations
        plot_metrics(report_df)
        plot_confusion_matrix(y_test, y_pred, sorted(y.unique()))
        plot_feature_importance(model, vectorizer)

        # Save trained artifacts
        model_path = settings.MODEL_PATH
        vectorizer_path = settings.VECTORIZER_PATH

        # Create directory if it doesn't exist
        settings.MODEL_PATH.parent.mkdir(exist_ok=True)
        
        # Save model and vectorizer using pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")

        # Save metrics for future reference
        save_classification_report(report_str, report_df)
        logger.info(f"All reports and visualizations saved to {REPORTS_DIR}")
        
        return True

    except Exception as e:
        logger.exception(f"Error in train_model: {str(e)}")
        return False


if __name__ == "__main__":
    train_model()
