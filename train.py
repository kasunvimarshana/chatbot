import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree, export_graphviz
import pickle
from datetime import datetime
from pathlib import Path
import logging
import graphviz
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
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot normalized confusion matrix with class labels"""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
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
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(REPORTS_DIR / "metrics_comparison.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_tfidf_heatmap(vectorizer, X, y, max_docs=50, max_terms=50):
    """Plot a heatmap of TF-IDF scores for a sample of documents and terms"""
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Convert sparse matrix to dense array for the sample
    dense_X = X[:max_docs].toarray()
    
    # Get most frequent terms across the sample
    term_counts = dense_X.sum(axis=0)
    top_term_indices = np.argsort(term_counts)[-max_terms:][::-1]
    top_terms = feature_names[top_term_indices]
    
    # Get the subset of the matrix for our sample
    subset_X = dense_X[:max_docs, top_term_indices]
    
    # Get the corresponding labels
    subset_y = y[:max_docs].values
    
    # Create dataframe for better labeling
    df = pd.DataFrame(subset_X, columns=top_terms, index=subset_y)
    df.index.name = 'Intent'
    
    # Plot heatmap
    plt.figure(figsize=(20, 12))
    sns.heatmap(
        df,
        cmap="YlGnBu",
        cbar_kws={'label': 'TF-IDF Score'},
        yticklabels=True
    )
    
    plt.title(f"TF-IDF Scores for Top {max_terms} Terms in First {max_docs} Documents")
    plt.xlabel("Terms")
    plt.ylabel("Document Intent")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "tfidf_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_random_forest_structure(model):
    """
    Visualize the structure of the Random Forest classifier.
    Includes depth distribution, node count, and sample tree visualization.
    """
    try:
        if not isinstance(model, RandomForestClassifier):
            logger.warning("Input is not a RandomForestClassifier")
            return
            
        if not hasattr(model, 'estimators_'):
            logger.warning("Model hasn't been trained yet - no estimators available")
            return
            
        plt.figure(figsize=(18, 12))
        
        # Plot 1: Tree depth distribution
        plt.subplot(2, 2, 1)
        depths = [tree.tree_.max_depth for tree in model.estimators_]
        sns.histplot(depths, bins=30, kde=True)
        plt.title('Distribution of Tree Depths')
        plt.xlabel('Tree Depth')
        plt.ylabel('Count')
        
        # Plot 2: Number of nodes distribution
        plt.subplot(2, 2, 2)
        nodes = [tree.tree_.node_count for tree in model.estimators_]
        sns.histplot(nodes, bins=30, kde=True)
        plt.title('Distribution of Node Counts')
        plt.xlabel('Number of Nodes')
        plt.ylabel('Count')
        
        # Plot 3: Leaf node samples distribution
        plt.subplot(2, 2, 3)
        leaf_samples = [tree.tree_.n_node_samples[-1] for tree in model.estimators_]
        sns.histplot(leaf_samples, bins=30, kde=True)
        plt.title('Distribution of Samples in Leaf Nodes')
        plt.xlabel('Samples in Leaf Nodes')
        plt.ylabel('Count')
        
        # Plot 4: Feature importance consistency
        plt.subplot(2, 2, 4)
        importances = np.array([tree.feature_importances_ for tree in model.estimators_])
        sns.heatmap(importances, cmap="viridis", cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance Across Trees')
        plt.xlabel('Feature Index')
        plt.ylabel('Tree Index')
        
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "random_forest_structure.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log statistics
        stats = {
            'avg_depth': np.mean(depths),
            'avg_nodes': np.mean(nodes),
            'avg_leaf_samples': np.mean(leaf_samples),
            'importance_std': np.std(importances, axis=0).mean()
        }
        logger.info(f"Random Forest stats: {stats}")
        
    except Exception as e:
        logger.exception(f"Error in plot_random_forest_structure: {str(e)}")

def export_interactive_tree(model, vectorizer, tree_index=0):
    """
    Export a decision tree as an interactive Graphviz visualization.
    Returns the Graphviz object and saves as DOT file.
    """
    try:
        dot_data = export_graphviz(
            model.estimators_[tree_index],
            out_file=None,
            feature_names=vectorizer.get_feature_names_out(),
            class_names=model.classes_,
            filled=True,
            rounded=True,
            special_characters=True,
            proportion=True,
            impurity=False
        )
        
        graph = graphviz.Source(dot_data)
        dot_path = REPORTS_DIR / f"decision_tree_{tree_index}.dot"
        graph.save(dot_path.as_posix())
        
        return graph
        
    except Exception as e:
        logger.exception(f"Error exporting interactive tree {tree_index}: {str(e)}")
        return None

def plot_single_tree(model, vectorizer, tree_index=0, max_depth=3, figsize=(25, 12), dpi=200):
    """
    Visualize a single decision tree with enhanced formatting.
    Saves both PNG and PDF versions for different use cases.
    """
    try:
        # Set the backend to Agg (non-interactive) before importing pyplot
        import matplotlib
        matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
        import matplotlib.pyplot as plt
        
        if tree_index >= len(model.estimators_):
            logger.warning(f"Tree index {tree_index} out of range")
            return
            
        if len(vectorizer.get_feature_names_out()) > 1000:
            logger.warning("Too many features for clear tree visualization")
            return
            
        plt.figure(figsize=figsize, dpi=dpi)
        plot_tree(
            model.estimators_[tree_index],
            feature_names=vectorizer.get_feature_names_out(),
            class_names=model.classes_,
            filled=True,
            max_depth=max_depth,
            fontsize=8,
            rounded=True,
            proportion=True,
            impurity=False
        )
        plt.title(f"Decision Tree {tree_index} (Max Depth Shown: {max_depth})")
        plt.tight_layout()
        
        base_path = REPORTS_DIR / f"decision_tree_{tree_index}"
        plt.savefig(f"{base_path}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{base_path}.pdf", bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.exception(f"Error plotting tree {tree_index}: {str(e)}")

def plot_multiple_trees(model, vectorizer, n_trees=3, max_depth=3):
    """
    Plot multiple trees sequentially (modified from parallel version)
    """
    try:
        n_trees = min(n_trees, len(model.estimators_))
        
        # Plot trees sequentially
        for i in range(n_trees):
            plot_single_tree(model, vectorizer, i, max_depth)
            
            # Export interactive version if feature count is reasonable
            if len(vectorizer.get_feature_names_out()) <= 100:
                export_interactive_tree(model, vectorizer, i)
                
    except Exception as e:
        logger.exception(f"Error in plot_multiple_trees: {str(e)}")

def save_classification_report(report_str, report_df):
    """Save classification reports to text and CSV files"""
    # Save text report
    with open(REPORTS_DIR / "classification_report.txt", "w") as f:
        f.write(report_str)
    
    # Save metrics dataframe
    report_df.to_csv(REPORTS_DIR / "classification_metrics.csv", index=True)


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
        vectorizer = TfidfVectorizer()

        X = vectorizer.fit_transform(data["text"])
        y = data["intent"]

        # Split data with stratification to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        logger.info(
            f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}"
        )

        # Initialize Random Forest classifier with balanced class weights
        model = RandomForestClassifier()

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
        plot_tfidf_heatmap(vectorizer, X, y)
        plot_random_forest_structure(model)
        plot_multiple_trees(model, vectorizer, n_trees=3, max_depth=3)

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
