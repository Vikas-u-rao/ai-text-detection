"""
AI vs Human Text Detector
Using Sentence Transformers + Lightweight Classifier

This script implements a complete pipeline for detecting AI-generated vs human-written text
using pretrained sentence embeddings and traditional ML classifiers.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sentence_transformers import SentenceTransformer
import gradio as gr
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class AITextDetector:
    """Main class for AI vs Human text detection"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the detector with a sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.sentence_model = None
        self.classifier = None
        self.is_trained = False
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and inspect the dataset
        
        Args:
            data_path: Path to the CSV dataset file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("📂 Loading dataset...")
        
        # Load the dataset
        df = pd.read_csv(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for label column (common names: 'label', 'generated', 'target', 'class')
        label_cols = ['label', 'generated', 'target', 'class', 'ai_generated']
        text_cols = ['text', 'content', 'message', 'article']
        
        label_col = None
        text_col = None
        
        for col in label_cols:
            if col in df.columns:
                label_col = col
                break
                
        for col in text_cols:
            if col in df.columns:
                text_col = col
                break
        
        if label_col is None:
            print("⚠️  No standard label column found. Available columns:", list(df.columns))
            # Try to use the second column as label if it exists
            if len(df.columns) >= 2:
                label_col = df.columns[1]
                print(f"Using '{label_col}' as label column")
        
        if text_col is None:
            print("⚠️  No standard text column found. Available columns:", list(df.columns))
            # Try to use the first column as text if it exists
            if len(df.columns) >= 1:
                text_col = df.columns[0]
                print(f"Using '{text_col}' as text column")
        
        # Rename columns for consistency
        df = df.rename(columns={text_col: 'text', label_col: 'label'})
        
        print(f"\n📊 Label distribution:")
        print(df['label'].value_counts())
        
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the text data
        
        Args:
            df: Input dataframe with 'text' and 'label' columns
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("🧹 Preprocessing data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove missing values
        initial_size = len(df_clean)
        df_clean = df_clean.dropna(subset=['text', 'label'])
        removed = initial_size - len(df_clean)
        if removed > 0:
            print(f"Removed {removed} rows with missing values")
        
        # Clean text minimally
        df_clean['text'] = df_clean['text'].astype(str)
        df_clean['text'] = df_clean['text'].str.strip()  # Remove leading/trailing spaces
        df_clean['text'] = df_clean['text'].str.lower()  # Convert to lowercase
        
        # Remove empty texts
        df_clean = df_clean[df_clean['text'].str.len() > 0]
        
        # Ensure binary labels: 0 = Human, 1 = AI
        unique_labels = df_clean['label'].unique()
        print(f"Unique labels before processing: {unique_labels}")
        
        # Convert labels to binary if needed
        if set(unique_labels) == {0, 1} or set(unique_labels) == {0.0, 1.0}:
            df_clean['label'] = df_clean['label'].astype(int)
        elif set(unique_labels) == {'human', 'ai'} or set(unique_labels) == {'Human', 'AI'}:
            df_clean['label'] = (df_clean['label'].str.lower() == 'ai').astype(int)
        elif len(unique_labels) == 2:
            # Map the labels to 0 and 1
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            df_clean['label'] = df_clean['label'].map(label_map)
            print(f"Mapped labels: {label_map}")
        
        print(f"Final dataset size: {len(df_clean)}")
        print(f"Final label distribution:")
        print(f"Human (0): {(df_clean['label'] == 0).sum()}")
        print(f"AI (1): {(df_clean['label'] == 1).sum()}")
        
        return df_clean
    
    def embed_text(self, texts: list) -> np.ndarray:
        """
        Convert texts to embeddings using sentence transformer
        
        Args:
            texts: List of text strings
            
        Returns:
            np.ndarray: Text embeddings
        """
        if self.sentence_model is None:
            print(f"🤖 Loading sentence transformer: {self.model_name}")
            self.sentence_model = SentenceTransformer(self.model_name)
        
        print(f"🔤 Generating embeddings for {len(texts)} texts...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
        
        print(f"Embedding shape: {embeddings.shape}")
        return embeddings
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray,
                        classifier_type: str = "random_forest") -> Dict[str, Any]:
        """
        Train a lightweight classifier on the embeddings
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            X_test: Test embeddings  
            y_test: Test labels
            classifier_type: Type of classifier ("random_forest" or "logistic_regression")
            
        Returns:
            Dict: Training results and metrics
        """
        print(f"🎯 Training {classifier_type} classifier...")
        
        # Initialize classifier
        if classifier_type == "random_forest":
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif classifier_type == "logistic_regression":
            self.classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError("classifier_type must be 'random_forest' or 'logistic_regression'")
        
        # Train the classifier
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classifier_type': classifier_type,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"\n📈 Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
        
        self.is_trained = True
        return results
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict if text is AI-generated or human-written
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict: Prediction results with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet. Call train_classifier() first.")
        
        # Preprocess text
        text_clean = str(text).strip().lower()
        
        # Generate embedding
        embedding = self.embed_text([text_clean])
        
        # Make prediction
        prediction = self.classifier.predict(embedding)[0]
        probabilities = self.classifier.predict_proba(embedding)[0]
        
        # Format results
        result = {
            'prediction': 'AI-Generated' if prediction == 1 else 'Human-Written',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'human': float(probabilities[0]),
                'ai': float(probabilities[1])
            },
            'prediction_label': int(prediction)
        }
        
        return result
    
    def save_model(self, filepath: str):
        """Save the trained model and sentence transformer"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'model_name': self.model_name,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        model_data = joblib.load(filepath)
        
        self.classifier = model_data['classifier']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        
        # Load sentence transformer
        self.sentence_model = SentenceTransformer(self.model_name)
        
        print(f"📥 Model loaded from {filepath}")


def main_training_pipeline(data_path: str, classifier_type: str = "random_forest"):
    """
    Complete training pipeline
    
    Args:
        data_path: Path to the dataset CSV file
        classifier_type: Type of classifier to use
    """
    # Initialize detector
    detector = AITextDetector()

    # Load and preprocess data
    df = detector.load_data(data_path)
    df_clean = detector.preprocess(df)

    # Optionally balance dataset (downsample majority class)
    balance = True  # Set to False to disable balancing
    if balance:
        print("⚖️ Balancing dataset by downsampling majority class...")
        min_count = df_clean['label'].value_counts().min()
        df_balanced = pd.concat([
            df_clean[df_clean['label'] == 0].sample(min_count, random_state=42),
            df_clean[df_clean['label'] == 1].sample(min_count, random_state=42)
        ])
        df_clean = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced label distribution:\n{df_clean['label'].value_counts()}")

    # Train/test split (80/20)
    print("📊 Splitting data (80% train, 20% test)...")
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df_clean['text'].tolist(),
        df_clean['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df_clean['label']
    )

    print(f"Train size: {len(train_texts)}")
    print(f"Test size: {len(test_texts)}")

    # Generate embeddings
    X_train = detector.embed_text(train_texts)
    X_test = detector.embed_text(test_texts)

    # Train classifier (optionally tune hyperparameters)
    # Example: classifier_type = "random_forest" or "logistic_regression"
    results = detector.train_classifier(
        X_train, train_labels,
        X_test, test_labels,
        classifier_type=classifier_type
    )

    # Print confusion matrix and classification report
    from sklearn.metrics import confusion_matrix, classification_report
    y_pred = results['predictions']
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, y_pred))
    print("\nClassification Report:")
    print(classification_report(test_labels, y_pred, target_names=['Human', 'AI']))

    # Save the model
    model_filename = f"ai_detector_{classifier_type}.joblib"
    detector.save_model(model_filename)

    return detector, results


def launch_gradio_app(detector: AITextDetector):
    """
    Launch Gradio web interface
    
    Args:
        detector: Trained AITextDetector instance
    """
    def predict_text(text):
        if not text.strip():
            return "Please enter some text to analyze.", "", ""
        
        try:
            result = detector.predict(text)
            
            prediction = result['prediction']
            confidence = f"{result['confidence']:.2%}"
            
            prob_details = f"Human: {result['probabilities']['human']:.2%} | AI: {result['probabilities']['ai']:.2%}"
            
            return prediction, confidence, prob_details
            
        except Exception as e:
            return f"Error: {str(e)}", "", ""
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=predict_text,
        inputs=gr.Textbox(
            label="Enter text to analyze",
            placeholder="Paste your text here...",
            lines=5
        ),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Textbox(label="Confidence"),
            gr.Textbox(label="Probability Breakdown")
        ],
        title="🤖 AI vs Human Text Detector",
        description="Detect whether text is AI-generated or human-written using sentence transformers and machine learning.",
        examples=[
            ["The quick brown fox jumps over the lazy dog. This is a simple test sentence."],
            ["In the rapidly evolving landscape of artificial intelligence, machine learning algorithms have demonstrated remarkable capabilities in natural language processing tasks."]
        ]
    )
    
    return interface


def launch_streamlit_app(detector: AITextDetector):
    """
    Launch Streamlit web interface
    
    Args:
        detector: Trained AITextDetector instance
    """
    st.title("🤖 AI vs Human Text Detector")
    st.write("Detect whether text is AI-generated or human-written using sentence transformers and machine learning.")
    
    # Text input
    user_text = st.text_area(
        "Enter text to analyze:",
        placeholder="Paste your text here...",
        height=200
    )
    
    if st.button("Analyze Text"):
        if user_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    result = detector.predict(user_text)
                    
                    # Display results
                    st.subheader("Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Prediction", 
                            value=result['prediction']
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence", 
                            value=f"{result['confidence']:.2%}"
                        )
                    
                    # Probability breakdown
                    st.subheader("Probability Breakdown")
                    prob_data = {
                        'Class': ['Human', 'AI'],
                        'Probability': [
                            result['probabilities']['human'],
                            result['probabilities']['ai']
                        ]
                    }
                    st.bar_chart(pd.DataFrame(prob_data).set_index('Class'))
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    # Example usage
    print("🚀 AI vs Human Text Detector")
    print("=" * 50)
    
    # You can run the training pipeline like this:
    # detector, results = main_training_pipeline("path/to/your/dataset.csv")
    
    # Or load a pre-trained model and launch the app:
    # detector = AITextDetector()
    # detector.load_model("ai_detector_random_forest.joblib")
    # app = launch_gradio_app(detector)
    # app.launch()
    
    print("To use this detector:")
    print("1. Run main_training_pipeline('your_dataset.csv') to train")
    print("2. Use launch_gradio_app(detector) or launch_streamlit_app(detector) for web interface")