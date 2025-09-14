"""
Prediction Module for AI vs Human Text Detection
Main prediction functionality with transformer models
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from model_training import AIHumanDetector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPredictor:
    """Main prediction class using transformer models"""
    
    def __init__(self, model_path: str):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to trained model
        """
        self.model_path = model_path
        
        # Initialize transformer model
        self.detector = AIHumanDetector()
        self.detector.load_trained_model(model_path)
    
    def predict_single_text(self, text: str) -> Dict:
        """
        Predict whether a single text is AI-generated or human-written.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Dict: Prediction results with confidence scores
        """
        # Get transformer model prediction
        transformer_result = self.detector.predict_text(text)
        
        # Add text length information
        word_count = len(text.split())
        char_count = len(text)
        
        result = {
            'text_preview': text[:200] + '...' if len(text) > 200 else text,
            'word_count': word_count,
            'character_count': char_count,
            'transformer_prediction': transformer_result['prediction'],
            'transformer_confidence': transformer_result['confidence'],
            'human_probability': transformer_result['human_probability'],
            'ai_probability': transformer_result['ai_probability']
        }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict for a batch of texts.
        
        Args:
            texts (List[str]): List of texts to classify
            
        Returns:
            List[Dict]: List of prediction results
        """
        logger.info(f"Processing batch of {len(texts)} texts...")
        
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.info(f"Processing text {i+1}/{len(texts)}")
            
            result = self.predict_single_text(text)
            result['text_id'] = i
            results.append(result)
        
        return results
    
    def evaluate_on_test_set(self, test_df: pd.DataFrame, text_column: str = 'text', 
                           label_column: str = 'label') -> Dict:
        """
        Evaluate model performance on a test dataset.
        
        Args:
            test_df (pd.DataFrame): Test dataset
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            Dict: Evaluation results including predictions and metrics
        """
        logger.info("Evaluating model on test set...")
        
        texts = test_df[text_column].tolist()
        true_labels = test_df[label_column].tolist()
        
        # Get predictions
        predictions = []
        probabilities = []
        
        for text in texts:
            result = self.predict_single_text(text)
            
            # Convert prediction to binary
            pred_binary = 1 if result['transformer_prediction'] == 'AI-Generated' else 0
            predictions.append(pred_binary)
            probabilities.append(result['ai_probability'])
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report
        
        accuracy = accuracy_score(true_labels, predictions)
        
        evaluation_results = {
            'test_accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels,
            'classification_report': classification_report(true_labels, predictions, 
                                                         target_names=['Human', 'AI'])
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return evaluation_results

def predict_text_simple(model_path: str, text: str) -> str:
    """
    Simple prediction function for quick testing.
    
    Args:
        model_path (str): Path to trained model
        text (str): Input text
        
    Returns:
        str: Simple prediction result
    """
    try:
        predictor = TextPredictor(model_path)
        result = predictor.predict_single_text(text)
        
        prediction = result['transformer_prediction']
        confidence = result['transformer_confidence']
        
        return f"{prediction} (Confidence: {confidence:.3f})"
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Example usage
    print("Prediction module loaded successfully!")
    print("\nAvailable classes:")
    print("- TextPredictor: Main prediction class with transformer models")
    print("\nQuick function:")
    print("- predict_text_simple(model_path, text): Simple prediction for testing")
    
    # Example
    print("\nExample usage:")
    print("predictor = TextPredictor('./model_output')")
    print("result = predictor.predict_single_text('Your text here')")
    print("print(result['transformer_prediction'])")