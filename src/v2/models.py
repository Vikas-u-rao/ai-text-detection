"""
Model management for Hugging Face detector and SetFit classifier
"""

import os
import pickle
from typing import Optional, Dict, Any
from transformers import pipeline
from setfit import SetFitModel, Trainer
from datasets import Dataset
import config

# Fix TensorFlow deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set TensorFlow environment variables to avoid deprecated function calls
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage

# Import tensorflow after setting environment variables
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    # Use compatibility mode for deprecated functions
    tf.compat.v1.disable_eager_execution()
except ImportError:
    pass

class HuggingFaceDetector:
    """Wrapper for Hugging Face text classification models"""
    
    def __init__(self, model_name: str = config.PRETRAINED_DETECTOR):
        self.model_name = model_name
        self.pipeline = None
    
    def load(self):
        """Load the Hugging Face model pipeline"""
        if self.pipeline is None:
            print(f"Loading Hugging Face detector: {self.model_name}")
            self.pipeline = pipeline("text-classification", model=self.model_name, device=-1)
        return self.pipeline
    
    def predict(self, text: str) -> float:
        """
        Get AI probability score from the detector
        
        Args:
            text: Input text
            
        Returns:
            AI probability (0-1)
        """
        if self.pipeline is None:
            self.load()
        
        result = self.pipeline(text)[0]
        label = result["label"].lower()
        score = result["score"]
        
        # Some models output 'LABEL_1' for AI, some 'LABEL_0' for human
        if "ai" in label or "1" in label:
            return score
        else:
            return 1 - score

class SetFitClassifier:
    """Wrapper for SetFit fine-tuned classifier"""
    
    def __init__(self, model_name: str = config.SETFIT_MODEL):
        self.model_name = model_name
        self.model = None
        self.model_path = config.SETFIT_MODEL_PATH
    
    def train(self, texts: list, labels: list):
        """
        Train the SetFit model using HuggingFace Dataset and Trainer
        Args:
            texts: List of training texts
            labels: List of training labels (0=human, 1=AI)
        """
        print(f"Original dataset size: {len(texts)} samples")
        
        # SetFit works best with very small datasets - use 50 examples per class
        import pandas as pd
        import numpy as np
        import gc
        
        df = pd.DataFrame({"text": texts, "label": labels})
        
        # Sample very small balanced subset for memory efficiency
        max_per_class = 50  # Reduced from 1000 to 50
        sampled_dfs = []
        for label in [0, 1]:
            class_df = df[df["label"] == label]
            n_samples = min(len(class_df), max_per_class)
            sampled_df = class_df.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled_df)
        
        df_sampled = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Clean up memory
        del df, sampled_dfs
        gc.collect()
        
        texts = df_sampled["text"].tolist()
        labels = df_sampled["label"].tolist()
        
        print(f"Training SetFit model with {len(texts)} samples (balanced subset)")

        # Create model
        self.model = SetFitModel.from_pretrained(self.model_name)

        # Prepare HuggingFace Dataset
        train_data = {"text": texts, "label": labels}
        train_dataset = Dataset.from_dict(train_data)

        # Use GPU if available
        import torch
        # Force CPU usage to avoid GPU memory issues
        device = "cpu"  # Changed from auto-detection to force CPU
        print(f"Using device: {device}")
        self.model.to(device)

        # Setup Trainer (new API)
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=train_dataset
        )

        # Train
        trainer.train()

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_pretrained(self.model_path)
        print(f"SetFit model saved to {self.model_path}")
    
    def load(self):
        """Load a pre-trained SetFit model"""
        if os.path.exists(self.model_path):
            print(f"Loading SetFit model from {self.model_path}")
            self.model = SetFitModel.from_pretrained(self.model_path)
        else:
            print(f"No saved SetFit model found at {self.model_path}")
            return False
        return True
    
    def predict(self, text: str) -> float:
        """
        Get AI probability from SetFit model
        
        Args:
            text: Input text
            
        Returns:
            AI probability (0-1)
        """
        if self.model is None:
            raise ValueError("SetFit model not loaded. Call load() or train() first.")
        
        # Get prediction probabilities
        probs = self.model.predict_proba([text])[0]
        return probs[1]  # Probability of class 1 (AI)

class EnsembleDetector:
    """Ensemble of Hugging Face detector and SetFit classifier"""
    
    def __init__(self):
        self.hf_detector = HuggingFaceDetector()
        self.setfit_classifier = SetFitClassifier()
    
    def load_models(self):
        """Load both models"""
        self.hf_detector.load()
        return self.setfit_classifier.load()
    
    def train_setfit(self, texts: list, labels: list):
        """Train the SetFit component"""
        self.setfit_classifier.train(texts, labels)
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Get ensemble prediction
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with prediction results
        """
        # Get individual scores
        hf_score = self.hf_detector.predict(text)
        setfit_score = self.setfit_classifier.predict(text)
        
        # Ensemble score (simple average)
        if config.ENSEMBLE_METHOD == "average":
            ensemble_score = (hf_score + setfit_score) / 2
        else:
            ensemble_score = (hf_score + setfit_score) / 2
        
        # Classification
        if ensemble_score >= config.THRESHOLD:
            prediction = "Likely AI"
            confidence = ensemble_score
        else:
            prediction = "Likely Human"
            confidence = 1 - ensemble_score
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "hf_score": hf_score,
            "setfit_score": setfit_score,
            "ensemble_score": ensemble_score
        }