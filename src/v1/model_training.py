"""
Model Training Module for AI vs Human Text Detection
Uses modern transformer models (RoBERTa, DistilBERT, DeBERTa) for binary classification
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import logging
import os
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """Custom dataset for text classification"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AIHumanDetector:
    """Main class for training and using the AI vs Human text detector"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        """
        Initialize the detector with a specified model.
        
        Args:
            model_name (str): HuggingFace model name (distilbert-base-uncased, roberta-base, microsoft/deberta-v3-base)
            max_length (int): Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load the tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,  # Binary classification
                problem_type="single_label_classification"
            )
            
            # Move model to device
            self.model.to(self.device)
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def prepare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """
        Prepare train and test datasets for training.
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            
        Returns:
            Tuple[Dataset, Dataset]: Train and test datasets
        """
        logger.info("Preparing datasets...")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_model_and_tokenizer() first.")
        
        train_dataset = TextClassificationDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        test_dataset = TextClassificationDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_model(self, 
                   train_dataset: Dataset, 
                   test_dataset: Dataset,
                   output_dir: str = "./model_output",
                   num_epochs: int = 3,
                   batch_size: int = 16,
                   learning_rate: float = 2e-5,
                   warmup_steps: int = 500,
                   weight_decay: float = 0.01,
                   save_steps: int = 500,
                   eval_steps: int = 500,
                   early_stopping_patience: int = 3) -> Dict:
        """
        Train the model with specified parameters.
        
        Args:
            train_dataset (Dataset): Training dataset
            test_dataset (Dataset): Test dataset
            output_dir (str): Directory to save model outputs
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate
            warmup_steps (int): Number of warmup steps
            weight_decay (float): Weight decay for regularization
            save_steps (int): Steps between model saves
            eval_steps (int): Steps between evaluations
            early_stopping_patience (int): Patience for early stopping
            
        Returns:
            Dict: Training history and metrics
        """
        logger.info("Starting model training...")
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,  # Effectively 4x larger batch size
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            save_steps=save_steps,
            save_total_limit=2,
            learning_rate=learning_rate,
            report_to="none",  # Disable wandb/tensorboard
            dataloader_num_workers=4,  # Increase for better data loading
            dataloader_pin_memory=True,  # Pin memory for faster GPU transfer
            fp16=True,  # Use mixed precision for faster training
        )
        
        # Set up trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        logger.info("Training started...")
        train_result = trainer.train()
        
        # Save the final model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        # Return training metrics
        return {
            'training_loss': train_result.training_loss,
            'training_time': train_result.metrics['train_runtime'],
            'model_path': output_dir
        }
    
    def load_trained_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path (str): Path to the saved model
        """
        logger.info(f"Loading trained model from {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Trained model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading trained model: {str(e)}")
            raise
    
    def predict_text(self, text: str) -> Dict[str, float]:
        """
        Predict whether a text is AI-generated or human-written.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Dict[str, float]: Prediction results with probabilities
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Load a model first.")
        
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Convert to numpy and get results
        probs = probabilities.cpu().numpy()[0]
        human_prob = float(probs[0])
        ai_prob = float(probs[1])
        
        # Determine prediction
        prediction = "AI-Generated" if ai_prob > human_prob else "Human-Written"
        confidence = max(human_prob, ai_prob)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'human_probability': human_prob,
            'ai_probability': ai_prob
        }

def get_recommended_models() -> Dict[str, Dict]:
    """
    Get recommended models with their characteristics.
    
    Returns:
        Dict: Model recommendations
    """
    return {
        'distilbert-base-uncased': {
            'description': 'Lightweight, fast, good performance',
            'parameters': '66M',
            'recommended_for': 'Quick training and inference'
        },
        'roberta-base': {
            'description': 'Better performance than BERT, robust',
            'parameters': '125M',
            'recommended_for': 'High accuracy requirements'
        },
        'microsoft/deberta-v3-base': {
            'description': 'State-of-the-art performance',
            'parameters': '86M',
            'recommended_for': 'Best possible accuracy'
        }
    }

if __name__ == "__main__":
    # Example usage
    print("Model training module loaded successfully!")
    print("\nRecommended models:")
    models = get_recommended_models()
    for name, info in models.items():
        print(f"- {name}: {info['description']} ({info['parameters']} parameters)")
    
    print("\nExample usage:")
    print("detector = AIHumanDetector('distilbert-base-uncased')")
    print("detector.load_model_and_tokenizer()")
    print("# ... prepare datasets and train ...")