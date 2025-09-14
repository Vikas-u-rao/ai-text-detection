"""
Training pipeline for the AI text detector
"""

import os
import warnings

# Fix TensorFlow deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set TensorFlow environment variables before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from data_utils import load_dataset, prepare_training_data
from models import EnsembleDetector
import config

def train_pipeline():
    """
    Complete training pipeline
    """
    print("Starting training pipeline...")
    
    # Load dataset
    try:
        df = load_dataset(config.DATA_PATH)
    except FileNotFoundError:
        print(f"Dataset not found at {config.DATA_PATH}")
        print("Please ensure your dataset is available or update DATA_PATH in config.py")
        return None
    
    # Prepare training data
    texts, labels = prepare_training_data(df)
    
    # Initialize ensemble detector
    detector = EnsembleDetector()
    
    # Load Hugging Face detector
    detector.hf_detector.load()
    
    # Train SetFit classifier
    detector.train_setfit(texts, labels)
    
    print("Training completed successfully!")
    return detector

if __name__ == "__main__":
    train_pipeline()