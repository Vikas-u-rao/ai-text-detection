"""
Configuration settings for the AI Text Detector
"""

# Data and Model Configuration
DATA_PATH = "../data/AI_Human.csv"
PRETRAINED_DETECTOR = "roberta-base-openai-detector"
SETFIT_MODEL = "sentence-transformers/paraphrase-mpnet-base-v2"

# Model artifacts paths
SETFIT_MODEL_PATH = "../models/setfit_model"
DETECTOR_CACHE_PATH = "../models/detector_cache"

# Prediction settings
THRESHOLD = 0.5
ENSEMBLE_METHOD = "average"

# Training settings
SETFIT_BATCH_SIZE = 32
SETFIT_ITERATIONS = 20
SETFIT_EPOCHS = 2

# UI settings
EXAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "In the rapidly evolving landscape of artificial intelligence, machine learning algorithms have demonstrated remarkable capabilities in natural language processing tasks."
]