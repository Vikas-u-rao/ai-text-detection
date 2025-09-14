"""
Data Loading and Preprocessing Module for AI vs Human Text Detection
"""
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load the AI vs Human dataset from CSV file.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        # Display basic information about the dataset
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values}")
        
        # Display class distribution
        if 'label' in df.columns or 'generated' in df.columns:
            label_col = 'label' if 'label' in df.columns else 'generated'
            class_counts = df[label_col].value_counts()
            logger.info(f"Class distribution:\n{class_counts}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def preprocess_text(text: str, remove_urls: bool = True, remove_special_chars: bool = False) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        remove_urls (bool): Whether to remove URLs
        remove_special_chars (bool): Whether to remove special characters
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs if specified
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters if specified (keep basic punctuation)
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def prepare_dataset(df: pd.DataFrame, text_column: str = 'text', label_column: str = None) -> pd.DataFrame:
    """
    Prepare the dataset for training by cleaning and standardizing labels.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column
        label_column (str): Name of the label column (auto-detect if None)
        
    Returns:
        pd.DataFrame: Prepared dataset with cleaned text and binary labels
    """
    df_clean = df.copy()
    
    # Auto-detect label column if not specified
    if label_column is None:
        possible_labels = ['label', 'generated', 'target', 'class', 'is_ai']
        label_column = None
        for col in possible_labels:
            if col in df_clean.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError(f"Could not find label column. Available columns: {df_clean.columns.tolist()}")
    
    logger.info(f"Using '{text_column}' as text column and '{label_column}' as label column")
    
    # Clean text data
    logger.info("Preprocessing text data...")
    df_clean[text_column] = df_clean[text_column].apply(preprocess_text)
    
    # Remove empty texts
    initial_size = len(df_clean)
    df_clean = df_clean[df_clean[text_column].str.len() > 0]
    removed_empty = initial_size - len(df_clean)
    if removed_empty > 0:
        logger.info(f"Removed {removed_empty} empty text entries")
    
    # Standardize labels to binary (0 = Human, 1 = AI)
    unique_labels = df_clean[label_column].unique()
    logger.info(f"Unique labels before processing: {unique_labels}")
    
    # Convert labels to binary
    if set(unique_labels) == {0, 1}:
        # Already binary
        df_clean['label'] = df_clean[label_column]
    elif set(unique_labels) == {'human', 'ai'} or set(unique_labels) == {'Human', 'AI'}:
        # String labels
        df_clean['label'] = (df_clean[label_column].str.lower() == 'ai').astype(int)
    elif set(unique_labels) == {True, False}:
        # Boolean labels
        df_clean['label'] = df_clean[label_column].astype(int)
    else:
        # Try to map other formats
        # Assume first unique value is Human (0) and second is AI (1)
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        df_clean['label'] = df_clean[label_column].map(label_mapping)
        logger.info(f"Label mapping applied: {label_mapping}")
    
    # Final check for label distribution
    final_distribution = df_clean['label'].value_counts().sort_index()
    logger.info(f"Final label distribution - Human (0): {final_distribution.get(0, 0)}, AI (1): {final_distribution.get(1, 0)}")
    
    # Keep only text and label columns
    df_final = df_clean[['text', 'label']].copy()
    
    return df_final

def split_dataset(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets.
    
    Args:
        df (pd.DataFrame): Dataset to split
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets
    """
    logger.info(f"Splitting dataset with test_size={test_size}")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution in both sets
    )
    
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    test_df = pd.DataFrame({'text': X_test, 'label': y_test})
    
    logger.info(f"Train set size: {len(train_df)}")
    logger.info(f"Test set size: {len(test_df)}")
    logger.info(f"Train set distribution: {train_df['label'].value_counts().sort_index().to_dict()}")
    logger.info(f"Test set distribution: {test_df['label'].value_counts().sort_index().to_dict()}")
    
    return train_df, test_df

def get_sample_texts(df: pd.DataFrame, n_samples: int = 5) -> dict:
    """
    Get sample texts from each class for inspection.
    
    Args:
        df (pd.DataFrame): Dataset
        n_samples (int): Number of samples per class
        
    Returns:
        dict: Sample texts by class
    """
    samples = {}
    
    for label in [0, 1]:
        label_name = "Human" if label == 0 else "AI"
        class_data = df[df['label'] == label]
        
        if len(class_data) >= n_samples:
            sample_texts = class_data['text'].sample(n_samples, random_state=42).tolist()
        else:
            sample_texts = class_data['text'].tolist()
        
        samples[label_name] = sample_texts
    
    return samples

if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module loaded successfully!")
    print("Available functions:")
    print("- load_data(data_path)")
    print("- preprocess_text(text)")
    print("- prepare_dataset(df)")
    print("- split_dataset(df)")
    print("- get_sample_texts(df)")