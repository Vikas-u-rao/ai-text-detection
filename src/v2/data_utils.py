"""
Data loading and preprocessing utilities
"""

import pandas as pd
import os
from typing import Tuple, List

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load and preprocess the dataset
    
    Args:
        path: Path to the CSV dataset
        
    Returns:
        Preprocessed DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    
    # Auto-detect text and label columns
    text_col = next(
        (c for c in df.columns if c.lower() in ["text", "content", "message", "article"]), 
        df.columns[0]
    )
    label_col = next(
        (c for c in df.columns if c.lower() in ["label", "generated", "target", "class", "ai_generated"]), 
        df.columns[1] if len(df.columns) > 1 else None
    )
    
    if label_col is None:
        raise ValueError("Could not find label column in dataset")
    
    # Rename columns for consistency
    df = df.rename(columns={text_col: "text", label_col: "label"})
    
    # Clean data
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    
    # Normalize labels to 0 (human) and 1 (AI)
    unique_labels = set(df["label"].unique())
    
    if unique_labels == {0, 1} or unique_labels == {0.0, 1.0}:
        df["label"] = df["label"].astype(int)
    elif unique_labels == {"human", "ai"} or unique_labels == {"Human", "AI"}:
        df["label"] = (df["label"].str.lower() == "ai").astype(int)
    elif len(unique_labels) == 2:
        # Map first unique label to 0 (human), second to 1 (AI)
        labels = sorted(list(unique_labels))
        df["label"] = df["label"].map({labels[0]: 0, labels[1]: 1})
    else:
        raise ValueError(f"Invalid label format. Found: {unique_labels}")
    
    print(f"Loaded {len(df)} samples")
    print(f"Label distribution: Human={sum(df['label']==0)}, AI={sum(df['label']==1)}")
    
    return df

def prepare_training_data(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Convert DataFrame to lists for training
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        
    Returns:
        Tuple of (texts, labels)
    """
    return df["text"].tolist(), df["label"].tolist()