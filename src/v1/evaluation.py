"""
Evaluation Module for AI vs Human Text Detection
Provides comprehensive metrics and visualizations for model performance
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import List, Dict, Tuple, Optional
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, save_plots: bool = True, output_dir: str = "./evaluation_results"):
        """
        Initialize the evaluator.
        
        Args:
            save_plots (bool): Whether to save plots to files
            output_dir (str): Directory to save evaluation results
        """
        self.save_plots = save_plots
        self.output_dir = output_dir
        
        if self.save_plots:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def evaluate_predictions(self, y_true: List[int], y_pred: List[int], y_prob: Optional[List[float]] = None) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            y_prob (Optional[List[float]]): Prediction probabilities for positive class
            
        Returns:
            Dict: Comprehensive evaluation metrics
        """
        logger.info("Computing evaluation metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_human': precision_per_class[0],
            'precision_ai': precision_per_class[1],
            'recall_human': recall_per_class[0],
            'recall_ai': recall_per_class[1],
            'f1_human': f1_per_class[0],
            'f1_ai': f1_per_class[1],
            'confusion_matrix': cm.tolist(),
            'support_human': int(np.sum(cm[0])),
            'support_ai': int(np.sum(cm[1]))
        }
        
        # Add AUC if probabilities are provided
        if y_prob is not None:
            auc = roc_auc_score(y_true, y_prob)
            metrics['auc_roc'] = auc
        
        return metrics
    
    def print_evaluation_report(self, metrics: Dict):
        """
        Print a formatted evaluation report.
        
        Args:
            metrics (Dict): Evaluation metrics dictionary
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION REPORT")
        print("="*60)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision_weighted']:.4f}")
        print(f"  Recall:       {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score:     {metrics['f1_weighted']:.4f}")
        
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
        
        print(f"\nPER-CLASS METRICS:")
        print(f"  Human Text (Class 0):")
        print(f"    Precision:  {metrics['precision_human']:.4f}")
        print(f"    Recall:     {metrics['recall_human']:.4f}")
        print(f"    F1-Score:   {metrics['f1_human']:.4f}")
        print(f"    Support:    {metrics['support_human']}")
        
        print(f"  AI Text (Class 1):")
        print(f"    Precision:  {metrics['precision_ai']:.4f}")
        print(f"    Recall:     {metrics['recall_ai']:.4f}")
        print(f"    F1-Score:   {metrics['f1_ai']:.4f}")
        print(f"    Support:    {metrics['support_ai']}")
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], normalize: bool = False) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            normalize (bool): Whether to normalize the confusion matrix
            
        Returns:
            plt.Figure: Confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=['Human', 'AI'], 
                   yticklabels=['Human', 'AI'],
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if self.save_plots:
            filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {os.path.join(self.output_dir, filename)}")
        
        return fig
    
    def plot_roc_curve(self, y_true: List[int], y_prob: List[float]) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true (List[int]): True labels
            y_prob (List[float]): Prediction probabilities for positive class
            
        Returns:
            plt.Figure: ROC curve plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {os.path.join(self.output_dir, 'roc_curve.png')}")
        
        return fig
    
    def plot_class_distribution(self, y_true: List[int], y_pred: List[int]) -> plt.Figure:
        """
        Plot class distribution comparison.
        
        Args:
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            
        Returns:
            plt.Figure: Class distribution plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # True distribution
        true_counts = np.bincount(y_true)
        ax1.bar(['Human', 'AI'], true_counts, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax1.set_title('True Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(true_counts):
            ax1.text(i, v + 0.01 * max(true_counts), str(v), 
                    ha='center', va='bottom', fontweight='bold')
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred)
        ax2.bar(['Human', 'AI'], pred_counts, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax2.set_title('Predicted Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for i, v in enumerate(pred_counts):
            ax2.text(i, v + 0.01 * max(pred_counts), str(v), 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {os.path.join(self.output_dir, 'class_distribution.png')}")
        
        return fig
    
    def plot_prediction_confidence(self, y_true: List[int], y_prob: List[float]) -> plt.Figure:
        """
        Plot prediction confidence distribution.
        
        Args:
            y_true (List[int]): True labels
            y_prob (List[float]): Prediction probabilities for positive class
            
        Returns:
            plt.Figure: Confidence distribution plot
        """
        # Convert probabilities to confidence scores
        confidence_scores = np.maximum(y_prob, 1 - np.array(y_prob))
        
        # Separate by true class
        human_confidence = confidence_scores[np.array(y_true) == 0]
        ai_confidence = confidence_scores[np.array(y_true) == 1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histograms
        ax.hist(human_confidence, bins=30, alpha=0.7, label='Human Text', color='skyblue', density=True)
        ax.hist(ai_confidence, bins=30, alpha=0.7, label='AI Text', color='lightcoral', density=True)
        
        ax.set_xlabel('Prediction Confidence', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Prediction Confidence Distribution by True Class', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_human_conf = np.mean(human_confidence)
        mean_ai_conf = np.mean(ai_confidence)
        
        ax.axvline(mean_human_conf, color='blue', linestyle='--', alpha=0.8, 
                  label=f'Mean Human Confidence: {mean_human_conf:.3f}')
        ax.axvline(mean_ai_conf, color='red', linestyle='--', alpha=0.8, 
                  label=f'Mean AI Confidence: {mean_ai_conf:.3f}')
        
        ax.legend()
        plt.tight_layout()
        
        if self.save_plots:
            plt.savefig(os.path.join(self.output_dir, 'confidence_distribution.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Confidence distribution plot saved to {os.path.join(self.output_dir, 'confidence_distribution.png')}")
        
        return fig
    
    def save_detailed_report(self, metrics: Dict, y_true: List[int], y_pred: List[int], 
                           y_prob: Optional[List[float]] = None, model_name: str = "Model"):
        """
        Save a detailed evaluation report to file.
        
        Args:
            metrics (Dict): Evaluation metrics
            y_true (List[int]): True labels
            y_pred (List[int]): Predicted labels
            y_prob (Optional[List[float]]): Prediction probabilities
            model_name (str): Name of the model
        """
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"DETAILED EVALUATION REPORT - {model_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  Accuracy:     {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision:    {metrics['precision_weighted']:.4f}\n")
            f.write(f"  Recall:       {metrics['recall_weighted']:.4f}\n")
            f.write(f"  F1-Score:     {metrics['f1_weighted']:.4f}\n")
            
            if 'auc_roc' in metrics:
                f.write(f"  AUC-ROC:      {metrics['auc_roc']:.4f}\n")
            
            f.write(f"\nPER-CLASS PERFORMANCE:\n")
            f.write(f"  Human Text (Class 0):\n")
            f.write(f"    Precision:  {metrics['precision_human']:.4f}\n")
            f.write(f"    Recall:     {metrics['recall_human']:.4f}\n")
            f.write(f"    F1-Score:   {metrics['f1_human']:.4f}\n")
            f.write(f"    Support:    {metrics['support_human']}\n\n")
            
            f.write(f"  AI Text (Class 1):\n")
            f.write(f"    Precision:  {metrics['precision_ai']:.4f}\n")
            f.write(f"    Recall:     {metrics['recall_ai']:.4f}\n")
            f.write(f"    F1-Score:   {metrics['f1_ai']:.4f}\n")
            f.write(f"    Support:    {metrics['support_ai']}\n\n")
            
            # Confusion matrix
            f.write("CONFUSION MATRIX:\n")
            cm = np.array(metrics['confusion_matrix'])
            f.write(f"                Predicted\n")
            f.write(f"                Human    AI\n")
            f.write(f"Actual Human    {cm[0,0]:5d}  {cm[0,1]:5d}\n")
            f.write(f"       AI       {cm[1,0]:5d}  {cm[1,1]:5d}\n\n")
            
            # Classification report
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write(classification_report(y_true, y_pred, target_names=['Human', 'AI']))
        
        logger.info(f"Detailed evaluation report saved to {report_path}")

def compare_models(evaluators_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Args:
        evaluators_results (Dict[str, Dict]): Dictionary of model names and their metrics
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, metrics in evaluators_results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'F1-Score': metrics['f1_weighted'],
            'Precision': metrics['precision_weighted'],
            'Recall': metrics['recall_weighted'],
            'AUC-ROC': metrics.get('auc_roc', 'N/A')
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    return df

if __name__ == "__main__":
    # Example usage
    print("Model evaluation module loaded successfully!")
    print("\nAvailable functions:")
    print("- ModelEvaluator.evaluate_predictions()")
    print("- ModelEvaluator.print_evaluation_report()")
    print("- ModelEvaluator.plot_confusion_matrix()")
    print("- ModelEvaluator.plot_roc_curve()")
    print("- ModelEvaluator.plot_class_distribution()")
    print("- compare_models()")