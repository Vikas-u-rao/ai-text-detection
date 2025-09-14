"""
Main Training Script for AI vs Human Text Detection
Orchestrates the complete training pipeline from data loading to model evaluation
"""
import os
import sys
import argparse
import logging
import time
from datetime import datetime
import json
import pandas as pd
import torch

# Import our custom modules
from data_preprocessing import load_data, prepare_dataset, split_dataset, get_sample_texts
from model_training import AIHumanDetector, get_recommended_models
from evaluation import ModelEvaluator, compare_models
from prediction import TextPredictor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Complete training pipeline for AI vs Human text detection"""
    
    def __init__(self, config: dict):
        """
        Initialize the training pipeline.
        
        Args:
            config (dict): Configuration dictionary with training parameters
        """
        self.config = config
        self.start_time = None
        self.results = {}
        
        # Create output directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['model_output_dir'], exist_ok=True)
        os.makedirs(self.config['evaluation_dir'], exist_ok=True)
        
        logger.info(f"Initialized training pipeline with config: {config}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        logger.info("="*60)
        logger.info("STEP 1: LOADING AND PREPROCESSING DATA")
        logger.info("="*60)
        
        # Load dataset
        df = load_data(self.config['data_path'])
        
        # Prepare dataset
        df_clean = prepare_dataset(
            df, 
            text_column=self.config.get('text_column', 'text'),
            label_column=self.config.get('label_column', None)
        )
        
        # Show sample texts
        samples = get_sample_texts(df_clean, n_samples=3)
        logger.info("Sample texts:")
        for label, texts in samples.items():
            logger.info(f"\n{label} examples:")
            for i, text in enumerate(texts[:2]):
                logger.info(f"  {i+1}. {text[:200]}...")
        
        # Split dataset
        train_df, test_df = split_dataset(
            df_clean, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Save processed datasets
        train_path = os.path.join(self.config['output_dir'], 'train_data.csv')
        test_path = os.path.join(self.config['output_dir'], 'test_data.csv')
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Processed datasets saved:")
        logger.info(f"  Train: {train_path}")
        logger.info(f"  Test: {test_path}")
        
        self.results['data_info'] = {
            'total_samples': len(df_clean),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_distribution': train_df['label'].value_counts().to_dict(),
            'test_distribution': test_df['label'].value_counts().to_dict()
        }
        
        return train_df, test_df
    
    def train_model(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train the AI vs Human detection model"""
        logger.info("="*60)
        logger.info("STEP 2: TRAINING MODEL")
        logger.info("="*60)
        
        # Initialize detector
        detector = AIHumanDetector(
            model_name=self.config['model_name'],
            max_length=self.config['max_length']
        )
        
        # Load model and tokenizer
        detector.load_model_and_tokenizer()
        
        # Prepare datasets
        train_dataset, test_dataset = detector.prepare_datasets(train_df, test_df)
        
        # Train model
        training_start = time.time()
        
        training_results = detector.train_model(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            output_dir=self.config['model_output_dir'],
            num_epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            save_steps=self.config['save_steps'],
            eval_steps=self.config['eval_steps'],
            early_stopping_patience=self.config['early_stopping_patience']
        )
        
        training_time = time.time() - training_start
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Model saved to {self.config['model_output_dir']}")
        
        self.results['training'] = {
            'training_time': training_time,
            'final_loss': training_results['training_loss'],
            'model_path': training_results['model_path']
        }
        
        return detector
    
    def evaluate_model(self, test_df: pd.DataFrame):
        """Evaluate the trained model"""
        logger.info("="*60)
        logger.info("STEP 3: EVALUATING MODEL")
        logger.info("="*60)
        
        # Load the trained model for evaluation
        predictor = TextPredictor(self.config['model_output_dir'])
        
        # Evaluate on test set
        evaluation_results = predictor.evaluate_on_test_set(test_df)
        
        # Create evaluator
        evaluator = ModelEvaluator(
            save_plots=True,
            output_dir=self.config['evaluation_dir']
        )
        
        # Get detailed metrics
        y_true = evaluation_results['true_labels']
        y_pred = evaluation_results['predictions']
        y_prob = evaluation_results['probabilities']
        
        metrics = evaluator.evaluate_predictions(y_true, y_pred, y_prob)
        
        # Print evaluation report
        evaluator.print_evaluation_report(metrics)
        
        # Generate plots
        logger.info("Generating evaluation plots...")
        evaluator.plot_confusion_matrix(y_true, y_pred)
        evaluator.plot_confusion_matrix(y_true, y_pred, normalize=True)
        evaluator.plot_roc_curve(y_true, y_prob)
        evaluator.plot_class_distribution(y_true, y_pred)
        evaluator.plot_prediction_confidence(y_true, y_prob)
        
        # Save detailed report
        evaluator.save_detailed_report(
            metrics, y_true, y_pred, y_prob, 
            model_name=self.config['model_name']
        )
        
        self.results['evaluation'] = metrics
        
        return metrics, evaluation_results
    
    def save_final_results(self):
        """Save final training results and configuration"""
        logger.info("="*60)
        logger.info("STEP 4: SAVING FINAL RESULTS")
        logger.info("="*60)
        
        # Calculate total training time
        total_time = time.time() - self.start_time
        
        # Compile final results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'total_training_time': total_time,
            'results': self.results,
            'model_path': self.config['model_output_dir'],
            'evaluation_path': self.config['evaluation_dir']
        }
        
        # Save results
        results_path = os.path.join(self.config['output_dir'], 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_path}")
        
        # Print summary
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Model: {self.config['model_name']}")
        logger.info(f"Test accuracy: {self.results.get('evaluation', {}).get('accuracy', 'N/A'):.4f}")
        logger.info(f"Test F1-score: {self.results.get('evaluation', {}).get('f1_weighted', 'N/A'):.4f}")
        logger.info(f"Model saved: {self.config['model_output_dir']}")
        logger.info(f"Evaluation plots: {self.config['evaluation_dir']}")
        logger.info("="*60)
        
        return final_results
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        self.start_time = time.time()
        
        try:
            logger.info("Starting complete AI vs Human text detection training pipeline...")
            
            # Step 1: Load and preprocess data
            train_df, test_df = self.load_and_preprocess_data()
            
            # Step 2: Train model
            detector = self.train_model(train_df, test_df)
            
            # Step 3: Evaluate model
            metrics, eval_results = self.evaluate_model(test_df)
            
            # Step 4: Save final results
            final_results = self.save_final_results()
            
            logger.info("Training pipeline completed successfully! 🎉")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

def get_default_config():
    """Get default configuration for training"""
    return {
        # Data settings
        'data_path': 'data/AI_Human.csv',
        'text_column': 'text',
        'label_column': None,  # Auto-detect
        'test_size': 0.2,
        'random_state': 42,
        
        # Model settings
        'model_name': 'distilbert-base-uncased',  # Options: distilbert-base-uncased, roberta-base, microsoft/deberta-v3-base
        'max_length': 512,
        
        # Training settings
        'num_epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01,
        'save_steps': 500,
        'eval_steps': 500,
        'early_stopping_patience': 3,
        
        # Output settings
        'output_dir': './output',
        'model_output_dir': './model_output',
        'evaluation_dir': './evaluation_results'
    }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Train AI vs Human Text Detector')
    
    parser.add_argument('--data_path', type=str, default='data/AI_Human.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                       choices=['distilbert-base-uncased', 'roberta-base', 'microsoft/deberta-v3-base'],
                       help='Model to use for training')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create configuration
    config = get_default_config()
    config.update({
        'data_path': args.data_path,
        'model_name': args.model_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'output_dir': args.output_dir,
        'model_output_dir': os.path.join(args.output_dir, 'model'),
        'evaluation_dir': os.path.join(args.output_dir, 'evaluation')
    })
    
    # Display recommended models
    print("Available models:")
    models = get_recommended_models()
    for name, info in models.items():
        print(f"  - {name}: {info['description']}")
    print()
    
    # Run training pipeline
    pipeline = TrainingPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    print(f"\nTraining completed! Results saved to {config['output_dir']}")
    print(f"To test the model, run: python prediction.py --model_path {config['model_output_dir']}")
    print(f"To launch web interface, run: python gradio_app.py")

if __name__ == "__main__":
    main()