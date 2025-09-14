"""
Gradio web application for AI text detection
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

import gradio as gr
from src.v2.models import EnsembleDetector
from src.v2.train import train_pipeline
import src.v2.config as config

# Global detector instance
detector = None

def initialize_app():
    """Initialize the application and models"""
    global detector
    
    print("Initializing AI Text Detector...")
    
    # Create detector instance
    detector = EnsembleDetector()
    
    # Try to load existing models
    models_loaded = detector.load_models()
    
    # If SetFit model doesn't exist, train it
    if not models_loaded:
        print("SetFit model not found. Starting training...")
        detector = train_pipeline()
        if detector is None:
            raise RuntimeError("Failed to train models. Please check your dataset.")
    
    print("App initialized successfully!")

def predict_text(text: str):
    """
    Predict whether text is AI-generated or human-written
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with prediction results for Gradio
    """
    if not text or not text.strip():
        return {
            "Result": "Please enter some text to analyze",
            "Confidence": "",
            "HF Detector Score": "",
            "SetFit Score": ""
        }
    
    try:
        # Get prediction
        result = detector.predict(text.strip())
        
        return {
            "Result": result["prediction"],
            "Confidence": f"{result['confidence']:.1%}",
            "HF Detector Score": f"{result['hf_score']:.1%}",
            "SetFit Score": f"{result['setfit_score']:.1%}"
        }
        
    except Exception as e:
        return {
            "Result": f"Error: {str(e)}",
            "Confidence": "",
            "HF Detector Score": "",
            "SetFit Score": ""
        }

def create_interface():
    """Create and return the Gradio interface"""
    
    interface = gr.Interface(
        fn=predict_text,
        inputs=gr.Textbox(
            label="Enter text to analyze",
            lines=5,
            placeholder="Paste your text here...",
            max_lines=10
        ),
        outputs=[
            gr.Textbox(label="Result", interactive=False),
            gr.Textbox(label="Confidence", interactive=False),
            gr.Textbox(label="HF Detector Score", interactive=False),
            gr.Textbox(label="SetFit Score", interactive=False)
        ],
        title="🤖 Modern AI vs Human Text Detector",
        description="""
        Detect whether text is AI-generated or human-written using:
        • Hugging Face pretrained detector
        • SetFit fine-tuned classifier
        • Ensemble scoring for improved accuracy
        """,
        examples=[[example] for example in config.EXAMPLES],
        theme=gr.themes.Soft(),
        allow_flagging="never"
    )
    
    return interface

def main():
    """Main function to run the application"""
    try:
        # Initialize models
        initialize_app()
        
        # Create and launch interface
        interface = create_interface()
        interface.launch(
            share=False,
            server_name="0.0.0.0",
            server_port=7860
        )
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Please check your configuration and dataset.")

if __name__ == "__main__":
    main()