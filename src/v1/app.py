"""
Deployment configuration for the AI vs Human Text Detector
This file can be renamed to app.py for Hugging Face Spaces deployment
"""

# For Hugging Face Spaces deployment, use the gradio interface
from gradio_app import launch_app
import os

if __name__ == "__main__":
    # Check if running on Hugging Face Spaces
    if os.getenv("SPACE_ID"):
        print("Running on Hugging Face Spaces")
        # Use pre-trained model or smaller model for demo
        model_path = "./model_output"  # Ensure model is included in the Space
    else:
        print("Running locally")
        model_path = "./model_output"
    
    # Launch the application
    launch_app(
        model_path=model_path,
        share=False,  # HF Spaces handles sharing
        port=7860
    )