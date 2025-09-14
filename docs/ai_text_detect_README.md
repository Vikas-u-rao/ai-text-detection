# AI Text Detection Web App

This web application uses the Hugging Face model "VSAsteroid/ai-text-detector-hc3" to detect whether text was written by AI or humans.

## Features

- **Single Text Analysis**: Analyze individual pieces of text with confidence scores and visual indicators
- **Batch Processing**: Analyze multiple text segments separated by empty lines
- **Clean Gradio Interface**: User-friendly web interface with examples and intuitive design
- **Confidence Visualization**: Color-coded confidence bars showing prediction certainty
- **Hugging Face Spaces Ready**: Configured for easy deployment

## Installation

1. Navigate to the main project directory
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Local Development
Run the application locally:
```bash
# From project root
python app.py
```

The app will be available at `http://localhost:7860`

### Legacy HF Implementation
If you want to run the original implementation directly:
```bash
# Navigate to the legacy directory
cd src/ai_text_detect

# Run the legacy app
python app.py
```

### Deployment on Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces)
2. Choose "Gradio" as the SDK
3. Upload `app.py` and `requirements.txt` from the project root
4. Your app will automatically deploy and be publicly accessible

## Code Structure

```
📦 ai_text_detection/
├── app.py                    # Main HF model application
├── requirements.txt          # Main application dependencies
└── 📦 src/ai_text_detect/    # Original HF implementation (legacy)
    ├── app.py               # Legacy detector app
    ├── test_model.py        # Model testing script
    ├── sample_human_text.txt # Sample human text
    ├── sample_markdown.md   # Sample markdown
    ├── sample_mixed_text.txt # Sample mixed text
    └── requirements.txt     # Legacy dependencies
```

- `AITextDetector` class: Main detector class with model loading and inference methods
- `load_model()`: Loads the VSAsteroid/ai-text-detector-hc3 model
- `detect_text()`: Performs inference on single text inputs
- `detect_batch()`: Handles multiple text inputs
- `create_interface()`: Builds the Gradio web interface

## Model Information

- **Model**: VSAsteroid/ai-text-detector-hc3
- **Task**: Text Classification
- **Labels**: AI-Generated vs Human-Written
- **Source**: Hugging Face Model Hub

## Example Usage

The app includes several example texts you can try:
- AI-generated technical content
- Human-written personal narratives
- Common test phrases

## Requirements

- Python 3.7+
- transformers >= 4.21.0
- torch >= 1.13.0
- gradio >= 3.50.0

## Notes

- First run may take longer as the model downloads (approximately 500MB)
- GPU acceleration is automatically used if available
- Results should be used as guidance, not definitive proof
- The model's accuracy depends on the type and length of text provided

## Troubleshooting

- If the model fails to load, ensure you have a stable internet connection
- For CUDA issues, make sure you have compatible PyTorch and CUDA versions
- If Gradio interface doesn't load, try refreshing the browser or checking firewall settings