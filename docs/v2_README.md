# Modern AI vs Human Text Detector

A deploy-ready Gradio web app for Hugging Face Spaces that detects whether text is AI-generated or human-written using:
- A Hugging Face pretrained detector (e.g., roberta-base-openai-detector)
- Fast SetFit fine-tuning on your dataset
- Ensemble scoring for improved accuracy

## Features
- Modular, well-structured codebase for easy extension
- Loads and prepares your Kaggle AI vs Human Text dataset
- Trains a SetFit classifier in minutes (no heavy GPU required)
- Combines Hugging Face detector and SetFit for robust predictions
- Model caching to avoid retraining on every restart
- Error handling and proper initialization
- Simple Gradio interface: text box, "Check" button, confidence scores
- Ready for Hugging Face Spaces deployment

## File Structure
```
📦 ai_text_detection/
└── 📦 src/v2/                    # Version 2 implementation (legacy)
    ├── app.py              # Main Gradio web application
    ├── config.py           # Configuration settings
    ├── data_utils.py       # Data loading and preprocessing
    ├── models.py           # Model management and ensemble logic
    ├── train.py            # Training pipeline
    ├── requirements.txt    # Dependencies
    └── README.md          # This guide
```

## Usage

### 1. Setup
Place your dataset as `data/AI_Human.csv` (or adjust `DATA_PATH` in `config.py`).

Install dependencies:
```bash
# From project root
pip install -r src/v2/requirements.txt
```

### 2. Run the App
```bash
# Navigate to v2 directory
cd src/v2

# Run the app
python app.py
```

The app will:
- Automatically load existing trained models if available
- Train SetFit model if not found (first run only)
- Launch Gradio interface on http://localhost:7860

### 3. Training Only
To just train models without running the web interface:
```bash
# From v2 directory
python train.py
```

## Customization

### Model Configuration (`config.py`)
- **PRETRAINED_DETECTOR**: Change Hugging Face detector model
- **SETFIT_MODEL**: Change SetFit base model
- **THRESHOLD**: Adjust decision threshold (0.5 = balanced)
- **ENSEMBLE_METHOD**: Modify ensemble strategy

### Training Parameters (`config.py`)
- **SETFIT_BATCH_SIZE**: Batch size for training
- **SETFIT_ITERATIONS**: Number of training iterations
- **SETFIT_EPOCHS**: Number of epochs

### Extending the Pipeline
- **Add new models**: Extend classes in `models.py`
- **New ensemble methods**: Modify `EnsembleDetector.predict()`
- **Custom preprocessing**: Update `data_utils.py`
- **Stylometric features**: Add to model classes

## Deployment

### Hugging Face Spaces
1. Create a new Space on Hugging Face
2. Upload all files from `src/v2/` directory
3. Set the app file to `app.py`
4. The Space will automatically install dependencies and run

### Local Development
The modular structure makes it easy to:
- Test individual components
- Add new models or features
- Debug specific parts of the pipeline
- Scale to larger datasets

## Extending Further
- **Ghostbuster integration**: Add to `models.py` as new detector class
- **DetectGPT features**: Extend ensemble logic
- **Custom features**: Add stylometric analysis in preprocessing
- **Better UI**: Enhance Gradio interface with plots, explanations

---
Created by GitHub Copilot for modern AI-text detection.
