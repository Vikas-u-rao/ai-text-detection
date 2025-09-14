# 🤖 AI vs Human Text Detector

A modern AI text detection system using pre-trained Hugging Face models to distinguish between AI-generated and human-written text.

## 🌟 Features

- **Fast Detection**: Uses pre-trained Hugging Face model for real-time AI text detection
- **Simple Interface**: Clean Gradio web interface with examples and confidence scores
- **No Training Required**: Works out-of-the-box with pre-trained model
- **Batch Processing**: Analyze multiple texts at once
- **Legacy Support**: Includes older implementations for comparison

## 📋 Requirements

- Python 3.8+
- See `requirements.txt` for main application dependencies
- Legacy versions (v1/v2) have their own `src/v1/requirements.txt` and `src/v2/requirements.txt`

## 🚀 Quick Start

### 0. Clone the Repository

```bash
git clone https://github.com/Vikas-u-rao/ai-text-detection
cd ai_text_detection
```

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate
```

### 2. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Run the main app
python app.py
```

The app will be available at `http://localhost:7860`

### Other Versions

- **Version 1 (Legacy)**: Transformer-based models (DistilBERT, RoBERTa) with comprehensive evaluation - see `docs/v1_README.md` and `src/v1/requirements.txt`
- **Version 2 (Legacy)**: SetFit fine-tuning with ensemble scoring - see `docs/v2_README.md` and `src/v2/requirements.txt`

## 📁 Project Structure

```
├── app.py                 # Main HF model application
├── requirements.txt       # Main application dependencies
├── src/                   # Source code for all versions
│   ├── ai_detector.py     # Legacy sentence transformer detector
│   ├── v1/               # Version 1 implementation (legacy)
│   │   └── requirements.txt  # V1 dependencies
│   ├── v2/               # Version 2 implementation (legacy)
│   │   └── requirements.txt  # V2 dependencies
│   └── ai_text_detect/   # Original HF implementation
├── apps/                  # Legacy web applications
├── docs/                  # Documentation and guides
│   ├── v1_README.md      # V1 documentation
│   ├── v2_README.md      # V2 documentation
│   └── ai_text_detect_README.md # Original HF docs
├── data/                  # Datasets and samples
├── tests/                 # Test files
└── docs/                  # Documentation and guides
```

## 💻 Usage Examples

### Basic Usage

The main application provides a web interface. Simply run:

```bash
python app.py
```

Then open your browser to `http://localhost:7860` to use the detection interface.

### Programmatic Usage

```python
import gradio as gr
from transformers import pipeline
import torch

# Load the AI detection model
classifier = pipeline(
    "text-classification",
    model="VSAsteroid/ai-text-detector-hc3",
    return_all_scores=True,
    device=0 if torch.cuda.is_available() else -1
)

# Detect AI-generated text
text = "Your text here..."
results = classifier(text)

# Process results
for result in results[0]:
    label = result['label']
    score = result['score']
    print(f"{label}: {score:.3f}")
```

## 🔧 How It Works

The application uses a pre-trained Hugging Face model (`VSAsteroid/ai-text-detector-hc3`) that has been fine-tuned on datasets distinguishing between AI-generated and human-written text. The model analyzes text and provides confidence scores for both categories.

## ⚠️ Limitations

- **Language**: Optimized for English text
- **Model Updates**: Detection effectiveness may change as AI models evolve
- **Context**: Performance may vary across different writing styles and topics
- **Length**: Works best with substantial text samples

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face Model**: `VSAsteroid/ai-text-detector-hc3` for AI text detection
- **Transformers Library**: Hugging Face transformers for model implementation
- **Gradio**: For the web interface

---

**Happy Detecting! 🚀**

