# AI vs Human Text Detector - Setup Instructions

## 📋 Complete Setup Guide

**Note**: This guide is specifically for **Version 1 (Transformer-based)** implementation located in `src/v1/`. For the main Hugging Face model, see the main `README.md`.

### 1. Environment Setup

```bash
# Navigate to project root
cd ai_text_detection

# Create virtual environment
python -m venv ai_detector_env

# Activate environment
# Windows:
ai_detector_env\Scripts\activate
# Mac/Linux:
source ai_detector_env/bin/activate

# Install v1 dependencies
pip install -r src/v1/requirements.txt
```

### 2. Dataset Preparation

1. **Download from Kaggle**: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
2. **Extract** the `AI_Human.csv` file
3. **Place** in `data/AI_Human.csv`

```bash
mkdir data
# Copy AI_Human.csv to data/AI_Human.csv
```

### 3. Training the Model

#### Quick Start (Recommended)
```bash
# Navigate to v1 directory
cd src/v1

# Train with DistilBERT (fast, good performance)
python train.py --model_name distilbert-base-uncased --num_epochs 3
```

#### Advanced Training Options
```bash
# High performance training
python train.py --model_name roberta-base --num_epochs 5 --batch_size 8

# State-of-the-art training (slower)
python train.py --model_name microsoft/deberta-v3-base --num_epochs 4

# Custom settings
python train.py \
  --model_name distilbert-base-uncased \
  --num_epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --output_dir ./my_output
```

### 4. Running the Web Interface

#### Gradio Interface (Recommended)
```bash
# From v1 directory
cd src/v1
python gradio_app.py
```
- Opens at `http://localhost:7860`
- Creates public sharing link
- Best for demos and testing

#### Streamlit Interface
```bash
# From v1 directory
cd src/v1
streamlit run streamlit_app.py
```
- Opens at `http://localhost:8501`
- More advanced features
- Better for analysis

### 5. Testing Predictions

#### Command Line Testing
```python
# Navigate to v1 directory first
import sys
sys.path.append('../src/v1')

# Quick test
from prediction import predict_text_simple

text = "Your text here..."
result = predict_text_simple('./model_output', text)
print(result)
```

#### Programmatic Testing
```python
# Navigate to v1 directory first
import sys
sys.path.append('../src/v1')

from prediction import TextPredictor

predictor = TextPredictor('./model_output')
result = predictor.predict_single_text("Test text here")
print(f"Prediction: {result['transformer_prediction']}")
print(f"Confidence: {result['transformer_confidence']:.1%}")
```

## 🚀 Deployment Options

### Option 1: Hugging Face Spaces (Recommended)

1. **Create new Space** at https://huggingface.co/new-space
2. **Select Gradio SDK**
3. **Upload files from `src/v1/`**:
   ```
   src/v1/app.py (gradio_app.py renamed)
   src/v1/requirements.txt
   src/v1/data_preprocessing.py
   src/v1/model_training.py
   src/v1/evaluation.py
   src/v1/prediction.py
   src/v1/model_output/ (your trained model folder)
   ```
4. **Space will auto-deploy**

### Option 2: Local Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "gradio_app.py"]
```

```bash
# Build and run
docker build -t ai-detector .
docker run -p 7860:7860 ai-detector
```

### Option 3: Cloud Deployment

#### Google Colab
```python
# In Colab notebook
!git clone your-repo-url
%cd ai_text_detection
!pip install -r src/v1/requirements.txt

# Train model
%cd src/v1
!python train.py

# Launch interface
!python gradio_app.py
```

#### AWS/Azure/GCP
- Use container services (ECS, Container Instances, Cloud Run)
- Include model files in container or load from storage

## 📊 Performance Tuning

### Memory Optimization
```python
# Reduce batch size for limited memory
python train.py --batch_size 8

# Use DistilBERT for faster training
python train.py --model_name distilbert-base-uncased
```

### Speed Optimization
```python
# Disable GPT-2 baseline for faster inference
python train.py --no_gpt2_baseline

# Reduce max sequence length
# Edit train.py: 'max_length': 256
```

### Accuracy Optimization
```python
# Use larger model
python train.py --model_name roberta-base --num_epochs 5

# Increase training epochs
python train.py --num_epochs 10 --early_stopping_patience 5
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train.py --batch_size 4
   ```

2. **Dataset Not Found**
   ```bash
   # Ensure correct path
   python train.py --data_path data/AI_Human.csv
   ```

3. **Model Loading Error**
   ```python
   # Check model path exists
   import os
   print(os.path.exists('./model_output'))
   ```

4. **Import Errors**
   ```bash
   # Ensure all dependencies installed
   pip install -r requirements.txt
   ```

### GPU Setup (Optional)

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📝 Project Structure

```
ai-vs-human-detector/
├── 📊 Data Processing
│   └── data_preprocessing.py
├── 🤖 Model Components
│   ├── model_training.py
│   ├── evaluation.py
│   └── prediction.py
├── 🌐 Web Interfaces
│   ├── gradio_app.py
│   └── streamlit_app.py
├── 🎯 Main Scripts
│   ├── train.py
│   └── app.py (for deployment)
├── 📁 Output Directories
│   ├── model_output/
│   ├── evaluation_results/
│   └── output/
└── 📋 Configuration
    ├── requirements.txt
    ├── README.md
    └── setup_guide.md
```

## 🎯 Next Steps

1. **Train your first model** with `python train.py`
2. **Test the web interface** with `python gradio_app.py`
3. **Customize the model** by editing training parameters
4. **Deploy to Hugging Face Spaces** for public access
5. **Integrate into your application** using the prediction API

## 💡 Tips for Best Results

- **Text Length**: Use texts >100 words for better accuracy
- **Model Selection**: DistilBERT for speed, RoBERTa for accuracy
- **Training Data**: More diverse training data improves robustness
- **Confidence Scores**: Pay attention to confidence levels
- **Multiple Models**: Try different models and compare results

Happy detecting! 🤖✨