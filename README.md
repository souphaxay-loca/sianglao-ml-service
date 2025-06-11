# SiangLao ML Service 🎤

A Flask-based REST API service for Lao language Automatic Speech Recognition (ASR) using state-of-the-art transformer models.

## 📖 Overview

SiangLao ML Service provides ASR inference for the Lao language using three pre-trained models:
- **XLS-R**: Cross-lingual Speech Representation model
- **XLSR-53**: Cross-lingual Speech Representation (53 languages)
- **HuBERT**: Hidden-Unit BERT for speech recognition

The service accepts WAV audio files and returns transcriptions with confidence scores.

## ✨ Features

- 🎯 **Multi-model inference**: Run predictions on individual models or all models simultaneously
- 🌐 **RESTful API**: Clean HTTP endpoints with JSON responses
- 📊 **Health monitoring**: Built-in health checks and status endpoints
- ⚡ **GPU/CPU support**: Automatic device detection with CUDA support
- 🔧 **Configurable**: Comprehensive configuration system
- 📝 **Detailed logging**: Extensive logging with emoji indicators
- 🚀 **Production ready**: Error handling, CORS support, and proper HTTP status codes

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd sianglao-ml-service
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download models**
```bash
python download_models.py
```

## 🚀 Usage

### Start the service

#### Normal Mode (16GB+ RAM recommended)
```bash
python app.py
```

#### Low Memory Mode (8GB RAM)
For machines with limited RAM, use model offloading mode:
```bash
MEMORY_MODE=low_memory python app.py
```

In low memory mode:
- Models are loaded only when needed
- Each model is unloaded after prediction
- Memory usage stays around 1.2GB per request
- Slower inference (5-10s loading time per model)

The service will start on `http://localhost:8000` by default.

### API Endpoints

#### 🏠 Root Endpoint
```http
GET /
```
Returns basic service information and available endpoints.

#### 💊 Health Check
```http
GET /health
```
Returns service health status and model loading information.

#### 📋 Models Information
```http
GET /models
```
Returns detailed information about all available models.

#### 🎯 Single Model Prediction
```http
POST /predict/<model_name>
Content-Type: multipart/form-data

Form data:
- audio: WAV file
```

Available model names: `xls-r`, `xlsr-53`, `hubert`

#### 🎯 All Models Prediction
```http
POST /predict
Content-Type: multipart/form-data

Form data:
- audio: WAV file
```

### Example Usage

#### Using curl
```bash
# Predict with all models
curl -X POST \
  http://localhost:5000/predict \
  -F "audio=@your_audio_file.wav"

# Predict with specific model
curl -X POST \
  http://localhost:5000/predict/xls-r \
  -F "audio=@your_audio_file.wav"

# Health check
curl http://localhost:5000/health
```

#### Using Python requests
```python
import requests

# Upload audio file
with open('audio.wav', 'rb') as f:
    files = {'audio': f}
    response = requests.post('http://localhost:5000/predict', files=files)
    
result = response.json()
print(result)
```

### Response Format

#### Success Response
```json
{
  "success": true,
  "timestamp": "2024-01-01T12:00:00",
  "service": "Sianglao ML Service",
  "results": {
    "xls-r": {
      "model": "xls-r",
      "prediction": "ສະບາຍດີ",
      "confidence": 0.95,
      "processing_time": 1.23,
      "success": true
    }
  },
  "processing_time": 1.25,
  "audio_duration": 2.5
}
```

#### Error Response
```json
{
  "success": false,
  "error": "Error message",
  "timestamp": "2024-01-01T12:00:00",
  "service": "Sianglao ML Service"
}
```

## ⚙️ Configuration

The service can be configured through `config.py`. Key configuration options:

- **Audio settings**: Sample rate, mono/stereo processing
- **Performance**: Device selection (CPU/GPU), batch processing
- **API behavior**: CORS, error handling, response formatting
- **Model paths**: Locations of downloaded models

## 👥 Authors

- **Souphaxay Naovalath**
- **Sounmy Chanthavong**

## 🙏 Acknowledgments

- Hugging Face Transformers library
- Facebook Research for Wav2Vec2 and HuBERT models
- The Lao language NLP community

---

**Version**: 1.0.0  
**Status**: Demo