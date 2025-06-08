# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Flask-based REST API service for Lao language Automatic Speech Recognition (ASR). It provides inference for Lao audio transcription using three pre-trained transformer models: XLS-R, XLSR-53, and HuBERT.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download pre-trained models (required before first run)
python download_models.py

# Start the service
python app.py
```

### Service Operations
- **Start server**: `python app.py` (runs on http://localhost:8000)
- **Download models**: `python download_models.py` (one-time setup)

## Architecture

### Core Components

- **app.py**: Main Flask application with API endpoints
- **models.py**: ModelManager class handling all model loading and inference
- **config.py**: Configuration system with environment-specific settings
- **download_models.py**: One-time model download script

### Model Architecture

Three Lao ASR models are supported:
- **XLS-R** (Cross-lingual Speech Representation): Best performance model
- **XLSR-53** (53 languages): Good multilingual support  
- **HuBERT** (Hidden-Unit BERT): Alternative architecture

Models are loaded into a singleton `ModelManager` instance that handles:
- Device selection (CPU/CUDA auto-detection)
- Lazy loading with verification
- Inference orchestration for single or multiple models

### API Endpoints

- `GET /`: Service info and available endpoints
- `GET /health`: Health check with model loading status
- `GET /models`: Detailed model information
- `POST /predict`: Run inference on all models
- `POST /predict/<model_name>`: Run inference on specific model

### Configuration System

The config uses environment-based classes:
- `DevelopmentConfig`: Debug enabled, verbose errors
- `ProductionConfig`: Debug disabled, error details hidden
- `TestingConfig`: Reduced timeouts

Key configuration areas in `config.py`:
- `MODEL_CONFIGS`: Model paths and metadata
- `AUDIO_CONFIG`: Sample rate, format validation
- `TIMEOUT_CONFIG`: Processing timeouts
- `RESPONSE_CONFIG`: Output formatting
- `PERFORMANCE_CONFIG`: Device selection, optimizations

### Audio Processing

- Accepts WAV files only via multipart/form-data
- Uses librosa for audio loading and preprocessing
- Standardizes to 16kHz mono audio
- Supports confidence scoring and timing metrics

### Error Handling

The service implements comprehensive error handling:
- Standardized JSON error responses
- Graceful degradation (partial model failures allowed)
- Environment-specific error detail exposure
- Request validation with proper HTTP status codes