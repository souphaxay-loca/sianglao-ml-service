#!/usr/bin/env python3
"""
Sianglao ML Service - Flask Application
Lao ASR Inference API with XLS-R, XLSR-53, and HuBERT models
"""

import os
import time
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np

# Import our modules
from config import config, AVAILABLE_MODELS, validate_model_paths
from models import (
    load_models, 
    predict_single_model, 
    predict_all_models,
    get_models_status,
    is_models_loaded,
    validate_model_name,
    validate_loaded_model
)

# ================================
# Flask App Initialization
# ================================
app = Flask(__name__)

# Enable CORS for development
if config.DEV_CONFIG["enable_cors"]:
    CORS(app)

# ================================
# Global Variables
# ================================
MODELS_LOADED = False
STARTUP_TIME = None
STARTUP_DURATION = None
SERVICE_INFO = {
    "service": "Sianglao ML Service",
    "description": "Lao ASR Inference API",
    "version": "1.0.0",
    "models": AVAILABLE_MODELS,
    "author": "Souphaxay Naovalath & Sounmy Chanthavong"
}

# ================================
# Utility Functions
# ================================
def load_audio_from_request() -> tuple:
    """Load audio from Flask request file upload"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return None, "No audio file provided"
        
        audio_file = request.files['audio']
        
        # Check if file is selected
        if audio_file.filename == '':
            return None, "No audio file selected"
        
        # Check file extension
        if not audio_file.filename.lower().endswith('.wav'):
            return None, "Only WAV files are supported"
        
        # Load audio with librosa
        print(f"🎵 Loading audio: {audio_file.filename}")
        audio_data, sample_rate = librosa.load(
            audio_file, 
            sr=config.AUDIO_CONFIG["sample_rate"], 
            mono=config.AUDIO_CONFIG["mono"]
        )
        
        # Basic validation
        if len(audio_data) == 0:
            return None, "Empty audio file"
        
        duration = len(audio_data) / sample_rate
        print(f"✅ Audio loaded: {duration:.1f}s, {sample_rate}Hz")
        
        return audio_data, None
        
    except Exception as e:
        error_msg = f"Error loading audio: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg


def create_error_response(message: str, status_code: int = 400, 
                         include_details: bool = None) -> tuple:
    """Create standardized error response"""
    if include_details is None:
        include_details = config.ERROR_CONFIG["include_error_details"]
    
    # Demo-friendly error messages
    demo_messages = {
        "No audio file provided": "Please provide a WAV audio file",
        "Models not loaded": "Inference models are loading, please wait",
        "Only WAV files are supported": "Please upload a WAV format audio file",
        "Empty audio file": "Audio file is empty or corrupted",
        "No audio file selected": "Please select an audio file to upload"
    }
    
    display_message = demo_messages.get(message, message)
    
    response = {
        "success": False,
        "error": display_message,
        "technical_error": message if display_message != message else None,
        "timestamp": datetime.now().isoformat(),
        "service": SERVICE_INFO["service"]
    }
    
    if include_details and config.FLASK_DEBUG:
        response["traceback"] = traceback.format_exc()
    
    return jsonify(response), status_code


def create_success_response(data: dict, processing_time: float = None, 
                          audio_duration: float = None) -> dict:
    """Create standardized success response"""
    response = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        "service": SERVICE_INFO["service"],
        **data
    }
    
    # Add timing information if requested
    if config.RESPONSE_CONFIG["include_timing"] and processing_time is not None:
        response["processing_time"] = round(processing_time, config.RESPONSE_CONFIG["decimal_precision"])
    
    # Add audio information if requested
    if config.RESPONSE_CONFIG["include_audio_info"] and audio_duration is not None:
        response["audio_duration"] = round(audio_duration, config.RESPONSE_CONFIG["decimal_precision"])
    
    # Add demo-friendly display info
    response["display_info"] = {
        "processing_time_text": f"{processing_time:.2f}s" if processing_time else None,
        "audio_duration_text": f"{audio_duration:.1f}s" if audio_duration else None,
        "timestamp_text": datetime.now().strftime("%H:%M:%S")
    }
    
    return jsonify(response)


# ================================
# Health Check Routes
# ================================
@app.route('/', methods=['GET'])
def root():
    """Root endpoint - basic service info"""
    uptime = time.time() - STARTUP_TIME if STARTUP_TIME else 0
    
    return jsonify({
        **SERVICE_INFO,
        "status": "running",
        "models_loaded": MODELS_LOADED,
        "startup_duration_seconds": STARTUP_DURATION,
        "uptime_seconds": round(uptime, 1),
        "endpoints": {
            "predict_all": "POST /predict",
            "predict_single": "POST /predict/<model_name>",
            "health": "GET /health", 
            "models": "GET /models"
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    uptime = time.time() - STARTUP_TIME if STARTUP_TIME else 0
    
    health_status = {
        "status": "healthy" if MODELS_LOADED else "unhealthy",
        "models_loaded": MODELS_LOADED,
        "startup_duration_seconds": STARTUP_DURATION,
        "uptime_seconds": round(uptime, 1),
        "available_models": AVAILABLE_MODELS,
        "loaded_models": list(get_models_status().get("loaded_models", [])),
        "timestamp": datetime.now().isoformat(),
        # Demo-specific info
        "inference_ready": MODELS_LOADED,
        "total_inferences": get_models_status().get("inference_count", 0),
        "service_uptime_minutes": round(uptime / 60, 1)
    }
    
    status_code = 200 if MODELS_LOADED else 503
    return jsonify(health_status), status_code


@app.route('/models', methods=['GET'])
def models_info():
    """Get detailed information about all models"""
    try:
        models_status = get_models_status()
        return jsonify({
            "success": True,
            "models_info": models_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return create_error_response(f"Error getting models info: {str(e)}", 500)


# ================================
# Prediction Routes
# ================================
@app.route('/predict', methods=['POST'])
def predict_all():
    """Predict using all available models"""
    if not MODELS_LOADED:
        return create_error_response("Models not loaded", 503)
    
    start_time = time.time()
    
    try:
        # Load audio from request
        audio_data, error = load_audio_from_request()
        if error:
            return create_error_response(error, 400)
        
        # Calculate audio duration
        duration = len(audio_data) / config.AUDIO_CONFIG["sample_rate"]
        
        # Run inference on all models
        print(f"🎯 Running prediction on all models...")
        results = predict_all_models(audio_data)
        
        processing_time = time.time() - start_time
        
        if results.get("success", False):
            response_data = {
                "results": results["results"],
                "summary": results["summary"]
            }
            return create_success_response(response_data, processing_time, duration)
        else:
            return create_error_response(
                results.get("message", "All models failed"), 
                500
            )
    
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


@app.route('/predict/<model_name>', methods=['POST'])
def predict_single(model_name):
    """Predict using a specific model"""
    if not MODELS_LOADED:
        return create_error_response("Models not loaded", 503)
    
    # Validate model name
    if not validate_model_name(model_name):
        return create_error_response(
            f"Invalid model name: {model_name}. Available: {AVAILABLE_MODELS}", 
            400
        )
    
    # Check if specific model is loaded
    if not validate_loaded_model(model_name):
        return create_error_response(f"Model {model_name} not loaded", 503)
    
    start_time = time.time()
    
    try:
        # Load audio from request
        audio_data, error = load_audio_from_request()
        if error:
            return create_error_response(error, 400)
        
        # Calculate audio duration
        duration = len(audio_data) / config.AUDIO_CONFIG["sample_rate"]
        
        # Run inference on specific model
        print(f"🎯 Running prediction on {model_name}...")
        result = predict_single_model(model_name, audio_data)
        
        processing_time = time.time() - start_time
        
        if result and result.get("success", False):
            response_data = {
                "result": result
            }
            return create_success_response(response_data, processing_time, duration)
        else:
            error_message = result.get("error", "Prediction failed") if result else "Prediction failed"
            return create_error_response(error_message, 500)
    
    except Exception as e:
        error_msg = f"Prediction failed for {model_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return create_error_response(error_msg, 500)


# ================================
# Error Handlers
# ================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return create_error_response("Endpoint not found", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return create_error_response("Method not allowed", 405)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return create_error_response("File too large", 413)


@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 errors"""
    return create_error_response("Internal server error", 500)


# ================================
# Application Startup
# ================================
def initialize_service():
    """Initialize the ML service"""
    global MODELS_LOADED, STARTUP_TIME, STARTUP_DURATION
    
    # Demo-friendly startup banner
    print("\n" + "🎤" * 30)
    print("   SIANGLAO LAO ASR INFERENCE SERVICE")
    print("🎤" * 30)
    print(f"📍 Service URL: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"🎯 Ready for Lao speech recognition!")
    print(f"🤖 Models to load: {', '.join(AVAILABLE_MODELS)}")
    print("🎤" * 30)
    
    STARTUP_TIME = time.time()
    
    # Validate model paths
    print("📁 Validating model paths...")
    missing_paths = validate_model_paths()
    if missing_paths:
        print("❌ Missing model paths:")
        for path in missing_paths:
            print(f"   {path}")
        print("\n💡 Run download_models.py first!")
        return False
    
    print("✅ All model paths exist")
    
    # Load models
    print("\n📦 Loading models...")
    MODELS_LOADED = load_models()
    
    # Calculate startup duration
    STARTUP_DURATION = round(time.time() - STARTUP_TIME, 2)
    
    if MODELS_LOADED:
        print(f"\n🎉 Service initialized successfully in {STARTUP_DURATION}s")
        print(f"🌐 Ready to serve requests on {config.FLASK_HOST}:{config.FLASK_PORT}")
        print(f"🚀 Loaded models: {', '.join(get_models_status().get('loaded_models', []))}")
        print("🎤" * 30 + "\n")
        return True
    else:
        print("\n❌ Service initialization failed")
        return False


# ================================
# Application Entry Point
# ================================
if __name__ == '__main__':
    try:
        # Initialize service
        if initialize_service():
            print(f"\n🔥 Starting Flask server...")
            print(f"📍 URL: http://{config.FLASK_HOST}:{config.FLASK_PORT}")
            print(f"🎯 Test endpoint: POST /predict")
            print(f"💊 Health check: GET /health")
            print("\n" + "=" * 60)
            
            # Start Flask app
            app.run(
                host=config.FLASK_HOST,
                port=config.FLASK_PORT,
                debug=config.FLASK_DEBUG,
                threaded=True  # Enable multiple requests
            )
        else:
            print("❌ Failed to initialize service. Exiting.")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Service interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        exit(1)