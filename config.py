#!/usr/bin/env python3
"""
Configuration settings for Sianglao ML Service
"""

import os
from pathlib import Path

class Config:
    """Main configuration class for the ML service"""
    
    # ================================
    # Flask Configuration
    # ================================
    FLASK_HOST = "0.0.0.0"
    FLASK_PORT = 8000
    FLASK_DEBUG = False
    
    # ================================
    # Model Configuration  
    # ================================
    MODEL_CONFIGS = {
        "xls-r": {
            "path": "./saved_models/xls-r",
            "model_class": "Wav2Vec2ForCTC",
            "description": "XLS-R-300M Lao ASR",
            "repo": "h3llohihi/xls-r-lao-asr",
            "expected_performance": 0.1514  # 15.14% CER
        },
        "xlsr-53": {
            "path": "./saved_models/xlsr-53",
            "model_class": "Wav2Vec2ForCTC", 
            "description": "XLSR-53 Lao ASR",
            "repo": "h3llohihi/xlsr-53-lao-asr",
            "expected_performance": 0.1622  # 16.22% CER
        },
        "hubert": {
            "path": "./saved_models/hubert",
            "model_class": "HubertForCTC",
            "description": "HuBERT-Large Lao ASR", 
            "repo": "h3llohihi/hubert-lao-asr",
            "expected_performance": 0.2537  # 25.37% CER
        }
    }
    
    # Available model names for validation
    AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())
    
    # ================================
    # Audio Processing Configuration
    # ================================
    AUDIO_CONFIG = {
        "sample_rate": 16000,           # Target sample rate
        "mono": True,                   # Convert to mono
        "normalize": True,              # Normalize audio  
        "supported_formats": [".wav"],  # Only WAV files
        "max_duration": 300,            # Max 5 minutes (handled elsewhere but for reference)
        "min_duration": 0.1,            # Min 0.1 seconds
    }
    
    # ================================
    # Timeout Configuration
    # ================================
    TIMEOUT_CONFIG = {
        "audio_processing": 10,         # Audio loading/preprocessing timeout
        "single_model_inference": 30,   # Single model prediction timeout  
        "all_models_inference": 60,     # All models prediction timeout
        "request_timeout": 90,          # Overall request timeout
        "model_loading": 120            # Model loading timeout (startup)
    }
    
    # ================================
    # Response Configuration
    # ================================
    RESPONSE_CONFIG = {
        "include_confidence": True,     # Include confidence scores
        "include_timing": True,         # Include processing times
        "include_audio_info": True,     # Include audio duration/info
        "clean_predictions": True,      # Apply <unk> → space cleaning
        "decimal_precision": 4          # Decimal places for confidence/timing
    }
    
    # ================================
    # Error Handling Configuration
    # ================================
    ERROR_CONFIG = {
        "log_errors": True,            # Log errors to console
        "include_error_details": True, # Include error details in response
        "fail_fast": True,            # Stop on first error vs continue
        "retry_attempts": 1,          # Number of retry attempts for inference
        "graceful_degradation": True  # Continue if some models fail
    }
    
    # ================================
    # Performance Configuration
    # ================================
    PERFORMANCE_CONFIG = {
        "device": "auto",              # auto, cpu, cuda, mps
        "batch_processing": False,     # Future: support batch inference
        "model_caching": True,         # Keep models in memory
        "gradient_checkpointing": False, # Memory optimization (not needed for inference)
        "half_precision": False,       # Use FP16 (can cause issues with vocab adaptation)
        "mps_fallback": True,          # Fallback to CPU for unsupported MPS operations
        "memory_mode": os.getenv("MEMORY_MODE", "normal").lower()  # normal, low_memory (model offloading)
    }
    
    # ================================
    # Development/Testing Configuration
    # ================================
    DEV_CONFIG = {
        "test_audio_dir": "./test_audio",
        "sample_test_files": [
            "sample1.wav",
            "sample2.wav", 
            "sample3.wav"
        ],
        "log_level": "INFO",           # DEBUG, INFO, WARNING, ERROR
        "enable_cors": True,           # Enable CORS for development
        "verbose_responses": True,     # Include extra info in responses
        # Demo-specific settings
        "demo_mode": True,             # Enable demo features
        "demo_console_output": True,   # Enhanced console logging
        "show_inference_timing": True, # Display timing info
        "friendly_error_messages": True # User-friendly errors
    }


class ProductionConfig(Config):
    """Production-specific configuration"""
    FLASK_DEBUG = False
    ERROR_CONFIG = {
        **Config.ERROR_CONFIG,
        "include_error_details": False,  # Hide error details in production
        "log_errors": True
    }
    DEV_CONFIG = {
        **Config.DEV_CONFIG,
        "log_level": "WARNING",
        "verbose_responses": False
    }


class DevelopmentConfig(Config):
    """Development-specific configuration"""
    FLASK_DEBUG = True
    ERROR_CONFIG = {
        **Config.ERROR_CONFIG,
        "include_error_details": True,
        "log_errors": True
    }
    DEV_CONFIG = {
        **Config.DEV_CONFIG,
        "log_level": "DEBUG",
        "verbose_responses": True
    }


class TestingConfig(Config):
    """Testing-specific configuration"""
    FLASK_DEBUG = True
    TIMEOUT_CONFIG = {
        **Config.TIMEOUT_CONFIG,
        "audio_processing": 5,
        "single_model_inference": 15,
        "all_models_inference": 30,
        "request_timeout": 45
    }


# ================================
# Configuration Selection
# ================================
def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development').lower()
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()


# ================================
# Utility Functions
# ================================
def validate_model_paths():
    """Validate that all model paths exist"""
    config = get_config()
    missing_paths = []
    
    for model_name, model_config in config.MODEL_CONFIGS.items():
        path = Path(model_config["path"])
        if not path.exists():
            missing_paths.append(f"{model_name}: {path}")
    
    return missing_paths


def get_model_info():
    """Get summary information about all models"""
    config = get_config()
    info = {}
    
    for model_name, model_config in config.MODEL_CONFIGS.items():
        path = Path(model_config["path"])
        info[model_name] = {
            "description": model_config["description"],
            "path": str(path),
            "exists": path.exists(),
            "model_class": model_config["model_class"],
            "expected_cer": model_config["expected_performance"]
        }
    
    return info


# ================================
# Export default config
# ================================
config = get_config()

# For backwards compatibility and easy imports
MODEL_CONFIGS = config.MODEL_CONFIGS
AVAILABLE_MODELS = config.AVAILABLE_MODELS
AUDIO_CONFIG = config.AUDIO_CONFIG
TIMEOUT_CONFIG = config.TIMEOUT_CONFIG