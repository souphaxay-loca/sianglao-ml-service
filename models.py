#!/usr/bin/env python3
"""
Model loading and inference logic for Sianglao ML Service
Handles all 3 Lao ASR models: XLS-R, XLSR-53, HuBERT
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import warnings

from transformers import (
    Wav2Vec2ForCTC,
    HubertForCTC, 
    Wav2Vec2Processor
)

from config import config, MODEL_CONFIGS, AVAILABLE_MODELS

warnings.filterwarnings("ignore")

class ModelManager:
    """Manages loading and inference for all Lao ASR models"""
    
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = self._get_device()
        self.loaded = False
        
        print(f"🚀 Initializing Model Manager")
        print(f"💻 Device: {self.device}")
        print(f"🎯 Models to load: {', '.join(AVAILABLE_MODELS)}")
    
    def _get_device(self) -> str:
        """Determine the best device for inference"""
        device_config = config.PERFORMANCE_CONFIG["device"]
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🔥 CUDA detected: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                print("💻 Using CPU for inference")
        else:
            device = device_config
            
        return device
    
    def load_all_models(self) -> bool:
        """Load all models on startup"""
        print("\n📦 Loading all models...")
        print("=" * 50)
        
        start_time = time.time()
        success_count = 0
        
        for model_name in AVAILABLE_MODELS:
            try:
                if self._load_single_model(model_name):
                    success_count += 1
                    print(f"✅ {model_name.upper()} loaded successfully")
                else:
                    print(f"❌ {model_name.upper()} failed to load")
                    
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
                
            print("-" * 30)
        
        loading_time = time.time() - start_time
        
        # Summary
        total_models = len(AVAILABLE_MODELS)
        print(f"📊 Loading Summary:")
        print(f"✅ Loaded: {success_count}/{total_models}")
        print(f"⏱️  Time: {loading_time:.1f}s")
        
        if success_count == total_models:
            print("🎉 All models loaded successfully!")
            self.loaded = True
            return True
        elif success_count > 0:
            print(f"⚠️  Partial success: {success_count}/{total_models} models loaded")
            self.loaded = True
            return True
        else:
            print("❌ No models loaded successfully!")
            self.loaded = False
            return False
    
    def _load_single_model(self, model_name: str) -> bool:
        """Load a single model and processor"""
        if model_name not in MODEL_CONFIGS:
            print(f"❌ Unknown model: {model_name}")
            return False
        
        config_item = MODEL_CONFIGS[model_name]
        model_path = Path(config_item["path"])
        model_class_name = config_item["model_class"]
        description = config_item["description"]
        
        print(f"🔄 Loading {description}...")
        print(f"📁 Path: {model_path}")
        
        try:
            # Check if path exists
            if not model_path.exists():
                print(f"❌ Model path not found: {model_path}")
                return False
            
            # Load processor (same for all models)
            print("📝 Loading processor...")
            processor = Wav2Vec2Processor.from_pretrained(model_path)
            
            # Load model based on class type
            print("🧠 Loading model...")
            if model_class_name == "HubertForCTC":
                model = HubertForCTC.from_pretrained(model_path)
            else:  # Wav2Vec2ForCTC
                model = Wav2Vec2ForCTC.from_pretrained(model_path)
            
            # Move to device and set eval mode
            model.to(self.device)
            model.eval()
            
            # Store in manager
            self.models[model_name] = model
            self.processors[model_name] = processor
            
            # Verify loading
            vocab_size = len(processor.tokenizer)
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            
            print(f"✅ Model loaded - Params: {param_count:.1f}M, Vocab: {vocab_size}")
            return True
            
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            return False
    
    def predict_single(self, model_name: str, audio_data: np.ndarray, 
                      sample_rate: int = 16000) -> Optional[Dict]:
        """Run inference on a single model"""
        if not self.loaded:
            return {"error": "Models not loaded"}
        
        if model_name not in self.models:
            return {"error": f"Model {model_name} not available"}
        
        try:
            start_time = time.time()
            
            # Get model and processor
            model = self.models[model_name]
            processor = self.processors[model_name]
            
            # Preprocess audio
            print(f"🎵 Processing audio for {model_name}...")
            inputs = processor(
                audio_data,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            print(f"🧠 Running inference on {model_name}...")
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Calculate confidence score
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_prob).item()
            
            # Decode prediction
            predicted_ids = torch.argmax(logits, dim=-1)
            prediction = processor.batch_decode(predicted_ids)[0]
            
            # Clean prediction (handle <unk> tokens)
            if config.RESPONSE_CONFIG["clean_predictions"]:
                cleaned_prediction = prediction.strip().replace("<unk>", " ")
            else:
                cleaned_prediction = prediction.strip()
            
            processing_time = time.time() - start_time
            
            result = {
                "model": model_name,
                "prediction": cleaned_prediction,
                "raw_prediction": prediction.strip(),
                "confidence": round(confidence, config.RESPONSE_CONFIG["decimal_precision"]),
                "processing_time": round(processing_time, config.RESPONSE_CONFIG["decimal_precision"]),
                "success": True
            }
            
            # print(f"✅ {model_name} prediction: '{cleaned_prediction}' (conf: {confidence:.3f})")
            return result
            
        except Exception as e:
            error_result = {
                "model": model_name,
                "error": str(e),
                "success": False
            }
            print(f"❌ {model_name} inference failed: {e}")
            return error_result
    
    def predict_all(self, audio_data: np.ndarray, 
                   sample_rate: int = 16000) -> Dict:
        """Run inference on all available models"""
        if not self.loaded:
            return {
                "success": False,
                "error": "Models not loaded",
                "results": {}
            }
        
        print(f"🎯 Running inference on all models...")
        start_time = time.time()
        
        results = {}
        success_count = 0
        
        # Run inference on each loaded model
        for model_name in AVAILABLE_MODELS:
            if model_name in self.models:
                result = self.predict_single(model_name, audio_data, sample_rate)
                results[model_name] = result
                
                if result.get("success", False):
                    success_count += 1
            else:
                results[model_name] = {
                    "model": model_name,
                    "error": "Model not loaded",
                    "success": False
                }
        
        total_time = time.time() - start_time
        
        # Determine overall success
        if success_count == 0:
            overall_success = False
            message = "All models failed"
        elif success_count == len(AVAILABLE_MODELS):
            overall_success = True
            message = f"All {success_count} models succeeded"
        else:
            overall_success = config.ERROR_CONFIG["graceful_degradation"]
            message = f"{success_count}/{len(AVAILABLE_MODELS)} models succeeded"
        
        response = {
            "success": overall_success,
            "message": message,
            "results": results,
            "summary": {
                "successful_models": success_count,
                "total_models": len(AVAILABLE_MODELS),
                "total_processing_time": round(total_time, config.RESPONSE_CONFIG["decimal_precision"])
            }
        }
        
        print(f"📊 All models completed: {message} in {total_time:.1f}s")
        return response
    
    def get_model_status(self) -> Dict:
        """Get status of all models"""
        status = {
            "loaded": self.loaded,
            "device": self.device,
            "available_models": AVAILABLE_MODELS,
            "loaded_models": list(self.models.keys()),
            "models": {}
        }
        
        for model_name in AVAILABLE_MODELS:
            model_config = MODEL_CONFIGS[model_name]
            is_loaded = model_name in self.models
            
            status["models"][model_name] = {
                "description": model_config["description"],
                "path": model_config["path"],
                "loaded": is_loaded,
                "expected_cer": model_config["expected_performance"]
            }
        
        return status
    
    def unload_models(self):
        """Unload all models from memory"""
        print("🧹 Unloading models...")
        
        for model_name in list(self.models.keys()):
            del self.models[model_name]
            del self.processors[model_name]
            print(f"🗑️  Unloaded {model_name}")
        
        self.models.clear()
        self.processors.clear()
        
        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("🧹 Cleared CUDA cache")
        
        self.loaded = False
        print("✅ All models unloaded")


# ================================
# Global Model Manager Instance
# ================================
model_manager = ModelManager()


# ================================
# Convenience Functions
# ================================
def load_models() -> bool:
    """Initialize and load all models"""
    return model_manager.load_all_models()


def predict_single_model(model_name: str, audio_data: np.ndarray, 
                        sample_rate: int = 16000) -> Optional[Dict]:
    """Predict using a single model"""
    return model_manager.predict_single(model_name, audio_data, sample_rate)


def predict_all_models(audio_data: np.ndarray, 
                      sample_rate: int = 16000) -> Dict:
    """Predict using all models"""
    return model_manager.predict_all(audio_data, sample_rate)


def get_models_status() -> Dict:
    """Get status of all models"""
    return model_manager.get_model_status()


def is_models_loaded() -> bool:
    """Check if models are loaded"""
    return model_manager.loaded


def get_available_models() -> list:
    """Get list of available model names"""
    return AVAILABLE_MODELS.copy()


# ================================
# Model Validation
# ================================
def validate_model_name(model_name: str) -> bool:
    """Validate if model name is available"""
    return model_name in AVAILABLE_MODELS


def validate_loaded_model(model_name: str) -> bool:
    """Check if specific model is loaded"""
    return model_name in model_manager.models