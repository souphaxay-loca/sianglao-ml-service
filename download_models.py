#!/usr/bin/env python3
"""
Download Lao ASR Models from HuggingFace
One-time setup script to download and organize models locally

Usage:
    python download_models.py
"""

import os
from pathlib import Path
from transformers import (
    Wav2Vec2ForCTC, 
    HubertForCTC, 
    Wav2Vec2Processor
)
import torch

class ModelDownloader:
    """Download and organize Lao ASR models from HuggingFace"""
    
    def __init__(self):
        self.model_configs = {
            "xls-r": {
                "repo": "SiangLao/xls-r-lao-asr",
                "local_path": "./saved_models/xls-r",
                "model_class": Wav2Vec2ForCTC,
                "description": "XLS-R-300M Lao ASR"
            },
            "xlsr-53": {
                "repo": "SiangLao/xlsr-53-lao-asr", 
                "local_path": "./saved_models/xlsr-53",
                "model_class": Wav2Vec2ForCTC,
                "description": "XLSR-53 Lao ASR"
            },
            "hubert": {
                "repo": "SiangLao/hubert-lao-asr",
                "local_path": "./saved_models/hubert", 
                "model_class": HubertForCTC,
                "description": "HuBERT-Large Lao ASR"
            }
        }
        
        print("🚀 Lao ASR Model Downloader")
        print("=" * 50)
    
    def check_folders(self):
        """Verify all required folders exist"""
        print("📁 Checking folder structure...")
        
        for model_name, config in self.model_configs.items():
            local_path = Path(config["local_path"])
            if not local_path.exists():
                print(f"❌ Missing folder: {local_path}")
                print(f"Please create it with: mkdir -p {local_path}")
                return False
            else:
                print(f"✅ {local_path}")
        
        return True
    
    def download_model(self, model_name: str):
        """Download a single model and processor"""
        config = self.model_configs[model_name]
        repo = config["repo"]
        local_path = config["local_path"]
        model_class = config["model_class"]
        description = config["description"]
        
        print(f"\n🔄 Downloading {description}...")
        print(f"📦 From: {repo}")
        print(f"💾 To: {local_path}")
        
        try:
            # Check if already exists
            if self.check_model_exists(local_path):
                print(f"✅ Model already exists in {local_path}")
                return True
            
            # Download model
            print("⬇️  Downloading model...")
            model = model_class.from_pretrained(repo)
            
            # Download processor  
            print("⬇️  Downloading processor...")
            processor = Wav2Vec2Processor.from_pretrained(repo)
            
            # Save locally
            print("💾 Saving model...")
            model.save_pretrained(local_path)
            
            print("💾 Saving processor...")
            processor.save_pretrained(local_path)
            
            # Verify download
            if self.verify_model(local_path, model_class):
                print(f"✅ {description} successfully downloaded!")
                return True
            else:
                print(f"❌ {description} verification failed!")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {description}: {e}")
            return False
    
    def check_model_exists(self, local_path: str) -> bool:
        """Check if model files already exist"""
        path = Path(local_path)
        required_files = ["config.json", "vocab.json"]
        
        # Check for model weights (either .bin or .safetensors)
        has_weights = (
            (path / "pytorch_model.bin").exists() or 
            (path / "model.safetensors").exists()
        )
        
        # Check for required config files
        has_configs = all((path / file).exists() for file in required_files)
        
        return has_weights and has_configs
    
    def verify_model(self, local_path: str, model_class) -> bool:
        """Verify downloaded model can be loaded"""
        try:
            print("🧪 Verifying model...")
            
            # Try loading model
            model = model_class.from_pretrained(local_path)
            processor = Wav2Vec2Processor.from_pretrained(local_path)
            
            # Basic checks
            assert model is not None, "Model is None"
            assert processor is not None, "Processor is None"
            assert len(processor.tokenizer) > 0, "Empty tokenizer"
            
            print(f"✅ Verification passed - vocab size: {len(processor.tokenizer)}")
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def download_all(self):
        """Download all Lao ASR models"""
        print("\n🎯 Starting download of all Lao ASR models...")
        
        # Check folder structure first
        if not self.check_folders():
            print("\n❌ Please create required folders first!")
            return False
        
        success_count = 0
        total_models = len(self.model_configs)
        
        # Download each model
        for model_name in self.model_configs.keys():
            if self.download_model(model_name):
                success_count += 1
            print("-" * 50)
        
        # Summary
        print(f"\n📊 Download Summary:")
        print(f"✅ Successful: {success_count}/{total_models}")
        print(f"❌ Failed: {total_models - success_count}/{total_models}")
        
        if success_count == total_models:
            print("\n🎉 All models downloaded successfully!")
            print("🚀 Ready to start Flask ML service!")
            return True
        else:
            print(f"\n⚠️  {total_models - success_count} model(s) failed to download")
            return False
    
    def show_model_info(self):
        """Display information about downloaded models"""
        print("\n📋 Model Information:")
        print("=" * 60)
        
        for model_name, config in self.model_configs.items():
            local_path = Path(config["local_path"])
            description = config["description"]
            repo = config["repo"]
            
            if self.check_model_exists(local_path):
                # Get folder size
                total_size = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file())
                size_mb = total_size / (1024 * 1024)
                
                print(f"✅ {description}")
                print(f"   📁 Path: {local_path}")
                print(f"   🔗 Repo: {repo}")
                print(f"   📦 Size: {size_mb:.1f} MB")
                print()
            else:
                print(f"❌ {description} - Not downloaded")
                print(f"   📁 Expected: {local_path}")
                print()


def main():
    """Main function"""
    try:
        downloader = ModelDownloader()
        
        # Download all models
        success = downloader.download_all()
        
        if success:
            # Show final model information
            downloader.show_model_info()
            
            print("\n🔥 Next Steps:")
            print("1. Models are ready in saved_models/ folder")
            print("2. Run your Flask ML service")
            print("3. Test with WAV files")
        
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()