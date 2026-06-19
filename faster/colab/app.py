#!/usr/bin/env python3
"""
Audio Transcription Tool
A simple tool to transcribe MP3 files using Faster-Whisper (optimized Whisper).
"""

import os
import sys
import traceback
from pathlib import Path
from faster_whisper import WhisperModel

# Global variable to store the current model
current_model = None
current_model_size = None

# Valid language codes for Faster-Whisper
VALID_LANGUAGES = [
    'auto', 'en', 'ro', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'
]

def load_whisper_model(model_size):
    """Load or switch the Whisper model"""
    global current_model, current_model_size
    
    if current_model is None or current_model_size != model_size:
        print(f"Loading Whisper model: {model_size}")
        try:
            # Faster-Whisper uses compute_type for optimization
            # "int8" is a good default for GPU; use "int8_float16" for better GPU performance
            # or "int8" for CPU/GPU compatibility
            current_model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
            current_model_size = model_size
            print(f"✅ Successfully loaded {model_size} model")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model '{model_size}': {str(e)}")
    
    return current_model

def check_whisper_installation():
    """Check if Faster-Whisper is properly installed and working"""
    try:
        from faster_whisper import WhisperModel
        print("✅ Faster-Whisper library found")
        
        print("Testing Faster-Whisper installation...")
        test_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Faster-Whisper is working correctly")
        return True
        
    except ImportError:
        print("❌ Faster-Whisper is not installed")
        print("Install with: pip install faster-whisper")
        return False
    except Exception as e:
        print(f"❌ Faster-Whisper test failed: {e}")
        print("This might be due to:")
        print("1. Missing FFmpeg - install from https://ffmpeg.org/")
        print("2. Insufficient disk space for model download")
        print("3. Network connectivity issues")
        return False

def validate_audio_file(file_path):
    """Validate if the audio file exists and has correct extension"""
    if not file_path:
        raise Exception("No file path provided")
    
    path = Path(file_path)
    
    if not path.exists():
        raise Exception(f"File does not exist: {file_path}")
    
    if not path.is_file():
        raise Exception(f"Path is not a file: {file_path}")
    
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    if path.suffix.lower() not in valid_extensions:
        raise Exception(f"Unsupported file format: {path.suffix}. Supported formats: {', '.join(valid_extensions)}")
    
    return str(path.absolute())

def transcribe_audio(audio_file_path, model_size, language):
    """Transcribe audio file using Faster-Whisper"""
    if not audio_file_path or not os.path.exists(audio_file_path):
        raise Exception("No valid audio file found")
    
    try:
        model = load_whisper_model(model_size)
        
        print(f"Transcribing audio using {model_size} model...")
        if language and language != "auto":
            print(f"Language set to: {language}")
        else:
            print("Auto-detecting language...")
        
        # Faster-Whisper API: transcribe() returns segments iterator
        # language=None means auto-detect
        lang_param = None if (not language or language == "auto") else language
        
        print(f"Transcribing ... be patient")
        segments, info = model.transcribe(
            audio_file_path,
            beam_size=5,
            language=lang_param,
            log_progress=True,
            vad_filter=True
        )
        
        # Print detected language if auto-detect was used
        detected_lang = info.language
        print(f"Detected language: {detected_lang}")
        print(f"Language probability: {info.language_probability:.2%}")
        
        # Collect all segments into final text
        full_text = []
        for segment in segments:
            full_text.append(segment.text)
        
        return " ".join(full_text)
        
    except Exception as e:
        print(f"Transcription error details: {traceback.format_exc()}")
        raise Exception(f"Transcription failed: {str(e)}")

def get_user_input():
    """Interactive CLI to get user preferences"""
    print("🎵 Audio Transcription Tool")
    print("=" * 50)
    
    # Get audio file path
    while True:
        file_path = input("\nEnter MP3 file path: ").strip()
        if not file_path:
            print("Please enter a valid file path")
            continue
        
        file_path = file_path.strip('"\'')
        
        try:
            validated_path = validate_audio_file(file_path)
            break
        except Exception as e:
            print(f"❌ {str(e)}")
            continue
    
    language = 'ro'
    # Get model size preference
    model_size = "large-v3"
    
    return validated_path, language, model_size

def save_transcription(text, file_path):
    """Save transcription to transcription.txt"""
    try:
        with open('transcription.txt', 'w', encoding='utf-8') as f:
            f.write(f"Transcription of: {file_path}\n")
            f.write("=" * 50 + "\n\n")
            f.write(text)
        print("✅ Transcription saved to: transcription.txt")
        return True
    except Exception as e:
        print(f"❌ Error saving transcription: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    missing = []
    
    if not check_whisper_installation():
        missing.append("faster-whisper")
    
    if missing:
        print("❌ Missing or broken dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """Main interactive function"""
    try:
        if not check_dependencies():
            sys.exit(1)
        
        file_path, language, model_size = get_user_input()
        
        print("\n" + "=" * 50)
        print("PROCESSING AUDIO")
        print("=" * 50)
        print(f"File: {file_path}")
        print(f"Language: {language}")
        print(f"Model: {model_size}")
        print()
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📁 File size: {file_size:.2f} MB")
        
        print("🎯 Starting transcription...")
        transcription = transcribe_audio(file_path, model_size, language)
        
        print("\n" + "=" * 50)
        print("TRANSCRIPTION COMPLETE")
        print("=" * 50)
        print(transcription)
        print("\n" + "=" * 50)
        
        save_transcription(transcription, file_path)
        
        print("\n✅ Process completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting Audio Transcription Tool...")
    main()
