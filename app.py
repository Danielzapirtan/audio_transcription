#!/usr/bin/env python3
"""
Audio Transcription Tool
A simple tool to transcribe MP3 files using OpenAI's Whisper model.
"""

import os
import sys
import whisper
import traceback
from pathlib import Path

# Global variable to store the current model
current_model = None
current_model_size = None

def load_whisper_model(model_size):
    """Load or switch the Whisper model"""
    global current_model, current_model_size
    
    # Only load if different from current model
    if current_model is None or current_model_size != model_size:
        print(f"Loading Whisper model: {model_size}")
        try:
            current_model = whisper.load_model(model_size)
            current_model_size = model_size
            print(f"✅ Successfully loaded {model_size} model")
        except Exception as e:
            raise Exception(f"Failed to load Whisper model '{model_size}': {str(e)}")
    
    return current_model

def check_whisper_installation():
    """Check if Whisper is properly installed and working"""
    try:
        import whisper
        print(f"Whisper version: {whisper.__version__}")
        
        # Test if load_model function exists
        if not hasattr(whisper, 'load_model'):
            raise AttributeError("whisper module doesn't have load_model function")
        
        # Try to load the smallest model as a test
        print("Testing Whisper installation...")
        test_model = whisper.load_model("tiny")
        print("✅ Whisper is working correctly")
        return True
        
    except ImportError:
        print("❌ Whisper is not installed")
        print("Install with: pip install openai-whisper")
        return False
    except AttributeError as e:
        print(f"❌ Whisper installation issue: {e}")
        print("Try reinstalling: pip uninstall openai-whisper && pip install openai-whisper")
        return False
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
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
    
    # Check file extension
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    if path.suffix.lower() not in valid_extensions:
        raise Exception(f"Unsupported file format: {path.suffix}. Supported formats: {', '.join(valid_extensions)}")
    
    return str(path.absolute())

def transcribe_audio(audio_file_path, model_size, language):
    """Transcribe audio file using Whisper"""
    if not audio_file_path or not os.path.exists(audio_file_path):
        raise Exception("No valid audio file found")
    
    try:
        # Load the appropriate model
        model = load_whisper_model(model_size)
        
        # Set transcription options based on language selection
        options = {
            'fp16': False,  # Disable half precision for better compatibility
            'verbose': False
        }
        
        if language and language != "auto":
            options["language"] = language
        
        print(f"Transcribing audio using {model_size} model...")
        if language and language != "auto":
            print(f"Language set to: {language}")
        else:
            print("Auto-detecting language...")
        
        # Transcribe with the selected options
        result = model.transcribe(audio_file_path, **options)
        
        # Print detected language if auto-detect was used
        if not language or language == "auto":
            detected_lang = result.get("language", "unknown")
            print(f"Detected language: {detected_lang}")
        
        return result["text"]
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
        
        # Remove quotes if present
        file_path = file_path.strip('"\'')
        
        try:
            validated_path = validate_audio_file(file_path)
            break
        except Exception as e:
            print(f"❌ {str(e)}")
            continue
    
    # Get language preference
    while True:
        print("\nLanguage options:")
        print("- auto: Auto-detect language")
        print("- en: English")
        print("- ro: Romanian")
        print("- es: Spanish")
        print("- fr: French")
        print("- de: German")
        print("- it: Italian")
        print("- pt: Portuguese")
        print("- ru: Russian")
        print("- ja: Japanese")
        print("- ko: Korean")
        print("- zh: Chinese")
        
        language = input("Enter language code (auto/en/ro/es/fr/de/it/pt/ru/ja/ko/zh): ").strip().lower()
        if language in ['auto', 'en', 'ro', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']:
            break
        else:
            print("Please enter a valid language code")
    
    # Get model size preference
    while True:
        print("\nAvailable models:")
        print("1. tiny - Fastest, least accurate (~39 MB)")
        print("2. base - Good balance (~74 MB) [recommended]")
        print("3. small - Better accuracy (~244 MB)")
        print("4. medium - High accuracy (~769 MB)")
        print("5. large - Highest accuracy, slowest (~1550 MB)")
        
        choice = input("Select model (1-5) or press Enter for auto-selection: ").strip()
        
        if choice == '':
            # Auto-select based on language
            if language == 'en':
                model_size = 'base'
                print(f"Auto-selected 'base' model for English")
            else:
                model_size = 'medium'
                print(f"Auto-selected 'medium' model for better multilingual support")
            break
        elif choice in ['1', '2', '3', '4', '5']:
            models = ['tiny', 'base', 'small', 'medium', 'large']
            model_size = models[int(choice) - 1]
            print(f"Selected '{model_size}' model")
            break
        else:
            print("Please enter a number 1-5 or press Enter")
    
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
    
    # Check Whisper specifically
    if not check_whisper_installation():
        missing.append("openai-whisper")
    
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
        # Check dependencies first
        if not check_dependencies():
            sys.exit(1)
        
        # Get user input
        file_path, language, model_size = get_user_input()
        
        print("\n" + "=" * 50)
        print("PROCESSING AUDIO")
        print("=" * 50)
        print(f"File: {file_path}")
        print(f"Language: {language}")
        print(f"Model: {model_size}")
        print()
        
        # Get file info
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"📁 File size: {file_size:.2f} MB")
        
        # Transcribe
        print("🎯 Starting transcription...")
        transcription = transcribe_audio(file_path, model_size, language)
        
        # Display results
        print("\n" + "=" * 50)
        print("TRANSCRIPTION COMPLETE")
        print("=" * 50)
        print(transcription)
        print("\n" + "=" * 50)
        
        # Save transcription to file
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