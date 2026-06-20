#!/usr/bin/env python3

import os
import sys
import traceback
from pathlib import Path
from faster_whisper import WhisperModel

# Global variable to store the current model
current_model = None
current_model_size = None

def error(ecode):
    print(f"Error {ecode}")
    sys.exit(2)

def load_whisper_model(model_size):
    global current_model, current_model_size
    if current_model is None or current_model_size != model_size:
        try:
            current_model = WhisperModel(model_size, device="cpu", compute_type="int8")
            current_model_size = model_size
        except Exception as e:
            error(23)
    return current_model

def validate_audio_file(file_path):
    if not file_path:
        error(11)
    path = Path(file_path)
    if not path.exists():
        error(13)
    if not path.is_file():
        error(17)
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    if path.suffix.lower() not in valid_extensions:
        error(19)
    return str(path.absolute())

def transcribe_audio(audio_file_path, model_size, language):
    try:
        model = load_whisper_model(model_size)
        file_path = validate_audio_file(audio_file_path)
        lang_param = language
        segments, _ = model.transcribe(
            file_path,
            language=lang_param,
            log_progress=True,
            beam_size=5,
            vad_filter=True
        )
        full_text = []
        for segment in segments:
            full_text.append(segment.text)
        return " ".join(full_text)
    except Exception as e:
        error(e)

def save_transcription(text, file_path):
    with open('transcription.txt', 'w', encoding='utf-8') as f:
        f.write(text)

def main():
    try:
        file_path = sys.argv[1]
        language = "ro"
        model_size = "large-v3"
        transcription = transcribe_audio(file_path, model_size, language)
        printf(transcription)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        error(2)
    except Exception as e:
        error(e)

if __name__ == "__main__":
    main()

