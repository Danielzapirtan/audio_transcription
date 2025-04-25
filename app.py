import os
import sys
import time
import whisper

def transcribe(audio_path, model_size="medium"):
    start_time = time.time()
    
    # Load the Whisper model
    print(f"Loading {model_size} model...")
    model = whisper.load_model(model_size)
    
    # Transcribe the audio
    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path)
    transcript = result["text"]
    
    # Save the transcript
    txt_filename = "transcript.txt"
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f"Transcript saved to: {txt_filename}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    
    return transcript

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) < 2:
        print("Usage: python app.py <audio_path> [--model <model_size>]")
        sys.exit(1)
    
    # Default model size
    model_size = "medium"
    
    # Parse arguments
    audio_path = sys.argv[1]
    if "--model" in sys.argv:
        try:
            model_index = sys.argv.index("--model")
            model_size = sys.argv[model_index + 1]
        except IndexError:
            print("Error: --model requires a size argument")
            sys.exit(1)
    
    # Verify audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Run transcription
    transcribe(audio_path, model_size)