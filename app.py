import argparse
import whisper
import time
import os

def transcribe(audio_path, model_size="medium"):
    start_time = time.time()
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript = result["text"]
    
    # Save transcript to a file
    txt_filename = "transcript.txt"
    with open(txt_filename, "w") as f:
        f.write(transcript)
    
    elapsed_time = time.time() - start_time
    print(f"Transcript: {transcript}")
    print(f"Transcript saved to: {txt_filename}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    return transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Transcription CLI")
    parser.add_argument("audio_path", help="Path to the audio file to transcribe")
    parser.add_argument("--model", "-m", default="medium", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Model size to use for transcription")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"Audio file not found: {args.audio_path}")
    
    transcribe(args.audio_path, args.model)
