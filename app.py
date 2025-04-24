import os
import sys
import time
import whisper

def transcribe(audio_path, model_size="medium"):
    start_time = time.time()
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    transcript = result["text"]
    txt_filename = "transcript.txt"
    with open(txt_filename, "w") as f:
        f.write(transcript)
    elapsed_time = time.time() - start_time
    print(f"Transcript: {transcript}")
    print(f"Transcript saved to: {txt_filename}")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    return transcript

if __name__ == "__main__":
    if argv.length >= 4:
        audio_path = sys.argv[1]
        model_size = sys.argv[3]
    else
        sys.exit(1)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    transcribe(audio_path, model_size)

