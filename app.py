import os
import subprocess
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the whisper.cpp executable
WHISPER_CPP_EXEC = "../../whisper.cpp/build/bin/whisper-cli"

# Model configuration
MODEL_BASE_PATH = "../../whisper.cpp/models"
MODEL_SIZES = {
    "tiny": "ggml-tiny.bin",
    "tiny-q5_1": "ggml-tiny-q5_1.bin",
    "tiny.en": "ggml-tiny.en.bin",
    "base": "ggml-base.bin",
    "base-q5_1": "ggml-base-q5_1.bin",
    "base-q8_0": "ggml-base-q8_0.bin",
    "base.en-q8_0": "ggml-base.en-q8_0.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small-q5_1": "ggml-small-q5_1.bin",
    "small.en": "ggml-small.en.bin",
    "small-q8_0": "ggml-small-q8_0.bin",
    "small.en-q5_1": "ggml-small.en-q5_1.en.bin",
    "medium": "ggml-medium.bin",
    "large-v1": "ggml-large-v1.bin",
    "large-v3-turbo-q8_0": "ggml-large-v3-turbo-q8_0.bin"
}

# Language configuration
SUPPORTED_LANGUAGES = {
    "en": "english",
    "ro": "romanian"
}

# Upload folder for audio files
UPLOAD_FOLDER = "/tmp/uploads"
CONVERTED_FOLDER = "/tmp/converted"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_FOLDER, exist_ok=True)

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_path(model_size):
    """
    Get the full path for the specified model size
    """
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Invalid model size: {model_size}")
    return os.path.join(MODEL_BASE_PATH, MODEL_SIZES[model_size])

def validate_language(language):
    """
    Validate the specified language
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}")
    return SUPPORTED_LANGUAGES[language]

def convert_to_wav(input_path, output_path):
    """
    Convert audio file to 16kHz WAV format using FFmpeg
    """
    try:
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            '-y',  # Overwrite output file if it exists
            output_path
        ]
        
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            raise Exception("FFmpeg conversion failed")
            
        logger.info(f"Successfully converted {input_path} to WAV")
        return True
        
    except Exception as e:
        logger.error(f"Error during audio conversion: {str(e)}")
        raise

@app.route('/')
def index():
    """
    Render the main index page
    """
    return render_template('index.html')

@app.route("/download-transcription", methods=["POST"])
def download_transcription():
    """
    Download transcription as a text file
    """
    transcription = request.form.get('transcription')
    
    if not transcription:
        return jsonify({"error": "No transcription provided"}), 400
    
    # Create a temporary file for the transcription
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='transcription_') as temp_file:
        temp_file.write(transcription)
        temp_file_path = temp_file.name
    
    try:
        return send_file(
            temp_file_path, 
            mimetype='text/plain', 
            as_attachment=True, 
            download_name='transcription.txt'
        )
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        # Validate model size, language, and get optional prompt
        model_size = request.form.get('model_size')
        language = request.form.get('language')
        initial_prompt = request.form.get('prompt', '').strip()

        if not model_size:
            return jsonify({"error": "Model size not specified"}), 400
        if not language:
            return jsonify({"error": "Language not specified"}), 400

        try:
            model_path = get_model_path(model_size)
            language = validate_language(language)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Check if executable and model exist
        if not os.path.isfile(WHISPER_CPP_EXEC):
            logger.error(f"Whisper executable not found at {WHISPER_CPP_EXEC}")
            return jsonify({"error": "Transcription service not properly configured"}), 500

        if not os.path.isfile(model_path):
            logger.error(f"Model file not found at {model_path}")
            return jsonify({"error": f"Transcription model {model_size} not found"}), 500

        # Check for FFmpeg installation
        try:
            subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError:
            logger.error("FFmpeg not found. Please install FFmpeg.")
            return jsonify({"error": "FFmpeg not installed"}), 500

        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed"}), 400

        # Secure the filename
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        wav_filename = os.path.splitext(filename)[0] + '.wav'
        output_path = os.path.join(CONVERTED_FOLDER, wav_filename)

        try:
            # Save the uploaded file
            file.save(input_path)
            os.chmod(input_path, 0o666)
            logger.info(f"Saved uploaded file: {input_path}")

            # Convert to WAV format
            convert_to_wav(input_path, output_path)
            logger.info(f"Converted to WAV: {output_path}")

            # Build command with optional prompt
            command = [
                WHISPER_CPP_EXEC,
                "-m", model_path,
                "-l", language,
                "-nt",
                "--output-txt",
                "-f", output_path
            ]

            # Add initial prompt if provided
            if initial_prompt:
                command.extend(["--prompt", initial_prompt])
                logger.info(f"Using initial prompt: {initial_prompt}")

            # Run whisper.cpp transcription
            logger.info(f"Starting transcription process with model {model_size} and language {language}")
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Check if transcription was successful
            if result.returncode != 0:
                logger.error(f"Transcription failed: {result.stderr}")
                return jsonify({"error": "Transcription failed: " + result.stderr.strip()}), 500

            # Extract transcription output
            transcription = result.stdout.strip()
            logger.info("Transcription completed successfully")
            return jsonify({"transcription": transcription})

        except subprocess.TimeoutExpired:
            logger.error("Transcription process timed out")
            return jsonify({"error": "Transcription process timed out"}), 500
        except subprocess.SubprocessError as e:
            logger.error(f"Subprocess error: {str(e)}")
            return jsonify({"error": "Failed to run transcription process"}), 500

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500

    finally:
        # Clean up uploaded and converted files
        try:
            if 'input_path' in locals() and os.path.exists(input_path):
                os.remove(input_path)
                logger.info(f"Cleaned up input file: {input_path}")
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
                logger.info(f"Cleaned up converted file: {output_path}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {str(e)}")

if __name__ == "__main__":
    # Ensure the upload and converted folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(CONVERTED_FOLDER, exist_ok=True)
    
    app.run(host="0.0.0.0", port=5050, debug=True)
