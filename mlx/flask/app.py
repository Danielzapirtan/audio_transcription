from flask import Flask, render_template, request, jsonify
import subprocess
import tempfile
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'wma', 'aac'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_romanian(audio_file_path):
    """
    Transcribe Romanian audio using whispermlx CLI tool with large-v3 model
    
    Args:
        audio_file_path: Path to audio file
    """
    if not audio_file_path or not os.path.exists(audio_file_path):
        return "Please upload an audio file.", ""
    
    try:
        # Build the whispermlx command
        cmd = [
            "whispermlx",
            "--model", "large-v3",
            "--language", "ro",
            "--hf_token", os.getenv("HF_TOKEN", ""),
            audio_file_path
        ]
        
        # Run whispermlx as subprocess and capture stdout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1500  # 25 minute timeout for large files
        )
        
        # Check for errors
        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error occurred"
            return f"Error: {error_msg}", ""
        
        # Get transcription from stdout
        full_text = result.stdout.strip()
        
        if not full_text:
            return "No transcription produced. Check if the audio file contains speech.", ""
        
        # Format with metadata for the details tab
        details = f"Model: large-v3\nLanguage: Romanian (ro)\nStatus: Success\n\n{full_text}"
        
        return full_text, details
    
    except subprocess.TimeoutExpired:
        return "Error: Transcription timed out (25 minute limit)", ""
    except FileNotFoundError:
        return "Error: whispermlx not found. Please install with: pip install whispermlx", ""
    except Exception as e:
        return f"Error during transcription: {str(e)}", ""

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handle audio file upload and transcription"""
    # Check if a file was uploaded
    if 'audio_file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No audio file provided',
            'transcription': '',
            'details': ''
        }), 400
    
    file = request.files['audio_file']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected',
            'transcription': '',
            'details': ''
        }), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
            'transcription': '',
            'details': ''
        }), 400
    
    try:
        # Create a unique filename to avoid collisions
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit('.', 1)[1].lower() if '.' in original_filename else 'wav'
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the uploaded file
        file.save(file_path)
        
        # Perform transcription
        transcription, details = transcribe_romanian(file_path)
        
        # Clean up - remove the temporary file
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors
        
        # Check if transcription was successful
        if transcription.startswith("Error:"):
            return jsonify({
                'success': False,
                'error': transcription,
                'transcription': '',
                'details': details
            }), 500
        
        return jsonify({
            'success': True,
            'error': '',
            'transcription': transcription,
            'details': details
        })
    
    except Exception as e:
        # Clean up on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'transcription': '',
            'details': ''
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    app.run(
        host='0.0.0.0',
        port=7860,
        debug=True
    )
