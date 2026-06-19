#!/usr/bin/env python3
"""
Flask Transcription App using Faster-Whisper
Supports Apple Silicon (Metal) and NVIDIA GPUs with automatic CPU fallback
"""

import os
import sys
import platform
import traceback
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, session
from faster_whisper import WhisperModel
import torch
import tempfile
import uuid
import atexit
from werkzeug.utils import secure_filename
import time

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global variables for model caching
current_model = None
current_model_size = None
device = None
compute_type = None
device_info = None

# Create upload and temp directories
UPLOAD_FOLDER = Path('uploads')
TEMP_FOLDER = Path('temp')
UPLOAD_FOLDER.mkdir(exist_ok=True)
TEMP_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['TEMP_FOLDER'] = str(TEMP_FOLDER)

# Cleanup function for temporary files
def cleanup_temp_files():
    """Clean up temporary files on app exit"""
    for folder in [UPLOAD_FOLDER, TEMP_FOLDER]:
        if folder.exists():
            for file in folder.glob('*'):
                try:
                    file.unlink()
                except:
                    pass

atexit.register(cleanup_temp_files)

def detect_device_and_compute_type():
    """
    Detect available hardware and return appropriate device and compute type.
    Supports: Apple Metal (MPS), NVIDIA CUDA, CPU fallback.
    """
    global device, compute_type, device_info
    
    if device is not None:
        return device, compute_type
    
    system = platform.system()
    
    # Check for Apple Silicon (MPS/Metal support)
    if system == "Darwin" and torch.backends.mps.is_available():
        try:
            torch.zeros(1).to('mps')
            device = "cpu"
            compute_type = "int8"
            device_info = f"🔵 Apple Silicon (CPU with int8 optimization)"
            return device, compute_type
        except Exception:
            pass
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        try:
            torch.zeros(1).to('cuda')
            device = "cuda"
            compute_type = "float16"
            device_info = f"🟢 NVIDIA GPU (CUDA - float16)"
            return device, compute_type
        except Exception:
            pass
    
    # Fallback to CPU
    device = "cpu"
    compute_type = "int8"
    device_info = f"🟡 CPU (int8)"
    return device, compute_type

def load_whisper_model(model_size):
    """
    Load or reuse the Whisper model based on the selected size.
    Handles device detection and model caching.
    """
    global current_model, current_model_size, device, compute_type, device_info
    
    # Detect device if not already done
    if device is None:
        detect_device_and_compute_type()
    
    # Reload model only if size changed
    if current_model is None or current_model_size != model_size:
        try:
            print(f"Loading {model_size} model... This may take a moment.")
            current_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            current_model_size = model_size
        except Exception as e:
            # Fallback to CPU if initial device fails
            if device != "cpu":
                print(f"Failed to load with {device}. Falling back to CPU.")
                device = "cpu"
                compute_type = "int8"
                device_info = "🟡 CPU (int8) - Fallback"
                
                try:
                    current_model = WhisperModel(
                        model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    current_model_size = model_size
                except Exception as e2:
                    print(f"Error loading model: {str(e2)}")
                    return None, str(e2)
            else:
                print(f"Error loading model: {str(e)}")
                return None, str(e)
    
    return current_model, None

def validate_audio_file(file_path):
    """
    Validate the uploaded audio file.
    Returns the file path if valid, or error message.
    """
    if not file_path:
        return None, "No file path provided"
    
    path = Path(file_path)
    
    if not path.exists():
        return None, "File does not exist"
    
    if not path.is_file():
        return None, "Path is not a file"
    
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    if path.suffix.lower() not in valid_extensions:
        return None, f"Invalid file format. Supported formats: {', '.join(valid_extensions)}"
    
    return str(path.absolute()), None

@app.route('/')
def index():
    """Render main page"""
    # Detect hardware on first load
    if device_info is None:
        detect_device_and_compute_type()
    
    return render_template('index.html', 
                         device_info=device_info,
                         current_model_size=current_model_size)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file:
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            file_extension = Path(original_filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = Path(app.config['UPLOAD_FOLDER']) / unique_filename
            
            # Save file
            file.save(str(file_path))
            
            # Validate file
            valid_path, error = validate_audio_file(str(file_path))
            if error:
                file_path.unlink(missing_ok=True)
                return jsonify({'error': error}), 400
            
            return jsonify({
                'success': True,
                'filename': unique_filename,
                'original_name': original_filename
            })
    
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Handle transcription request"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        model_size = data.get('model_size', 'large-v3')
        language = data.get('language', None)
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Construct file path
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        
        # Validate file
        valid_path, error = validate_audio_file(str(file_path))
        if error:
            return jsonify({'error': error}), 400
        
        # Load model
        model, error = load_whisper_model(model_size)
        if error:
            return jsonify({'error': f'Model loading error: {error}'}), 500
        
        # Set language parameter
        lang_param = language if language and language != "auto" else None
        
        # Perform transcription
        print(f"Transcribing audio file: {filename}")
        segments, info = model.transcribe(
            str(valid_path),
            language=lang_param,
            beam_size=5,
            vad_filter=True
        )
        
        # Collect transcription text
        full_text = []
        for segment in segments:
            full_text.append(segment.text)
        
        transcription = " ".join(full_text)
        
        # Save transcription to temp file for download
        transcription_filename = f"transcription_{uuid.uuid4()}.txt"
        transcription_path = Path(app.config['TEMP_FOLDER']) / transcription_filename
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        return jsonify({
            'success': True,
            'transcription': transcription,
            'download_filename': transcription_filename,
            'language_detected': info.language if hasattr(info, 'language') else None
        })
    
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'Transcription error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download transcription file"""
    try:
        file_path = Path(app.config['TEMP_FOLDER']) / filename
        
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            str(file_path),
            as_attachment=True,
            download_name='transcription.txt',
            mimetype='text/plain'
        )
    
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get current app status"""
    return jsonify({
        'device_info': device_info,
        'current_model': current_model_size,
        'device': device,
        'compute_type': compute_type
    })

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up uploaded and temp files"""
    try:
        data = request.get_json()
        filename = data.get('filename') if data else None
        
        if filename:
            # Clean specific file
            file_path = Path(app.config['UPLOAD_FOLDER']) / filename
            file_path.unlink(missing_ok=True)
        
        return jsonify({'success': True})
    
    except Exception as e:
        return jsonify({'error': f'Cleanup error: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File is too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Detect hardware on startup
    detect_device_and_compute_type()
    print(f"Hardware detected: {device_info}")
    print(f"Starting Flask transcription server...")
    
    # Run the app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5003,
        debug=True  # Set to False in production
    )
