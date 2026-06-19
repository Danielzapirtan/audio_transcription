#!/usr/bin/env python3
"""
Gradio Transcription App using Faster-Whisper
Supports Apple Silicon (Metal) and NVIDIA GPUs with automatic CPU fallback
"""

import os
import sys
import platform
import traceback
from pathlib import Path
import gradio as gr
from faster_whisper import WhisperModel
import torch
import tempfile

# Global variables for model caching
current_model = None
current_model_size = None
device = None
compute_type = None
device_info = ""

def detect_device_and_compute_type():
    """
    Detect available hardware and return appropriate device and compute type.
    Supports: Apple Metal (MPS), NVIDIA CUDA, CPU fallback.
    """
    system = platform.system()
    
    # Check for Apple Silicon (MPS/Metal support)
    if system == "Darwin" and torch.backends.mps.is_available():
        try:
            # Test MPS availability
            torch.zeros(1).to('mps')
            return "cpu", "int8"  # faster-whisper uses CPU with int8 for best compatibility
        except Exception:
            pass
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        try:
            # Test CUDA availability
            torch.zeros(1).to('cuda')
            return "cuda", "float16"
        except Exception:
            pass
    
    # Fallback to CPU
    return "cpu", "int8"

def load_whisper_model(model_size):
    """
    Load or reuse the Whisper model based on the selected size.
    Handles device detection and model caching.
    """
    global current_model, current_model_size, device, compute_type, device_info
    
    # Detect device if not already done
    if device is None:
        device, compute_type = detect_device_and_compute_type()
        
        # Store device info for display
        if device == "cuda":
            device_info = f"🟢 NVIDIA GPU (CUDA - {compute_type})"
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            device_info = f"🔵 Apple Silicon (CPU with {compute_type} optimization)"
        else:
            device_info = f"🟡 CPU ({compute_type})"
    
    # Reload model only if size changed
    if current_model is None or current_model_size != model_size:
        
        try:
            print(f"Loading {model_size} model... This may take a moment on first run.")
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
                    raise Exception(f"Error loading model: {str(e2)}")
            else:
                raise Exception(f"Error loading model: {str(e)}")
    
    return current_model

def validate_audio_file(file_path):
    """
    Validate the uploaded audio file.
    Returns the file path if valid.
    """
    if not file_path:
        raise ValueError("No file path provided")
    
    path = Path(file_path)
    
    if not path.exists():
        raise ValueError("File does not exist")
    
    if not path.is_file():
        raise ValueError("Path is not a file")
    
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")
    
    return str(path.absolute())

def transcribe_audio(audio_file, model_size, language, progress=gr.Progress()):
    """
    Transcribe audio file using the Whisper model.
    Returns the transcription text and status updates.
    """
    global device_info
    
    try:
        if audio_file is None:
            return "Please upload an audio file first.", "", device_info
        
        progress(0.1, desc="Loading model...")
        model = load_whisper_model(model_size)
        
        if model is None:
            return "Error loading model. Please try again.", "", device_info
        
        # Validate file path
        file_path = validate_audio_file(audio_file)
        
        # Set language parameter
        lang_param = language if language != "Auto-detect" else None
        
        progress(0.2, desc="Transcribing audio... This may take a few minutes.")
        print("Transcribing audio... This may take a few minutes.")
        
        # Create a progress callback for the transcribe function
        def progress_callback(progress_info):
            # progress_info is a tuple of (completed, total)
            if hasattr(progress_info, '__len__') and len(progress_info) == 2:
                completed, total = progress_info
                if total > 0:
                    # Map from 0.2 to 0.9 for the transcription phase
                    fraction = 0.2 + (completed / total) * 0.7
                    progress(fraction, desc=f"Transcribing... {completed}/{total} segments")
        
        segments, _ = model.transcribe(
            file_path,
            language=lang_param,
            beam_size=5,
            vad_filter=True,
            log_progress=True
        )
        
        progress(0.9, desc="Processing transcription...")
        full_text = []
        for segment in segments:
            full_text.append(segment.text)
        
        transcription = " ".join(full_text)
        
        progress(0.95, desc="Saving transcription...")
        # Save transcription to temporary file for download
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(transcription)
            temp_path = tmp_file.name
        
        progress(1.0, desc="Transcription complete!")
        return transcription, temp_path, device_info
            
    except Exception as e:
        error_msg = f"Transcription error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return error_msg, "", device_info

def create_app():
    """
    Create and configure the Gradio interface.
    """
    # Detect initial device info
    global device_info
    if not device_info:
        detect_device_and_compute_type()
    
    with gr.Blocks(title="Audio Transcription", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎙️ Audio Transcription App
            Powered by Faster-Whisper
            """
        )
        
        # Device information display
        device_display = gr.Markdown(f"**Hardware:** {device_info}" if device_info else "")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                
                # Model size selection
                model_size = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="large-v3",
                    label="Model Size",
                    info="Larger models are more accurate but slower"
                )
                
                # Language selection
                language = gr.Dropdown(
                    choices=["Auto-detect", "en", "ro", "fr", "de", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko"],
                    value="ro",
                    label="Language",
                    info="Select source language or auto-detect"
                )
                
                gr.Markdown("### 📋 Model Information")
                model_info = gr.Markdown(f"- Selected model: **large-v3**\n- Language: **ro**")
            
            with gr.Column(scale=2):
                # Audio input
                audio_input = gr.Audio(
                    label="Upload Audio File",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                # Transcribe button
                transcribe_btn = gr.Button("🔍 Transcribe Audio", variant="primary", size="lg")
                
                # Transcription output
                transcription_output = gr.Textbox(
                    label="📝 Transcription",
                    placeholder="Transcription will appear here...",
                    lines=10,
                    interactive=False
                )
                
                # Download button for transcription
                download_output = gr.File(
                    label="📥 Download Transcription",
                    visible=True
                )
        
        # Footer
        gr.Markdown(
            """
            ---
            <div style='text-align: center; color: #888;'>
                <small>Transcription app using Faster-Whisper | 
                Supports GPU acceleration on Apple Silicon and NVIDIA GPUs</small>
            </div>
            """
        )
        
        # Update model info when selections change
        def update_model_info(model_size, language):
            global device_info
            return f"- Selected model: **{model_size}**\n- Language: **{language}**\n- Hardware: {device_info}"
        
        model_size.change(
            update_model_info,
            inputs=[model_size, language],
            outputs=[model_info]
        )
        
        language.change(
            update_model_info,
            inputs=[model_size, language],
            outputs=[model_info]
        )
        
        # Handle transcription
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, model_size, language],
            outputs=[transcription_output, download_output, device_display],
            show_progress=True
        )
        
        # Update device display on load
        app.load(
            fn=lambda: f"**Hardware:** {device_info}" if device_info else "",
            outputs=[device_display]
        )
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",  # Make accessible on local network
        server_port=7860,
        share=False,  # Set to True to create a public link
        debug=True
    )
