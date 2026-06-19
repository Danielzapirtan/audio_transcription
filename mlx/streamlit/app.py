#!/usr/bin/env python3
"""
Streamlit Transcription App using Faster-Whisper
Supports Apple Silicon (Metal) and NVIDIA GPUs with automatic CPU fallback
"""

import os
import sys
import platform
import traceback
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel
import torch
import tempfile

# Page configuration
st.set_page_config(
    page_title="Audio Transcription",
    page_icon="🎙️",
    layout="centered"
)

# Global variables for model caching
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_model_size' not in st.session_state:
    st.session_state.current_model_size = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'compute_type' not in st.session_state:
    st.session_state.compute_type = None

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
    # Detect device if not already done
    if st.session_state.device is None:
        device, compute_type = detect_device_and_compute_type()
        st.session_state.device = device
        st.session_state.compute_type = compute_type
        
        # Store device info for display
        if device == "cuda":
            st.session_state.device_info = f"🟢 NVIDIA GPU (CUDA - {compute_type})"
        elif platform.system() == "Darwin" and torch.backends.mps.is_available():
            st.session_state.device_info = f"🔵 Apple Silicon (CPU with {compute_type} optimization)"
        else:
            st.session_state.device_info = f"🟡 CPU ({compute_type})"
    
    # Reload model only if size changed
    if (st.session_state.current_model is None or 
        st.session_state.current_model_size != model_size):
        
        try:
            with st.spinner(f"Loading {model_size} model... This may take a moment on first run."):
                st.session_state.current_model = WhisperModel(
                    model_size,
                    device=st.session_state.device,
                    compute_type=st.session_state.compute_type
                )
                st.session_state.current_model_size = model_size
        except Exception as e:
            # Fallback to CPU if initial device fails
            if st.session_state.device != "cpu":
                st.warning(f"Failed to load with {st.session_state.device}. Falling back to CPU.")
                st.session_state.device = "cpu"
                st.session_state.compute_type = "int8"
                st.session_state.device_info = "🟡 CPU (int8) - Fallback"
                
                try:
                    st.session_state.current_model = WhisperModel(
                        model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    st.session_state.current_model_size = model_size
                except Exception as e2:
                    st.error(f"Error loading model: {str(e2)}")
                    return None
            else:
                st.error(f"Error loading model: {str(e)}")
                return None
    
    return st.session_state.current_model

def validate_audio_file(file_path):
    """
    Validate the uploaded audio file.
    Returns the file path if valid.
    """
    if not file_path:
        st.error("No file path provided")
        return None
    
    path = Path(file_path)
    
    if not path.exists():
        st.error("File does not exist")
        return None
    
    if not path.is_file():
        st.error("Path is not a file")
        return None
    
    valid_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
    if path.suffix.lower() not in valid_extensions:
        st.error(f"Invalid file format. Supported formats: {', '.join(valid_extensions)}")
        return None
    
    return str(path.absolute())

def transcribe_audio(audio_file_path, model_size, language):
    """
    Transcribe audio file using the Whisper model.
    Returns the transcription text.
    """
    try:
        model = load_whisper_model(model_size)
        
        if model is None:
            return None
        
        file_path = validate_audio_file(audio_file_path)
        if file_path is None:
            return None
        
        # Set language parameter
        lang_param = language if language != "Auto-detect" else None
        
        with st.spinner("Transcribing audio... This may take a few minutes."):
            segments, _ = model.transcribe(
                file_path,
                language=lang_param,
                beam_size=5,
                vad_filter=True
            )
            
            full_text = []
            for segment in segments:
                full_text.append(segment.text)
            
            return " ".join(full_text)
            
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def save_transcription(text, output_path="transcription.txt"):
    """
    Save transcription to a text file.
    Returns the file content as bytes for download.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    with open(output_path, 'rb') as f:
        return f.read()

def main():
    """
    Main Streamlit application.
    """
    st.title("🎙️ Audio Transcription App")
    st.markdown("Powered by Faster-Whisper")
    
    # Display device information
    #if st.session_state.device_info:
    #   st.info(f"**Hardware:** {st.session_state.device_info}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model size selection
        model_size = st.selectbox(
            "Model Size",
            options=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
            index=5,  # Default to large-v3
            help="Larger models are more accurate but slower"
        )
        
        # Language selection
        languages = ["Auto-detect", "en", "ro", "fr", "de", "es", "it", "pt", "nl", "ru", "zh", "ja", "ko"]
        language = st.selectbox(
            "Language",
            options=languages,
            index=2 if "ro" in languages else 0,  # Default to Romanian if available
            help="Select source language or auto-detect"
        )
        
        st.divider()
        
        st.markdown("### 📋 Model Information")
        st.markdown(f"- Current model: **{model_size}**")
        st.markdown(f"- Language: **{language}**")
        
        if st.session_state.current_model_size:
            st.markdown(f"- Loaded model: **{st.session_state.current_model_size}**")
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format=f'audio/{Path(uploaded_file.name).suffix[1:]}')
        
        # Transcribe button
        if st.button("🔍 Transcribe Audio", type="primary", use_container_width=True):
            transcription = transcribe_audio(temp_path, model_size, language)
            
            if transcription:
                st.success("✅ Transcription completed successfully!")
                
                # Display transcription
                st.markdown("### 📝 Transcription")
                st.text_area(
                    "Transcription Result",
                    transcription,
                    height=300,
                    key="transcription_result"
                )
                
                # Save and provide download button
                file_content = save_transcription(transcription)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    st.download_button(
                        label="📥 Download as TXT",
                        data=file_content,
                        file_name="transcription.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Copy to clipboard (using JavaScript)
                    st.markdown(f"""
                    <button onclick="navigator.clipboard.writeText(`{transcription.replace('`', '\\`')}`)" 
                            style="width:100%; padding:0.5rem; border-radius:0.5rem; border:1px solid #ccc; background-color:#f0f2f6; cursor:pointer;">
                        📋 Copy to Clipboard
                    </button>
                    """, unsafe_allow_html=True)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <small>Transcription app using Faster-Whisper | 
        Supports GPU acceleration on Apple Silicon and NVIDIA GPUs</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
