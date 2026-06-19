import streamlit as st
import subprocess
import tempfile
import os
from pathlib import Path

def transcribe_romanian(audio_file):
    """
    Transcribe Romanian audio using whispermlx CLI tool with large-v3 model
    
    Args:
        audio_file: Path to audio file from Streamlit
    """
    if audio_file is None:
        return "Please upload an audio file.", ""
    
    try:
        # Build the whispermlx command
        cmd = [
            "whispermlx",
            "--model", "large-v3",
            "--language", "ro",
            "--hf_token", os.getenv("HF_TOKEN"),
            audio_file
        ]
        
        # Run whispermlx as subprocess and capture stdout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1500  # 5 minute timeout for large files
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
        return "Error: Transcription timed out (5 minute limit)", ""
    except FileNotFoundError:
        return "Error: whispermlx not found. Please install with: pip install whispermlx", ""
    except Exception as e:
        return f"Error during transcription: {str(e)}", ""

# Streamlit UI
st.set_page_config(
    page_title="WhisperMLX - Transcriere Română",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.markdown(
    """
    # 🎙️ Transcriere Audio în Română cu WhisperMLX
    Transcriere locală folosind modelul **large-v3** pentru limba română.
    Procesare complet locală pe MacBook-ul tău.
    """
)

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    # Audio input
    audio_file = st.file_uploader(
        "Încarcă fișierul audio",
        type=["wav", "mp3", "m4a", "flac"],
        help="Formate suportate: WAV, MP3, M4A, FLAC"
    )
    
    # Record audio option (using microphone)
    audio_bytes = st.audio_input("Înregistrează audio")
    
    # Transcribe button
    transcribe_btn = st.button(
        "Transcrie Audio",
        type="primary",
        use_container_width=True
    )
    
    # Info about the model
    st.markdown(
        """
        ### 📋 Configurare
        - **Model**: large-v3
        - **Limbă**: Română (ro)
        - **Procesare**: Locală, fără internet
        """
    )

with col2:
    # Create tabs for output
    tab1, tab2 = st.tabs(["Transcriere Completă", "Detalii"])
    
    with tab1:
        output_text = st.text_area(
            label="Text transcris",
            height=250,
            placeholder="Transcrierea va apărea aici...",
            key="transcription"
        )
    
    with tab2:
        segments_output = st.text_area(
            label="Informații adiționale",
            height=200,
            placeholder="Detalii despre transcriere...",
            key="details"
        )

# Status
status_placeholder = st.empty()
status_placeholder.markdown("✓ Gata de transcriere")

# Process when button is clicked
if transcribe_btn:
    # Determine which audio source to use
    audio_path = None
    temp_file = None
    
    if audio_file is not None:
        # Save uploaded file to temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix)
        temp_file.write(audio_file.getvalue())
        temp_file.close()
        audio_path = temp_file.name
    
    elif audio_bytes is not None:
        # Save recorded audio to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.write(audio_bytes)
        temp_file.close()
        audio_path = temp_file.name
    
    if audio_path is None:
        st.error("Vă rugăm să încărcați sau să înregistrați un fișier audio.")
    else:
        with st.spinner("Se transcrie audio... Acest proces poate dura câteva minute."):
            try:
                # Update status
                status_placeholder.markdown("🔄 Se procesează...")
                
                # Call transcription function
                full_text, details = transcribe_romanian(audio_path)
                
                # Update outputs
                st.session_state.transcription = full_text
                st.session_state.details = details
                
                # Update status
                if full_text.startswith("Error"):
                    status_placeholder.markdown("❌ Eroare la transcriere")
                    st.error(full_text)
                else:
                    status_placeholder.markdown("✅ Transcriere finalizată cu succes!")
                    
            except Exception as e:
                st.error(f"Eroare: {str(e)}")
                status_placeholder.markdown("❌ Eroare la transcriere")
            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file.name):
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass

# Usage instructions
st.markdown(
    """
    ---
    ### 💡 Instrucțiuni
    1. **Instalați whispermlx**: `pip install whispermlx`
    2. **Formate suportate**: WAV, MP3, M4A, FLAC
    3. **Prima rulare**: Modelul large-v3 va fi descărcat automat (~3GB)
    4. **Performanță**: Optimizat pentru Apple Silicon (M1/M2/M3)
    5. **Confidențialitate**: Toată procesarea este locală, datele nu părăsesc dispozitivul
    """
)
