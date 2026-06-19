import gradio as gr
import subprocess
import tempfile
import os
from pathlib import Path

def transcribe_romanian(audio_file):
    """
    Transcribe Romanian audio using whispermlx CLI tool with large-v3 model
    
    Args:
        audio_file: Path to audio file from Gradio
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

# Create the Gradio interface
with gr.Blocks(title="WhisperMLX - Transcriere Română", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎙️ Transcriere Audio în Română cu WhisperMLX
        Transcriere locală folosind modelul **large-v3** pentru limba română.
        Procesare complet locală pe MacBook-ul tău.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # Audio input
            audio_input = gr.Audio(
                label="Încarcă fișierul audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            # Transcribe button
            transcribe_btn = gr.Button(
                "Transcrie Audio", 
                variant="primary", 
                size="lg"
            )
            
            # Info about the model
            gr.Markdown(
                """
                ### 📋 Configurare
                - **Model**: large-v3
                - **Limbă**: Română (ro)
                - **Procesare**: Locală, fără internet
                """
            )
        
        with gr.Column(scale=2):
            # Output tabs
            with gr.Tabs():
                with gr.TabItem("Transcriere Completă"):
                    output_text = gr.Textbox(
                        label="Text transcris",
                        lines=10,
                        max_lines=30,
                        placeholder="Transcrierea va apărea aici..."
                    )
                
                with gr.TabItem("Detalii"):
                    segments_output = gr.Textbox(
                        label="Informații adiționale",
                        lines=8,
                        max_lines=20,
                        placeholder="Detalii despre transcriere..."
                    )
    
    # Status
    status_box = gr.Markdown("✓ Gata de transcriere")
    
    # Connect the button
    transcribe_btn.click(
        fn=transcribe_romanian,
        inputs=[audio_input],
        outputs=[output_text, segments_output]
    )
    
    # Usage instructions
    gr.Markdown(
        """
        ### 💡 Instrucțiuni
        1. **Instalați whispermlx**: `pip install whispermlx`
        2. **Formate suportate**: WAV, MP3, M4A, FLAC
        3. **Prima rulare**: Modelul large-v3 va fi descărcat automat (~3GB)
        4. **Performanță**: Optimizat pentru Apple Silicon (M1/M2/M3)
        5. **Confidențialitate**: Toată procesarea este locală, datele nu părăsesc dispozitivul
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

