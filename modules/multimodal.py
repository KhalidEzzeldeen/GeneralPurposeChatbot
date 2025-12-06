import os
from faster_whisper import WhisperModel
import ollama
from PIL import Image
import io

def transcribe_audio(file_path: str, model_size="medium") -> list:
    """
    Transcribe audio file and return segments with timestamps.
    Returns: [{'text': str, 'start': float, 'end': float}, ...]
    """
    try:
        print(f"Loading Whisper model ({model_size})...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print(f"Transcribing {file_path}...")
        segments, info = model.transcribe(file_path, beam_size=5)
        
        transcript_data = []
        for segment in segments:
            transcript_data.append({
                "text": segment.text,
                "start": segment.start,
                "end": segment.end
            })
            
        print(f"Transcription complete. {len(transcript_data)} segments.")
        return transcript_data
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return []

def describe_image(image_path: str, model="qwen2.5-vl") -> str:
    """
    Generate a description for an image using Qwen2.5-VL via Ollama.
    """
    try:
        print(f"Analyzing image {image_path} with {model}...")
        
        with open(image_path, 'rb') as file:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                    'role': 'user',
                    'content': 'Extract all text, fields, and numbers from this image and describe it in detail for searchability.',
                    'images': [file.read()],
                    }
                ]
            )
        
        description = response['message']['content']
        print("Image analysis complete.")
        return description
    except Exception as e:
        print(f"Error describing image with {model}: {e}")
        # Fallback to Llava if Qwen-VL fails or is not present, though we should pin to Qwen-VL
        return ""
