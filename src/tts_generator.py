"""
Text-to-Speech Generator Module

Converts text scripts to natural-sounding audio using Deepgram API.
"""
from typing import Dict, Any, Optional
import requests
import json
import boto3
from datetime import datetime
from datetime import datetime as dt
import random
import string
import tempfile
import os

def _upload_to_r2(audio_data, r2_config: Dict[str, str], extension="wav"):
    """Upload audio data to R2 storage and return public URL"""
    if not r2_config or not all(k in r2_config for k in ['endpoint_url', 'access_key_id', 'secret_access_key', 'bucket_name', 'public_domain']):
        print("Error: R2 configuration is incomplete.")
        return None
        
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=r2_config['endpoint_url'],
            aws_access_key_id=r2_config['access_key_id'],
            aws_secret_access_key=r2_config['secret_access_key']
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
        object_name = f"tts_output_{timestamp}_{random_str}.{extension}"
        
        s3.put_object(
            Bucket=r2_config['bucket_name'],
            Key=object_name,
            Body=audio_data,
            ContentType=f'audio/{extension}'
        )
        
        return f"https://{r2_config['public_domain']}/{object_name}"
    except Exception as e:
        print(f"Error uploading to R2: {e}")
        return None

def _text_to_speech_deepgram(text: str,
                            api_key: str,
                            model: str,
                            r2_config: Dict[str, str]) -> tuple[int, Optional[str]]:
    """
    Converts text to speech using Deepgram API.

    Args:
        text: The text to convert (up to 2000 characters)
        api_key: Deepgram API key
        model: Deepgram voice model name
        r2_config: R2 storage configuration

    Returns:
        tuple: (status_code, audio_url) where:
               status_code is HTTP status code
               audio_url is public R2 URL if upload succeeded, else None
    """
    if len(text) > 2000:
        raise ValueError("Deepgram text length cannot exceed 2000 characters")

    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": text
    }

    try:
        response = requests.post(
            f"https://api.deepgram.com/v1/speak?model={model}",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        audio_url = None
        if response.status_code == 200:
            audio_url = _upload_to_r2(response.content, r2_config, extension="mp3")
            if audio_url:
                print(f"Audio uploaded to R2: {audio_url}")
            else:
                print("Audio generation succeeded but R2 upload failed")

        return (response.status_code, audio_url)

    except requests.exceptions.RequestException as e:
        print(f"Deepgram API call failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            return (e.response.status_code, None)
        return (500, None)

def _text_to_speech_allvoicelab(text: str, 
                                api_key: str, 
                                endpoint: str, 
                                voice_id: int, 
                                model_id: str, 
                                r2_config: Dict[str, str],
                                language_code: str = None, 
                                speed: float = 1.0):
    """
    Converts text to speech using the AllVoiceLab API.

    Args:
        text (str): The text to convert (up to 5000 characters).
        voice_id (int): The ID of the voice to use.
        model_id (str, optional): The model ID. Defaults to "tts-multilingual".
        language_code (str, optional): The language code (e.g., 'en', 'fr'). Defaults to None.
        speed (float, optional): The speech speed (0.5 to 1.5). Defaults to 1.0.

    Returns:
        tuple: (response_status, audio_url) where:
               response_status is the HTTP status code
               audio_url is the public R2 URL if upload succeeded, else None
    """
    """Internal function for AllVoiceLab TTS call."""
    if len(text) > 5000:
        raise ValueError("Text length cannot exceed 5000 characters.")
    if not (0.5 <= speed <= 1.5):
        raise ValueError("Speed must be between 0.5 and 1.5.")

    headers = {
        "ai-api-key": api_key, # Use passed API key
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice_id": voice_id,
        "model_id": model_id,
        "voice_settings": {
            "speed": speed
        }
    }

    if language_code:
        payload["language_code"] = language_code

    try:
        response = requests.post(endpoint, headers=headers, json=payload) # Use passed endpoint
        response.raise_for_status()
        
        audio_url = None
        if response.status_code == 200:
            # Pass R2 config to upload function
            audio_url = _upload_to_r2(response.content, r2_config) 
            if audio_url:
                print(f"Audio uploaded to R2: {audio_url}")
            else:
                print("Audio generation succeeded but R2 upload failed")
        
        return (response.status_code, audio_url)
        
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            return (e.response.status_code, None)
        return (500, None)

class TTSGenerator:
    """Handles text-to-speech conversion using configured provider."""
    
    def __init__(self, 
                 tts_provider: str, 
                 tts_config: Dict[str, Any], 
                 api_key: str, 
                 r2_config: Dict[str, str]):
        """
        Initialize the TTS generator.
        
        Args:
            tts_provider: Name of the TTS provider (e.g., 'allvoicelab').
            tts_config: Dictionary with provider-specific config (endpoint, voice_id, model).
            api_key: API key for the provider.
            r2_config: R2 storage configuration dictionary.
        """
        self.provider = tts_provider
        self.config = tts_config
        self.api_key = api_key
        self.r2_config = r2_config
        
        # Validate required config based on provider
        if self.provider == "allvoicelab":
            if not all(k in self.config for k in ["allvoicelab_endpoint", "voice_id", "model"]):
                 raise ValueError("Missing required AllVoiceLab configuration (endpoint, voice_id, model).")
        elif self.provider == "deepgram":
            if not all(k in self.config for k in ["model"]):
                 raise ValueError("Missing required Deepgram configuration (model).")
        else:
            raise ValueError(f"Unsupported TTS provider: {self.provider}")

    def generate_audio(self, text: str, max_chars: int = 4000) -> Dict[str, Any]:
        """
        Generate audio from text, splitting long text into chunks.
        
        Args:
            text: Text to convert to speech
            max_chars: Maximum characters per API call
            
        Returns:
            Dictionary with audio information including URL
        """
        print(f"Generating audio for text ({len(text.split())} words)...")
        
        # Split text into chunks if too long
        if len(text) > max_chars:
            print(f"Splitting long text into chunks...")
            chunks = []
            current_chunk = ""
            for sentence in text.split('.'):
                if len(current_chunk) + len(sentence) < max_chars:
                    current_chunk += sentence + '.'
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence + '.'
            if current_chunk:
                chunks.append(current_chunk.strip())
        else:
            chunks = [text]
        
        # Process each chunk
        audio_urls = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            status_code = 500
            audio_url = None

            if self.provider == "allvoicelab":
                status_code, audio_url = _text_to_speech_allvoicelab(
                    text=chunk,
                    api_key=self.api_key,
                    endpoint=self.config["allvoicelab_endpoint"],
                    voice_id=self.config["voice_id"],
                    model_id=self.config["model"],
                    r2_config=self.r2_config,
                    language_code='en',
                    speed=1.0
                )
            elif self.provider == "deepgram":
                status_code, audio_url = _text_to_speech_deepgram(
                    text=chunk,
                    api_key=self.api_key,
                    model=self.config["model"],
                    r2_config=self.r2_config
                )
            
            if status_code == 200 and audio_url:
                audio_urls.append(audio_url)
            else:
                error_msg = f"Failed to generate audio chunk (Status: {status_code})"
                if status_code == 422:
                    error_msg += " - Text may contain unsupported characters or formatting"
                raise Exception(error_msg)
        
        # For now just return first chunk's URL
        # In a real implementation you'd concatenate the audio files
        return {
            "url": audio_urls[0],
            "word_count": len(text.split()),
            "timestamp": datetime.now().isoformat(),
            "url": audio_urls[0] if audio_urls else None, # Return first URL for single chunk case
            "all_urls": audio_urls, # Keep all URLs if needed later
            "word_count": len(text.split()),
            "timestamp": datetime.now().isoformat(),
            "chunks": len(chunks)
        }

    def generate_audio_from_script(self, script_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate audio for a single full script.
        
        Args:
            script_data: Script data from ScriptWriter, expected to have a 'full_script' key.
            
        Returns:
            Dictionary containing audio information for the full script.
            Example: {'url': '...', 'all_urls': ['...'], 'word_count': N, ...}
        """
        if "full_script" not in script_data or not isinstance(script_data["full_script"], str):
            raise ValueError("script_data must contain a 'full_script' string.")
            
        full_script_text = script_data["full_script"]
        
        print("Generating audio for the full script...")
        
        if not full_script_text.strip():
            print("--- Script text is empty. Skipping audio generation. ---")
            return {
                "url": None,
                "all_urls": [],
                "word_count": 0,
                "timestamp": datetime.now().isoformat(),
                "chunks": 0,
                "error": "Empty script text"
            }

        try:
            audio_info = self.generate_audio(full_script_text)
            
            if not audio_info.get("url"):
                # This case should ideally be handled within generate_audio or by it raising an exception
                print(f"--- Warning: Audio generation for full script succeeded but no primary URL was returned. ---")
                # Ensure a consistent structure even on failure to get a URL
                return {
                    "url": None,
                    "all_urls": audio_info.get("all_urls", []),
                    "word_count": audio_info.get("word_count", 0),
                    "timestamp": datetime.now().isoformat(),
                    "chunks": audio_info.get("chunks", 0),
                    "error": "No primary URL returned from generate_audio"
                }

            print(f"--- Full script audio generated: {audio_info['url']} ---")
            return audio_info # This already contains url, all_urls, word_count, timestamp, chunks

        except Exception as e:
            print(f"--- Error generating audio for full script: {e} ---")
            # Return a structured error
            return {
                "url": None,
                "all_urls": [],
                "word_count": len(full_script_text.split()), # Best effort word count
                "timestamp": datetime.now().isoformat(),
                "chunks": 0, # Assuming 0 chunks if generation failed
                "error": str(e)
            }
