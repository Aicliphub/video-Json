"""Transcriber Module

Creates timestamped segments from audio files using Deepgram API.
"""
import os
import json
import time
import requests
from dotenv import load_dotenv
load_dotenv()
import tempfile
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import requests

from src.utils.api_manager import ApiKeyManager
from src.utils.config import ConfigManager
from src.utils.storage import StorageManager

class Transcriber:
    """Handles transcription of audio to timestamped segments."""
    
    def __init__(self, 
                 config_manager: ConfigManager, # Keep for potential future use
                 storage_manager: StorageManager,
                 api_key: str,
                 transcriber_config: Dict[str, str]):
        """
        Initialize the transcriber.
        
        Args:
            config_manager: Configuration manager.
            storage_manager: Storage manager for saving transcripts.
            api_key: Deepgram API key.
            transcriber_config: Dictionary containing transcriber settings (e.g., 'model').
        """
        self.config_manager = config_manager
        self.storage_manager = storage_manager
        self.api_key = api_key
        self.config = transcriber_config
        self.model_name = self.config.get("model", "nova-3") # Get model from config
        self.max_retries = 3 # Could make this configurable too
        self.retry_delay = 2  # seconds
            
        if not self.api_key:
            raise ValueError("Deepgram API key is required for Transcriber.")
        if not self.model_name:
             raise ValueError("Deepgram model name is required for Transcriber.")
        print(f"Transcriber initialized (Model: {self.model_name}).")

    def download_audio(self, audio_url: str) -> str:
        """
        Download audio file from URL to temporary file.
        
        Args:
            audio_url: URL of the audio file
            
        Returns:
            Path to downloaded file
        """
        print(f"Downloading audio from {audio_url}...")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()
        
        # Download file
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Audio downloaded to {temp_path}")
        return temp_path
    
    def transcribe_audio(self, 
                        audio_path: str, 
                        script_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file to timestamped segments using Deepgram.
        
        Args:
            audio_path: Path to audio file
            script_text: Optional script text (not used for Deepgram)
            
        Returns:
            Dictionary with transcription information
        """
        print(f"Transcribing audio file: {audio_path}")
        
        with open(audio_path, "rb") as audio:
            # Read audio file
            audio_data = audio.read()
            
            # Call Deepgram API directly
            headers = {
                'Authorization': f'Token {self.api_key}', # Use passed API key
                'Content-Type': 'audio/wav'
            }
            params = {
                'model': self.model_name, # Use model from config
                'smart_format': 'true',
                'punctuate': 'true',
                'utterances': 'true'
            }
            
            response = requests.post(
                'https://api.deepgram.com/v1/listen',
                headers=headers,
                params=params,
                data=audio_data
            )
            response.raise_for_status()
            response = response.json()
        
        # Get utterance segments from Deepgram response
        segments = []
        for utterance in response["results"]["utterances"]:
            segments.append({
                "text": utterance["transcript"],
                "start_time": utterance["start"],
                "end_time": utterance["end"],
                "confidence": utterance["confidence"],
                "words": len(utterance["words"])
            })
        
        # Create transcription result
        transcription = {
            "segments": segments,
            "duration": segments[-1]["end_time"] if segments else 0,
            "word_count": len(segments),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcription_{timestamp}.json"
        self.storage_manager.save_json(transcription, filename, "scripts")
        
        return transcription
    
    def _post_process_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Post-process utterance segments (no longer needed as we get full utterances)
        """
        return segments
    
    def process_audio(self, 
                     audio_info: Dict[str, Any], 
                     script_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process audio to create timestamped segments.
        
        Args:
            audio_info: Audio information from TTSGenerator. Expected to have 'all_urls' (list of audio chunk URLs)
                         and optionally 'url' (primary/first chunk URL).
            script_data: Script data from ScriptWriter (currently not heavily used by this method post-refactor).
            
        Returns:
            Dictionary with combined transcription information from all audio chunks.
        """
        if not audio_info.get("all_urls") and not audio_info.get("url"):
            raise ValueError("audio_info must contain 'all_urls' or a primary 'url'.")

        audio_urls_to_process = audio_info.get("all_urls", [])
        if not audio_urls_to_process and audio_info.get("url"): # Fallback to single URL if all_urls is empty but url exists
            audio_urls_to_process = [audio_info["url"]]

        if not audio_urls_to_process:
            print("No audio URLs found to process in audio_info.")
            return {
                "segments": [],
                "duration": 0,
                "word_count": 0,
                "timestamp": datetime.now().isoformat(),
                "error": "No audio URLs provided"
            }
            
        all_segments_combined = []
        cumulative_duration = 0.0
        total_word_count = 0 # Will be based on transcribed words

        print(f"Processing {len(audio_urls_to_process)} audio chunk(s) for transcription...")

        for i, audio_url in enumerate(audio_urls_to_process):
            if not audio_url:
                print(f"Skipping empty audio URL for chunk {i+1}.")
                continue
            
            print(f"--- Transcribing audio chunk {i+1}/{len(audio_urls_to_process)} from {audio_url} ---")
            audio_path = None
            try:
                audio_path = self.download_audio(audio_url)
                chunk_transcription = self.transcribe_audio(audio_path)
                
                if chunk_transcription and chunk_transcription.get("segments"):
                    for segment in chunk_transcription["segments"]:
                        segment["start_time"] += cumulative_duration
                        segment["end_time"] += cumulative_duration
                        # 'part' is no longer relevant as it's a single script
                        all_segments_combined.append(segment)
                    
                    chunk_duration = chunk_transcription.get("duration", 0)
                    cumulative_duration += chunk_duration
                    # Deepgram provides word count per utterance, sum them up
                    total_word_count += sum(s.get("words", 0) for s in chunk_transcription["segments"])

                else:
                    print(f"Warning: Transcription for chunk {i+1} ({audio_url}) produced no segments.")

            except Exception as e:
                print(f"Error processing audio chunk {i+1} ({audio_url}): {e}")
                # Optionally, decide if one failed chunk should stop the whole process
            finally:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    print(f"Cleaned up temporary file: {audio_path}")
        
        final_transcription = {
            "segments": all_segments_combined,
            "duration": cumulative_duration,
            "word_count": total_word_count, # Based on actual transcribed words
            "timestamp": datetime.now().isoformat()
        }
        
        # Save the final combined transcription
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_transcription_{timestamp}.json"
        self.storage_manager.save_json(final_transcription, filename, "scripts")
        print(f"Full transcription saved: {filename}")

        return final_transcription
