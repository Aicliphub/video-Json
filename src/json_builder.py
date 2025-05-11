"""
JSON Builder Module

Creates and updates the JSON file for Remotion.
"""
import os
import json
import random # Added for random effect selection
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from src.utils.storage import StorageManager

class JsonBuilder:
    """Handles creation and updating of the JSON file for Remotion."""
    
    def __init__(self, storage_manager: StorageManager):
        """
        Initialize the JSON builder.
        
        Args:
            storage_manager: Storage manager for saving JSON
        """
        self.storage_manager = storage_manager
        self.unique_id = str(uuid.uuid4())
        self.filename = f"video_{self.unique_id}.json"
        self.error_filename = f"video_error_{self.unique_id}.json"
        self.json_data = self._create_initial_json()
    
    def _create_initial_json(self) -> Dict[str, Any]:
        """
        Create initial JSON structure.
        
        Returns:
            Initial JSON data
        """
        return {
            "metadata": {
                "title": "",
                "description": "",
                "topic": "",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "audio": {
                "url": "",
                "duration": 0
            },
            "script": {
                "full_text": "",
                "word_count": 0
            },
            "segments": []
        }
    
    def update_metadata(self, 
                       title: str, 
                       description: str, 
                       topic: str) -> None:
        """
        Update metadata in the JSON and auto-save.
        
        Args:
            title: Video title
            description: Video description
            topic: Video topic
        """
        self.json_data["metadata"]["title"] = title
        self.json_data["metadata"]["description"] = description
        self.json_data["metadata"]["topic"] = topic
        self.json_data["metadata"]["updated_at"] = datetime.now().isoformat()
        self._auto_save()
    
    def update_script(self, script_data: Dict[str, Any]) -> None:
        """
        Update script information in the JSON and auto-save.
        
        Args:
            script_data: Script data from ScriptWriter, expected to have 'full_script' and 'metadata.length_words'.
        """
        self.json_data["script"]["full_text"] = script_data.get("full_script", "")
        if "metadata" in script_data and "length_words" in script_data["metadata"]:
            self.json_data["script"]["word_count"] = script_data["metadata"]["length_words"]
        else:
            # Fallback if length_words is not directly available, calculate from full_script
            self.json_data["script"]["word_count"] = len(script_data.get("full_script", "").split())
            
        if "metadata" in script_data and "cleaned_topic" in script_data["metadata"]:
            self.json_data["metadata"]["topic"] = script_data["metadata"]["cleaned_topic"]
            self.json_data["metadata"]["title"] = script_data["metadata"]["cleaned_topic"] # Default title to topic
        
        self.json_data["metadata"]["updated_at"] = datetime.now().isoformat()
        self._auto_save()
    
    def update_audio(self, audio_info: Dict[str, Any]) -> None:
        """
        Update audio information in the JSON and auto-save.
        
        Args:
            audio_info: Audio information from TTSGenerator. Expected to have 'url', 'all_urls', 'chunks'.
        """
        self.json_data["audio"]["url"] = audio_info.get("url", "") # Primary URL (e.g., first chunk or full)
        
        # Store additional info if available
        if "all_urls" in audio_info:
            self.json_data["audio"]["all_urls"] = audio_info["all_urls"]
        if "chunks" in audio_info:
            self.json_data["audio"]["num_chunks"] = audio_info["chunks"]
        if "word_count" in audio_info: # This would be from the original script text fed to TTS
            self.json_data["audio"]["source_word_count"] = audio_info["word_count"]

        # Remove old "parts" structure if it exists from a previous format
        if "parts" in self.json_data["audio"]:
            del self.json_data["audio"]["parts"]
            
        self.json_data["metadata"]["updated_at"] = datetime.now().isoformat()
        self._auto_save()
    
    def update_segments(self,
                       transcription: Dict[str, Any],
                       prompt_data: Optional[Dict[str, Any]] = None,
                       image_data: Optional[Dict[str, Dict[str, Optional[str]]]] = None) -> None: # Changed image_data type
        """
        Update segments in the JSON and auto-save.
        
        Args:
            transcription: Transcription data from Transcriber
            prompt_data: Optional prompt data from PromptGenerator
            image_data: Optional image data from ImageGenerator
        """
        # Update audio duration
        self.json_data["audio"]["duration"] = transcription["duration"]
        
        # Create segments
        segments = []
        
        for i, segment in enumerate(transcription["segments"]):
            # Create segment
            new_segment = {
                "id": i,
                "text": segment["text"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration": segment["end_time"] - segment["start_time"],
                "words": segment["words"]
            }
            
            # Add part if available
            if "part" in segment:
                new_segment["part"] = segment["part"]
            
            # Add image prompt if available
            if prompt_data and i < len(prompt_data["segments"]):
                new_segment["image_prompt"] = prompt_data["segments"][i]["image_prompt"]
            
            # Add image URL and depth map URL with content-based matching
            if image_data:
                import hashlib
                clean_text = segment["text"].strip().lower().encode('utf-8')
                content_hash = hashlib.md5(clean_text).hexdigest()
                
                segment_image_data: Optional[Dict[str, Optional[str]]] = None

                # 1. Try exact ID match (segment index as string key)
                if str(i) in image_data and image_data[str(i)]:
                    segment_image_data = image_data[str(i)]
                    # print(f"Matched image data for segment {i} by exact ID key '{str(i)}'")
                
                # 2. Try content hash match (key format like "id-hash-content_hash")
                else:
                    matching_key_with_hash = next(
                        (key for key in image_data.keys() if f"hash-{content_hash}" in key),
                        None
                    )
                    if matching_key_with_hash and image_data[matching_key_with_hash]:
                        segment_image_data = image_data[matching_key_with_hash]
                        # print(f"Matched image data for segment {i} by content hash key '{matching_key_with_hash}'")
                
                # 3. Fallback to positional matching if keys are just indices or simple strings
                    # This assumes image_data keys might correspond to segment order if not hashed
                    elif len(image_data) > i:
                        # This part is tricky if keys are not predictable (e.g. "0", "1"...)
                        # Let's assume mastermind.py prepares keys that can be somewhat matched.
                        # The keys from image_generator are segment_id, which mastermind maps from prompt_data.
                        # The prompt_data keys in mastermind are "segment_id-hash-content_hash".
                        # So, the hash match should be the primary way.
                        # If we reach here, it means the hash didn't match or keys are different.
                        # This positional fallback might be unreliable if keys are arbitrary.
                        # For now, we'll rely on the hash matching primarily.
                        # If no match by ID or hash, we might not have data for this specific segment.
                        pass # Positional matching is less reliable with complex keys.

                if segment_image_data:
                    new_segment["image_url"] = segment_image_data.get("image_url")
                    new_segment["depth_map_url"] = segment_image_data.get("depth_map_url")
                    if new_segment["image_url"]:
                        print(f"✓ Image & Depth Map URLs updated for segment {i}")
                    elif new_segment["image_url"]:
                         print(f"✓ Image URL updated for segment {i} (no depth map)")
                else:
                    print(f"✗ No image data found for segment {i} (text: '{segment['text'][:30]}...') using ID or hash.")
            
            segments.append(new_segment)
        
        # Update combined segments
        self.json_data["segments"] = segments
        
        # Remove section_transcripts if it exists (not needed in final output)
        if "section_transcripts" in self.json_data:
            del self.json_data["section_transcripts"]
             
        self.json_data["metadata"]["updated_at"] = datetime.now().isoformat()
        self._auto_save()
    
    def _auto_save(self) -> None:
        """
        Auto-save the JSON file after updates.
        Uses the unique instance filename.
        """
        try:
            self.storage_manager.save_json(self.json_data, self.filename)
            # print(f"Auto-saved to {self.filename}") # Optional: for debugging
        except Exception as e:
            print(f"Auto-save to {self.filename} failed: {str(e)}")

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save the JSON file.
        
        Args:
            filename: Optional. Filename for the JSON. If None, uses instance's unique filename.
            
        Returns:
            Path to the saved file
        """
        save_to_filename = filename if filename is not None else self.filename
        return self.storage_manager.save_json(self.json_data, save_to_filename)
    
    def load(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Load the JSON file.
        
        Args:
            filename: Optional. Filename for the JSON. If None, uses instance's unique filename.
            
        Returns:
            Loaded JSON data or an empty dict if load fails.
        """
        load_from_filename = filename if filename is not None else self.filename
        loaded_data = self.storage_manager.load_json(load_from_filename)
        
        if loaded_data:
            self.json_data = loaded_data
            return self.json_data
        
        # If loading self.filename fails (e.g., new instance, file doesn't exist yet),
        # return current self.json_data (which is the initial structure).
        # Or, if loading a specified file fails, return an empty dict or raise error.
        # For now, returning self.json_data (initial if load failed for self.filename)
        # or an empty dict if a specific file load failed.
        if load_from_filename == self.filename:
            return self.json_data # Return initial if unique file not found
        return {} # Return empty if specified file not found


    def save_minimal(self, error_info: Dict[str, Any]) -> str:
        """
        Save a minimal valid JSON structure when errors occur.
        
        Args:
            error_info: Dictionary containing error information
            
        Returns:
            Path to the saved file
        """
        minimal_data = {
            "metadata": {
                "title": "Error in video generation",
                "description": "",
                "topic": self.json_data.get("metadata", {}).get("topic", ""),
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "error": True,
                "original_filename_attempt": self.filename 
            },
            "error_info": error_info,
            "segments": []
        }
        return self.storage_manager.save_json(minimal_data, self.error_filename)
