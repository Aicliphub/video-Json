
import os
import sys
import json
import argparse
import time
from typing import Dict, Any, Optional
from pathlib import Path
import uuid
import shutil
# import google.generativeai as genai
from openai import OpenAI # For Mastermind's own LLM client

from src.utils.api_manager import ApiKeyManager # ApiKeyManager might be less relevant if all LLMs are DeepSeek with one key
from src.style_parser import StyleParser
from src.utils.config import ConfigManager, ProjectConfig
from src.utils.storage import StorageManager
from src.script_writer import ScriptWriter
from src.tts_generator import TTSGenerator
from src.transcriber import Transcriber
from src.prompt_generator import DeepSeekPromptGenerator as PromptGenerator
from src.image_generator import ImageGenerator
from src.json_builder import JsonBuilder

class Mastermind:
    """Main coordinator for the YouTube video generation system."""
    
    def __init__(self):
        """Initialize the mastermind."""
        # Load project configuration
        self.config = ProjectConfig()
        
        try:
            # Validate configuration
            self.config.validate()
        except ValueError as e:
            print(f"Configuration error: {str(e)}")
            sys.exit(1)
        
        # Initialize core managers first
        self.api_key_manager = ApiKeyManager(self.config.gemini_api_keys) # Retain for now, StyleParser might still use it if not fully switched
        self.config_manager = ConfigManager(self.config.llm_config) # For generation params like temperature
        self.storage_manager = StorageManager(self.config.r2_config)

        # Initialize Mastermind's own LLM client for input parsing (DeepSeek)
        try:
            self.input_parser_llm = OpenAI(
                api_key=self.config.deepseek_api_key,
                base_url=self.config.deepseek_config.get("base_url")
            )
            self.input_parser_model = self.config.deepseek_config.get("chat_model", "deepseek-chat")
            print("Mastermind initialized its own DeepSeek client for input parsing.")
        except Exception as e:
            print(f"Error initializing Mastermind's DeepSeek client: {e}")
            self.input_parser_llm = None # Handle potential failure

        # Initialize style system components
        self.style_parser = StyleParser( # StyleParser now uses DeepSeek
            deepseek_api_key=self.config.deepseek_api_key,
            deepseek_config=self.config.deepseek_config,
            default_style=getattr(self.config, 'default_style', None)
        )
        # self.style_config = None # This instance variable is set per video
        
        # Then initialize all other components using centralized config
        # Assuming DeepSeek is the primary LLM for script writing for now
        self.script_writer = ScriptWriter(
            config_manager=self.config_manager,
            storage_manager=self.storage_manager,
            llm_provider="deepseek", 
            deepseek_config=self.config.deepseek_config,
            deepseek_api_key=self.config.deepseek_api_key
        )
        self.tts_generator = TTSGenerator(
            tts_provider="allvoicelab", # Force allvoicelab since it's the only implemented provider
            tts_config=self.config.tts_config,
            api_key=self.config.allvoicelab_api_key, # Pass the correct key for allvoicelab
            r2_config=self.config.r2_config # Pass R2 config
        )
        self.transcriber = Transcriber(
            config_manager=self.config_manager,
            storage_manager=self.storage_manager,
            api_key=self.config.deepgram_api_key,
            transcriber_config=self.config.transcriber_config
        )
        # Pass relevant config to PromptGenerator (currently DeepSeek)
        self.prompt_generator = PromptGenerator(
            storage_manager=self.storage_manager,
            deepseek_config=self.config.deepseek_config,
            deepseek_api_key=self.config.deepseek_api_key
            # Add Gemini config/keys here if needed later
        )
        self.image_generator = ImageGenerator(
            storage_manager=self.storage_manager,
            api_key=self.config.freeflux_api_key,
            image_generator_config=self.config.image_generator_config
        )
        self.json_builder = JsonBuilder(self.storage_manager)
        
        # State tracking
        self.state = {
            "input_prompt": None, # Changed from topic
            "parsed_topic": None,
            "parsed_style_directive": None,
            "script_data": None,
            "audio_info": None,
            "transcription": None,
            "prompt_data": None,
            "image_data": None,
            "json_path": None,
            "video_path": None,
            "status": "initialized",
            "start_time": time.time(),
            "end_time": None
        }
    
    def _parse_input_prompt(self, input_prompt_str: str) -> Dict[str, Optional[str]]:
        """Parse the combined input prompt into topic and style directive using an LLM."""
        if not self.input_parser_llm:
            print("Warning: Input parser LLM not initialized. Attempting basic split.")
            # Basic fallback: assume last part after " in " or " with " is style
            parts = input_prompt_str.lower().split(" in ")
            if len(parts) > 1:
                return {"topic": " in ".join(parts[:-1]), "style_directive": parts[-1]}
            parts = input_prompt_str.lower().split(" with ")
            if len(parts) > 1:
                 return {"topic": " with ".join(parts[:-1]), "style_directive": parts[-1]}
            return {"topic": input_prompt_str, "style_directive": None}

        parsing_prompt_messages = [
            {"role": "system", "content": "You are an expert at parsing user requests. Extract the main topic and any visual style instructions. Return a valid JSON object with two keys: 'topic' (string) and 'style_directive' (string, or null if no style is mentioned)."},
            {"role": "user", "content": f"Parse the following user request: '{input_prompt_str}'. Example: User request 'A funny video about cats in a cartoon style' -> {{\"topic\": \"A funny video about cats\", \"style_directive\": \"cartoon style\"}}. User request 'History of Rome' -> {{\"topic\": \"History of Rome\", \"style_directive\": null}}."}
        ]
        try:
            completion = self.input_parser_llm.chat.completions.create(
                model=self.input_parser_model,
                messages=parsing_prompt_messages,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            parsed_result = json.loads(completion.choices[0].message.content)
            return {
                "topic": parsed_result.get("topic"),
                "style_directive": parsed_result.get("style_directive")
            }
        except Exception as e:
            print(f"Error parsing input prompt with LLM: {e}. Using basic fallback.")
            # Fallback to basic split on error
            parts = input_prompt_str.lower().split(" in ")
            if len(parts) > 1:
                return {"topic": " in ".join(parts[:-1]).strip(), "style_directive": parts[-1].strip()}
            parts = input_prompt_str.lower().split(" with ")
            if len(parts) > 1:
                 return {"topic": " with ".join(parts[:-1]).strip(), "style_directive": parts[-1].strip()}
            return {"topic": input_prompt_str.strip(), "style_directive": None}

    def generate_video(self, input_prompt_str: str) -> Dict[str, Any]:
        """
        Generate a complete video from a combined input prompt.
        
        Args:
            input_prompt_str: Combined topic, character, and style directive.
            
        Returns:
            State dictionary with results including json_data
        """
        try:
            self.state["input_prompt"] = input_prompt_str
            print(f"\n=== Received Input Prompt: '{input_prompt_str}' ===\n")

            # Step 0: Parse Input Prompt
            print(f"\n=== Step 0: Parsing Input Prompt ===\n")
            parsed_input = self._parse_input_prompt(input_prompt_str)
            actual_topic = parsed_input.get("topic")
            style_directive_from_input = parsed_input.get("style_directive")

            if not actual_topic:
                raise ValueError("Failed to extract a topic from the input prompt.")
            
            self.state["parsed_topic"] = actual_topic
            self.state["parsed_style_directive"] = style_directive_from_input
            print(f"Parsed Topic: '{actual_topic}'")
            print(f"Parsed Style Directive: '{style_directive_from_input}'")

            # Parse style directive string into structured config if present
            style_config_for_prompts = None
            if style_directive_from_input:
                print(f"\n=== Parsing Style Directive String for Prompts: '{style_directive_from_input}' ===\n")
                try:
                    style_config_for_prompts = self.style_parser.parse_style(style_directive_from_input)
                    if not self.style_parser.validate_style(style_config_for_prompts): # Though validate_style is a TODO
                        print("Warning: Parsed style configuration failed validation.")
                        # Decide if to proceed with potentially invalid style or fallback
                except Exception as e:
                    print(f"Warning: Failed to parse style directive string into structured config: {e}. Proceeding without structured style for prompts.")
                    # Fallback: use the raw style_directive_from_input directly for prompt_generator if it can handle it,
                    # or pass None. For now, pass None if parsing fails.
                    style_config_for_prompts = None 
            
            # Step 1: Generate script (style_config is None to keep script style-agnostic)
            print(f"\n=== Step 1: Generating Script for '{actual_topic}' ===\n")
            script_data = self.script_writer.generate_full_script(actual_topic, style_config=None)
            self.state["script_data"] = script_data
            
            # Update JSON
            self.json_builder.update_script(script_data)
            self.json_builder.save()
            
            # Step 2: Generate TTS using the updated generator method
            print(f"\n=== Step 2: Generating Text-to-Speech ===\n")
            self.state["status"] = "generating_tts"
            
            # Call the updated method which handles parts internally
            audio_info = self.tts_generator.generate_audio_from_script(script_data)
            
            self.state["audio_info"] = audio_info
            
            # Update JSON
            self.json_builder.update_audio(audio_info)
            self.json_builder.save()
            
            # Step 3: Transcribe audio using the updated transcriber method
            print(f"\n=== Step 3: Transcribing Audio ===\n")
            self.state["status"] = "transcribing_audio"
            
            # Call the updated method which handles parts internally based on audio_info structure
            transcription = self.transcriber.process_audio(audio_info, script_data)
            
            self.state["transcription"] = transcription
            
            # Update JSON
            self.json_builder.update_segments(transcription)
            self.json_builder.save()
            
            # Step 4: Generate image prompts
            print(f"\n=== Step 4: Generating Image Prompts ===\n")
            self.state["status"] = "generating_prompts"
            
            # The prompt_generator now takes input_prompt_str directly
            # It will parse out topic, character, and video_type internally.
            # We still need transcription and style_config_for_prompts.
            
            if not transcription or not isinstance(transcription, dict) or "segments" not in transcription:
                raise ValueError("Invalid transcription data for prompt generation")

            try:
                # Pass the original input_prompt_str, transcription, and parsed style_config
                prompt_data = self.prompt_generator.generate_prompts(
                    transcription=transcription,
                    input_prompt_text=input_prompt_str, # Pass the full original prompt
                    style_config=style_config_for_prompts
                )
                self.state["prompt_data"] = prompt_data
                
                # Update JSON
                self.json_builder.update_segments(transcription, prompt_data)
                self.json_builder.save()
            except Exception as e:
                print(f"Warning: Prompt generation failed: {str(e)}")
                # Create minimal prompt data structure
                prompt_data = {
                    "segments": [{
                        "text": s["text"],
                        "start_time": s["start_time"],
                        "end_time": s["end_time"],
                        "image_prompt": f"Visual representation of: {s['text'][:100]}..."
                    } for s in transcription["segments"]],
                    "count": len(transcription["segments"]),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "batch_generated": False,
                    "error": str(e)
                }
                self.state["prompt_data"] = prompt_data
            
            # Step 5: Generate images
            print(f"\n=== Step 5: Generating Images ===\n")
            self.state["status"] = "generating_images"
            
            # Extract prompts from prompt_data with fallback index
            prompts = {}
            for i, segment in enumerate(prompt_data["segments"]):
                if "image_prompt" in segment:
                    # Create ID+hash key for better matching
                    import hashlib
                    clean_text = segment["text"].strip().lower().encode('utf-8')
                    content_hash = hashlib.md5(clean_text).hexdigest()
                    key = f"{segment.get('id', i+1)}-hash-{content_hash}"
                    prompts[key] = segment["image_prompt"]
            
            # Generate all images in batch
            if prompts:
                print(f"Generating {len(prompts)} images and their depth maps...")
                try:
                    image_data = self.image_generator.generate_batch(prompts)
                    self.state["image_data"] = image_data
                    
                    # Adjust print statement to count successful image_urls
                    successful_images = len([data for data in image_data.values() if data and data.get("image_url")])
                    successful_depth_maps = len([data for data in image_data.values() if data and data.get("depth_map_url")])
                    print(f"Successfully generated {successful_images}/{len(prompts)} images and {successful_depth_maps}/{successful_images} depth maps.")
                
                except ConnectionError as ce:
                    print(f"ERROR: Image generation API connection failed: {str(ce)}")
                    print("Please check your FreeFlux API key and network connectivity in render.com environment variables")
                    print("Falling back to placeholder images")
                    
                    # Create placeholder image data
                    image_data = {
                        key: {
                            "image_url": None,
                            "depth_map_url": None,
                            "error": str(ce),
                            "placeholder": True
                        } for key in prompts.keys()
                    }
                    self.state["image_data"] = image_data
                    self.state["image_generation_error"] = str(ce)
                
                except Exception as e:
                    print(f"ERROR: Image generation failed: {str(e)}")
                    image_data = {
                        key: {
                            "image_url": None,
                            "depth_map_url": None,
                            "error": str(e)
                        } for key in prompts.keys()
                    }
                    self.state["image_data"] = image_data
                    self.state["image_generation_error"] = str(e)
            else:
                print("Warning: No image prompts found to generate")
                image_data = {}
                self.state["image_data"] = image_data
            
            # Update JSON with all data
            json_path = None
            try:
                if not prompt_data or not isinstance(prompt_data, dict):
                    raise ValueError("Invalid prompt_data format")
                if not transcription or not isinstance(transcription, dict):
                    raise ValueError("Invalid transcription format")
                
                # Ensure segments exist and are lists
                if "segments" not in prompt_data or not isinstance(prompt_data["segments"], list):
                    prompt_data["segments"] = []
                if "segments" not in transcription or not isinstance(transcription["segments"], list):
                    transcription["segments"] = []
                
                # Match segment counts if needed
                if len(prompt_data["segments"]) != len(transcription["segments"]):
                    min_len = min(len(prompt_data["segments"]), len(transcription["segments"]))
                    prompt_data["segments"] = prompt_data["segments"][:min_len]
                    transcription["segments"] = transcription["segments"][:min_len]
                
                self.json_builder.update_segments(transcription, prompt_data, image_data)
                json_path = self.json_builder.save()
            except Exception as e:
                print(f"Warning: Failed to update JSON segments: {str(e)}")
                # Create minimal valid JSON structure
                json_path = self.json_builder.save_minimal({
                    "error": str(e),
                    "prompt_data": bool(prompt_data),
                    "transcription": bool(transcription),
                    "image_data": bool(image_data)
                })
            
            self.state["json_path"] = json_path
            
            # Skip video generation step
            print(f"\n=== Step 6: Video Generation (Skipped) ===\n")
            print(f"JSON file with all video assets is available at: {json_path}")

            # Clean up all files except JSON
            current_json_path_str = self.state.get("json_path")
            if current_json_path_str:
                preserved_json_path_obj = Path(current_json_path_str)
                assets_root_path = preserved_json_path_obj.parent
                print(f"Preserving JSON file: '{preserved_json_path_obj}'")

                # Clean up asset subdirectories
                asset_subdirs_to_clean = ["audio", "images", "scripts"]
                print(f"Cleaning asset subdirectories: {', '.join(asset_subdirs_to_clean)}")
                for subdir_name in asset_subdirs_to_clean:
                    subdir_path = assets_root_path / subdir_name
                    if subdir_path.exists() and subdir_path.is_dir():
                        for item_name in os.listdir(subdir_path):
                            item_path = subdir_path / item_name
                            try:
                                if item_path.is_file():
                                    os.remove(item_path)
                                elif item_path.is_dir():
                                    shutil.rmtree(item_path)
                            except OSError as oe_clean:
                                print(f"Error removing '{item_path}': {oe_clean}")
                        print(f"Cleaned contents of '{subdir_path}'")
                    else:
                        print(f"Subdirectory '{subdir_path}' not found or not a directory, skipping cleanup for it.")
                
                # Clean up other files in the root assets directory except JSON
                print(f"Cleaning other files in '{assets_root_path}' (except JSON)...")
                if assets_root_path.exists() and assets_root_path.is_dir():
                    for item_name in os.listdir(assets_root_path):
                        item_path = assets_root_path / item_name
                        # Skip the preserved JSON file and subdirectories
                        if item_path.is_file() and item_path != preserved_json_path_obj:
                            try:
                                os.remove(item_path)
                                print(f"Removed file: '{item_path}'")
                            except OSError as oe_clean_root:
                                print(f"Error removing root asset file '{item_path}': {oe_clean_root}")
                print("Asset cleanup process finished.")
            else:
                print("Warning: 'json_path' not found in state, skipping asset cleanup.")

            # Update state
            self.state["status"] = "completed"
            self.state["end_time"] = time.time()
            
            print(f"\n=== Processing Complete ===\n")
            print(f"Total time: {self.state['end_time'] - self.state['start_time']:.2f} seconds")
            
            # Include the json data in the returned state
            return {
                **self.state,
                "json_data": self.json_builder.json_data
            }
            
        except Exception as e:
            # Update state with error
            self.state["status"] = "error"
            self.state["error"] = str(e)
            self.state["end_time"] = time.time()
            
            print(f"\n=== Error: {str(e)} ===\n")
            
            return self.state

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YouTube Video Generation System")
    
    # Combined input prompt
    parser.add_argument("--input_prompt", type=str, required=True,
                        help="Full input prompt including topic, character description, and style (e.g., 'A story about a brave knight named Sir Reginald, who is tall and wears shining silver armor, in a medieval fantasy art style')")
    
    # Advanced options
    parser.add_argument("--model", type=str, default="gemini-1.5-pro-latest",
                        help="Gemini model to use")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Temperature for generation")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create mastermind
    mastermind = Mastermind()
    
    # Generate video using the combined input prompt
    mastermind.generate_video(args.input_prompt)

if __name__ == "__main__":
    main()
