"""
DeepSeek Prompt Generator
Hardcoded implementation for generating image prompts using DeepSeek API.
"""
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI
from src.utils.storage import StorageManager

class DeepSeekPromptGenerator:
    def __init__(self, 
                 storage_manager: StorageManager,
                 deepseek_config: Dict[str, str],
                 deepseek_api_key: str,
                 # TODO: Add gemini_config, gemini_api_keys if Gemini support is needed
                 ):
        """
        Initialize the Prompt Generator.
        
        Args:
            storage_manager: Storage manager instance.
            deepseek_config: Dictionary with 'base_url', 'chat_model'.
            deepseek_api_key: API key for DeepSeek.
            # Add other provider configs/keys as needed
        """
        self.storage = storage_manager
        self.max_retries = 3
        self.retry_delay = 2
        
        # Initialize clients based on provided config
        self.apis = []
        
        # DeepSeek Client
        if deepseek_api_key and deepseek_config:
            try:
                deepseek_client = OpenAI(
                    api_key=deepseek_api_key,
                    base_url=deepseek_config.get("base_url")
                )
                self.apis.append({
                    "name": "DeepSeek",
                    "client": deepseek_client,
                    "model": deepseek_config.get("chat_model", "deepseek-chat")
                })
                print(f"PromptGenerator initialized with DeepSeek client (Model: {deepseek_config.get('chat_model', 'deepseek-chat')}).")
            except Exception as e:
                print(f"Warning: Failed to initialize DeepSeek client for PromptGenerator: {e}")
        
        # Gemini Client (Optional - requires config/keys to be passed)
        # gemini_client = self._init_gemini(gemini_api_keys) # Pass keys if available
        # if gemini_client:
        #     self.apis.append({
        #         "name": "Gemini",
        #         "client": gemini_client,
        #         "model": gemini_config.get("model", "gemini-pro") # Get model from config
        #     })
        #     print(f"PromptGenerator initialized with Gemini client (Model: {gemini_config.get('model', 'gemini-pro')}).")

        if not self.apis:
             raise ValueError("PromptGenerator requires at least one valid LLM configuration (DeepSeek or Gemini).")
        
        self.system_prompt = """You are a very talented Senior Art Director with decades of experience in filmmaking and visual storytelling. Your task is to conceptualize and describe compelling image prompts for a video. Each prompt should be a masterpiece of visual direction.

        **Core Directives for Each Prompt:**
        1.  **Visual Storytelling:** Each image must tell a part of the story or convey a key concept with clarity and impact. Think about the narrative arc and how each shot contributes.
        2.  **Emotional Resonance:** Infuse prompts with elements that evoke the desired emotion for the scene (e.g., wonder, tension, joy, intrigue).
        3.  **Composition & Framing:** Describe the shot with a director's eye. Consider camera angles (low angle, high angle, close-up, wide shot), subject placement, depth of field, and leading lines. Ensure all compositions are for a **vertical 9:16 aspect ratio**.
        4.  **Lighting & Atmosphere:** Specify lighting to create mood (e.g., dramatic Rembrandt lighting, soft morning light, mysterious chiaroscuro, vibrant neon glow). Describe the overall atmosphere.
        5.  **Stylistic Cohesion:** While each prompt is unique, ensure they can collectively form a stylistically coherent sequence if an overall style is provided in the user prompt.
        6.  **Detail & Specificity:** Be specific about key visual elements, textures, and details that will make the image striking and memorable. Avoid vague descriptions.
        7.  **Conciseness & Power:** While detailed, prompts should be concise and use powerful, evocative language.
        8.  **Technical Adherence:** The final output for each segment's prompt must be a string. The collective output for all segments must be a single JSON object where keys are segment IDs (e.g., "1", "2") and values are the corresponding prompt strings. Example: `{"1": "prompt text for segment 1", "2": "prompt text for segment 2"}`.
        
        Think like a master artist and a seasoned director. Your prompts are the blueprint for visual excellence.
        """

    # Modified _init_gemini to accept keys - uncomment if Gemini support is added back
    # def _init_gemini(self, gemini_api_keys: List[str]):
    #     if not gemini_api_keys or not gemini_api_keys[0]:
    #         return None
    #     try:
    #         import google.generativeai as genai
    #         # Use the first key for configuration, ApiKeyManager handles rotation if needed elsewhere
    #         genai.configure(api_key=gemini_api_keys[0].strip())
    #         return genai # Return the configured module/client object
    #     except ImportError:
    #         print("Warning: google.generativeai package not installed. Gemini support disabled.")
    #         return None
    #     except Exception as e:
    #          print(f"Warning: Failed to configure Gemini: {e}")
    #          return None

    def _parse_input_prompt(self, input_prompt_text: str) -> Dict[str, Optional[str]]:
        """
        Parse the input prompt string to extract video topic, character description, and video type using LLM.
        """
        system_message = """You are an expert at analyzing user prompts for video generation.
        Your task is to extract the core video topic, a detailed character description (if any), and infer a video type.
        Return a single JSON object with the keys "video_topic", "character_description", and "video_type".
        If a character description is multi-faceted or describes multiple characters, combine them into a single descriptive string for "character_description".
        If no specific character description is found, "character_description" should be null.
        If no specific video topic is clear, "video_topic" should be null.
        Infer "video_type" from the overall prompt (e.g., storytelling, educational, animation, documentary, product-showcase). If not clear, set to "general".
        """
        
        user_message_content = f"""
        Analyze the following input prompt and extract the video topic, character description, and video type.

        Input Prompt:
        "{input_prompt_text}"

        Output JSON format:
        {{
            "video_topic": "string or null",
            "character_description": "string or null",
            "video_type": "string (e.g., storytelling, educational, animation, documentary, product-showcase, general) or null"
        }}
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_content}
        ]

        # Use the first configured API (assuming DeepSeek for this task)
        api_to_use = self.apis[0] if self.apis else None
        if not api_to_use or not api_to_use["client"]:
            print("Warning: No suitable LLM client found for parsing input prompt.")
            return {"video_topic": None, "character_description": None, "video_type": "general"}

        for attempt in range(self.max_retries):
            try:
                if api_to_use["name"] == "DeepSeek": # Or generalize if other LLMs support JSON mode
                    response = api_to_use["client"].chat.completions.create(
                        model=api_to_use["model"],
                        messages=messages,
                        temperature=0.2,
                        response_format={"type": "json_object"}
                    )
                    parsed_content = json.loads(response.choices[0].message.content)
                    return {
                        "video_topic": parsed_content.get("video_topic"),
                        "character_description": parsed_content.get("character_description"),
                        "video_type": parsed_content.get("video_type", "general")
                    }
                # Add other API handlers if necessary, though DeepSeek with JSON mode is ideal
            except Exception as e:
                print(f"Error parsing input prompt with {api_to_use['name']} (attempt {attempt + 1}): {e}")
                time.sleep(self.retry_delay)
        
        print("Failed to parse input prompt after multiple retries.")
        return {"video_topic": input_prompt_text[:100], "character_description": None, "video_type": "general"}


    def generate_prompts(self, 
                        transcription: Dict[str, Any], 
                        input_prompt_text: str, # Changed from script_data
                        style_config: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
        """
        Generate image prompts for all segments
        
        Args:
            transcription: Transcription data with segments.
            input_prompt_text: Raw text prompt from the user containing topic and character info.
            style_config: Optional style configuration dictionary.
            
        Returns:
            Dictionary with generated prompts for each segment
        """
        
        parsed_input = self._parse_input_prompt(input_prompt_text)
        video_topic = parsed_input.get("video_topic") or "Video" # Fallback topic
        character_description = parsed_input.get("character_description")
        video_type = parsed_input.get("video_type")

        segments = transcription["segments"]
        
        # Try batch generation first with style config, character description, and video type
        batch_prompts = self.generate_batch(segments, 
                                            video_topic, # Pass parsed topic
                                            style_config, 
                                            character_description, 
                                            video_type)
        
        # Prepare output structure
        segments_with_prompts = []
        for i, segment in enumerate(segments):
            segment_with_prompt = segment.copy()
            segment_id = str(i+1)
            segment_with_prompt["image_prompt"] = batch_prompts.get(segment_id,
                f"A visual representation of: {segment['text'][:100]}...")
            
            if "part" in segment:
                segment_with_prompt["part"] = segment["part"]
            
            segments_with_prompts.append(segment_with_prompt)
        
        # Create prompt data structure
        prompt_data = {
            "segments": segments_with_prompts,
            "count": len(segments_with_prompts),
            "timestamp": datetime.now().isoformat(),
            "batch_generated": True,
            "sections": {
                "intro": [p for p in segments_with_prompts if p.get("part") == "intro"],
                "main": [p for p in segments_with_prompts if p.get("part") == "main"],
                "conclusion": [p for p in segments_with_prompts if p.get("part") == "conclusion"]
            } if any("part" in p for p in segments_with_prompts) else {}
        }
        
        # Save prompt data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prompts_{timestamp}.json"
        # Include parsed input in the saved data for traceability
        prompt_data_to_save = prompt_data.copy()
        prompt_data_to_save["parsed_input_prompt"] = parsed_input
        self.storage.save_json(prompt_data_to_save, filename, "scripts")
        
        return prompt_data

    def generate_batch(self, 
                      segments: List[Dict[str, Any]],
                      video_topic: str, # Added video_topic
                      style_config: Optional[Dict[str, Any]] = None,
                      character_description: Optional[str] = None,
                      video_type: Optional[str] = None) -> Dict[str, str]:
        """Generate prompts for multiple segments"""
        # topic parameter is now video_topic, passed from generate_prompts
        
        # Consolidate style details from style_config into a string for the LLM
        final_style_details_for_prompt_prefix = ""
        if style_config:
            visual_style = style_config.get("visual_style", {})
            art_style = visual_style.get("art_style")
            color_palette = visual_style.get("color_palette")
            lighting = visual_style.get("lighting")
            composition = visual_style.get("composition")
            
            style_parts = [s for s in [art_style, color_palette, lighting, composition] if s and s != "null"]
            if style_parts:
                final_style_details_for_prompt_prefix = ", ".join(style_parts)

        # Type-specific guidelines
        type_specific_guidelines_str = ""
        if video_type:
            if video_type == "storytelling":
                type_specific_guidelines_str = "\nFor this storytelling video, emphasize character emotions, dynamic poses, and key narrative moments in the visuals."
            elif video_type == "educational":
                type_specific_guidelines_str = "\nFor this educational video, focus on clear visual representation of concepts, use helpful diagrams or metaphors, and ensure legibility if text is implied in the visuals."
            elif video_type == "documentary": # Assuming 'documentary' might be a type
                type_specific_guidelines_str = "\nFor this documentary video, aim for realistic or historically accurate depictions, and convey a sense of authenticity and context."
            # Add more types as needed
        
        # Prepare segments with previous text context
        segments_with_context = []
        previous_text = None
        for i, s in enumerate(segments):
            segments_with_context.append({
                "id": str(i+1),
                "text": s['text'],
                "previous_text": previous_text
            })
            previous_text = s['text'] # Update previous_text for the next iteration

        # Construct the user prompt for the LLM
        user_prompt_content = f"""
Generate an image prompt for each segment provided below. Use the 'previous_text' for narrative context.

**CRITICAL INSTRUCTIONS FOR EACH GENERATED PROMPT (MANDATORY):**
1.  **CHARACTER CONSISTENCY IS PARAMOUNT:** If a `Character Description` is provided below, **EVERY SINGLE generated image prompt MUST start with this EXACT `Character Description` verbatim.** This is crucial for maintaining visual consistency of the character(s) across all scenes.
2.  **OVERALL STYLE CONSISTENCY:** If an `Overall Visual Style` is provided below, **EVERY SINGLE generated image prompt MUST include these EXACT style details, immediately following the `Character Description` (if one was provided) or as the very first part of the prompt (if no `Character Description` was provided).** This ensures a cohesive look and feel.
3.  **NARRATIVE ACCURACY (MOST IMPORTANT for segment details):** After the mandatory prefixes (Character Description and/or Overall Visual Style), the appended descriptive details **MUST PRIMARILY AND ACCURATELY VISUALIZE THE SPECIFIC CONTENT, ACTIONS, OBJECTS, AND SETTING described in the current segment's `text` field.** While `previous_text` provides context for flow, the visual elements of the prompt must directly reflect what is happening in the *current* segment's `text`.
4.  **NO CHARACTER NAMES:** Do NOT use character names (e.g., "Alice," "Bob") in the prompts. Instead, rely on the provided `Character Description`.
{type_specific_guidelines_str}

**Provided Prefixes (Use these in EVERY prompt as instructed above):**
Character Description: `{character_description if character_description else "Not specified. If not specified, do not invent one."}`
Overall Visual Style: `{final_style_details_for_prompt_prefix if final_style_details_for_prompt_prefix else "Not specified. If not specified, focus on art direction principles for each segment."}`

**Segments for Prompt Generation:**
{json.dumps(segments_with_context, indent=2)}

**Example of a single, correctly constructed image prompt (if Character, Style, and segment text "The cat jumps onto the table" are specified):**
`"{character_description if character_description else "Suppose a character was described as: A fluffy ginger cat"}. {final_style_details_for_prompt_prefix if final_style_details_for_prompt_prefix else "Suppose an overall style was: whimsical cartoon style, bright colors"}. The fluffy ginger cat, mid-air, gracefully leaping onto a wooden kitchen table with a bowl of fruit on it."`
*(The parts in "Suppose..." are just for this example's illustration; you will use the actual provided Character Description and Overall Visual Style. The key is that "The cat jumps onto the table" from the segment text is directly visualized after the prefixes.)*

Return a single JSON object where keys are the segment 'id' strings (e.g., "1", "2") and values are the fully constructed image prompts.
Example JSON output: {{"1": "prompt text 1", "2": "prompt text 2", ...}}
"""

        messages = [
            {"role": "system", "content": self.system_prompt}, # System prompt has general guidelines for prompt quality
            {"role": "user", "content": user_prompt_content}
        ]
        
        for api in self.apis:
            if not api["client"]:
                continue
                
            for attempt in range(self.max_retries):
                try:
                    if api["name"] == "DeepSeek":
                        response = api["client"].chat.completions.create(
                            model=api["model"],
                            messages=messages,
                            temperature=0.7,
                            response_format={"type": "json_object"}
                        )
                        return json.loads(response.choices[0].message.content)
                    else:  # Gemini
                        model = api["client"].GenerativeModel(api["model"])
                        response = model.generate_content(
                            contents=messages[1]["content"],
                            generation_config={"temperature": 0.7}
                        )
                        return {str(i+1): response.text for i in range(len(segments))}
                except Exception as e:
                    print(f"{api['name']} batch generation attempt {attempt+1} failed: {str(e)}")
                    time.sleep(self.retry_delay)
        
        # Fallback if all APIs fail
        return {str(i+1): f"Visual representation of: {s['text'][:100]}..." 
                for i, s in enumerate(segments)}

    def save_results(self, prompts: Dict[str, str], prefix: str = "prompts"):
        """Save generated prompts to storage"""
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.storage.save_json({
            "prompts": prompts,
            "timestamp": datetime.now().isoformat(),
            "model": self.model
        }, filename)
