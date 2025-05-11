"""
Script Writer Module

Generates video scripts using the Gemini API with different sections (intro, main, conclusion).
"""
import os
import time
import json
import datetime
from typing import Dict, Any, List, Optional
from openai import OpenAI # Changed import
# import google.generativeai as genai # Commented out Gemini

from src.utils.api_manager import ApiKeyManager
from src.utils.config import ConfigManager
from src.utils.storage import StorageManager

# System instruction for a single, complete short script (approx 1 min) - MrBeast Style YouTube Scriptwriter Persona
SYSTEM_INSTRUCTION_FULL_SCRIPT = """
You are an ELITE, SENIOR YouTube SCRIPTWRITER, channeling the high-energy, ultra-engaging style of creators like MrBeast. Your goal is to write a script for a short video (approx. 1 minute) that GRABS the viewer's attention from THE VERY FIRST WORD and KEEPS THEM HOOKED until the end. This script is for Text-to-Speech (TTS) narration.

**CORE MISSION: MAXIMUM VIEWER RETENTION.**

**CRITICAL SCRIPTING RULES (NON-NEGOTIABLE):**
1.  **TTS CLEANLINESS:**
    *   ABSOLUTELY NO visual markers like `(**Visual:)`, `[SCENE START]`, etc.
    *   NO formatting symbols or special characters: `*`, `_`, `"`, `'`. (Do NOT use double quotes or single quotes/apostrophes).
    *   NO stage directions or parentheticals like `(excitedly)`, `(whispers)`.
    *   ABSOLUTELY NO PREFIXES OR LABELS like "Script:", "Narrator:", "Host:", "(Script):", etc. The output must be ONLY the raw narrative text itself, starting directly with the first word of the script.
    *   The script must be a single, continuous block of text. Pure, unadulterated narration.

2.  **MRBEAST-STYLE ENGAGEMENT:**
    *   **MEGA-HOOK (First 3-5 Seconds CRITICAL):** Start with an INSANELY compelling hook. This could be:
        *   A shocking statement or question directly related to the topic.
        *   A bold claim or a peek at an unbelievable outcome.
        *   An immediate presentation of a high-stakes challenge or a fascinating premise.
        *   Example: "I just buried $100,000 in this backyard, and these ten people have 24 hours to find it!" (Adapt this energy to the given topic).
    *   **FAST PACING & HIGH ENERGY:** Keep the energy UP. Use short, punchy sentences mixed with slightly longer ones for rhythm. Avoid lulls. Every sentence must drive interest.
    *   **CLEAR VALUE/PREMISE:** Make it immediately obvious what the video is about and why the viewer should care. What will they see, learn, or experience?
    *   **CONSTANT CURIOSITY GAPS & MINI-PAYOFFS:** Tease what's coming. Build anticipation. Deliver on small promises throughout the script to keep them invested.
    *   **CONVERSATIONAL & DIRECT:** Speak directly to the viewer. Use "you," "we," "I." Make it feel personal and urgent.
    *   **SIMPLE & CLEAR LANGUAGE:** Avoid jargon or overly complex vocabulary unless the topic demands it and it can be quickly explained. The language must be instantly understandable.

3.  **STRUCTURE (Approx. 1 Minute / 140-170 words):**
    *   **THE HOOK (0-5 seconds):** As described above. Utterly captivating.
    *   **THE SETUP/CHALLENGE (5-20 seconds):** Quickly elaborate on the hook. Explain the core premise, challenge, or main point of the video. Build initial intrigue.
    *   **THE DEVELOPMENT/JOURNEY (20-50 seconds):** Deliver the main content. If it's a story, show progression. If it's informational, reveal key facts or steps. Maintain high energy and quick cuts in information or action.
    *   **THE CLIMAX/PAYOFF/CONCLUSION (50-60 seconds):** A satisfying resolution, a key takeaway, a final reveal, or a call to action (if appropriate for a generic script, otherwise a strong concluding statement). Make it memorable.

**OUTPUT EXPECTATION:**
-   A single block of text.
-   Approximately 140-170 words.
-   Ready for direct TTS narration.
-   Adhering to ALL rules above.

Let's create something VIRAL!
"""

class ScriptWriter:
    """Handles the generation of video scripts with error handling and coherence."""

    def __init__(self, 
                 config_manager: ConfigManager,
                 storage_manager: StorageManager,
                 llm_provider: str = "deepseek", # Assume deepseek default for now
                 deepseek_config: Optional[Dict[str, str]] = None,
                 deepseek_api_key: Optional[str] = None,
                 # Add gemini_config, gemini_api_keys if supporting both needed
                 ):
        """
        Initialize the script writer.
        
        Args:
            config_manager: Configuration manager for generation params (temp, etc.)
            storage_manager: Storage manager for saving scripts
            llm_provider: Name of the LLM provider ('deepseek', 'gemini', etc.)
            deepseek_config: Dictionary with 'base_url', 'chat_model' for DeepSeek.
            deepseek_api_key: API key for DeepSeek.
            # Add other provider configs/keys as needed
        """
        self.config_manager = config_manager
        self.storage_manager = storage_manager
        self.llm_provider = llm_provider
        self.max_retries = 5 # General retry limit
        self.retry_delay = 5  # seconds
        self.backoff_factor = 1.5

        self.llm_client = None
        self.model_name = None

        if self.llm_provider == "deepseek":
            if not deepseek_config or not deepseek_api_key:
                raise ValueError("DeepSeek configuration and API key are required for ScriptWriter.")
            self.llm_client = OpenAI(
                api_key=deepseek_api_key,
                base_url=deepseek_config.get("base_url")
            )
            self.model_name = deepseek_config.get("chat_model", "deepseek-chat")
            print(f"ScriptWriter initialized with DeepSeek client (Model: {self.model_name}).")
        # Add elif blocks here for other providers like Gemini if needed
        # elif self.llm_provider == "gemini":
            # Initialize Gemini client using gemini_config and gemini_api_keys
            # self.model_name = gemini_config.get("model") ... etc.
        else:
             raise ValueError(f"Unsupported LLM provider for ScriptWriter: {self.llm_provider}")

    def preprocess_topic(self, topic: str) -> Dict[str, Any]:
        """
        Clean, analyze and enhance the topic for better prompt quality.
        
        Args:
            topic: Raw topic string
            
        Returns:
            Dictionary with processed topic information including:
            - cleaned_topic: Stripped topic without enhancements
            - enhancement: Any style/tonal enhancements requested
            - original_topic: Original input
            - topic_type: Detected category (educational, entertainment, etc.)
            - tone: Detected tone (serious, humorous, dramatic)
            - style_keywords: Any style-related keywords found
        """
        # Remove any special formatting or unnecessary characters
        cleaned_topic = topic.strip()
        cleaned_topic = cleaned_topic.replace("(.", "").replace(".)", "")

        # Initialize analysis results
        topic_type = "general"
        tone = "neutral" 
        style_keywords = []
        enhancement = ""

        # Detect topic type
        topic_lower = cleaned_topic.lower()
        if any(x in topic_lower for x in ["how to", "tutorial", "guide", "explain"]):
            topic_type = "educational"
        elif any(x in topic_lower for x in ["story", "tale", "narrative", "experience"]):
            topic_type = "storytelling"
        elif any(x in topic_lower for x in ["review", "compare", "vs", "versus"]):
            topic_type = "comparison"
        elif any(x in topic_lower for x in ["funny", "humor", "comedy", "joke"]):
            topic_type = "entertainment"
            tone = "humorous"

        # Detect tone indicators
        if any(x in topic_lower for x in ["serious", "important", "critical"]):
            tone = "serious"
        elif any(x in topic_lower for x in ["dramatic", "emotional", "heartfelt"]):
            tone = "dramatic"
        elif any(x in topic_lower for x in ["light", "fun", "entertaining"]):
            tone = "lighthearted"

        # Extract style preferences
        style_phrases = {
            "documentary": ["documentary", "factual", "real story"],
            "vlog": ["vlog", "personal", "my experience"],
            "explainer": ["explainer", "whiteboard", "simple explanation"],
            "news": ["news", "report", "breaking"],
            "ad": ["ad", "commercial", "promo"]
        }
        for style, phrases in style_phrases.items():
            if any(x in topic_lower for x in phrases):
                style_keywords.append(style)

        # Extract generic enhancements
        enhancement_phrases = ["make it exciting", "make it so exciting", "make it interesting"]
        for phrase in enhancement_phrases:
            if cleaned_topic.lower().endswith(phrase):
                cleaned_topic = cleaned_topic[:-(len(phrase))].strip()
                enhancement = phrase
                break

        return {
            "cleaned_topic": cleaned_topic,
            "enhancement": enhancement,
            "original_topic": topic,
            "topic_type": topic_type,
            "tone": tone,
            "style_keywords": style_keywords
        }

    def _generate_single_script_text(self, 
                                   topic_info: Dict[str, str],
                                   style_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a single complete script text.
        
        Args:
            topic_info: Processed topic information.
            style_config: Optional style configuration.
            
        Returns:
            Generated script text.
        """
        system_instruction = SYSTEM_INSTRUCTION_FULL_SCRIPT
        # Use default config or apply overrides if any were defined for a single script agent
        # For now, using general config_manager settings.
        # If specific overrides for a "full_script_agent" were needed, they'd be defined similar to AGENTS
        config_overrides = {"temperature": 0.85, "max_output_tokens": 4000} # Example, adjust as needed for full short script
        generation_config = self.config_manager.get_config(**config_overrides)

        # Prepare the prompt for a single script
        prompt = self._prepare_single_prompt(topic_info, style_config)

        # Try to generate with retries and exponential backoff
        for attempt in range(self.max_retries):
            # Calculate delay with exponential backoff
            current_delay = self.retry_delay * (self.backoff_factor ** attempt)

            try:
                if self.llm_provider == "deepseek" and self.llm_client:
                    messages = [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ]
                    api_params = {
                        "model": self.model_name,
                        "messages": messages,
                        "temperature": generation_config.get("temperature", 0.85),
                    }
                    if "max_output_tokens" in generation_config:
                         api_params["max_tokens"] = generation_config["max_output_tokens"]

                    response = self.llm_client.chat.completions.create(**api_params)
                    
                    if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                        raw_script = response.choices[0].message.content.strip()
                        cleaned_script = self._clean_script_text(raw_script)
                        return cleaned_script
                    else:
                        print(f"Empty or malformed response from {self.llm_provider}, retrying ({attempt+1}/{self.max_retries})...")
                
                # Add elif block here for Gemini or other providers if supported
                # elif self.llm_provider == "gemini" and self.llm_client:
                    # ... Gemini API call logic ...
                    
                else:
                    # Should not happen if __init__ validation works
                    return f"[LLM provider {self.llm_provider} not supported or client not initialized]"

            except Exception as e:
                error_msg = str(e)
                print(f"Error generating script with {self.llm_provider} model {self.model_name} (attempt {attempt+1}/{self.max_retries}): {error_msg}")

                if "quota" in error_msg.lower() or "rate limit" in error_msg.lower() or "429" in error_msg:
                    print(f"Rate limit or quota error detected.")
                
                if attempt < self.max_retries - 1:
                    print(f"Waiting {current_delay:.1f} seconds before retrying...")
                    time.sleep(current_delay)
                else:
                    print(f"Max retries reached for script generation.")

            if attempt > self.max_retries // 2:
                prompt = self._modify_prompt_for_retry(prompt, attempt)
        
        return f"[Unable to generate script content after {self.max_retries} attempts using {self.llm_provider}. The API may be experiencing issues or the prompt is problematic.]"

    def _clean_script_text(self, text: str) -> str:
        """
        Removes problematic symbols and common script prefixes for TTS from the script text.
        Targets: ", ', * and prefixes like "Script:", "Narrator:".
        """
        import re # Import re for regular expressions

        # Remove specific symbols
        text_no_symbols = text.replace('"', '')  # Remove double quotes
        text_no_symbols = text_no_symbols.replace("'", "")  # Remove single quotes/apostrophes
        text_no_symbols = text_no_symbols.replace("*", "")  # Remove asterisks

        # Define common prefixes to remove (case-insensitive)
        # This list can be expanded. Handles optional parentheses, asterisks, and colons.
        prefixes_to_strip = [
            r"^\s*\(?\*?\s*script\s*\*?\)?\s*:\s*",
            r"^\s*\(?\*?\s*narrator\s*\*?\)?\s*:\s*",
            r"^\s*\(?\*?\s*full script\s*\*?\)?\s*:\s*",
            r"^\s*\(?\*?\s*scene\s*\*?\)?\s*:\s*",
            r"^\s*\(?\*?\s*text\s*\*?\)?\s*:\s*",
            r"^\s*\[script\]\s*",
            r"^\s*\[narrator\]\s*",
        ]

        cleaned_text = text_no_symbols
        for prefix_pattern in prefixes_to_strip:
            cleaned_text = re.sub(prefix_pattern, "", cleaned_text, flags=re.IGNORECASE | re.MULTILINE).lstrip()
            # Check if the script starts with the actual content after stripping a prefix
            # This helps avoid accidentally stripping parts of the script if it legitimately starts with "Script" for example.
            # However, given the prompt, the LLM should not be doing this.
            # For now, simple stripping is likely sufficient.

        # Final trim of any leading/trailing whitespace that might have been left or introduced
        return cleaned_text.strip()

    def _modify_prompt_for_retry(self, original_prompt: str, attempt: int) -> str:
        """
        Slightly modify the prompt for retry attempts to avoid same failure.
        
        Args:
            original_prompt: Original prompt that failed
            attempt: Current attempt number
            
        Returns:
            Modified prompt
        """
        # Add some variation based on the attempt number
        variations = [
            "Please provide a creative and detailed response to the following: ",
            "I need your help writing content about: ",
            "As an expert scriptwriter, please create content on: ",
            "Let's approach this from a different angle: ",
            "Considering this topic from a fresh perspective: "
        ]

        # Select a variation based on the attempt number
        prefix = variations[attempt % len(variations)]

        # For later attempts, simplify the prompt if it's very long
        if attempt > 5 and len(original_prompt) > 500:
            # Extract just the core topic from the prompt
            lines = original_prompt.split('\n')
            for line in lines:
                if line.startswith("Topic:"):
                    return f"{prefix}{line.replace('Topic:', '').strip()}"

        return f"{prefix}{original_prompt}"

    def _prepare_single_prompt(self, 
                               topic_info: Dict[str, Any],
                               style_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Prepare a prompt for generating a single complete script.
        
        Args:
            topic_info: Processed topic information including analysis.
            style_config: Optional style configuration from mastermind.
            
        Returns:
            Prepared prompt.
        """
        cleaned_topic = topic_info["cleaned_topic"]
        topic_type = topic_info["topic_type"]
        tone = topic_info["tone"] # Original tone from topic analysis
        style_keywords = topic_info["style_keywords"] # Original style from topic analysis

        # Override with style_config if provided
        if style_config:
            if "style_name" in style_config and style_config["style_name"]:
                # Prepend to existing or set as primary
                if style_keywords:
                    style_keywords.insert(0, style_config["style_name"])
                else:
                    style_keywords = [style_config["style_name"]]
            if "tone" in style_config and style_config["tone"]:
                tone = style_config["tone"] # Override tone
            if "key_elements" in style_config and style_config["key_elements"]:
                 # Add to enhancement or create one
                enhancement_str = f"Incorporate these key elements: {', '.join(style_config['key_elements'])}."
                if topic_info["enhancement"]:
                    topic_info["enhancement"] += f" Also, {enhancement_str.lower()}"
                else:
                    topic_info["enhancement"] = enhancement_str


        prompt = f"TOPIC: {cleaned_topic}\n"
        prompt += f"REQUESTED VIDEO TYPE: Short video (under 60 seconds)\n"
        prompt += f"TOPIC TYPE (for context): {topic_type}\n"
        prompt += f"DESIRED TONE: {tone}\n"
        if style_keywords:
            prompt += f"DESIRED STYLE: {', '.join(list(set(style_keywords)))}\n" # Use set to avoid duplicates

        if topic_info["enhancement"]:
            prompt += f"\nENHANCEMENT INSTRUCTIONS: {topic_info['enhancement']}\n"

        prompt += "\nSCRIPT REQUIREMENTS:\n"
        prompt += "- Generate a complete, single, continuous script.\n"
        prompt += "- The script should be engaging and suitable for a video of approximately 1 minute (55-65 seconds read time).\n"
        prompt += "- Ensure a clear beginning (strong hook), middle (engaging body), and end (satisfying conclusion).\n"
        prompt += "- Adhere strictly to 'Pure text only for TTS' - no visual cues, formatting, or stage directions.\n"

        # Add Topic-Type Specific Scripting Guidelines
        prompt += "\nTOPIC-SPECIFIC GUIDELINES:\n"
        if topic_type == "educational":
            prompt += f"- Your primary goal is to clearly explain '{cleaned_topic}'.\n"
            prompt += "- Use analogies, simple language, and structure the information logically for learning.\n"
            prompt += "- Focus on 1-2 key concise points due to the short format.\n"
        elif topic_type == "storytelling":
            prompt += f"- Your primary goal is to tell an engaging story about '{cleaned_topic}'.\n"
            prompt += "- Focus on creating a clear narrative arc (beginning, rising action, climax/turning point, resolution).\n"
            prompt += "- Develop characters (if any) briefly but effectively, focusing on their motivations or key traits.\n"
            prompt += "- Build emotional engagement appropriate to the story.\n"
        elif topic_type == "comparison":
            prompt += f"- Your primary goal is to compare elements related to '{cleaned_topic}'.\n"
            prompt += "- Present pros and cons, key differences, or similarities in a balanced and clear way.\n"
            prompt += "- Ensure the comparison is easy to follow and leads to a clear understanding for the listener.\n"
        elif topic_type == "entertainment" and tone == "humorous": # Example for a sub-type
             prompt += f"- Your primary goal is to entertain and be humorous about '{cleaned_topic}'.\n"
             prompt += "- Use wit, wordplay, or amusing anecdotes. Keep the energy light and engaging.\n"
        else: # Default or general type
            prompt += f"- Present information or tell a story about '{cleaned_topic}' in an engaging and clear manner.\n"
            prompt += "- Ensure the content is well-structured and easy to follow.\n"
        
        return prompt

    def generate_full_script(self, topic: str, style_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a complete script with all parts.
        
        Args:
            topic: Topic for the script
            style_config: Optional style configuration dictionary
            
        Returns:
            Dictionary with full script, parts, and metadata including analysis info
        """
        # Preprocess and analyze the topic
        topic_info = self.preprocess_topic(topic)
        print(f"Generating single short script ({topic_info['tone']} tone) for: {topic_info['cleaned_topic']}")

        try:
            # Generate the single, complete script text
            print("Generating full short script text...")
            # Pass style_config to _generate_single_script_text if it needs to influence prompt further
            full_script_text = self._generate_single_script_text(topic_info, style_config)

            if not full_script_text or full_script_text.startswith("[Unable to generate"):
                 raise ValueError(f"Script generation failed: {full_script_text}")

            # Metadata
            metadata = {
                "topic": topic_info["original_topic"],
                "cleaned_topic": topic_info["cleaned_topic"],
                "analysis": { # Retain original analysis for context
                    "type": topic_info["topic_type"],
                    "tone": topic_info["tone"], # This might be overridden by style_config for generation
                    "styles": topic_info["style_keywords"] # This might be augmented by style_config
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "model": self.model_name,
                "length_words": len(full_script_text.split())
                # "parts" metadata is no longer relevant
            }
            
            # If style_config was used and modified tone/style for generation, reflect it in metadata
            if style_config:
                metadata["applied_style_config"] = {
                    "requested_style_name": style_config.get("style_name"),
                    "requested_tone": style_config.get("tone"),
                    "requested_key_elements": style_config.get("key_elements")
                }


            # Save the script
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            topic_slug = topic_info["cleaned_topic"].lower().replace(" ", "_")[:30]
            filename = f"{timestamp}_{topic_slug}_fullscript.json" # Indicate it's a full script
            
            script_data = {
                "full_script": full_script_text, # This is the main script content
                # "parts" is removed as it's a single script
                "metadata": metadata
            }
            
            self.storage_manager.save_json(script_data, filename, "scripts")
            print(f"Full script saved: {filename}")
            return script_data

        except Exception as e:
            print(f"Error generating full script: {str(e)}")
            return {
                "error": str(e),
                "topic_info": topic_info
            }
