import json
from openai import OpenAI # Changed for DeepSeek
from typing import Dict, Any

class StyleParser:
    """Interprets natural language style directives into structured config using DeepSeek."""
    
    def __init__(self, deepseek_api_key: str, deepseek_config: Dict[str, str], default_style=None):
        if not deepseek_api_key or not deepseek_config:
            raise ValueError("DeepSeek API key and config are required for StyleParser.")
        
        self.llm = OpenAI(
            api_key=deepseek_api_key,
            base_url=deepseek_config.get("base_url")
        )
        self.model_name = deepseek_config.get("chat_model", "deepseek-chat")
        print(f"StyleParser initialized with DeepSeek client (Model: {self.model_name}).")
        
        self.default_style = default_style or {
            "video_style": {
                "genre": "documentary",
                "pace": "medium"
            },
            "visual_style": {
                "art_style": "realistic",
                "color_palette": "natural"
            },
            "audio_style": {
                "voice_tone": "neutral",
                "music_style": "ambient"
            }
        }
        
    def parse_style(self, style_directive: str) -> Dict[str, Any]:
        """Convert natural language style to structured config."""
        prompt = f"""
        Convert this video style description to structured JSON:
        "{style_directive}"
        
        Output format (must be valid JSON):
        {{
            "video_style": {{
                "genre": "string or null",
                "pace": "string (e.g., slow, medium, fast) or null",
                "intro_style": "string or null"
            }},
            "visual_style": {{
                "art_style": "string (e.g., Pixar, photorealistic, anime) or null",
                "color_palette": "string (e.g., vibrant, monochrome, pastel) or null",
                "lighting": "string (e.g., cinematic, dramatic, soft) or null",
                "composition": "string (e.g., wide shots, close-ups) or null"
            }},
            "audio_style": {{
                "voice_tone": "string (e.g., authoritative, friendly, energetic) or null",
                "music_style": "string (e.g., orchestral, electronic, none) or null",
                "sound_effects": "string (e.g., realistic, minimal, stylized) or null"
            }}
        }}
        Ensure all fields are present, using null if no specific information is found for a field.
        Focus on extracting visual style elements like art_style, color_palette, lighting, and composition accurately.
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at converting natural language video style descriptions into structured JSON. Adhere strictly to the provided JSON output format."},
            {"role": "user", "content": prompt}
        ]
        
        response_text = ""
        try:
            completion = self.llm.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}, # Request JSON output
                temperature=0.2 # Low temperature for more deterministic parsing
            )
            response_text = completion.choices[0].message.content
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse style JSON from LLM: {str(e)}\nLLM Response was: {response_text}")
        except Exception as e:
            # Catch-all for other API errors or issues
            raise ValueError(f"Style parsing with LLM failed: {str(e)}\nLLM Response was: {response_text}")

    def validate_style(self, style_config: Dict[str, Any]) -> bool:
        """Validate style config against available capabilities."""
        # TODO: Implement validation logic
        return True
