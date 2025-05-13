"""
Configuration Management Utilities
"""
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ConfigManager:
    """Manages model and generation configurations."""

    DEFAULT_CONFIG = {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    def __init__(self, custom_config=None):
        """Initialize with default or custom configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        if custom_config:
            self.config.update(custom_config)

    def get_config(self, **overrides):
        """Get configuration with optional overrides."""
        if not overrides:
            return self.config.copy()

        config = self.config.copy()
        config.update(overrides)
        return config

    @staticmethod
    def from_env(prefix=""):
        """Load configuration from environment variables with optional prefix."""
        config = ConfigManager.DEFAULT_CONFIG.copy()

        # Map of environment variable names to config keys
        env_mapping = {
            f"{prefix}TEMPERATURE": "temperature",
            f"{prefix}TOP_P": "top_p",
            f"{prefix}TOP_K": "top_k",
            f"{prefix}MAX_TOKENS": "max_output_tokens",
            f"{prefix}MAX_RETRIES": "max_retries",
            f"{prefix}RETRY_DELAY": "retry_delay",
            f"{prefix}BACKOFF_FACTOR": "backoff_factor",
            f"{prefix}KEY_FAILURE_TIMEOUT": "key_failure_timeout",
        }

        # Check for environment variables
        for env_var, config_key in env_mapping.items():
            if os.getenv(env_var):
                # Convert to appropriate type
                value = os.getenv(env_var)
                if config_key in ["temperature", "top_p", "retry_delay", "backoff_factor"]:
                    config[config_key] = float(value)
                elif config_key in ["top_k", "max_output_tokens", "max_retries", "key_failure_timeout"]:
                    config[config_key] = int(value)
                else:
                    config[config_key] = value

        return ConfigManager(config)


class ProjectConfig:
    """Central configuration for the entire project."""
    
    def __init__(self):
        """Initialize project configuration from environment variables."""
        # Load environment variables
        load_dotenv()
        
        # API Keys
        self.gemini_api_keys = os.getenv("GEMINI_API_KEYS", "").split(",")
        self.allvoicelab_api_key = os.getenv("ALLVOICELAB_API_KEY", "ak_85a73e532c7798494e280ad243f114e12100") # Default matches hardcoded value
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "sk-ca74a33b93cd4c30be05b27b1f1b5128") # Default matches hardcoded value
        self.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY") # No default, should be set
        # self.freeflux_api_key = os.getenv("FREEFLUX_API_KEY", "084bf5ff-cd3b-4c09-abaa-d2334322f562") # No longer loaded from .env
        
        # Style Configuration
        self.default_style = os.getenv("DEFAULT_STYLE", "")
        
        # R2 Storage Configuration
        self.r2_config = {
            "access_key_id": os.getenv("R2_ACCESS_KEY_ID", ""),
            "secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY", ""),
            "endpoint_url": os.getenv("R2_ENDPOINT_URL", ""),
            "bucket_name": os.getenv("R2_BUCKET_NAME", ""),
            "public_domain": os.getenv("R2_PUBLIC_DOMAIN", "")
        }
        
        # TTS Configuration (AllVoiceLab specific + general)
        self.tts_config = {
            "provider": os.getenv("TTS_PROVIDER", "allvoicelab"), # Default provider
            "voice_id": int(os.getenv("TTS_VOICE_ID", "267361701140104083")), # Default voice
            "model": os.getenv("TTS_MODEL", "tts-multilingual"), # Default model
            "allvoicelab_endpoint": os.getenv("ALLVOICELAB_API_ENDPOINT", "https://api.allvoicelab.com/v1/text-to-speech/create") # Default endpoint
            # Note: Existing Zyphra-specific TTS settings in .env are not loaded here unless TTS_PROVIDER is changed
        }
        
        # LLM Configuration (Gemini specific + DeepSeek specific)
        self.llm_config = ConfigManager.from_env("GEMINI_").config # Gemini specific LLM params
        self.deepseek_config = {
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "chat_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat") # Matches .env default
        }
        
        # Transcriber Configuration (Deepgram specific)
        self.transcriber_config = {
            "model": os.getenv("DEEPGRAM_MODEL", "nova-3") # Default model
        }

        # Image Generator Configuration (FreeFlux specific) - No longer loaded from .env
        # self.image_generator_config = {
        #     "endpoint": os.getenv("FREEFLUX_API_ENDPOINT", "https://api.freeflux.ai/v1/images/generate"),
        #     "model": os.getenv("FREEFLUX_MODEL", "flux_1_schnell")
        # }

        # Project paths
        self.paths = {
            "assets": "assets",
            "audio": "assets/audio",
            "images": "assets/images",
            "scripts": "assets/scripts",
            "reference": "assets/reference",
            "output": "output"
        }
        
    def validate(self):
        """Validate that all required configuration is present."""
        # Check API keys
        if not self.gemini_api_keys or not any(self.gemini_api_keys):
            raise ValueError("No Gemini API keys found in configuration")
        
        # Check required API keys
        required_keys = {
            "AllVoiceLab": self.allvoicelab_api_key,
            "DeepSeek": self.deepseek_api_key,
            "Deepgram": self.deepgram_api_key
            # "FreeFlux": self.freeflux_api_key # No longer validated here
        }
        for service, key in required_keys.items():
            if not key:
                 # Allow Gemini keys to be optional if DeepSeek is primary LLM? For now, require all.
                 # if service == "Gemini" and self.deepseek_api_key: continue 
                 raise ValueError(f"Missing API key for {service} in configuration (check .env)")

        # Check R2 configuration
        for key, value in self.r2_config.items():
            if not value:
                raise ValueError(f"Missing R2 configuration: {key}")
        
        # Check TTS configuration
        if not self.tts_config["voice_id"]:
            raise ValueError("No TTS voice_id specified")
        if not self.tts_config["allvoicelab_endpoint"]:
             raise ValueError("Missing AllVoiceLab API endpoint")

        # Check DeepSeek config
        if not self.deepseek_config["base_url"] or not self.deepseek_config["chat_model"]:
             raise ValueError("Missing DeepSeek base URL or model name")
             
        # Check Deepgram config
        if not self.transcriber_config["model"]:
             raise ValueError("Missing Deepgram model name")

        # Check FreeFlux config - No longer validated here
        # if not self.image_generator_config["endpoint"] or not self.image_generator_config["model"]:
        #      raise ValueError("Missing FreeFlux endpoint or model name")

        return True
