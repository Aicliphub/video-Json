"""
Video Generation Module

Handles the generation of the final video using Remotion.
"""
import os
import json
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

class VideoGenerator:
    """Handles generation of the final video using Remotion."""
    
    def __init__(self, 
                 remotion_path: Optional[str] = None,
                 output_dir: str = "output"):
        """
        Initialize the video generator.
        
        Args:
            remotion_path: Path to Remotion project
            output_dir: Directory for output videos
        """
        self.remotion_path = remotion_path or self._find_remotion_path()
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _find_remotion_path(self) -> str:
        """
        Find the Remotion project path.
        
        Returns:
            Path to Remotion project
        """
        # Check common locations
        possible_paths = [
            "remotion",
            "../remotion",
            "../../remotion",
            os.path.expanduser("~/remotion")
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "package.json")):
                return path
        
        # Default to "remotion" and let the user know
        print("Remotion project not found. Please specify the path when calling generate_video().")
        return "remotion"
    
    def generate_video(self, 
                      json_path: str, 
                      output_filename: Optional[str] = None,
                      resolution: str = "1080p",
                      fps: int = 30) -> str:
        """
        Generate video using Remotion.
        
        Args:
            json_path: Path to JSON file
            output_filename: Optional output filename
            resolution: Video resolution
            fps: Frames per second
            
        Returns:
            Path to generated video
        """
        # Check if Remotion project exists
        if not os.path.exists(self.remotion_path):
            raise ValueError(f"Remotion project not found at {self.remotion_path}")
        
        # Check if JSON file exists
        if not os.path.exists(json_path):
            raise ValueError(f"JSON file not found at {json_path}")
        
        # Load JSON to get metadata
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        
        # Generate output filename if not provided
        if not output_filename:
            topic_slug = json_data["metadata"]["topic"].lower().replace(" ", "_")[:30]
            output_filename = f"{topic_slug}.mp4"
        
        # Ensure output filename has .mp4 extension
        if not output_filename.endswith(".mp4"):
            output_filename += ".mp4"
        
        # Full path to output file
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Get resolution dimensions
        width, height = self._get_resolution_dimensions(resolution)
        
        # Build Remotion command
        # Note: This is a placeholder. In a real implementation, you would need to
        # configure Remotion correctly and use the appropriate command.
        command = [
            "npx",
            "remotion",
            "render",
            "src/index.tsx",
            "Main",
            output_path,
            "--props",
            f"file={json_path}",
            "--width",
            str(width),
            "--height",
            str(height),
            "--fps",
            str(fps)
        ]
        
        print(f"Generating video with command: {' '.join(command)}")
        print("Note: This is a placeholder. In a real implementation, you would need to configure Remotion correctly.")
        
        # In a real implementation, you would run the command:
        # subprocess.run(command, cwd=self.remotion_path, check=True)
        
        # For now, just return the output path
        return output_path
    
    def _get_resolution_dimensions(self, resolution: str) -> tuple:
        """
        Get width and height for a resolution.
        
        Args:
            resolution: Resolution string (e.g., "1080p", "4K")
            
        Returns:
            Tuple of (width, height)
        """
        resolutions = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "4K": (3840, 2160)
        }
        
        return resolutions.get(resolution, (1920, 1080))
