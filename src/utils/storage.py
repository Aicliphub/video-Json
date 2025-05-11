"""
Storage utilities for managing files and cloud storage
"""
import os
import json
import boto3
import random
import string
from datetime import datetime
from typing import Dict, Any, Optional, BinaryIO, Union
from pathlib import Path

class StorageManager:
    """Manages file storage both locally and in cloud storage (R2)."""
    
    def __init__(self, r2_config: Dict[str, str], local_base_path: str = "assets"):
        """
        Initialize the storage manager.
        
        Args:
            r2_config: Dictionary with R2 configuration
            local_base_path: Base path for local storage
        """
        self.r2_config = r2_config
        self.local_base_path = local_base_path
        
        # Create local directories if they don't exist
        for subdir in ["audio", "images", "scripts"]:
            os.makedirs(os.path.join(local_base_path, subdir), exist_ok=True)
        
        # Initialize R2 client if configuration is provided
        self.s3_client = None
        if all(r2_config.values()):
            self.s3_client = boto3.client(
                's3',
                endpoint_url=r2_config['endpoint_url'],
                aws_access_key_id=r2_config['access_key_id'],
                aws_secret_access_key=r2_config['secret_access_key']
            )
    
    def save_local_file(self, content: Union[str, bytes], 
                        directory: str, 
                        filename: Optional[str] = None,
                        extension: str = "txt") -> str:
        """
        Save content to a local file.
        
        Args:
            content: Content to save (string or bytes)
            directory: Subdirectory under local_base_path
            filename: Optional filename (generated if not provided)
            extension: File extension
            
        Returns:
            Path to the saved file
        """
        # Create directory if it doesn't exist
        full_dir = os.path.join(self.local_base_path, directory)
        os.makedirs(full_dir, exist_ok=True)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            filename = f"{timestamp}_{random_str}.{extension}"
        elif not filename.endswith(f".{extension}"):
            filename = f"{filename}.{extension}"
            
        # Full path to file
        file_path = os.path.join(full_dir, filename)
        
        # Save the file
        mode = "wb" if isinstance(content, bytes) else "w"
        encoding = None if isinstance(content, bytes) else "utf-8"
        
        with open(file_path, mode=mode, encoding=encoding) as f:
            f.write(content)
            
        return file_path
    
    def upload_to_r2(self, 
                     content: Union[str, bytes, BinaryIO],
                     object_name: Optional[str] = None,
                     content_type: Optional[str] = None,
                     directory: Optional[str] = None) -> str:
        """
        Upload content to R2 storage.
        
        Args:
            content: Content to upload (string, bytes, or file-like object)
            object_name: Optional object name (generated if not provided)
            content_type: Optional content type
            directory: Optional directory prefix
            
        Returns:
            Public URL of the uploaded object
        """
        if not self.s3_client:
            raise ValueError("R2 client not initialized. Check your configuration.")
            
        # Generate object name if not provided
        if not object_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
            
            # Determine extension based on content_type
            extension = "txt"
            if content_type:
                if content_type.startswith("image/"):
                    extension = content_type.split("/")[1]
                elif content_type == "audio/wav":
                    extension = "wav"
                elif content_type == "audio/mp3":
                    extension = "mp3"
                    
            object_name = f"{timestamp}_{random_str}.{extension}"
            
        # Add directory prefix if provided
        if directory:
            object_name = f"{directory}/{object_name}"
            
        # Upload to R2
        try:
            # Convert string to bytes if needed
            if isinstance(content, str):
                content = content.encode('utf-8')
                if not content_type:
                    content_type = "text/plain"
                    
            # Prepare upload parameters
            upload_args = {
                'Bucket': self.r2_config['bucket_name'],
                'Key': object_name,
                'Body': content
            }
            
            # Add content type if provided
            if content_type:
                upload_args['ContentType'] = content_type
                
            # Upload the file
            self.s3_client.put_object(**upload_args)
            
            # Generate public URL
            url = f"https://{self.r2_config['public_domain']}/{object_name}"
            return url
            
        except Exception as e:
            print(f"Error uploading to R2: {str(e)}")
            raise
    
    def save_json(self, data: Dict[str, Any], filename: str, directory: str = "") -> str:
        """
        Save data as JSON file.
        
        Args:
            data: Dictionary to save as JSON
            filename: Filename (without extension)
            directory: Optional subdirectory
            
        Returns:
            Path to the saved file
        """
        # Ensure filename has .json extension
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
            
        # Full directory path
        full_dir = self.local_base_path
        if directory:
            full_dir = os.path.join(full_dir, directory)
            
        # Create directory if it doesn't exist
        os.makedirs(full_dir, exist_ok=True)
        
        # Full path to file
        file_path = os.path.join(full_dir, filename)
        
        # Save the file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return file_path
    
    def save_image(self, image_bytes: bytes, filename: str, folder: str = "images") -> str:
        """
        Save image bytes directly to R2 storage and return public URL.
        
        Args:
            image_bytes: Image data as bytes
            filename: Name for the image file
            folder: Storage folder
            
        Returns:
            Public URL of the saved image
            
        Raises:
            ValueError: If R2 client is not configured
        """
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
            
        if not self.s3_client:
            raise ValueError("R2 client not configured - cannot upload image")
            
        return self.upload_to_r2(
            content=image_bytes,
            object_name=filename,
            content_type="image/png",
            directory=folder
        )

    def load_json(self, filename: str, directory: str = "") -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            filename: Filename (without extension)
            directory: Optional subdirectory
            
        Returns:
            Dictionary with loaded data
        """
        # Ensure filename has .json extension
        if not filename.endswith(".json"):
            filename = f"{filename}.json"
            
        # Full directory path
        full_dir = self.local_base_path
        if directory:
            full_dir = os.path.join(full_dir, directory)
            
        # Full path to file
        file_path = os.path.join(full_dir, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return {}
        
        # Load the file
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
