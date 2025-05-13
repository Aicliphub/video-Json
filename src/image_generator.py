import os
import time
import requests
import base64
import json
from typing import Dict, List, Any, Optional # Added Optional
from datetime import datetime
from src.utils.storage import StorageManager
from src.depth_map_generator import DepthMapGenerator # Added import

class ImageGenerator:
    def __init__(self, 
                 storage_manager: StorageManager, 
                 api_key: str, 
                 image_generator_config: Dict[str, str]):
        """
        Initialize the Image Generator.
        
        Args:
            storage_manager: Storage manager instance.
            api_key: API key for the image generation service (FreeFlux).
            image_generator_config: Dictionary with 'endpoint' and 'model'.
        """
        self.storage = storage_manager
        self.api_key = api_key
        self.config = image_generator_config
        self.endpoint = self.config.get("endpoint")
        self.model = self.config.get("model", "flux_1_schnell") # Default if not in config
        
        if not self.api_key:
            raise ValueError("FreeFlux API key is required for ImageGenerator.")
        if not self.endpoint:
             raise ValueError("FreeFlux endpoint is required for ImageGenerator.")
             
        self.headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {self.api_key}', # Use passed API key
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36' # Keep user-agent
        }

        # Validate API connectivity
        try:
            test_response = requests.get(
                self.endpoint.replace('/generate', '/health'),
                headers=self.headers,
                timeout=5
            )
            if test_response.status_code != 200:
                raise ConnectionError(f"API health check failed with status {test_response.status_code}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to FreeFlux API: {str(e)}. Please check your API key and network connectivity.")
        self.max_retries = 3
        self.retry_delay = 2
        self.depth_map_generator = DepthMapGenerator() # Instantiate DepthMapGenerator
        print(f"ImageGenerator initialized (Endpoint: {self.endpoint}, Model: {self.model}). API connection validated. DepthMapGenerator also initialized.")

    def generate_image(self, prompt: str, segment_id: str) -> Dict[str, Optional[str]]:
        """
        Generate a single image from prompt and its corresponding depth map.
        Returns a dictionary with 'image_url' and 'depth_map_url'.
        
        Raises:
            ConnectionError: If API is unreachable after retries
            ValueError: If API returns invalid response
        """
        files = {
            'prompt': (None, prompt),
            'model': (None, self.model), # Use model from config
            'size': (None, '9_16'), # Use horizontal aspect ratio (matches working API)
            'lora': (None, ''),
            'style': (None, 'no_style'),
            'color': (None, ''),
            'lighting': (None, ''),
            'composition': (None, '')
        }

        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint, # Use endpoint from config
                    headers=self.headers,
                    files=files,
                    timeout=10
                )

                if response.status_code == 200:
                    image_data_url = response.json().get('result')
                    if image_data_url and image_data_url.startswith("data:image/png;base64,"):
                        base64_image_data = image_data_url.split(",")[1]
                        try:
                            image_bytes = base64.b64decode(base64_image_data)
                        except Exception as e:
                            raise ValueError(f"Invalid image data received from API: {str(e)}")
                        
                        image_r2_url: Optional[str] = None
                        try:
                            # Save directly to R2 and get URL
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"image_{segment_id}_{timestamp}"
                            image_r2_url = self.storage.save_image(image_bytes, filename, "images")
                        except ValueError as e:
                            print(f"Failed to upload to R2: {str(e)}")
                            # If R2 fails, return the data URL directly
                            return {"image_url": image_data_url, "depth_map_url": None}

                        if image_r2_url:
                            print(f"Successfully generated and uploaded image for {segment_id}: {image_r2_url}")
                            depth_map_url = self.depth_map_generator.generate_depth_map(image_r2_url)
                            return {"image_url": image_r2_url, "depth_map_url": depth_map_url}
                        else: # Should not happen if save_image doesn't raise error and returns None
                             print(f"Image R2 URL is None for {segment_id} even after successful save_image call. Skipping depth map.")
                             return {"image_url": None, "depth_map_url": None} # Or handle as error

                # Handle non-200 responses
                error_msg = f"API returned status {response.status_code}"
                if response.text:
                    try:
                        error_detail = response.json().get('error', response.text)
                        error_msg += f": {error_detail}"
                    except:
                        error_msg += f": {response.text[:200]}"
                last_error = error_msg
                print(f"Attempt {attempt+1} failed for {segment_id}: {error_msg}")
                
            except requests.exceptions.RequestException as e:
                last_error = f"Network error: {str(e)}"
                print(f"Attempt {attempt+1} network error for {segment_id}: {str(e)}")
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                print(f"Attempt {attempt+1} unexpected error for {segment_id}: {str(e)}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1)) # Exponential backoff
        
        raise ConnectionError(
            f"Failed to generate image for {segment_id} after {self.max_retries} attempts. "
            f"Last error: {last_error}. Please check your API key and network connectivity."
        )

    def generate_batch(self, prompts: Dict[str, str], batch_size: int = 20) -> Dict[str, Dict[str, Optional[str]]]:
        """
        Generate multiple images and their depth maps from prompts dictionary {segment_id: prompt}.
        Returns a dictionary {segment_id: {'image_url': url, 'depth_map_url': url}}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results: Dict[str, Dict[str, Optional[str]]] = {}
        prompt_items = list(prompts.items())
        total_batches = (len(prompt_items) + batch_size - 1) // batch_size
        
        # Adaptive concurrency settings
        MIN_WORKERS = 2
        MAX_WORKERS = 20
        current_workers = 15  # Further increased initial workers
        backoff_factor = 1
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(prompt_items))
            batch_items = prompt_items[start_idx:end_idx]
            
            print(f"\nProcessing batch {batch_num + 1}/{total_batches} ({len(batch_items)} images @ {current_workers} workers)")
            
            batch_results = {}
            rate_limited = False
            
            with ThreadPoolExecutor(max_workers=current_workers) as executor:
                futures = {
                    executor.submit(self.generate_image, prompt, segment_id): segment_id
                    for segment_id, prompt in batch_items
                }
                
                for future in as_completed(futures):
                    segment_id = futures[future]
                    try:
                        image_data = future.result() # This will be {'image_url': ..., 'depth_map_url': ...}
                        batch_results[segment_id] = image_data
                        if image_data.get("image_url"):
                            print(f"✓ Generated image (and attempted depth map) for {segment_id}")
                        else:
                            print(f"✗ Image generation failed for {segment_id}, depth map skipped.")
                        
                        # Successful request - gradually increase concurrency
                        if not rate_limited and current_workers < MAX_WORKERS:
                            current_workers = min(current_workers + 1, MAX_WORKERS)
                            
                    except Exception as e:
                        print(f"✗ Failed to process image generation task for {segment_id}: {str(e)}")
                        batch_results[segment_id] = {"image_url": None, "depth_map_url": None}
                        
                        # If rate limited, reduce concurrency
                        if "429" in str(e): # Check if the exception string indicates a rate limit
                            rate_limited = True
                            current_workers = max(current_workers // 2, MIN_WORKERS)
                            backoff_factor = min(backoff_factor * 2, 60)  # Max 60s backoff
                            time.sleep(backoff_factor)
                        else:
                            backoff_factor = 1
            
            # Save batch results
            results.update(batch_results)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"image_batch_{batch_num+1}_of_{total_batches}_{timestamp}.json"
            self.storage.save_json({
                "results": batch_results,
                "batch_number": batch_num + 1,
                "total_batches": total_batches,
                "timestamp": timestamp,
                "success_rate": f"{len([res for res in batch_results.values() if res.get('image_url')])}/{len(batch_results)} images generated",
                "depth_map_success_rate": f"{len([res for res in batch_results.values() if res.get('depth_map_url')])}/{len([res for res in batch_results.values() if res.get('image_url')])} depth maps generated for successful images",
                "concurrency": current_workers
            }, filename, "images")
        
        return results
