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
    def __init__(self, storage_manager: StorageManager):
        """
        Initialize the Image Generator with hardcoded settings.
        
        Args:
            storage_manager: Storage manager instance.
        """
        self.storage = storage_manager
        self.api_key = "084bf5ff-cd3b-4c09-abaa-d2334322f562"  # Using key from your .env; REPLACE IF THIS IS NOT YOUR VALID KEY
        self.endpoint = "https://api.freeflux.ai/v1/images/generate"
        self.model = "flux_1_schnell"
        
        # It's good practice to check if the key looks like a placeholder or a known default,
        # but for now, we'll assume the user intends to use this key or will replace it.
        # Consider adding a more robust check or warning if this key is known to be non-functional.
        print(f"ImageGenerator using API Key: {self.api_key}")

        self.headers = {
            'accept': 'application/json',
            'authorization': f'Bearer {self.api_key}',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
        }
        self.max_retries = 3
        self.retry_delay = 2
        self.depth_map_generator = DepthMapGenerator() # Instantiate DepthMapGenerator
        print(f"ImageGenerator initialized (Endpoint: {self.endpoint}, Model: {self.model}). DepthMapGenerator also initialized.")

    def generate_image(self, prompt: str, segment_id: str) -> Dict[str, Optional[str]]:
        """
        Generate a single image from prompt and its corresponding depth map.
        Returns a dictionary with 'image_url' and 'depth_map_url'.
        """
        files = {
            'prompt': (None, prompt),
            'model': (None, self.model), # Use model from config
            'size': (None, '9_16'), # Keep vertical aspect ratio
            'lora': (None, ''),
            'style': (None, 'no_style'),
            'color': (None, ''),
            'lighting': (None, ''),
            'composition': (None, '')
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.endpoint, # Use endpoint from config
                    headers=self.headers,
                    files=files
                )

                if response.status_code == 200:
                    image_data_url = response.json().get('result')
                    if image_data_url and image_data_url.startswith("data:image/png;base64,"):
                        base64_image_data = image_data_url.split(",")[1]
                        image_bytes = base64.b64decode(base64_image_data)
                        
                        image_r2_url: Optional[str] = None
                        try:
                            # Save directly to R2 and get URL
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"image_{segment_id}_{timestamp}"
                            image_r2_url = self.storage.save_image(image_bytes, filename, "images")
                        except ValueError as e:
                            print(f"Failed to upload to R2: {str(e)}")
                            # If R2 fails, we can't generate a depth map from a public URL
                            # Consider if image_data_url should be used for depth map or if it's better to skip
                            print(f"Skipping depth map generation for {segment_id} due to R2 upload failure.")
                            return {"image_url": image_data_url, "depth_map_url": None} # Return data URL if R2 fails

                        if image_r2_url:
                            print(f"Successfully generated and uploaded image for {segment_id}: {image_r2_url}")
                            depth_map_url = self.depth_map_generator.generate_depth_map(image_r2_url)
                            return {"image_url": image_r2_url, "depth_map_url": depth_map_url}
                        else: # Should not happen if save_image doesn't raise error and returns None
                             print(f"Image R2 URL is None for {segment_id} even after successful save_image call. Skipping depth map.")
                             return {"image_url": None, "depth_map_url": None} # Or handle as error
                else:
                    print(f"Attempt {attempt+1} failed to generate image for {segment_id}. Status: {response.status_code}, Response: {response.text}")
            
            except requests.exceptions.RequestException as e: # More specific exception for network errors
                print(f"RequestException on attempt {attempt+1} for {segment_id}: {str(e)}")
            except Exception as e: # General exception
                print(f"Unexpected error on attempt {attempt+1} for {segment_id}: {str(e)}")

            # Common retry logic
            if attempt < self.max_retries - 1:
                print(f"Retrying image generation for {segment_id} (attempt {attempt+2}/{self.max_retries})...")
                time.sleep(self.retry_delay)
            else:
                print(f"Failed to generate image for {segment_id} after {self.max_retries} attempts.")
        
        # This print statement might be redundant if the one in the loop's else branch covers all failures.
        # print(f"Failed to generate image for {segment_id} after multiple attempts.") 
        return {"image_url": None, "depth_map_url": None} # Return None for both if image generation fails

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
