import os
import time
import requests
import base64
import json
import traceback
from typing import Dict, List, Any, Optional # Added Optional
from datetime import datetime
from src.utils.storage import StorageManager
from src.depth_map_generator import DepthMapGenerator # Added import

class ImageGenerator:
    def __init__(self, storage_manager: StorageManager):
        """
        Initialize the Image Generator with configurable providers.
        
        Args:
            storage_manager: Storage manager instance.
        """
        self.storage = storage_manager
        self.provider = {
            'name': 'FreeFlux',
            'api_key': "084bf5ff-cd3b-4c09-abaa-d2334322f562",
            'endpoint': "https://api.freeflux.ai/v1/images/generate",
            'model': "flux_1_schnell",
            'headers': {
                'accept': 'application/json',
                'authorization': 'Bearer 084bf5ff-cd3b-4c09-abaa-d2334322f562',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'
            }
        }
        self.service_available = True
        
        # Test API connectivity during initialization
        try:
            test_response = requests.get(
                "https://api.freeflux.ai",
                headers={'accept': 'application/json'},
                timeout=5
            )
            if test_response.status_code != 200:
                error_msg = f"FreeFlux API may be unavailable (Status: {test_response.status_code})"
                print(f"ERROR: {error_msg}")
                print(f"DEBUG: API Response: {test_response.text}")
                self.service_available = False
        except Exception as e:
            error_msg = f"FreeFlux API connectivity test failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"DEBUG: Exception Type: {type(e).__name__}")
            self.service_available = False

        self.max_retries = 3
        self.retry_delay = 2
        self.depth_map_generator = DepthMapGenerator() # Instantiate DepthMapGenerator
        print(f"ImageGenerator initialized with FreeFlux provider")

    def generate_image(self, prompt: str, segment_id: str) -> Dict[str, Optional[str]]:
        """
        Generate a single image from prompt and its corresponding depth map.
        Returns a dictionary with 'image_url' and 'depth_map_url'.
        
        Will return {'image_url': None, 'depth_map_url': None} if:
        - No providers are available
        - Generation fails after max retries
        - Storage upload fails
        """
        if not self.service_available:
            print(f"Skipping image generation for {segment_id} - FreeFlux API unavailable")
            return {"image_url": None, "depth_map_url": None}
            
        provider = self.provider
        files = {
            'prompt': (None, prompt),
            'model': (None, provider['model']),
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
                    provider['endpoint'],
                    headers=provider['headers'],
                    files=files,
                    timeout=30
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        image_data_url = response_data.get('result')
                        if not image_data_url:
                            print(f"ERROR: No image data in API response for {segment_id}")
                            print(f"DEBUG: Full API Response: {response_data}")
                            continue
                            
                        base64_image_data = image_data_url.split(",")[1]
                        image_bytes = base64.b64decode(base64_image_data)
                        
                        image_r2_url: Optional[str] = None
                        try:
                            # Save directly to R2 and get URL
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"image_{segment_id}_{timestamp}"
                            image_r2_url = self.storage.save_image(image_bytes, filename, "images")
                            
                            if image_r2_url:
                                print(f"Successfully generated and uploaded image for {segment_id}: {image_r2_url}")
                                depth_map_url = self.depth_map_generator.generate_depth_map(image_r2_url)
                                return {"image_url": image_r2_url, "depth_map_url": depth_map_url}
                            else:
                                print(f"CRITICAL: Image R2 URL is None for {segment_id} after successful save_image call")
                                print("ACTION: This indicates a potential bug in the storage manager")
                                return {"image_url": None, "depth_map_url": None}
                                
                        except ValueError as e:
                            print(f"ERROR: Failed to upload image to R2 for {segment_id}: {str(e)}")
                            print(f"DEBUG: Image Size: {len(image_bytes)} bytes")
                            print(f"DEBUG: Filename: {filename}")
                            print(f"ACTION: Skipping depth map generation for {segment_id} due to R2 upload failure")
                            return {"image_url": image_data_url, "depth_map_url": None} # Return data URL if R2 fails
                            
                    except json.JSONDecodeError as e:
                        print(f"ERROR: Failed to parse API response JSON for {segment_id}")
                        print(f"DEBUG: Response Text: {response.text}")
                        print(f"EXCEPTION: {str(e)}")
                        continue
                else:
                    print(f"ERROR: Attempt {attempt+1} failed for {segment_id}")
                    print(f"STATUS: {response.status_code}")
                    print(f"RESPONSE: {response.text}")
                    print(f"HEADERS: {response.headers}")
            
            except requests.exceptions.RequestException as e:
                print(f"NETWORK ERROR: Attempt {attempt+1} for {segment_id}")
                print(f"EXCEPTION: {type(e).__name__}: {str(e)}")
                if hasattr(e, 'request'):
                    print(f"REQUEST: {e.request.method} {e.request.url}")
            except Exception as e:
                print(f"UNEXPECTED ERROR: Attempt {attempt+1} for {segment_id}")
                print(f"EXCEPTION: {type(e).__name__}: {str(e)}")
                print(f"STACKTRACE: {traceback.format_exc()}")

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
