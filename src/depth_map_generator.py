import requests
import time
from typing import Dict, Optional

class DepthMapGenerator:
    def __init__(self, api_url: str = "https://sagesight-ai--depth-map-api-generate-depth-map.modal.run"):
        """
        Initialize the Depth Map Generator.

        Args:
            api_url: The API endpoint for the depth map generation service.
        """
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json"
        }
        self.max_retries = 3
        self.retry_delay = 5  # seconds

    def generate_depth_map(self, image_url: str) -> Optional[str]:
        """
        Generates a depth map for a given image URL.

        Args:
            image_url: The URL of the image to process.

        Returns:
            The URL of the generated depth map image, or None if an error occurs.
        """
        if not image_url:
            print("Error: No image URL provided for depth map generation.")
            return None

        data = {
            "image_url": image_url
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=data, timeout=60) # Added timeout
                response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

                depth_map_url: Optional[str] = None
                
                # Try to parse as JSON
                try:
                    parsed_response = response.json()

                    # Check if the parsed response is a list and take the first element if it's a dict
                    actual_response_data = None
                    if isinstance(parsed_response, list) and len(parsed_response) > 0:
                        if isinstance(parsed_response[0], dict):
                            actual_response_data = parsed_response[0]
                            print(f"API response for {image_url[:50]}... is a list, using first element (dict).")
                        # If the first element is not a dict, but the list itself might be the data (less likely for this API)
                        # elif len(parsed_response) == 1 and isinstance(parsed_response[0], str) and parsed_response[0].startswith("http"):
                        #     depth_map_url = parsed_response[0] # e.g. ["http://..."]
                    elif isinstance(parsed_response, dict):
                        actual_response_data = parsed_response # Response is directly a dict
                        print(f"API response for {image_url[:50]}... is a dict.")

                    if actual_response_data and isinstance(actual_response_data, dict):
                        possible_keys = ["depth_map_url", "output_url", "url", "result", "image_url"]
                        for key in possible_keys:
                            value = actual_response_data.get(key)
                            if isinstance(value, str) and value.startswith("http"):
                                depth_map_url = value
                                print(f"Found depth map URL in JSON response using key '{key}' for {image_url[:50]}...")
                                break
                        
                        if depth_map_url:
                            print(f"Successfully generated depth map for {image_url[:50]}...")
                            return depth_map_url
                    
                except requests.exceptions.JSONDecodeError:
                    print(f"Response for {image_url[:50]}... is not valid JSON. Raw text: {response.text[:200]}")
                    # Fallback: Check if the raw response text itself is a URL (less likely now we've seen the format)
                    if response.text and response.text.strip().startswith("http"):
                        depth_map_url = response.text.strip()
                        print(f"Using raw response text as depth map URL for {image_url[:50]}...")
                        return depth_map_url

                # If no URL found after trying JSON parsing and raw text
                if not depth_map_url:
                    print(f"Error: Depth map URL not extracted from API response for {image_url[:50]}....")
                    print(f"API Status Code: {response.status_code}")
                    print(f"API Response Text (first 500 chars): {response.text[:500]}")
                    return None

            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred while generating depth map for {image_url[:50]}... (Attempt {attempt + 1}/{self.max_retries}): {http_err}")
                # Log response text on HTTP error as well, if available
                if hasattr(http_err, 'response') and http_err.response is not None:
                    print(f"Error Response Text: {http_err.response.text[:500]}")
                if response.status_code == 429: # Rate limiting
                    print(f"Rate limited. Retrying in {self.retry_delay * (attempt + 1)} seconds...")
                    time.sleep(self.retry_delay * (attempt + 1))
                elif attempt == self.max_retries - 1:
                    print(f"Failed to generate depth map for {image_url[:50]}... after {self.max_retries} attempts due to HTTP error: {http_err}. Response: {response.text}")
                    return None
            except requests.exceptions.RequestException as req_err:
                print(f"Request exception occurred while generating depth map for {image_url[:50]}... (Attempt {attempt + 1}/{self.max_retries}): {req_err}")
                if attempt == self.max_retries - 1:
                    print(f"Failed to generate depth map for {image_url[:50]}... after {self.max_retries} attempts due to request error: {req_err}")
                    return None
            except Exception as e:
                print(f"An unexpected error occurred while generating depth map for {image_url[:50]}... (Attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    print(f"Failed to generate depth map for {image_url[:50]}... after {self.max_retries} attempts due to unexpected error: {e}")
                    return None
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)
        
        print(f"Failed to generate depth map for {image_url[:50]}... after all retries.")
        return None

if __name__ == '__main__':
    # Example usage:
    generator = DepthMapGenerator()
    test_image_url = "https://pub-09f5881ec7024e0faaff06eba5fb1e05.r2.dev/generated_image_1743124133.png" # Example URL
    depth_map_result_url = generator.generate_depth_map(test_image_url)

    if depth_map_result_url:
        print(f"Generated Depth Map URL: {depth_map_result_url}")
    else:
        print("Failed to generate depth map.")

    # Test with a potentially problematic URL or scenario
    # test_invalid_url = "http://invalid-url-for-testing.com/image.png"
    # depth_map_invalid_result = generator.generate_depth_map(test_invalid_url)
    # if depth_map_invalid_result:
    #     print(f"Generated Depth Map URL (invalid test): {depth_map_invalid_result}")
    # else:
    #     print("Failed to generate depth map for invalid URL (as expected).")

    # test_no_url = ""
    # depth_map_no_url_result = generator.generate_depth_map(test_no_url)
    # if depth_map_no_url_result:
    #     print(f"Generated Depth Map URL (no URL test): {depth_map_no_url_result}")
    # else:
    #     print("Failed to generate depth map for empty URL (as expected).")
