import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_video_generation():
    # Test POST /generate
    print("Testing video generation API...")
    prompt = "A tutorial about Python programming in educational style"
    
    try:
        # Start generation
        print(f"Sending request with prompt: {prompt}")
        response = requests.post(
            f"{BASE_URL}/generate",
            json={"input_prompt": prompt},
            headers={"Content-Type": "application/json"}
        )
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        job_data = response.json()
        print(f"Started generation job: {job_data['job_id']}")
        
        # Check status periodically
        job_id = job_data['job_id']
        max_checks = 10
        for i in range(max_checks):
            status_response = requests.get(f"{BASE_URL}/status/{job_id}")
            status_data = status_response.json()
            
            print(f"Check {i+1}/{max_checks}: Status = {status_data['status']}")
            
            if status_data['status'] == 'completed':
                # Get final result
                try:
                    result_response = requests.get(f"{BASE_URL}/result/{job_id}")
                    result_response.raise_for_status()
                    result_data = result_response.json()
                    print("\nGeneration completed successfully!")
                    
                    # Save the JSON assets to file
                    output_file = "video_assets.json"
                    with open(output_file, 'w') as f:
                        json.dump(result_data['assets'], f, indent=2)
                    
                    print(json.dumps(result_data['assets']['content'], indent=2))
                    return
                except Exception as e:
                    print(f"\nFailed to get result: {str(e)}")
                    print(f"Response content: {result_response.text if 'result_response' in locals() else 'N/A'}")
                    return
            elif status_data['status'] == 'failed':
                print("\nGeneration failed!")
                return
            
            time.sleep(5)  # Wait 5 seconds between checks
            
        print("\nMax checks reached without completion")
        
    except requests.exceptions.RequestException as e:
        print(f"\nAPI request failed: {e}")

if __name__ == "__main__":
    test_video_generation()
