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
        
        # Check status periodically until completion
        job_id = job_data['job_id']
        check_count = 0
        while True:
            check_count += 1
            status_response = requests.get(f"{BASE_URL}/status/{job_id}")
            status_data = status_response.json()
            
            print(f"Check {check_count}: Status = {status_data['status']}")
            
            if status_data['status'] == 'completed':
                # Get final result
                try:
                    result_response = requests.get(f"{BASE_URL}/result/{job_id}")
                    if result_response.status_code == 200:
                        result_data = result_response.json()
                        print("\nGeneration completed successfully!")
                        
                        # Save the JSON result to file
                        output_file = "video_assets.json"
                        with open(output_file, 'w') as f:
                            json.dump(result_data, f, indent=2)
                        
                        print(json.dumps(result_data, indent=2))
                        return
                    else:
                        print(f"\nGeneration completed but result not available (Status: {result_response.status_code})")
                        
                        # Try to find the latest assets file
                        try:
                            assets_dir = "assets"
                            latest_file = max(
                                (os.path.join(assets_dir, f) for f in os.listdir(assets_dir) 
                                 if f.startswith("video_") and f.endswith(".json")),
                                key=os.path.getmtime
                            )
                            print(f"\nFound generated assets file: {latest_file}")
                            print("Contents:")
                            with open(latest_file) as f:
                                assets = json.load(f)
                            print(json.dumps(assets, indent=2))
                            
                            # Also save a copy locally
                            with open("video_assets.json", 'w') as f:
                                json.dump(assets, f, indent=2)
                            print("\nSaved copy as video_assets.json")
                        except Exception as e:
                            print(f"\nCould not locate assets file: {str(e)}")
                            print("Check assets directory manually for generated files")
                        return
                except Exception as e:
                    print(f"\nError getting result: {str(e)}")
                    print(f"Check assets directory for generated files")
                    return
            elif status_data['status'] == 'failed':
                print("\nGeneration failed!")
                return
            
            time.sleep(5)  # Wait 5 seconds between checks
        
    except requests.exceptions.RequestException as e:
        print(f"\nAPI request failed: {e}")

if __name__ == "__main__":
    test_video_generation()
