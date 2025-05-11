import requests
import threading
import time

BASE_URL = "http://localhost:8000"
PROMPTS = [
    "Python tutorial for beginners",
    "Advanced machine learning concepts", 
    "Web development with FastAPI",
    "Data science fundamentals",
    "Cloud computing overview"
]

def make_request(prompt):
    start = time.time()
    print(f"Starting request for: {prompt}")
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"input_prompt": prompt}
    )
    job_id = response.json()["job_id"]
    print(f"Job {job_id} started for: {prompt} (took {time.time()-start:.2f}s)")

if __name__ == "__main__":
    threads = []
    for prompt in PROMPTS:
        t = threading.Thread(target=make_request, args=(prompt,))
        threads.append(t)
        t.start()
        time.sleep(0.5)  # Stagger requests slightly
    
    for t in threads:
        t.join()
