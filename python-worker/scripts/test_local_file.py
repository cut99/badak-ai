import os
import time
import requests
import threading
import http.server
import socketserver
from pathlib import Path

# Configuration
# Configuration
API_URL = "http://localhost:8000/api/batch-process"
API_KEY = "your-secure-api-key-here"  # Ensure this matches your .env or default
# We will look for images in this directory
SAMPLE_IMAGE_DIR = "sampleimage"
LOCAL_PORT = 9000

def get_local_ip():
    """Get local IP address that the API can reach."""
    # Since API is likely running locally, localhost or 127.0.0.1 is fine.
    # But if API is in docker, we might need host.docker.internal
    return "127.0.0.1"

def serve_directory(directory, port):
    """Serve a directory over HTTP in a background thread."""
    handler = http.server.SimpleHTTPRequestHandler
    
    # Change to root dir
    current_dir = os.getcwd()
    os.chdir(directory)
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"Temporary file server running at http://localhost:{port}")
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48: # Address already in use
             print(f"Port {port} already in use, assuming server is already running...")
        else:
            raise
    finally:
        os.chdir(current_dir)


def main():
    # 1. Setup paths
    # We assume script is run from project root or we can find sampleimage
    possible_paths = [
        "sampleimage",
        "../sampleimage",
        "/Users/classic/webku/BADAK/AI-Face-worker/sampleimage"
    ]
    
    image_dir = None
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            image_dir = os.path.abspath(path)
            break
            
    if not image_dir:
        print("Error: Could not find 'sampleimage' directory.")
        return

    print(f"Using image directory: {image_dir}")

    # 2. Start local HTTP server in background
    server_thread = threading.Thread(
        target=serve_directory,
        args=(image_dir, LOCAL_PORT),
        daemon=True
    )
    server_thread.start()
    
    # Give server a moment to start
    time.sleep(1)

    # 3. Construct the batch payload
    images_payload = []
    
    # We want im1.jpeg to im7.jpeg
    local_ip = get_local_ip()
    base_url = f"http://{local_ip}:{LOCAL_PORT}"
    
    print(f"Preparing batch request for 7 images...")
    
    for i in range(7, 13):
        filename = f"im{i}.jpeg"
        file_path = os.path.join(image_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {filename} not found, skipping.")
            continue
            
        image_url = f"{base_url}/{filename}"
        images_payload.append({
            "file_id": f"file-uuid-{i}",
            "image_url": image_url
        })
        print(f" - Added {filename} as {image_url}")

    payload = {
        "images": images_payload
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print("\nSending batch request to API...")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("\nSuccess! Response:")
            import json
            print(json.dumps(response.json(), indent=2))
        else:
            print("\nFailed! Response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"\nError: Could not connect to API at {API_URL}")
        print("Make sure uvicorn is running: 'uvicorn main:app --reload'")

    print("\nKeeping local file server running for 20 seconds to allow worker to download files...")
    time.sleep(20)

if __name__ == "__main__":
    main()
