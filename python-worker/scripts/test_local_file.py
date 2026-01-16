import os
import time
import requests
import threading
import http.server
import socketserver
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000/api/process"
API_KEY = "your-secure-api-key-here"  # Ensure this matches your .env or default
IMAGE_PATH = "sampleimage/im9.jpeg"
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
    os.chdir(directory)
    
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Temporary file server running at http://localhost:{port}")
        httpd.serve_forever()


def main():
    # 1. Setup paths
    # We want to access ../sampleimage/im1.jpeg relative to python-worker
    # Or just use the absolute path provided by user metadata
    target_image = "/Users/classic/webku/BADAK/AI-Face-worker/sampleimage/im9.jpeg"
    
    if not os.path.exists(target_image):
        # Try relative if absolute doesn't work (fallback)
        target_image = "../sampleimage/im9.jpeg"
        if not os.path.exists(target_image):
             print(f"Error: Image not found at {target_image}")
             return

    abs_image_path = os.path.abspath(target_image)
    image_dir = os.path.dirname(abs_image_path)
    image_filename = os.path.basename(abs_image_path)
    
    print(f"Found image at: {abs_image_path}")

    # 2. Start local HTTP server in background
    # Serve ONLY the directory containing the image
    server_thread = threading.Thread(
        target=serve_directory,
        args=(image_dir, LOCAL_PORT),
        daemon=True
    )
    server_thread.start()
    
    # Give server a moment to start
    time.sleep(1)

    # 3. Construct the "fake" HTTP URL
    # Can access file directly at root of server
    local_url = f"http://{get_local_ip()}:{LOCAL_PORT}/{image_filename}"
    print(f"Generated local URL: {local_url}")

    # 4. Call the API
    payload = {
        "file_id": f"test-local-{int(time.time())}",
        "image_url": local_url
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    print("\nSending request to API...")
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

if __name__ == "__main__":
    main()
