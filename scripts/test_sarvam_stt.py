import os
import time
import argparse
import httpx
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Minimal Sarvam STT API Test")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    # Read SARVAM_API_KEY from root .env
    load_dotenv()
    api_key = os.getenv("SARVAM_API_KEY")
    
    if not api_key:
        print("Error: SARVAM_API_KEY environment variable is missing.")
        return

    audio_path = args.audio
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    endpoint = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": api_key
    }
    
    print(f"Sending audio file '{audio_path}' to Sarvam API (saaras:v3)...")
    
    start_time = time.perf_counter()
    
    # Use httpx to send the request
    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"model": "saaras:v3"}
            
            with httpx.Client(timeout=30.0) as client:
                response = client.post(endpoint, headers=headers, data=data, files=files)
                
    except Exception as e:
        print(f"Request failed: {e}")
        return
        
    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    
    print("\n--- Result ---")
    print(f"HTTP Status:     {response.status_code}")
    print(f"Request Latency: {latency_ms} ms")
    
    if response.status_code == 200:
        result_json = response.json()
        print(f"Transcript:      {result_json.get('transcript', '')}")
        print(f"Language Code:   {result_json.get('language_code', 'unknown')}")
    else:
        print(f"Error Response:  {response.text}")

if __name__ == "__main__":
    main()
