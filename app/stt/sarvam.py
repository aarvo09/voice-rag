import os
import time
import requests
from typing import Dict, Any, Optional

from app.stt.interface import STTProvider, STTResult

class SarvamSTTProvider(STTProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable is not set.")
        
        self.endpoint = "https://api.sarvam.ai/speech-to-text"
        self.model = "saaras:v3"

    def transcribe(self, audio_path: str, kwargs: Optional[Dict[str, Any]] = None) -> STTResult:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.perf_counter()
        
        headers = {
            "api-subscription-key": self.api_key
        }
        
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            data = {"model": self.model}
            
            # Additional params like language_code if provided
            if kwargs and "language_code" in kwargs:
                data["language_code"] = kwargs["language_code"]
                
            response = requests.post(self.endpoint, headers=headers, files=files, data=data)
            
            if response.status_code != 200:
                raise RuntimeError(f"Sarvam API error ({response.status_code}): {response.text}")
                
            result_json = response.json()
            
        latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        
        # Typically Sarvam returns transcript and language_code
        transcript = result_json.get("transcript", "")
        detected_lang = result_json.get("language_code", "unknown")
        
        return STTResult(
            transcript=transcript,
            detected_language=detected_lang,
            latency_ms=latency_ms,
            provider="sarvam",
            model=self.model,
            raw_response=response.text
        )