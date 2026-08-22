from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class STTResult:
    transcript: str
    detected_language: str
    latency_ms: float
    provider: str
    model: str
    raw_response: Optional[str] = None

class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str, kwargs: Optional[Dict[str, Any]] = None) -> STTResult:
        pass