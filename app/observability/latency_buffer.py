import time
import math
from typing import List, Dict, Any, Optional

class LatencyBuffer:
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []

    def add_record(self, record: Dict[str, Any]):
        record["timestamp"] = time.time()
        self.buffer.append(record)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def _percentile(self, data: List[float], p: float) -> float:
        if not data:
            return 0.0
        data.sort()
        if p == 100:
            return data[-1]
        k = (len(data) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return data[int(k)]
        d0 = data[int(f)] * (c - k)
        d1 = data[int(c)] * (k - f)
        return round(d0 + d1, 2)

    def get_summary(self, sample_size: int = 10) -> Dict[str, Any]:
        samples = self.buffer[-sample_size:]
        count = len(samples)
        
        metrics = ["stt_ms", "retrieval_ms", "generation_ms", "grounding_ms", "total_rag_ms", "total_e2e_ms"]
        result = {
            "sample_count": count,
            "updated_at": time.time(),
            "source": "live_requests"
        }
        
        for m in metrics:
            data = [s.get(m, 0) for s in samples]
            result[m] = {
                "p50": self._percentile(data, 50),
                "p70": self._percentile(data, 70),
                "p100": self._percentile(data, 100)
            }
            
        return result

# Global singleton
latency_buffer = LatencyBuffer(max_size=100)
