"""
Lightweight Google Gemini API Connection Test Utility (TASK 15).
Verifies GOOGLE_API_KEY environment variable and tests direct model connectivity.

Usage:
  python scripts/test_gemini_connection.py
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path=dotenv_path)

from google import genai
from app.generation.config import LLMConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n[ERROR] GOOGLE_API_KEY environment variable is missing.")
        print("Please set GOOGLE_API_KEY in your environment before running live API tests.")
        sys.exit(1)

    # Print masked key confirmation safely
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
    print(f"\n[INFO] Found GOOGLE_API_KEY (masked: {masked_key})")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    print(f"[INFO] Initializing Gemini Client with model: '{model_name}'")

    client = genai.Client(api_key=api_key)

    test_prompt = "Hello! Reply with a 5-word confirmation in Hindi and English."

    print("\nSending test prompt to Gemini API...")
    t_start = time.perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=test_prompt
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        print("\n==================================================")
        print("GEMINI CONNECTION TEST SUCCESSFUL")
        print("==================================================")
        print(f"Model Name:         {model_name}")
        print(f"Request Latency:    {latency_ms} ms")
        print(f"Response Content:   {response.text.strip()}")
        print("==================================================\n")

    except Exception as err:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        print("\n==================================================")
        print("GEMINI CONNECTION TEST FAILED")
        print("==================================================")
        print(f"Model Name:         {model_name}")
        print(f"Latency before err: {latency_ms} ms")
        print(f"Error Details:      {err}")
        print("==================================================\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
