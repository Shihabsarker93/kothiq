import os

import requests
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _transcribe_colab(audio_bytes: bytes, filename: str) -> str:
    """Transcribe via a self-hosted Whisper server (e.g. the Colab notebook)."""
    url = os.environ["COLAB_ASR_URL"].rstrip("/") + "/transcribe"
    response = requests.post(
        url, files={"audio": (filename, audio_bytes)}, timeout=60
    )
    response.raise_for_status()
    return response.json()["text"].strip()


def _transcribe_openai(audio_bytes: bytes, filename: str) -> str:
    client = _get_client()
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes),
        language="bn",
    )
    return result.text.strip()


def transcribe(audio_bytes: bytes, filename: str) -> str:
    """Transcribe Bangla/Banglish speech audio to text."""
    provider = os.environ.get("ASR_PROVIDER", "openai")
    if provider == "colab":
        return _transcribe_colab(audio_bytes, filename)
    return _transcribe_openai(audio_bytes, filename)
