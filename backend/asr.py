import os
import tempfile

import requests
from openai import OpenAI

_client: OpenAI | None = None
_local_model = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _get_local_model():
    """Lazily load faster-whisper in-process (CPU). Free, no external server."""
    global _local_model
    if _local_model is None:
        from faster_whisper import WhisperModel

        model_size = os.environ.get("WHISPER_MODEL_SIZE", "small")
        _local_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _local_model


def _transcribe_colab(audio_bytes: bytes, filename: str) -> str:
    """Transcribe via a self-hosted Whisper server (e.g. the Colab notebook)."""
    url = os.environ["COLAB_ASR_URL"].rstrip("/") + "/transcribe"
    response = requests.post(
        url, files={"audio": (filename, audio_bytes)}, timeout=60
    )
    response.raise_for_status()
    return response.json()["text"].strip()


def _transcribe_local(audio_bytes: bytes, filename: str) -> str:
    """Transcribe with faster-whisper running directly in this process."""
    model = _get_local_model()
    suffix = os.path.splitext(filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(audio_bytes)
        f.flush()
        segments, _info = model.transcribe(f.name, language="bn")
        return " ".join(seg.text for seg in segments).strip()


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
    if provider == "local":
        return _transcribe_local(audio_bytes, filename)
    return _transcribe_openai(audio_bytes, filename)
