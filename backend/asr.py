import os
import tempfile

import requests
from openai import OpenAI

_client: OpenAI | None = None
_local_model = None
_mlx_repo: str | None = None


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


def _decode_to_array(audio_bytes: bytes, filename: str):
    """Decode arbitrary audio bytes (webm/wav/…) to a 16kHz mono float32 numpy
    array using PyAV (bundled with faster-whisper), so no system ffmpeg is needed."""
    from faster_whisper.audio import decode_audio

    suffix = os.path.splitext(filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(audio_bytes)
        f.flush()
        return decode_audio(f.name, sampling_rate=16000)


def _transcribe_gemini(audio_bytes: bytes, filename: str) -> str:
    """Transcribe via Gemini's native audio understanding. Uses the same key as the
    LLM, is fast (cloud), and handles Bengali/Banglish well — a solid MVP bridge
    until we own the ASR. Decodes the browser's webm to WAV first (Gemini doesn't
    accept webm), reusing PyAV so no system ffmpeg is required."""
    import io

    import soundfile as sf
    from google import genai
    from google.genai import types

    audio = _decode_to_array(audio_bytes, filename)
    buf = io.BytesIO()
    sf.write(buf, audio, 16000, format="WAV")

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=buf.getvalue(), mime_type="audio/wav"),
            "Transcribe this Bangla/Banglish (mixed Bangla-English) phone audio "
            "verbatim. Output ONLY the transcription in natural Bengali script — "
            "no translation, no commentary, no quotes. If there is no intelligible "
            "speech, output nothing.",
        ],
    )
    return (response.text or "").strip()


def _transcribe_mlx(audio_bytes: bytes, filename: str) -> str:
    """Transcribe with mlx-whisper on the Apple Silicon GPU. Much faster than
    faster-whisper on CPU for Mac; lets us run a large, accurate model in real time."""
    global _mlx_repo
    import mlx_whisper

    if _mlx_repo is None:
        _mlx_repo = os.environ.get(
            "MLX_WHISPER_REPO", "mlx-community/whisper-large-v3-mlx"
        )
    audio = _decode_to_array(audio_bytes, filename)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=_mlx_repo,
        language="bn",
        condition_on_previous_text=False,
    )
    return result["text"].strip()


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
        segments, _info = model.transcribe(
            f.name,
            language="bn",
            vad_filter=True,
            condition_on_previous_text=False,
        )
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
    if provider == "gemini":
        return _transcribe_gemini(audio_bytes, filename)
    if provider == "mlx":
        return _transcribe_mlx(audio_bytes, filename)
    if provider == "colab":
        return _transcribe_colab(audio_bytes, filename)
    if provider == "local":
        return _transcribe_local(audio_bytes, filename)
    return _transcribe_openai(audio_bytes, filename)
