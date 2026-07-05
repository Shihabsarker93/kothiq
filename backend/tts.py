import os

import requests
from google.cloud import texttospeech

_client: texttospeech.TextToSpeechClient | None = None


def _get_client() -> texttospeech.TextToSpeechClient:
    global _client
    if _client is None:
        _client = texttospeech.TextToSpeechClient()
    return _client


def _synthesize_colab(text: str) -> bytes:
    """Synthesize via a self-hosted TTS server (e.g. the Colab notebook). Returns WAV."""
    url = os.environ["COLAB_TTS_URL"].rstrip("/") + "/synthesize"
    response = requests.post(url, data={"text": text}, timeout=60)
    response.raise_for_status()
    return response.content


def _synthesize_google(text: str) -> bytes:
    client = _get_client()
    voice_name = os.environ.get("TTS_VOICE_NAME", "bn-IN-Wavenet-A")

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="bn-IN",
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return response.audio_content


def synthesize(text: str) -> bytes:
    """Synthesize Bangla text to speech audio bytes."""
    provider = os.environ.get("TTS_PROVIDER", "google")
    if provider == "colab":
        return _synthesize_colab(text)
    return _synthesize_google(text)


def mime_type() -> str:
    """Audio MIME type of what synthesize() returns for the active provider."""
    provider = os.environ.get("TTS_PROVIDER", "google")
    return "audio/wav" if provider == "colab" else "audio/mpeg"
