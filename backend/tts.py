import io
import os

import requests
from google.cloud import texttospeech

_client: texttospeech.TextToSpeechClient | None = None
_local_model = None
_local_tokenizer = None


def _get_client() -> texttospeech.TextToSpeechClient:
    global _client
    if _client is None:
        _client = texttospeech.TextToSpeechClient()
    return _client


def _get_local_model():
    """Lazily load Meta MMS-TTS (Bengali) in-process on CPU. Free, no external server."""
    global _local_model, _local_tokenizer
    if _local_model is None:
        from transformers import AutoTokenizer, VitsModel

        _local_model = VitsModel.from_pretrained("facebook/mms-tts-ben")
        _local_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    return _local_model, _local_tokenizer


def _synthesize_colab(text: str) -> bytes:
    """Synthesize via a self-hosted TTS server (e.g. the Colab notebook). Returns WAV."""
    url = os.environ["COLAB_TTS_URL"].rstrip("/") + "/synthesize"
    response = requests.post(url, data={"text": text}, timeout=60)
    response.raise_for_status()
    return response.content


def _synthesize_local(text: str) -> bytes:
    """Synthesize with Meta MMS-TTS running directly in this process. Returns WAV."""
    import soundfile as sf
    import torch

    model, tokenizer = _get_local_model()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    wav = waveform.squeeze().cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, wav, model.config.sampling_rate, format="WAV")
    return buf.getvalue()


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
    if provider == "local":
        return _synthesize_local(text)
    return _synthesize_google(text)


def mime_type() -> str:
    """Audio MIME type of what synthesize() returns for the active provider."""
    provider = os.environ.get("TTS_PROVIDER", "google")
    return "audio/wav" if provider in ("colab", "local") else "audio/mpeg"
