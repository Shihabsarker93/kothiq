import asyncio
import base64
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import asr
import llm
import tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kothiq")

app = FastAPI(title="Kothiq Voice MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory per-session conversation history. Fine for a single-process demo;
# not a substitute for real session storage in production.
MAX_HISTORY_TURNS = 12
sessions: dict[str, list[dict]] = {}

# Safety net so a stuck local ASR/TTS model gives the user a clear error
# instead of the request hanging forever with no feedback.
STAGE_TIMEOUT_SECONDS = 45


@app.get("/health")
def health():
    """Simple health check endpoint to verify the backend is running."""
    return {"status": "ok"}


@app.post("/api/converse")
async def converse(session_id: str = Form(...), audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    logger.info(
        "Received audio: %d bytes, content_type=%s, filename=%s, header=%r",
        len(audio_bytes), audio.content_type, audio.filename, audio_bytes[:16],
    )

    try:
        transcript = await asyncio.wait_for(
            asyncio.to_thread(
                asr.transcribe, audio_bytes, audio.filename or "audio.webm"
            ),
            timeout=STAGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("ASR timed out after %ss", STAGE_TIMEOUT_SECONDS)
        return JSONResponse({"error": "asr_timeout"}, status_code=504)
    except Exception:
        logger.exception("ASR failed")
        return JSONResponse({"error": "asr_failed"}, status_code=502)

    if not transcript:
        return JSONResponse({"error": "empty_transcript"}, status_code=422)

    history = sessions.get(session_id, [])

    try:
        reply_text = await asyncio.wait_for(
            asyncio.to_thread(llm.generate_reply, history, transcript),
            timeout=STAGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("LLM timed out after %ss", STAGE_TIMEOUT_SECONDS)
        return JSONResponse({"error": "llm_timeout"}, status_code=504)
    except Exception:
        logger.exception("LLM failed")
        return JSONResponse({"error": "llm_failed"}, status_code=502)

    history = history + [
        {"role": "user", "text": transcript},
        {"role": "model", "text": reply_text},
    ]
    sessions[session_id] = history[-MAX_HISTORY_TURNS:]

    try:
        audio_reply = await asyncio.wait_for(
            asyncio.to_thread(tts.synthesize, reply_text),
            timeout=STAGE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error("TTS timed out after %ss", STAGE_TIMEOUT_SECONDS)
        return JSONResponse({"error": "tts_timeout"}, status_code=504)
    except Exception:
        logger.exception("TTS failed")
        return JSONResponse({"error": "tts_failed"}, status_code=502)

    return JSONResponse(
        {
            "transcript": transcript,
            "reply_text": reply_text,
            "audio_base64": base64.b64encode(audio_reply).decode("ascii"),
            "audio_mime": tts.mime_type(),
        }
    )


frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
