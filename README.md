# Kothiq — Voice MVP (Phase 1)

A browser voice demo that proves the core loop before any telephony work:

```
mic recording -> ASR -> Gemini (LLM) -> TTS -> playback
```

ASR and TTS are provider-switchable: paid APIs (OpenAI Whisper / Google Cloud TTS)
or free self-hosted open models run on a Colab GPU (Whisper via `faster-whisper` +
Meta MMS-TTS). See [CLAUDE.md](CLAUDE.md) for the full product architecture and why
this stack was chosen for Phase 1.

## Setup

1. **Create a virtualenv and install deps**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Get a Gemini key** (always needed, the LLM stays rented — see CLAUDE.md):
   https://aistudio.google.com/apikey

3. **Choose ASR + TTS providers** — pick per component in `.env` via
   `ASR_PROVIDER` / `TTS_PROVIDER`:

   - **`colab` (free, recommended to start)** — run [backend/colab_server.ipynb](backend/colab_server.ipynb)
     in Google Colab (free GPU runtime). It self-hosts Whisper (ASR) and Meta
     MMS-TTS Bengali (TTS) and exposes them via a free `cloudflared` tunnel — no
     signup needed. The last cell prints a public URL; paste it into `backend/.env`
     as both `COLAB_ASR_URL` and `COLAB_TTS_URL`. **This URL changes every time you
     rerun the notebook** — update `.env` and restart the local backend each session.
   - **`openai` / `google` (paid, better quality/latency today)**:
     - **OpenAI** (ASR): https://platform.openai.com/api-keys
     - **Google Cloud TTS**: a *separate* credential from the Gemini key.
       1. Create/select a project at https://console.cloud.google.com
       2. Enable the "Cloud Text-to-Speech API"
       3. Create a service account, add the "Cloud Text-to-Speech User" role
       4. Create and download a JSON key for it

   You can mix providers per component (e.g. free Colab ASR + paid Google TTS).

4. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Fill in `GEMINI_API_KEY`, then whichever ASR/TTS credentials your chosen
   providers need.

5. **Run it**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Open http://localhost:8000, allow mic access, hold the button and speak
   Bangla or Banglish, release to hear the reply.

## What this is / isn't

This is a turn-based (not streaming) demo to validate ASR -> LLM -> TTS quality
and prove the loop works end-to-end. It does **not** yet include:
- Real-time streaming / partial responses (needed to hit the ~1s latency feel)
- Telephony (SIP/phone calls) — see CLAUDE.md Phase 1 vs later phases
- Fine-tuned ASR/TTS on real Bangladeshi call data (that's Phase 2/3, once real
  call data exists — the free Colab models here are off-the-shelf open weights,
  not tuned on your users yet)

Conversation history is kept in memory per browser session and is lost on
server restart — fine for a demo, not for production.
