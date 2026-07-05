# Kothiq — Bangla/Banglish Voice AI Help Desk

## What this is
A voice AI call-center product for Bangladesh, competing with Speaklar/Phonely, but
Bangla/Banglish-first. Sold as a service, priced per minute — so COGS per minute and
concurrency-per-GPU are first-order product metrics, not just engineering concerns.

## Architecture (cascaded pipeline, not true speech-to-speech)
Caller <-> Telephony (SIP/RTP) <-> Orchestrator <-> ASR -> LLM (+RAG) -> TTS -> back to caller

- **Telephony**: local Bangladeshi SIP/IPTSP carrier (not Twilio — kills the ~$0.02/min
  markup, gives a local caller ID that actually gets answered, and is the BTRC-compliant
  path). Terminates into FreeSWITCH/Asterisk or LiveKit's/Pipecat's native SIP layer.
- **Orchestration**: LiveKit Agents or Pipecat, self-hosted. Do NOT hand-roll VAD /
  turn-taking / barge-in — the frameworks already solve this well.
- **ASR**: fine-tuned Bangla Whisper (e.g. BengaliAI checkpoints), self-hosted. This is
  a real product moat — generic Whisper's reported low WER hides poor real-world accuracy
  on Bengali, and code-switched Banglish adds ~30-50% relative WER on top of that. Must be
  fine-tuned on real telephone-quality (8kHz, narrowband) Banglish audio, not clean studio
  data, to be competitive.
- **LLM**: rent a frontier API (Claude/GPT/Gemini) with prompt caching. This is NOT where
  Bangla-ness comes from (ASR/TTS carry the language) — the LLM just reasons and calls
  tools (orders, CRM, bookings). Self-hosting a small model here trades quality for a
  savings that isn't real until very high sustained volume, and small models are
  meaningfully worse at reliable tool-calling — a bad tool call is a wrong order, not a typo.
  Revisit self-hosting only for hard data-residency requirements or genuinely high volume.
- **TTS**: cloned or self-hosted Bangladeshi (Dhaka-accent) voice. Generic Bangla TTS
  voices read as Indian-accented/overly formal to Bangladeshi listeners. This is the other
  real moat and the biggest per-minute cost lever if rented (ElevenLabs-quality can be
  3-7x pricier than alternatives).

## Cost reality (2026, rough per-minute COGS)
- ASR (API): $0.003-0.008/min; self-hosted fine-tune: near-free at volume.
- LLM: <$0.01/min with prompt caching, any frontier model, small/mid context.
- TTS: the biggest swing factor, ~7x range between providers.
- Telephony: local SIP beats Twilio by ~$0.02/min.
- Orchestration: ~$0.01/min managed (Pipecat/LiveKit Cloud), ~free self-hosted.
- Target retail (per Speaklar's public pricing): ~$0.04-0.05/min. A fully-rented
  API stack (Twilio+ElevenLabs+managed orchestration) does not close at that price;
  self-hosting ASR+TTS is what makes the margin work.
- Latency budget: full round trip (caller speaks -> hears AI reply) needs to land in
  ~800ms-1.5s to feel like a real conversation, not a walkie-talkie. Stream partial
  LLM output into TTS rather than waiting for full sentences.

## Build-vs-rent decision (locked in)
| Component      | Decision                          | Why |
|----------------|------------------------------------|-----|
| ASR            | Build/own (fine-tune)              | Moat + cheap at volume + APIs are weak for Bangla |
| TTS            | Build/own (clone or self-host)      | Moat + biggest cost lever |
| Orchestration  | Self-operate open source (no rebuild) | LiveKit/Pipecat already solve VAD/turn-taking |
| Telephony      | Self-operate via local BD carrier   | Cost + compliance + answer-rate |
| LLM            | Rent frontier API                   | Self-hosting costs quality (tool-calling), doesn't save money until huge volume |

## MVP sequencing (chicken-and-egg: need live calls to get Banglish data to fine-tune on)
1. **Phase 1 — MVP on existing parts**: open Bangla ASR + open/cloned Bangla TTS +
   rented LLM + LiveKit/Pipecat + local SIP. Where an open piece is too rough, bridge
   with an API temporarily for just that piece. Goal: get real calls flowing.
2. **Phase 2 — own ASR**: fine-tune on real collected call recordings, swap in once
   it beats the Phase 1 baseline.
3. **Phase 3 — own TTS**: record/train a Dhaka-accent voice, swap in.
LLM stays rented throughout. Success is measured as business outcome (call resolved or
cleanly handed to a human), not research-benchmark WER.

## Prior mistake to avoid
An earlier draft plan self-hosted the LLM (Qwen2.5-7B via vLLM) on a single RTX 4090
alongside Whisper + Coqui TTS simultaneously. Rejected: hardware-driven model choice
hurt tool-calling/Banglish quality, real concurrency was far below what the config
implied (KV-cache bound, not `--max-num-seqs` bound), and three real-time models don't
fit together on one consumer GPU for a production concurrency product anyway. Repurpose
that GPU for ASR/TTS self-hosting instead, not the LLM.
