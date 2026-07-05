import os

from google import genai
from google.genai import types

SYSTEM_PROMPT = """\
তুমি Kothiq — বাংলাদেশের একটি কাস্টমার হেল্প ডেস্কের ভয়েস AI সহকারী।

নিয়মাবলী:
- স্বাভাবিক কথ্য বাংলা বা প্রয়োজনে বাংলিশে (বাংলা-ইংরেজি মিশ্রিত) উত্তর দাও, যেভাবে
  ঢাকার একজন মানুষ ফোনে কথা বলে।
- উত্তর সংক্ষিপ্ত রাখো (১-৩ বাক্য) — এটি ফোন কলে জোরে পড়ে শোনানো হবে, লম্বা উত্তর
  শুনতে অস্বস্তিকর লাগে।
- প্রশ্নের সরাসরি উত্তর দাও, অপ্রয়োজনীয় ভূমিকা বা পুনরাবৃত্তি এড়িয়ে চলো।
- নিশ্চিত না হলে, অনুমান না করে স্পষ্টভাবে জিজ্ঞেস করো বা মানুষ এজেন্টের কাছে
  হস্তান্তরের কথা বলো।
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def generate_reply(history: list[dict], user_text: str) -> str:
    """Generate the assistant's next reply given prior turns and the new user utterance.

    history entries look like {"role": "user" | "model", "text": str}.
    """
    client = _get_client()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    contents = [
        types.Content(role=turn["role"], parts=[types.Part(text=turn["text"])])
        for turn in history
    ]
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text.strip()
