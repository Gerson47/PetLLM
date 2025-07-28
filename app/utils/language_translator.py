import asyncio
import logging
import time
from langdetect import detect
from googletrans import Translator
from async_lru import alru_cache

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("translation")

# Create the translator instance only ONCE
translator = Translator()

ALLOWED_LANGUAGES = {"en", "ko", "ja", "tl"}

# --- Asynchronous and Cached Functions ---

@alru_cache(maxsize=128)
async def detect_language(text: str) -> str:
    """
    Asynchronously detects the language of a text.
    It's now an async function and must be awaited.
    """
    if not text:
        return "en"
    loop = asyncio.get_event_loop()
    try:
        # Run the synchronous, CPU-bound 'detect' in an executor to not block the event loop
        lang = await loop.run_in_executor(None, detect, text)
        if lang in ALLOWED_LANGUAGES:
            return lang
        else:
            logger.warning(f"Detected unsupported language '{lang}', defaulting to 'en'.")
            return "en"
    except Exception as e:
        logger.error(f"[LangDetect Error]: {e} for text '{text[:50]}...'")
        return "en"

@alru_cache(maxsize=1024)
async def translate_to_english(text: str) -> str:
    """Translates a text to English, detecting the source language automatically."""
    try:
        # THE FIX IS HERE: We must 'await' the async detect_language function
        src_lang = await detect_language(text)

        if not text or src_lang == "en":
            return text

        translated = await translator.translate(text, src=src_lang, dest="en")
        logger.info(f"[Translate → EN] ({src_lang}): {text} → {translated.text}")
        return translated.text
    except Exception as e:
        logger.error(f"[Translation to EN Error]: {e} for text: '{text}'")
        return text

@alru_cache(maxsize=1024)
async def translate_to_user_language(text: str, target_lang: str) -> str:
    """Translates a text (assumed English) to a target language."""
    if not text or target_lang == "en":
        return text
    try:
        translated = await translator.translate(text, src="en", dest=target_lang)
        logger.info(f"[Translate → {target_lang.upper()}]: {text} → {translated.text}")
        return translated.text
    except Exception as e:
        logger.error(f"[Translation to User Lang Error]: {e} for text: '{text}'")
        return text