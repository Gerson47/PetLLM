
from decouple import config
import logging
from groq import AsyncGroq, GroqError
import json
 
client = AsyncGroq(
    api_key= config("GROQ_API_KEY"),
)
 
SITE_URL = config("SITE_URL", default="http://localhost")
SITE_TITLE = config("SITE_TITLE", default="Librarian Chatbot")

# Production-ready model but not quite as good as the qwen3-32b
# MODEL_NAME = "llama-3.1-8b-instant" 
# Backup model for testing purposes. reasoning_effort="low,medium,high" 
MODEL_NAME = "openai/gpt-oss-20b" 
# Default model. Uncomment reasoning_format and reasoning_effort to use qwen3-32b
# MODEL_NAME = "qwen/qwen3-32b"
logger = logging.getLogger("llm_client")
 
async def generate_response(system_prompt: str, prompt: str) -> str:

    try:
        logger.info(f"Recieved system prompt: {system_prompt}\n")
        chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=MODEL_NAME,
                temperature=0.7,
                max_tokens=250,
                top_p=0.9,
                reasoning_format="hidden",
                reasoning_effort="low"
            )

        response_content = chat_completion.choices[0].message.content
        success_response = {
            "status": "success",
            "data": {
                "response": response_content
            }
        }

        logger.info(f"LLM response: {response_content}")
        return json.dumps(success_response)

    except GroqError as e:
        # Prefer returning a structured JSON error so callers can parse it.
        logger.error("Groq API error: %s - %s", e.__class__.__name__, e)
        msg = str(e)
        # Heuristic mapping from exception message to an error code the route can interpret.
        code = "AI_SERVICE_ERROR"
        lower = msg.lower()
        if "401" in msg or "unauthor" in lower or "auth" in lower:
            code = "AI_AUTH_ERROR"
        elif "rate" in lower or "429" in msg:
            code = "AI_RATE_LIMIT"
        elif "503" in msg or "unavail" in lower or "timeout" in lower:
            code = "AI_UNAVAILABLE"

        error_response = {
            "status": "error",
            "error": {
                "message": msg,
                "code": code,
            },
        }
        return json.dumps(error_response)
    except Exception as e:
        # Catch-all: return structured unavailable error so the route can map to 503.
        logger.exception("An unexpected error occurred while calling LLM: %s", e)
        error_response = {
            "status": "error",
            "error": {
                "message": "The AI service is currently unavailable. Please try again later.",
                "code": "AI_UNAVAILABLE",
            },
        }
        return json.dumps(error_response)

