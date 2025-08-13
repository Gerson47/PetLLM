
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
 
async def generate_response(prompt: str) -> str:
 
    try:
        chat_completion = await client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=MODEL_NAME,
                temperature=0.6,
                max_tokens=2024,
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
        logger.error(f"Groq API error: {e.__class__.__name__} - {e}")
        return "[ERROR]: The AI service returned an error. Please check the logs."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "[ERROR]: The AI service is currently unavailable. Please try again later."
 
