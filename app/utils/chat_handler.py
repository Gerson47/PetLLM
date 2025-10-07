
from decouple import config
import logging
from groq import AsyncGroq, GroqError
import json
 
client = AsyncGroq(
    api_key= config("GROQ_API_KEY"),
)
 
SITE_URL = config("SITE_URL", default="http://localhost")
SITE_TITLE = config("SITE_TITLE", default="Librarian Chatbot")

MODEL_NAME = "openai/gpt-oss-20b" 
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
       
        # response_content = chat_completion.usage
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
 
