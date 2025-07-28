import httpx
from decouple import config

OPENROUTER_API_KEY = config("OPENROUTER_API_KEY4")
SITE_URL = config("SITE_URL", default="http://localhost")        
SITE_TITLE = config("SITE_TITLE", default="PetPal Chatbot")      

async def generate_response(prompt: str, use_mock=True) -> str:
    if use_mock:
        return f"*Wags tail happily* You said: '{prompt[-50:]}' 🐾"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_TITLE
    }

    payload = {
        "model": "mistralai/mistral-small-3.1-24b-instruct:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.6,
        "top_p": 0.9
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"[ERROR]: {response.text}"

    except Exception as e:
        return f"[ERROR]: OpenRouter API failed — {str(e)}"
