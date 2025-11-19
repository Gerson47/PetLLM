import os
from fastapi import APIRouter, Depends, HTTPException, status
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

router = APIRouter(prefix="/ai", tags=["ai"])

async def generate_response(prompt: str, persona: str):  
    try:
        response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ],
        config = types.GenerateContentConfig(
            max_output_tokens=2500,
            temperature=0.2,
            thinking_config = types.ThinkingConfig(
                thinking_budget=0, #set to 1 for thinking mode.
            ),
            system_instruction=[
                types.Part.from_text(text=persona),
            ],
        )
)
        success_response = {
            "status": "success",
            "data": {
                "response": response.text
            }
        }

        return success_response
        
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))