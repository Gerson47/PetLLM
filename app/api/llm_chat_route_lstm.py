# app/routes/llm_chat_route_lstm.py

import logging
from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

from app.schema.main_schema import ChatResponse
from app.utils.prompt_builder import build_pet_prompt
from app.utils.chat_handler import generate_response
from app.utils.extract_response import extract_response_features
from app.utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language
)
from app.utils.chat_retention import save_message_and_get_context
from app.utils.php_service import get_user_by_id, get_pet_by_id, get_pet_status_by_id
from app.utils.user_operations import get_or_create_user_profile

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Model and Dependencies (No Changes) ---
class ChatForm(BaseModel):
    user_id: int
    pet_id: int
    message: str

async def get_chat_form(form: ChatForm):
    return form

async def get_auth_token(authorization: str = Header(...)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return authorization

# --- Main Chat Route ---
@router.post("/chat", response_model=ChatResponse)
async def chat(
    form: ChatForm = Depends(get_chat_form), 
    authorization: str = Depends(get_auth_token)
):
    user_id = form.user_id
    pet_id = form.pet_id
    message = form.message
    
    logger.info("\n--- Chat Request Received ---\nUser ID: %s | Pet ID: %s", user_id, pet_id)

    # --- Data Fetching and Profile Management ---
    try:
        user_data_from_php = await get_user_by_id(user_id, authorization)
        if not user_data_from_php: raise ValueError("User not found.")
        user_profile = await get_or_create_user_profile(user_id, user_data_from_php)
        if not user_profile: raise ValueError("Profile creation failed.")
        pet_data = await get_pet_by_id(pet_id, authorization)
        if not pet_data: raise ValueError("Pet not found.")
        pet_status_data = await get_pet_status_by_id(pet_id, authorization)
    except Exception as e:
        logger.error("Data fetching error: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving core data.")

    owner_name = user_profile.get("first_name", "Friend")
    pet_name = pet_data.get("name", "Your Pet")

    logger.info("\n\n--- User Profile Loaded for Request ---")
    logger.info(f"User ID: {user_profile.get('user_id')}")
    logger.info(f"Name: {user_profile.get('first_name')}")

    # Log the learned facts from the biography object
    biography = user_profile.get("biography", {})
    if biography:
        logger.info("Learned Facts (Biography):")
        for key, value in biography.items():
            logger.info(f"  - {key}: {value}")
    else:
        logger.info("Learned Facts (Biography): None yet.")

    # Log the static preferences from the preferences object
    preferences = user_profile.get("preferences", {})
    if preferences:
        logger.info("Static Preferences:")
        for key, value in preferences.items():
            logger.info(f"  - {key}: {value}")
    else:
        logger.info("Static Preferences: None set.")
    logger.info("-------------------------------------\n\n")
    
    # --- Language and Chat Context ---
    user_lang = detect_language(message)
    translated_message = await translate_to_english(message, user_lang)

    # This function will now run the fact extraction synchronously (it will wait)
    conversation_context = await save_message_and_get_context(
        user_id=user_id,
        pet_id=pet_id,
        sender="user",
        message=translated_message
    )
    
    history_snippet = "\n".join(
        f"{owner_name}: {msg['text']}" if msg['sender'] == 'user' else f"{pet_name}: {msg['text']}"
        for msg in conversation_context
    )

    # --- LLM Call for Chat Response ---
    prompt = build_pet_prompt(pet_data, owner_name, memory_snippet=history_snippet, pet_status=pet_status_data)
    prompt += f"\n{pet_name}:"
    response = await generate_response(prompt, use_mock=False)

    logger.info("\n\nPrompt sent to LLM:\n%s", prompt)
    
    if response.startswith("[ERROR]"):
        logger.error(f"LLM Response Error: {response}")
        raise HTTPException(status_code=502, detail="AI response unavailable")
    
    # --- Save AI Response ---
    await save_message_and_get_context(
        user_id=user_id,
        pet_id=pet_id,
        sender="ai",
        message=response.strip()
    )
    
    # --- Final Translation and Return ---
    translated_response = await translate_to_user_language(response.strip(), user_lang)
    features = extract_response_features(response)
    
    return {"response": translated_response, "features": features}