import logging
import json
import re
from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks, Form, Request
from pydantic import BaseModel  

from app.models.main_schema import ChatResponse  
from app.utils.prompt_builder import build_pet_prompt
from app.utils.chat_handler import generate_response
from app.utils.extract_response import extract_response_features

from app.utils.chat_retention import save_message_and_get_context
from app.utils.php_service import get_user_by_id, get_pet_by_id, get_pet_status_by_id
from app.utils.user_operations import get_or_create_user_profile
from app.utils.fact_extractor import extract_and_save_user_facts

router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Auth dependency ---
async def get_auth_token(authorization: str = Header(...)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return authorization


# --- Main Chat Route ---
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    user_id: int = Form(...),
    pet_id: int = Form(...),
    message: str = Form(...),
    authorization: str = Depends(get_auth_token),
):
    logger.info("=== [CHAT REQUEST RECEIVED] ===")
    logger.info("User ID: %s | Pet ID: %s", user_id, pet_id)
    logger.info("Content-Type: %s", request.headers.get("content-type"))

    # --- Data Fetching and Profile Management ---
    try:
        user_data_from_php = await get_user_by_id(user_id, authorization)
        if not user_data_from_php:
            raise ValueError("User not found.")

        user_profile = await get_or_create_user_profile(user_id, user_data_from_php)
        if not user_profile:
            raise ValueError("Profile creation failed.")

        pet_data = await get_pet_by_id(pet_id, authorization)
        if not pet_data:
            raise ValueError("Pet not found.")

        pet_status_data = await get_pet_status_by_id(pet_id, authorization)

    except Exception as e:
        logger.error("[ERROR] Data fetching failed: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving core data.")

    owner_name = user_profile.get("first_name", "Friend")
    pet_name = pet_data.get("name", "Your Pet")

    logger.info("=== [USER PROFILE LOADED] ===")
    logger.info("User ID: %s", user_profile.get("user_id"))
    logger.info("Name: %s", user_profile.get("first_name"))

    biography = user_profile.get("biography", {})
    if biography:
        logger.info("Biography Facts:")
        for key, value in biography.items():
            logger.info("  - %s: %s", key, value)
    else:
        logger.info("Biography Facts: None")

    preferences = user_profile.get("preferences", {})
    if preferences:
        logger.info("Static Preferences:")
        for key, value in preferences.items():
            logger.info("  - %s: %s", key, value)
    else:
        logger.info("Static Preferences: None")

    logger.info("================================")

    # --- Language and Chat Context ---
    user_lang = message
    logger.info("[INFO] Scheduling fact extraction task for user_id=%s", user_id)
    background_tasks.add_task(extract_and_save_user_facts, user_id, user_lang)

    conversation_context = await save_message_and_get_context(
        user_id=user_id,
        pet_id=pet_id,
        sender="user",
        message=user_lang,
    )

    history_snippet = "\n".join(
        f"{owner_name}: {msg['text']}" if msg["sender"] == "user" else f"{pet_name}: {msg['text']}"
        for msg in conversation_context
    )

    # --- LLM Call for Chat Response ---
    prompt = build_pet_prompt(
        pet_data,
        owner_name,
        memory_snippet=history_snippet,
        pet_status=pet_status_data,
        message = user_lang,
        biography_snippet=biography,
    )
    prompt += f"\n{pet_name}:"

    logger.info("=== [LLM PROMPT SENT] ===")
    logger.info("%s", prompt)

    llm_json_string = await generate_response(prompt)

    try:
        response_data = json.loads(llm_json_string)
    except json.JSONDecodeError:
        logger.error("[ERROR] Malformed JSON from LLM: %s", llm_json_string)
        raise HTTPException(status_code=502, detail="AI service returned malformed data.")

    if response_data.get("status") == "error":
        error_message = response_data.get("error", {}).get("message", "Unknown AI error")
        logger.error("[ERROR] LLM Service: %s", error_message)
        raise HTTPException(status_code=502, detail=error_message)

    ai_response_text = response_data.get("data", {}).get("response")
    if not ai_response_text:
        logger.error("[ERROR] Missing 'data.response' in LLM output: %s", response_data)
        raise HTTPException(status_code=502, detail="AI service returned an incomplete response.")

    # --- Response Cleaning (only prefix removal, no emoji stripping) ---
    prefix_pattern = rf"^{re.escape(pet_name)}\s*:\s*"
    cleaned_response = re.sub(prefix_pattern, "", ai_response_text, count=1).strip()

    logger.info("=== [AI RESPONSE CLEANED] ===")
    logger.info("User Query: %s", user_lang)
    logger.info("AI Response: %s", cleaned_response)

    # Save AI response
    await save_message_and_get_context(
        user_id=user_id,
        pet_id=pet_id,
        sender="ai",
        message=cleaned_response,
    )

    # Extract features
    features = extract_response_features(cleaned_response)

    return {"response": cleaned_response, "features": features}