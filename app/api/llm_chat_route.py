import logging
import json
import re
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks, Form, Request

# --- App Imports ---
from app.models.main_schema import ChatResponse
from app.utils.prompt_builder import build_pet_prompt, system_prompt
from app.utils.chat_handler_test import generate_response
from app.utils.extract_response import extract_response_features
from app.utils.chat_retention import save_message_and_get_context
from app.utils.php_service import get_user_by_id, get_pet_by_id, get_pet_status_by_id
from app.utils.user_operations import get_or_create_user_profile
from app.utils.fact_extractor import extract_and_save_user_facts

# --- Basic Setup ---
router = APIRouter()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Auth dependency ---
async def get_auth_token(authorization: str = Header(...)):
    if not authorization:
           raise HTTPException(status_code=401, detail={
              "message": "Authorization header missing",
              "code": "AUTH_MISSING",
           })
    return authorization

# --- Helper Functions ---

def _log_user_profile(profile: dict):
    """
    Logs detailed user profile information for debugging purposes.
    """
    logger.debug("=== [USER PROFILE LOADED] ===")
    logger.debug("User ID: %s, Name: %s", profile.get("user_id"), profile.get("first_name"))
    
    biography = profile.get("biography", {})
    if biography:
        logger.debug("Biography Facts:")
        for key, value in biography.items():
            logger.debug("  - %s: %s", key, value)
    else:
        logger.debug("Biography Facts: None")

async def _fetch_chat_data(user_id: int, pet_id: int, token: str) -> dict:
    user_data_from_php = await get_user_by_id(user_id, token)
    if not user_data_from_php:
        raise HTTPException(status_code=404, detail={
            "message": "User not found.",
            "code": "USER_NOT_FOUND",
        })

    user_profile = await get_or_create_user_profile(user_id, user_data_from_php)
    if not user_profile:
        raise HTTPException(status_code=500, detail={
            "message": "Profile creation or retrieval failed.",
            "code": "PROFILE_ERROR",
        })

    if "biography" not in user_profile:
        user_profile["biography"] = {}
        
    profession = user_data_from_php.get("profession")
    if profession:
        user_profile["biography"]["profession"] = profession.upper()

    gender_map = {"0": "he/him", "1": "she/her", "2": "they/them"}
    gender_code = str(user_data_from_php.get("gender", "")) 
    if gender_code in gender_map:
        user_profile["biography"]["gender"] = gender_map[gender_code]
        
    birth_date_str = user_data_from_php.get("birth_date")
    if birth_date_str:
        try:
            birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            user_profile["biography"]["age"] = age
        except (ValueError, TypeError):
            logger.warning("Could not parse birth_date: %s", birth_date_str)    
    
    pet_data = await get_pet_by_id(pet_id, token)
    if not pet_data:
        raise HTTPException(status_code=404, detail={
            "message": "Pet not found.",
            "code": "PET_NOT_FOUND",
        })

    pet_status_data = await get_pet_status_by_id(pet_id, token)

    return {"user": user_profile, "pet": pet_data, "status": pet_status_data}

async def _call_ai_service(user_prompt_str: str, system_prompt_str: str) -> str:
    """
    Handles the LLM call, checks the response dictionary, and returns the text.
    Raises HTTPException on errors.
    """
    # 1. FIX: Call generate_response with the CORRECT keyword arguments
    #    (system_prompt and user_prompt)
    llm_response_dict = await generate_response(
        prompt=user_prompt_str,
        persona=system_prompt_str, 
    )

    # 2. FIX: Check the status on the dictionary directly
    if llm_response_dict.get("status") != "success":
        error_details = llm_response_dict.get("data", "Unknown AI error")
        logger.error("[ERROR] LLM Service returned a non-success status: %s", error_details)
        raise HTTPException(status_code=502, detail={
            "message": f"AI service error: {error_details}",
            "code": "AI_SERVICE_ERROR",
        })

    ai_response_text = llm_response_dict.get("data", {}).get("response")

    if not ai_response_text:
        logger.error("[ERROR] Missing 'data.response' in LLM output: %s", llm_response_dict)
        raise HTTPException(status_code=502, detail={
            "message": "AI service returned an incomplete response.",
            "code": "AI_INCOMPLETE_RESPONSE",
        })
        
    return ai_response_text

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
    logger.info("=== [CHAT REQUEST RECEIVED] User ID: %s | Pet ID: %s ===", user_id, pet_id)

    # Fetch all data
    try:
        data = await _fetch_chat_data(user_id, pet_id, authorization)
        user_profile = data["user"]
        pet_data = data["pet"]
        pet_status_data = data["status"]
    except HTTPException as e:
        # _fetch_chat_data now raises structured HTTPExceptions; log and re-raise
        logger.error("[ERROR] Data fetching failed for user %s: %s", user_id, e.detail)
        raise
    except Exception as e:
        logger.exception("[ERROR] Unexpected error while fetching data for user %s", user_id)
        raise HTTPException(status_code=500, detail={
            "message": "Unexpected error while fetching data.",
            "code": "DATA_FETCH_ERROR",
        })

    # Log profile details for debugging 
    _log_user_profile(user_profile)

    # Start background tasks and prepare context
    background_tasks.add_task(extract_and_save_user_facts, user_id, message)
    
    conversation_context = await save_message_and_get_context(user_id, pet_id, "user", message)
    
    owner_name = user_profile.get("first_name", "Friend")
    pet_name = pet_data.get("name", "Your Pet")
    
    history_snippet = "\n".join(
        f"{owner_name}: {msg['text']}" if msg["sender"] == "user" else f"{pet_name}: {msg['text']}"
        for msg in conversation_context
    )

    # Build the prompts
    build_system_prompt = system_prompt(pet_data, owner_name)
    prompt = build_pet_prompt(
        pet_data,
        owner_name,
        memory_snippet=history_snippet,
        pet_status=pet_status_data,
        message=message,
        biography_snippet=user_profile.get("biography", {}),
    )
    prompt += f"\n{pet_name}:"

    # Call the AI
    try:
        ai_response_text = await _call_ai_service(build_system_prompt, prompt)
    except HTTPException:
        # _call_ai_service raises structured HTTPExceptions for AI issues; re-raise after logging
        logger.error("[ERROR] AI service error for user %s | pet %s", user_id, pet_id)
        raise
    except Exception:
        logger.exception("[ERROR] Unexpected AI error for user %s | pet %s", user_id, pet_id)
        raise HTTPException(status_code=500, detail={
            "message": "Internal server error during AI processing.",
            "code": "AI_INTERNAL_ERROR",
        })

    # The final response
    cleaned_response = re.sub(rf"^{re.escape(pet_name)}\s*:\s*", "", ai_response_text, count=1).strip()

    await save_message_and_get_context(user_id, pet_id, "ai", cleaned_response)

    features = extract_response_features(cleaned_response)

    logger.info(f"=== [AI PROMPT RECEIVED] ===\n {prompt}\n === [SYSTEM PROMPT] ===\n {build_system_prompt}\n")

    logger.info("=== [RESPONSE SENT] AI Response: %s ===", cleaned_response)
    
    return {"response": cleaned_response, "features": features}