import logging
import json 
import re
from datetime import datetime 
from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks, Form, Request

# --- App Imports ---
from app.models.main_schema import ChatResponse
from app.utils.prompt_builder import build_pet_prompt, system_prompt
from app.utils.chat_handler import generate_response
from app.utils.extract_response import extract_response_features
from app.utils.chat_retention import save_message_and_get_context
from app.utils.php_service import get_user_by_id, get_pet_by_id, get_pet_status_by_id
from app.utils.user_operations import get_or_create_user_profile
from app.utils.fact_extractor import extract_and_save_user_facts

# --- Basic Setup ---
router = APIRouter()
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s') 
logger = logging.getLogger(__name__)

# --- Auth dependency ---
async def get_auth_token(authorization: str = Header(...)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return authorization

# --- Helper Functions ---
def _clean_response_text(text: str) -> str:
    """
    Removes emojis and common text-based emoticons from a string.
    """
    if not text:
        return ""
        
    logger.info(f"===Response before cleaning {text}===")
    
    # 1. Remove Unicode Emojis using a comprehensive regex
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\u2600-\u26FF"          # miscellaneous symbols
        u"\u2700-\u27BF"          # dingbats
        u"\uFE0F"                 # variation selector
        u"\u200d"                 # zero width joiner
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # 2. Remove common text-based emoticons (can be expanded)
    emoticons = [":)", ":-)", ":(", ":-(", ";)", ";-)", ":D", "XD", "<3", ":P"]
    for emoticon in emoticons:
        text = text.replace(emoticon, "")
        
    # 3. Clean up any resulting double spaces and strip whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def _log_user_profile(profile: dict):
    """
    Logs detailed user profile information with improved formatting.
    """
    logger.info("." * 60)
    logger.info("USER PROFILE LOADED")
    logger.info(f"  - User ID:    {profile.get('user_id')}")
    logger.info(f"  - First Name: {profile.get('first_name')}")

    biography = profile.get("biography", {})
    if biography:
        logger.info("  - Biography:")
        pretty_bio = json.dumps(biography, indent=4).split('\n')
        for line in pretty_bio:
            logger.info(f"      {line}")
    else:
        logger.info("  - Biography: None")
    logger.info("." * 60)

async def _fetch_chat_data(user_id: int, pet_id: int, token: str) -> dict:
    user_data_from_php = await get_user_by_id(user_id, token)
    if not user_data_from_php:
        raise ValueError("User not found.")

    user_profile = await get_or_create_user_profile(user_id, user_data_from_php)
    if not user_profile:
        raise ValueError("Profile creation or retrieval failed.")

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
        raise ValueError("Pet not found.")

    pet_status_data = await get_pet_status_by_id(pet_id, token)

    return {"user": user_profile, "pet": pet_data, "status": pet_status_data}

async def _call_ai_service(system_prompt_str: str, user_prompt_str: str) -> str:
    llm_json_string = await generate_response(system_prompt_str, user_prompt_str)

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

    logger.info("\n" + "="*25 + " [CHAT REQUEST START] " + "="*25)
    logger.info(f"User ID: {user_id} | Pet ID: {pet_id} | Message: \"{message}\"")

    # Fetch all data 
    try:
        data = await _fetch_chat_data(user_id, pet_id, authorization)
        user_profile = data["user"]
        pet_data = data["pet"]
        pet_status_data = data["status"]
    except ValueError as e:
        logger.error("[ERROR] Data fetching failed for user %s: %s", user_id, e)
        raise HTTPException(status_code=404, detail=str(e))

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
    ai_response_text = await _call_ai_service(build_system_prompt, prompt)

    logger.info(f"=== [PROMPT BUILT] ===\nSystem Prompt: \n{build_system_prompt} \nUser Prompt: \n{prompt}\n=== [END PROMPT] ===\n")

    # The final response
    text_without_prefix = re.sub(rf"^{re.escape(pet_name)}\s*:\s*", "", ai_response_text, count=1).strip()
    cleaned_response = _clean_response_text(text_without_prefix)

    await save_message_and_get_context(user_id, pet_id, "ai", cleaned_response)

    features = extract_response_features(cleaned_response)

    logger.info("RESPONSE SENT")
    logger.info(f"  -> AI Response: {cleaned_response}")
    logger.info("="*70 + "\n")

    return {"response": cleaned_response, "features": features}