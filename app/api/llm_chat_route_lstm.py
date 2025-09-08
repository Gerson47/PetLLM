import logging
from fastapi import APIRouter, Depends, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel
import json
import re

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
from app.api.get_history import get_history
from app.utils.php_service import get_user_by_id, get_pet_by_id, get_pet_status_by_id
from app.utils.user_operations import get_or_create_user_profile
from app.utils.fact_extractor import extract_and_save_user_facts

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
    background_tasks: BackgroundTasks,
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
    user_lang = message

    logger.info(f"Adding fact extraction to background tasks for user_id: {user_id}")
    background_tasks.add_task(extract_and_save_user_facts, user_id, user_lang)

    # This function will now run the fact extraction synchronously (it will wait)
    conversation_context = await save_message_and_get_context(
        user_id=user_id,
        pet_id=pet_id,
        sender="user",
        message=user_lang
    )

    logger.info("Retrieving full conversation history for logging purposes...")
    full_history = await get_history(user_id=user_id, pet_id=pet_id)
    if full_history:
        logger.info(f"--- Full History (User: {user_id}, Pet: {pet_id}) ---")
        for i, msg in enumerate(full_history):
            sender = msg.get('sender', 'unknown').upper()
            text = msg.get('text', '[no text]')
            timestamp = msg.get('timestamp', 'N/A')
            logger.info(f"  [{i+1}] ({sender} @ {timestamp}): {text}")
        logger.info("-----------------------------------------------------")
    else:
        logger.info("No extensive chat history found.")
    
    history_snippet = "\n".join(
        f"{owner_name}: {msg['text']}" if msg['sender'] == 'user' else f"{pet_name}: {msg['text']}"
        for msg in conversation_context
    )

    # --- LLM Call for Chat Response ---
    prompt = build_pet_prompt(pet_data, owner_name, memory_snippet=history_snippet, pet_status=pet_status_data,  biography_snippet=biography)
    prompt += f"\n{pet_name}:"
    llm_json_string = await generate_response(prompt)
    logger.info("\n\nPrompt sent to LLM:\n%s", prompt)

    try:
        response_data = json.loads(llm_json_string)
    except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM service: {llm_json_string}")
            raise HTTPException(status_code=502, detail="AI service returned malformed data.")

        # ---> 3. Check for errors using the 'status' key in the parsed JSON
    if response_data.get("status") == "error":
            error_message = response_data.get("error", {}).get("message", "Unknown AI error")
            logger.error(f"LLM Service Error: {error_message}")
            raise HTTPException(status_code=502, detail=error_message)
        
        # ---> 4. Extract the actual AI text response
    ai_response_text = response_data.get("data", {}).get("response")
    if not ai_response_text:
            logger.error(f"AI service response missing 'data.response' field: {response_data}")
            raise HTTPException(status_code=502, detail="AI service returned an incomplete response.")

    prefix_pattern = f"^{re.escape(pet_name)}\s*:\s*"
    text_without_prefix = re.sub(prefix_pattern, "", ai_response_text, count=1)

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251" 
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002620-\U000026FF"  # Miscellaneous Symbols
        "\U0000200D"             # Zero-width joiner for complex emojis
        "]+",
        flags=re.UNICODE,
    )
    text_without_emojis = emoji_pattern.sub(r'', text_without_prefix)

    cleaned_response = text_without_emojis.strip().replace('\n', ' ')
        # --- Save AI Response ---
        # ---> 5. Use the extracted text response here
    await save_message_and_get_context(
            user_id=user_id,
            pet_id=pet_id,
            sender="ai",
            message=cleaned_response
        )
        
        # --- Final Translation and Return ---
        # ---> 6. Use the extracted text response for feature extraction
    features = extract_response_features(cleaned_response)
    logger.info("\n\nUser query: \n%s", user_lang)
    logger.info("\nAI Response: \n%s", cleaned_response)
    
    # ---> 7. Return the extracted AI response text, not the user's message
    return {"response": cleaned_response, "features": features}