import logging
import asyncio
import random
import os
from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, Header, UploadFile, File, Form

from app.schema.main_schema import get_chat_form, ChatResponse
from app.utils.prompt_builder import build_pet_prompt
from app.utils.chat_handler import generate_response
from app.utils.php_service import get_user_by_id, get_pet_by_id, get_pet_status_by_id
from app.utils.extract_response import extract_response_features, extract_digit_label
# from app.utils.digit_memory_engine import DigitMemoryEngine
from app.db.connection import chats_collection, get_db
from app.utils.language_translator import (
    detect_language,
    translate_to_english,
    translate_to_user_language
)
# from app.utils.digit_classifier import predict_digit
# from ip_features.content_moderator import is_flagged_content  # ⛔ disabled for testing

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads/numbers"
os.makedirs(UPLOAD_DIR, exist_ok=True)

db = get_db()
flashcards_collection = db.k_numbers

# Soft warning responses (unused for now since moderation is disabled)
SOFT_WARNINGS = [
    "(confused) {tilt head} <whimper>\nHmm? That didn’t sound right. Let's talk about something fun!",
    "(anxious) {crouch down} <whimper>\nUmm... I don’t think I understand that. Can we talk about something else?",
    "(confused) {perk ears} <sniff sniff>\nThat feels weird. Maybe try saying it differently?",
    "(sleepy) {lie down} <yawn>\nThat’s a bit too strange for me. Let’s cuddle instead.",
    "(anxious) {sit beside} <whimper>\nI'm not sure I like that. Can we change the topic?",
    "(sad) {bow head} <whimper>\nThat makes me feel icky. Want to play instead?"
]


async def get_auth_token(authorization: str = Header(...)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return authorization

@router.post("/chat", response_model=ChatResponse)
async def chat(
    user_id: str = Form(...),
    pet_id: str = Form(...),
    message: str = Form(...),
    image: UploadFile = File(None),
    authorization: str = Depends(get_auth_token)
):
    logger.info("\n--- Chat Request Received ---\nUser ID: %s | Pet ID: %s", user_id, pet_id)

    # === FETCH USER & PET INFO ===
    try:
        user_data = await get_user_by_id(user_id, authorization)
        owner_name = user_data["first_name"]
        pet_data = await get_pet_by_id(pet_id, authorization)
        pet_status_data = await get_pet_status_by_id(pet_id, authorization)
    except Exception as e:
        logger.error("Fetch error: %s", e)
        raise HTTPException(status_code=500, detail="User or pet data retrieval failed.")

    # === TRANSLATE INPUT ===
    user_lang = detect_language(message)
    translated_message = translate_to_english(message, user_lang)

    # === DigitMemoryEngine Setup ===
    engine = DigitMemoryEngine(flashcards_collection)
    digit_knowledge = ""

    # === IMAGE + TEACH/RECOGNIZE HANDLING ===
    if image:
        filename = f"{uuid4()}_{image.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(await image.read())

        # --- Attempt to extract label from message ---
        label = extract_digit_label(translated_message)

        if label:
            logger.info(f"[TEACHING MODE] Saving label '{label}' from message: {translated_message}")
            flashcard_data = {
                "pet_id": pet_id,
                "label": label,
                "image_path": file_path,
            }
            result = await flashcards_collection.insert_one(flashcard_data)
            logger.info(f"[MongoDB] Inserted flashcard with _id: {result.inserted_id}")
            digit_knowledge = await engine.summarize_teaching(pet_id, label)

        else:
            logger.info("[RECOGNITION MODE] No label extracted. Predicting image.")
            predicted_label = str(predict_digit(file_path))
            logger.info(f"[Classifier] Predicted label: {predicted_label}")
            digit_knowledge = await engine.summarize_recognition(pet_id, predicted_label)
            
    # === SHORT-TERM MEMORY SNIPPET ===
    recent_chats = await chats_collection.find({
        "user_id": user_id,
        "pet_id": pet_id
    }).sort("timestamp", -1).to_list(5)

    def truncate(text, limit=250): return text[:limit].rsplit(" ", 1)[0] + "..." if len(text) > limit else text

    memory_snippet = "\n".join(
        f"User: {truncate(c['user_message'])}\nPet: {truncate(c['pet_response'])}"
        for c in reversed(recent_chats)
    )

    if memory_snippet:
        logger.info("\n===== MEMORY SNIPPET START =====\n%s\n===== MEMORY SNIPPET END =====", memory_snippet)

    # === PROMPT GENERATION ===
    prompt = build_pet_prompt(
        pet=pet_data,
        owner_name=owner_name,
        memory_snippet=memory_snippet,
        pet_status=pet_status_data,
        digit_knowledge=digit_knowledge
    )
    prompt += f"\n\nUser: {translated_message}\n{pet_data.get('species', 'pet').capitalize()}:"

    logger.info("\n--- Prompt Sent to LLM ---\n%s", prompt)

    # === LLM RESPONSE ===
    response = await generate_response(prompt, use_mock=False)
    if response.startswith("[ERROR]"):
        raise HTTPException(status_code=502, detail="AI response unavailable")

    translated_response = translate_to_user_language(response.strip(), user_lang)

    # === STORE CHAT ===
    features = extract_response_features(response)
    await chats_collection.insert_one({
        "user_id": user_id,
        "pet_id": pet_id,
        "timestamp": datetime.utcnow(),
        "user_message": message,
        "pet_response": translated_response
    })

    return {
        "response": translated_response,
        "features": features
    }