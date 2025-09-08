import logging
from typing import List, Dict
from app.db.connection import chats_collection
from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)

LIMIT_MESSAGES = 100  # Number of recent messages to retrieve

@router.post("/history", response_model=List[Dict])
async def get_history(user_id: int, pet_id: int) -> List[Dict]:
    """
    Retrieves the full message history for a user and pet.
    """
    try:
        query = {"user_id": user_id, "pet_id": pet_id}
        document = await chats_collection.find_one(query)

        if not document or "messages" not in document:
            logger.warning(f"No chat history found for user {user_id} and pet {pet_id}")
            return []

        return document.get("messages", [][-LIMIT_MESSAGES:]) 

    except Exception as e:
        logger.error(f"Error retrieving history for user {user_id}, pet {pet_id}: {e}", exc_info=True)
        return []