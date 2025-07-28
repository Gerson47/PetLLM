import logging
from datetime import datetime
from typing import List, Dict

from app.db.connection import chats_collection

logger = logging.getLogger("chat_retention")

RECENT_MESSAGES_LIMIT = 10 

async def save_message_and_get_context(
    user_id: int, 
    pet_id: int, 
    sender: str, 
    message: str
) -> List[Dict]:
    """
    Saves a message to chat history and returns a recent slice of the conversation.
    Fact extraction is handled by the route layer via BackgroundTasks.
    """
    try:
        query = {"user_id": user_id, "pet_id": pet_id}
        msg_obj = {"text": message, "sender": sender, "timestamp": datetime.utcnow()}
        
        await chats_collection.update_one(
            query,
            {
                "$push": {"messages": msg_obj},
                "$set": {"lastUpdatedAt": datetime.utcnow()}
            },
            upsert=True
        )

        updated_document = await chats_collection.find_one(
            query,
            projection={"messages": {"$slice": -RECENT_MESSAGES_LIMIT}}
        )
        
        if not updated_document or "messages" not in updated_document:
            return []

        return updated_document.get("messages", [])

    except Exception as e:
        logger.error(f"Error in save_message_and_get_context for user {user_id}, pet {pet_id}: {e}", exc_info=True)
        return []