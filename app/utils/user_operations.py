import logging
from datetime import datetime
from app.db.connection import user_profiles_collection

logger = logging.getLogger("user_operations")

async def get_or_create_user_profile(user_id: int, user_data_from_php: dict):
    """
    Finds a user profile by user_id. If it doesn't exist, it creates one
    using the data fetched from the external service. This is the entry point
    for ensuring a user exists in our local database.
    """
    try:
        # 1. Try to find the user in our database
        profile = await user_profiles_collection.find_one({"user_id": user_id})

        if profile:
            # The user already exists, so we just return their profile
            logger.info(f"Found existing profile for user_id: {user_id}")
            return profile

        # 2. If not found, we create the new document
        logger.info(f"Creating new profile for user_id: {user_id}")
        
        new_profile_doc = {
            "user_id": user_id,
            "email": user_data_from_php.get("email"),
            "first_name": user_data_from_php.get("first_name"),
            "pet_ids": [], # Can be populated later
            "biography": {}, # IMPORTANT: Start with an empty biography object
            "preferences": {
                "default_language": "en"
            },
            "metadata": {
                "createdAt": datetime.utcnow()
            }
        }
        
        await user_profiles_collection.insert_one(new_profile_doc)
        
        return new_profile_doc

    except Exception as e:
        logger.error(f"Error in get_or_create_user_profile for user_id {user_id}: {e}", exc_info=True)
        return None