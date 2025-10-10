import logging
import json
import asyncio
import random
from app.db.connection import user_profiles_collection
from app.utils.chat_handler import generate_response
from app.utils.prompt_builder import system_prompt

logger = logging.getLogger("fact_extractor")

FACT_EXTRACTION_PROMPT = """
Analyze the user's message to identify any personal facts about the user, such as their name, gender, location, preferences, likes, or dislikes.
Extract these facts into a valid JSON object. The keys should be snake_case.
- If the user says "My name is John", extract {{"name": "John"}}.
- If they say "I love listening to rock music", extract {{"favorite_music": "rock"}}.
- If they say "I live in California", extract {{"location": "California"}}.
- If no personal facts are mentioned, return an empty JSON object: {{}}.

User's message: "{user_message}"

JSON output:
"""

async def extract_and_save_user_facts(user_id: int, user_message: str):
    """
    Analyzes a user's message to find personal facts and saves them to their
    user_profile document. This function is now robust against API errors.
    """
    try:
        logger.info(f"BACKGROUND TASK: Starting fact extraction for user_id {user_id}")
        
        build_system_prompt = "You are a helpful assistant that extracts personal facts about the user from their messages into a strict JSON format."
        prompt = FACT_EXTRACTION_PROMPT.format(user_message=user_message)
        
        llm_json_string = await generate_response(build_system_prompt, prompt)
        response_data = json.loads(llm_json_string)

        if response_data.get("status") == "error":
            error_message = response_data.get("error", {}).get("message", "Unknown AI error")
            logger.error(f"LLM call failed inside fact_extractor for user_id {user_id}. API Error: {error_message}")
            return
        
        actual_llm_output = response_data.get("data", {}).get("response")
        
        if not actual_llm_output:
            logger.warning(f"LLM response for user {user_id} was empty or malformed.")
            return

        json_str = actual_llm_output.strip()
        start = json_str.find('{')
        end = json_str.rfind('}')
        
        if start == -1 or end == -1:
            logger.warning(f"Could not find a JSON object in the LLM response for user {user_id}: {json_str}")
            return
            
        json_str = json_str[start : end + 1]
        extracted_facts = json.loads(json_str)

        if not isinstance(extracted_facts, dict) or not extracted_facts:
            logger.info("No new facts to save for user_id: %s", user_id)
            return

        logger.info(f"Found facts for user_id {user_id}: {extracted_facts}")
        
        # --- FIX IS HERE: Replace dictionary comprehension with a safe loop ---
        update_fields = {}
        for key, value in extracted_facts.items():
            # 1. Validate the key is a non-empty string
            if not isinstance(key, str) or not key.strip():
                logger.warning(f"Ignoring invalid (empty or non-string) key from LLM for user {user_id}: '{key}'")
                continue # Skip this invalid key-value pair

            # 2. Sanitize the key (remove whitespace, ensure snake_case if desired)
            sanitized_key = key.strip() 

            # 3. Build the update path
            update_fields[f"biography.{sanitized_key}"] = value

        # No valid fields were created after validation, so exit.
        if not update_fields:
            logger.info("No valid facts remained after sanitization for user_id: %s", user_id)
            return

        # Handle the special 'name' case
        if "biography.name" in update_fields:
            update_fields["first_name"] = update_fields["biography.name"]

        await user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": update_fields}
        )
        logger.info(f"BACKGROUND TASK FINISHED SUCCESSFULLY for user_id {user_id}.")

    except json.JSONDecodeError:
        logger.warning(f"Fact extractor could not parse final JSON from LLM response for user {user_id}: {actual_llm_output}")
    except Exception as e:
        logger.error(
            f"--- FATAL ERROR IN BACKGROUND TASK for user_id {user_id} ---",
            exc_info=True
        )