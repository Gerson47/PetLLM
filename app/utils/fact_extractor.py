import logging
import json
from app.db.connection import user_profiles_collection
from app.utils.chat_handler import generate_response

logger = logging.getLogger("fact_extractor")

# The prompt is correct, no changes needed here.
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
    # --- MODIFIED: Added a comprehensive try/except block for debugging ---
    try:
        logger.info(f"BACKGROUND TASK STARTED for user_id {user_id}")

        prompt = FACT_EXTRACTION_PROMPT.format(user_message=user_message)
        llm_response = await generate_response(prompt, use_mock=False)

        # --- MODIFIED: Critical error checking BEFORE parsing JSON ---
        if llm_response.strip().startswith("[ERROR]"):
            logger.error(f"LLM call failed inside fact_extractor for user_id {user_id}. Response: {llm_response}")
            return # Exit the function gracefully

        # Clean up the response to only get the JSON part
        json_str = llm_response.strip()
        if '```json' in json_str:
            json_str = json_str.split('```json\n')[1].split('\n```')[0]

        extracted_facts = json.loads(json_str)

        if not isinstance(extracted_facts, dict) or not extracted_facts:
            logger.info("No new facts to save for user_id: %s", user_id)
            return

        logger.info(f"Found facts for user_id {user_id}: {extracted_facts}")
        
        update_fields = {f"biography.{key}": value for key, value in extracted_facts.items()}
        
        if "name" in extracted_facts:
            update_fields["first_name"] = extracted_facts["name"]

        await user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": update_fields}
        )
        logger.info(f"BACKGROUND TASK FINISHED SUCCESSFULLY for user_id {user_id}.")

    except json.JSONDecodeError:
        # This will catch malformed JSON that wasn't an [ERROR] string
        logger.warning(f"Fact extractor could not parse JSON from LLM response for user {user_id}: {llm_response}")
    except Exception as e:
        # This will catch ANY other unexpected error and log it with a full traceback.
        logger.error(
            f"--- FATAL ERROR IN BACKGROUND TASK for user_id {user_id} ---",
            exc_info=True  # This includes the full error traceback in your logs
        )