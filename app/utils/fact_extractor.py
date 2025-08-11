import logging
import json
import asyncio
import random
from app.db.connection import user_profiles_collection
from app.utils.chat_handler import generate_response

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
        
        # Optional: Keep a small delay if you want to avoid rate-limiting
        # await asyncio.sleep(random.uniform(1.0, 1.5))

        prompt = FACT_EXTRACTION_PROMPT.format(user_message=user_message)
        
        # ---> 1. This now returns a JSON STRING, not plain text
        llm_json_string = await generate_response(prompt)

        # ---> 2. Parse the outer JSON from generate_response
        response_data = json.loads(llm_json_string)

        # ---> 3. Check for errors and get the actual content
        if response_data.get("status") == "error":
            error_message = response_data.get("error", {}).get("message", "Unknown AI error")
            logger.error(f"LLM call failed inside fact_extractor for user_id {user_id}. API Error: {error_message}")
            return
        
        # This is the string we actually want to process (e.g., `{"name": "John"}`)
        actual_llm_output = response_data.get("data", {}).get("response")
        
        if not actual_llm_output:
            logger.warning(f"LLM response for user {user_id} was empty or malformed.")
            return

        # ---> 4. Make the JSON cleanup much more robust to prevent crashes
        # It finds the first '{' and the last '}' to isolate the JSON object
        json_str = actual_llm_output.strip()
        start = json_str.find('{')
        end = json_str.rfind('}')
        
        if start == -1 or end == -1:
            logger.warning(f"Could not find a JSON object in the LLM response for user {user_id}: {json_str}")
            return
            
        json_str = json_str[start : end + 1]

        # ---> 5. Now, parse the cleaned, extracted JSON string
        extracted_facts = json.loads(json_str)

        if not isinstance(extracted_facts, dict) or not extracted_facts:
            logger.info("No new facts to save for user_id: %s", user_id)
            return

        logger.info(f"Found facts for user_id {user_id}: {extracted_facts}")
        
        update_fields = {f"biography.{key}": value for key, value in extracted_facts.items()}
        
        if "name" in extracted_facts:
            # Also update the top-level first_name field for convenience
            update_fields["first_name"] = extracted_facts["name"]

        await user_profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": update_fields}
        )
        logger.info(f"BACKGROUND TASK FINISHED SUCCESSFULLY for user_id {user_id}.")

    except json.JSONDecodeError:
        # This will catch malformed JSON from the LLM's *actual_llm_output*
        logger.warning(f"Fact extractor could not parse final JSON from LLM response for user {user_id}: {actual_llm_output}")
    except Exception as e:
        # This will catch ANY other unexpected error
        logger.error(
            f"--- FATAL ERROR IN BACKGROUND TASK for user_id {user_id} ---",
            exc_info=True  # This includes the full error traceback in your logs
        )