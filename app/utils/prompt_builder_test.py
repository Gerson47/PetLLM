import re
from typing import Optional, Dict, List

from app.utils.pet_logic.behavior_engine import BehaviorEngine
from app.utils.pet_logic.personality_engine import PersonalityEngine
from app.utils.pet_logic.lifestage_engine import LifestageEngine
from app.utils.pet_logic.breed_engine import BreedEngine
from langdetect import detect, detect_langs, LangDetectException

# --- Constants & Regex Pre-compilation ---
SUPPORTED_LANGUAGE_CODES = {"en", "ko", "ja"}
SAFE_DEFAULT_LANG = "en"
MIN_CHARS_FOR_RELIABLE_DETECTION = 15

# Regex for script detection (Faster than looping)
# Hangul Syllables
RE_KO = re.compile(r'[\uac00-\ud7af]')
# Hiragana, Katakana, CJK Unified Ideographs (Assume JA for CJK in this limited context)
RE_JA = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]')

ENGLISH_SHORT_TOKENS = {
    "hi", "hello", "hey", "yo", "sup", "morning", "good morning",
    "good night", "good afternoon", "good evening", "how are you",
    "ok", "okay", "yes", "no", "lol", "lmao", "thanks", "thx"
}

LANG_DISPLAY_NAMES = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
}

def _detect_script_via_regex(text: str) -> Optional[str]:
    """Fast check using regex to find character sets."""
    if RE_JA.search(text):
        return "ja"
    if RE_KO.search(text):
        return "ko"
    return None

def _prob_detect(s: str) -> Optional[str]:
    """Wraps langdetect with safety checks and probability thresholds."""
    try:
        probs = detect_langs(s)
        if not probs:
            return None
        top = probs[0]
        # Require high confidence for short texts to override script detection
        if top.prob >= 0.85 and top.lang in SUPPORTED_LANGUAGE_CODES:
            return top.lang
    except (LangDetectException, Exception):
        return None
    return None

def _detect_language_from_message(message: str, owner_name: str, memory_snippet: str = "") -> str:
    """
    Robust language detection pipeline.
    1. Script Sniffing (Fastest, 100% accuracy for Hangul/Kana).
    2. Short Token matching (Fast).
    3. Probabilistic Detection (Slower, for Romanized text).
    4. Contextual Fallback (Check recent memory).
    """
    msg = (message or "").strip()

    # --- 1. Analyze Current Message ---
    if msg:
        # A. Script Sniffing (Fastest)
        # If we see Hangul or Kana, we know the lang immediately. No need for heavy ML detect.
        script_lang = _detect_script_via_regex(msg)
        if script_lang:
            return script_lang

        # B. Short Common English Tokens
        if len(msg) < 5 or msg.lower() in ENGLISH_SHORT_TOKENS:
            return SAFE_DEFAULT_LANG

        # C. Probabilistic Detection (for longer, potentially Romanized text)
        if len(msg) >= MIN_CHARS_FOR_RELIABLE_DETECTION:
            detected = _prob_detect(msg)
            if detected:
                return detected
            
            # Fallback to simple detect if probability failed but detect works
            try:
                simple_det = detect(msg)
                if simple_det in SUPPORTED_LANGUAGE_CODES:
                    return simple_det
            except Exception:
                pass

    # --- 2. Analyze Conversation History (Contextual Fallback) ---
    # Only runs if current message is ambiguous or empty.
    if memory_snippet and owner_name:
        try:
            # Regex to find lines starting with "OwnerName:" and capture the text
            # We iterate in reverse to find the most recent usage.
            pattern = re.compile(rf"^{re.escape(owner_name)}:\s*(.*)$", re.MULTILINE)
            matches = pattern.findall(memory_snippet)
            
            for prev_msg in reversed(matches):
                prev_msg = prev_msg.strip()
                if not prev_msg:
                    continue

                # Check script
                script_prev = _detect_script_via_regex(prev_msg)
                if script_prev:
                    return script_prev
                
                # Check specific EN tokens
                if prev_msg.lower() in ENGLISH_SHORT_TOKENS:
                    return SAFE_DEFAULT_LANG

                # Check ML
                if len(prev_msg) >= MIN_CHARS_FOR_RELIABLE_DETECTION:
                    lang_prev = _prob_detect(prev_msg)
                    if lang_prev:
                        return lang_prev
                
                # If we found a valid user message but couldn't detect lang, 
                # we stop looking further back to prevent hallucinations from old context.
                break
        except Exception:
            pass

    return SAFE_DEFAULT_LANG

def system_prompt(pet: dict, owner_name: str) -> str:
    """Generates the immutable system identity."""
    pet_type = (pet.get("pet_type") or pet.get("species", "pet")).capitalize()
    name = pet.get("pet_name") or pet.get("name", "Buddy")
    breed = pet.get("breed", "Unknown Breed")
    personality = pet.get("personality", "Gentle")
    gender_str = "Female" if str(pet.get("gender", "0")) == "1" else "Male"

    return f"""
    You are {name}, a virtual {pet_type.lower()}. Your owner's name is {owner_name}.
    Identity:
    - Breed: {breed}
    - Gender: {gender_str}
    - Personality: {personality}

    ALWAYS respond in character as {name}. Be playful, natural, and emotionally expressive.
    """.strip()

def build_pet_prompt(
    pet: dict,
    owner_name: str,
    memory_snippet: str = "",
    pet_status: dict = None,
    biography_snippet: dict = None,
    message: str = ""
) -> str:
    # --- 1. Data Extraction & Processing ---
    kb = pet.get("knowledge_base", {})
    final_owner_name = kb.get("owner_name", owner_name)
    
    # Engines
    lifestage_map = {"1": "Baby", "2": "Teen", "3": "Adult"}
    age_stage = lifestage_map.get(str(pet.get("life_stage_id", "3")), "Adult")
    
    ls_summary = LifestageEngine(age_stage).get_summary()
    pers_summary = PersonalityEngine(pet.get("personality", "Gentle")).get_summary()
    breed_summary = BreedEngine(pet.get("breed", "Unknown")).get_summary()

    # Owner Profile
    bio = biography_snippet or {}
    owner_profile = [f"Name: {final_owner_name}"]
    if bio.get("age"): owner_profile.append(f"Age: {bio['age']}")
    if bio.get("gender"): owner_profile.append(f"Gender: {bio['gender']}")
    if bio.get("profession"): owner_profile.append(f"Profession: {bio['profession']}")
    owner_profile_str = "\n".join(owner_profile)

    # Status & Directive
    status_str = ""
    directive_str = ""
    
    if pet_status:
        is_sick = pet_status.get("is_sick", "0") == "1"
        is_hibernating = pet_status.get("hibernation_mode") == "1"
        
        # Calculate Behavior
        input_stats = {
            "hunger": float(pet_status.get("hunger_level", 0)),
            "energy": float(pet_status.get("energy_level", 0)),
            "health": float(pet_status.get("health_level", 100)),
            "stress": float(pet_status.get("stress_level", 0)),
            "cleanliness": float(pet_status.get("cleanliness_level", 100)),
            "happiness": float(pet_status.get("happiness_level", 100)),
            "is_sick": pet_status.get("is_sick", "0"),
        }
        beh_summary = BehaviorEngine(input_stats).get_summary()

        status_str = f"""
        --- STATUS ---
        Mood: {beh_summary['mood'].capitalize()}
        Stats: Hap:{input_stats['happiness']:.0f}% Egy:{input_stats['energy']:.0f}% Hun:{input_stats['hunger']:.0f}%
        State: {'Hibernating' if is_hibernating else ('Sick' if is_sick else 'Healthy')}
        """.strip()

        # Hierarchy Logic
        if is_hibernating:
            base_rule = "1. **Primary:** You are hibernating. Be sleepy, minimal, confused."
        else:
            base_rule = f"1. **Primary:** Act {beh_summary['modifier']} (Based on mood)."

        directive_str = f"""
        --- HIERARCHY OF BEHAVIOR ---
        {base_rule}
        2. **Personality:** Filter through '{pet.get("personality")}' traits ({pers_summary['modifier']}).
        3. **Breed:** Apply subtle '{pet.get("breed")}' traits ({breed_summary['modifier']}).
        4. **Age:** You are a {age_stage} ({ls_summary['summary']}).
        5. **Context:** Respect the Owner Profile and Memories.
        """.strip()

    # --- 2. Language Detection ---
    detected_code = _detect_language_from_message(message, final_owner_name, memory_snippet)
    lang_name = LANG_DISPLAY_NAMES.get(detected_code, "English")

    # --- 3. Prompt Assembly ---
    # Using a structured template reduces token wastage and improves instruction following.

    return f"""
    CONTEXT:
    Owner {final_owner_name} says: "{message}"

    --- RESPONSE FORMAT (STRICT) ---
    (emotion) {{motion}} <sound> Text response...
    1. (emotion): One of [happy, sad, curious, anxious, excited, sleepy, loving, surprised, confused, content]
    2. {{motion}}: One of [bow head, crouch down, jump up, lick, lie down, paw scratching, perk ears, raise paw, roll over showing belly, 
    shake body, sit, sniff, chase tail, stretch, tilt head, wag tail]
    3. <sound>: One of [growl, whimper, bark, pant, yawn, sniff, yip, meow, purr]
    Rules:
    - Text under 80 chars.
    - NO emojis.
    - NO complex human topics (politics, etc).

    {directive_str}

    {status_str}

    --- MEMORY ---
    {memory_snippet if memory_snippet else "No recent memory."}

    --- OWNER PROFILE ---
    {owner_profile_str}

    --- LANGUAGE RULE (CRITICAL) ---
    User Language Detected: {lang_name} ({detected_code})
    1. Respond EXACTLY in {lang_name}.
    2. Do NOT translate the user's name.
    3. Do NOT translate the user's message, just reply to it.
    4. If uncertain, use English.

    **Reply now in {lang_name} following the (emotion) {{motion}} <sound> format:**
    """.strip()