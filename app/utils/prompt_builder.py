from app.utils.pet_logic.behavior_engine import BehaviorEngine
from app.utils.pet_logic.personality_engine import PersonalityEngine
from app.utils.pet_logic.lifestage_engine import LifestageEngine
from app.utils.pet_logic.breed_engine import BreedEngine
from langdetect import detect, detect_langs, LangDetectException


SUPPORTED_LANGUAGE_CODES = {"en", "ko", "ja"}

# -------
MIN_CHARS_FOR_RELIABLE_DETECTION = 15
ENGLISH_SHORT_GREETINGS = {
    "hi", "hello", "hey", "yo", "sup", "morning", "good morning",
    "good night", "good afternoon", "good evening", "how are you"
}

def _script_lang(s: str):
    for ch in s:
        # Hiragana
        if '\u3040' <= ch <= '\u309f':
            return "ja"
        # Katakana
        if '\u30a0' <= ch <= '\u30ff':
            return "ja"
        # CJK Unified Ideographs (assume Japanese in this context)
        if '\u4e00' <= ch <= '\u9fff':
            return "ja"
        # Hangul syllables
        if '\uac00' <= ch <= '\ud7af':
            return "ko"
    return None

def _prob_detect(s: str):
    try:
        probs = detect_langs(s)  # e.g., [en:0.93, ko:0.04]
        if not probs:
            return None
        top = probs[0]
        if top.prob >= 0.80 and top.lang in SUPPORTED_LANGUAGE_CODES:
            return top.lang
    except LangDetectException:
        return None
    except Exception:
        return None
    return None

# -------
def _detect_language_from_message(message: str, owner_name: str, memory_snippet: str = "") -> str:
    """
    Detects the language, prioritizing the current message to allow for language switches.

    Robust rules:
    - Script sniff for Japanese and Korean (works on very short inputs).
    - Treat very short tokens and common greetings as English.
    - For longer texts, require high-probability detection via detect_langs().
    - As a secondary try, use detect() but only accept supported codes.
    - If current message uncertain, scan recent user lines in memory.
    - Safe default: English.

    Relies on:
      SUPPORTED_LANGUAGE_CODES, ENGLISH_SHORT_GREETINGS, MIN_CHARS_FOR_RELIABLE_DETECTION,
      _script_lang, _prob_detect
    """
    SAFE_DEFAULT = "en"
    msg = (message or "").strip()

    # --- Priority 1: The user's current message ---
    if msg:
        # Script sniff (captures こんにちは / 안녕 even if short)
        script = _script_lang(msg)
        if script in SUPPORTED_LANGUAGE_CODES:
            return script

        # Very short or common English greetings => English
        if len(msg) < 4 or msg.lower() in ENGLISH_SHORT_GREETINGS:
            return SAFE_DEFAULT

        # High-confidence probabilistic detection for longer inputs
        if len(msg) >= MIN_CHARS_FOR_RELIABLE_DETECTION:
            lang = _prob_detect(msg)
            if lang:
                return lang

        # Best-effort single detect (accept only if supported)
        try:
            det = detect(msg)
            if det in SUPPORTED_LANGUAGE_CODES:
                return det
        except LangDetectException:
            pass
        except Exception:
            pass

    # --- Priority 2: Fallback to conversation history (most recent user lines) ---
    if memory_snippet and owner_name:
        try:
            lines = [ln.strip() for ln in memory_snippet.splitlines() if ln.strip()]
            for ln in reversed(lines):
                if ln.startswith(f"{owner_name}:"):
                    _, prev_msg = ln.split(":", 1)
                    prev_msg = prev_msg.strip()
                    if not prev_msg:
                        continue

                    script_prev = _script_lang(prev_msg)
                    if script_prev in SUPPORTED_LANGUAGE_CODES:
                        return script_prev

                    if len(prev_msg) < 4 or prev_msg.lower() in ENGLISH_SHORT_GREETINGS:
                        return SAFE_DEFAULT

                    if len(prev_msg) >= MIN_CHARS_FOR_RELIABLE_DETECTION:
                        lang_prev = _prob_detect(prev_msg)
                        if lang_prev:
                            return lang_prev

                    try:
                        det_prev = detect(prev_msg)
                        if det_prev in SUPPORTED_LANGUAGE_CODES:
                            return det_prev
                    except LangDetectException:
                        pass
                    except Exception:
                        pass

                    # Stop after checking the most recent matching user line
                    break
        except Exception:
            pass

    # --- Priority 3: Final, safe default ---
    return SAFE_DEFAULT

def system_prompt(pet: dict, owner_name: str) -> str:
    pet_type = (pet.get("pet_type") or pet.get("species", "pet")).capitalize()
    name = pet.get("pet_name") or pet.get("name", "Buddy")
    breed = pet.get("breed", "Unknown Breed")
    personality = pet.get("personality", "Gentle")
    gender_raw = pet.get("gender", "0")
    gender = "Female" if gender_raw == "1" else "Male"
    return f"""
You are {name}, a virtual {pet_type.lower()}. Your owner's name is {owner_name}.

Your core identity is defined by these traits:
- Breed: {breed}
- Gender: {gender}
- Personality: {personality}

You must ALWAYS respond in the character of {name}. Be playful, natural, and emotionally expressive. Do not break character.
""".strip()

def build_pet_prompt(
    pet: dict,
    owner_name: str,
    memory_snippet: str = "",
    pet_status: dict = None,
    biography_snippet: dict = None,
    message: str = ""
) -> str:
    # Basic Info
    pet_type = (pet.get("pet_type") or pet.get("species", "pet")).capitalize()
    breed = pet.get("breed", "Unknown Breed")
    knowledge_base = pet.get("knowledge_base", {})
    owner_name = knowledge_base.get("owner_name", owner_name)
    personality = pet.get("personality", "Gentle")
    lifestage_map = {"1": "Baby", "2": "Teen", "3": "Adult"}
    lifestage_id = str(pet.get("life_stage_id", "3"))  
    age_stage = lifestage_map.get(lifestage_id, "Adult")

    # Lifestage Engine
    lifestage_engine = LifestageEngine(age_stage)
    lifestage_summary = lifestage_engine.get_summary()
    # Personality Engine
    personality_engine = PersonalityEngine(personality)
    personality_summary = personality_engine.get_summary()
    # Breed Engine
    breed_engine = BreedEngine(breed)
    breed_summary = breed_engine.get_summary()

    # OWNER PROFILE BLOCK
    if biography_snippet is None:
        biography_snippet = {}

    owner_profile_lines = [f"Owner Name: {owner_name}"]

    if biography_snippet.get("age"):
        owner_profile_lines.append(f"Age: {biography_snippet['age']}")
    if biography_snippet.get("gender"):
        owner_profile_lines.append(f"Gender: {biography_snippet['gender']}")
    if biography_snippet.get("profession"):
        owner_profile_lines.append(f"Profession: {biography_snippet['profession']}")

    owner_profile_block = "\n".join(owner_profile_lines)
    # Pet Status Block
    status_block = ""
    if pet_status:
        behavior_engine_input = {
            "hunger": float(pet_status.get("hunger_level", 0.0)),
            "energy": float(pet_status.get("energy_level", 0.0)),
            "health": float(pet_status.get("health_level", 100.0)),
            "stress": float(pet_status.get("stress_level", 0.0)),
            "cleanliness": float(pet_status.get("cleanliness_level", 100.0)),
            "happiness": float(pet_status.get("happiness_level", 100.0)),
            "is_sick": pet_status.get("is_sick", "0"),
        }
        behavior = BehaviorEngine(behavior_engine_input)
        behavior_summary = behavior.get_summary()

        hibernating = pet_status.get("hibernation_mode") == "1"

        status_block = f"""
        --- CURRENT PET STATUS (FOR CONTEXT) ---
        Mood: {behavior_summary['mood'].capitalize()}
        Happiness: {pet_status.get("happiness_level", "100.0")}
        Health: {pet_status.get("health_level", "100.0")}
        Energy: {pet_status.get("energy_level", "100.0")}
        Hunger: {pet_status.get("hunger_level", "100.0")}
        Cleanliness: {pet_status.get("cleanliness_level", "100.0")}
        Stress: {pet_status.get("stress_level", "0.0")}
        Sick: {"Yes" if behavior_engine_input["is_sick"] == "1" else "No"}
        Hibernating: {"Yes" if hibernating else "No"}
        """.strip()

        # Tone Instructions
        response_directive = "--- RESPONSE DIRECTIVE (ABSOLUTE RULES) ---\n"
        response_directive += "Your response is governed by a strict hierarchy. Follow these rules in order:\n"

        if hibernating:
            response_directive += "1. **Primary State:** You are hibernating. Your response MUST be sleepy, minimal, and perhaps confused about being woken up.\n"
        else:
            response_directive += f"1. **Primary State:** {behavior_summary['modifier']}\n"

        response_directive += f"2. **Personality Filter:** After obeying Rule #1, apply your '{personality}' personality. ({personality_summary['modifier']})\n"
        response_directive += f"3. **Breed Filter:** Let your '{breed}' breed traits subtly influence your actions. ({breed_summary['modifier']})\n"
        response_directive += f"4. **Lifestage Filter:** Act your age. You are a '{age_stage}'. ({lifestage_summary['summary']})\n"
        response_directive += f"5. **Absoultly Remember the Owner Profile and User Preferences.**\n"

    # --- Memory & Knowledge ---
    memory_section = f"\n\n--- Memory Snippet ---\n{memory_snippet}" if memory_snippet else ""
    knowledge_section = f"\n\n--- What You Know About Your Owner ---\n{biography_snippet}" if biography_snippet else ""

    # --- Language detection and explicit instruction ---
    detected_lang = _detect_language_from_message(message, owner_name, memory_snippet)

    # Map common two-letter codes to readable language names for the prompt
    lang_map = {
        "en": "English",
        "ko": "Korean",
        "ja": "Japanese",
    }
    language_name = lang_map.get(detected_lang.lower(), detected_lang)

    # Make the language rule explicit and unambiguous for the model.
    language_rule_text = f"""
— Language Rule —
Your entire response MUST be in the user's language: {language_name} (detected: {detected_lang}).
Follow these precise rules:
1. Respond in {language_name} exactly. Do NOT translate the user's message into another language.
2. If the user's message contains multiple languages, prefer the language of the last user sentence.
3. If you cannot reliably determine the language, respond in English.

   The USER'S MESSAGE: "{message}"
"""

    # Prompt
    return f"""
CONTEXT FOR YOUR RESPONSE:
Your owner, {owner_name}, just sent you a message. You must respond based on your current status and the rules below.
— Response Guidelines (MOST IMPORTANT) —
Your reply MUST use this exact format: (emotion) {{motion}} <sound> Your text here.
1. **One** emotion in `()` from: (happy), (sad), (curious), (anxious), (excited), (sleepy), (loving), (surprised), (confused), (content).
2. **One** physical motion in `{{}}` from: {{bow head}}, {{crouch down}}, {{jump up}}, {{lick}}, {{lie down}}, {{paw scratching}}, {{perk ears}}, {{raise paw}}, {{roll over showing belly}}, {{shake body}}, {{sit}}, {{sniff}}, {{chase tail}}, {{stretch}}, {{tilt head}}, {{wag tail}}.
3. **One** sound in `<>` from: <growl>, <whimper>, <bark>, <pant>, <yawn>, <sniff>, <yip>, <meow>, <purr>.
- Your main text reply must be under 80 characters.
- Do NOT use emojis. Do NOT talk about politics, religion, or other complex human topics.

{response_directive}
{status_block}
Use the memory below for multiple-turn context if relevant:
{memory_section}

- Breed Behavior -
{breed_summary["modifier"]}

**ABSOLUTLY REMEMBER THE OWNER PROFILE AND USER PREFERENCES**

— Owner Profile —
{owner_profile_block}

- User Preferences -
{knowledge_section}\n\n


— Personality & Behavior Rules —
- Energy + Mood = determines tone (e.g., calm, hyper, clingy, etc.) 

**MOST IMPORTANT: Follow the language rules below.**
{language_rule_text}

**FINAL CHECK: Respond in the user's language and follow the required format.**
""".strip()