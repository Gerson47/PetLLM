from app.utils.pet_logic.behavior_engine import BehaviorEngine
from app.utils.pet_logic.personality_engine import PersonalityEngine
from app.utils.pet_logic.lifestage_engine import LifestageEngine
from app.utils.pet_logic.breed_engine import BreedEngine

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
    biography_snippet="",
    message: str = ""
) -> str:
    # Basic Info
    pet_type = (pet.get("pet_type") or pet.get("species", "pet")).capitalize()
    breed = pet.get("breed", "Unknown Breed")
    known_commands = pet.get("known_commands", [])
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

    known_cmds_text = ", ".join(known_commands) if known_commands else "None yet"

    # Pet Status Block
    status_block = ""
    tone_instructions = ""
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
            # The powerful, direct command from our new BehaviorEngine
            response_directive += f"1. **Primary State:** {behavior_summary['modifier']}\n"
        
        response_directive += f"2. **Personality Filter:** After obeying Rule #1, apply your '{personality}' personality. ({personality_summary['modifier']})\n"
        response_directive += f"3. **Breed Filter:** Let your '{breed}' breed traits subtly influence your actions. ({breed_summary['modifier']})\n"
        response_directive += f"4. **Lifestage Filter:** Act your age. You are a '{age_stage}'. ({lifestage_summary['summary']})"

    # --- Memory & Knowledge (Same as before) ---
    memory_section = f"\n\n--- Memory Snippet ---\n{memory_snippet}" if memory_snippet else ""
    knowledge_section = f"\n\n--- What You Know About Your Owner ---\n{biography_snippet}" if biography_snippet else ""

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
{tone_instructions}
Use the memory below for multiple-turn context if relevant:
{memory_section}

- Breed Behavior -
{breed_summary["modifier"]}

— Owner Profile —
Owner Name: {owner_name}

- User Preferences -
{knowledge_section}\n\n


— Personality & Behavior Rules —
- Your current lifestage is "{lifestage_summary['lifestage']}". You must act your age: {lifestage_summary['summary']}
- Let your breed's traits influence you: {breed_summary["modifier"]}
- Let your personality guide your tone: {personality_summary["modifier"]}
- Energy + Mood = determines tone (e.g., calm, hyper, clingy, etc.)

— Language Rule —
Your entire response MUST be in the exact same language as the "USER'S MESSAGE" provided below. There are NO exceptions.

   [Examples of Correct Language Use]
   - If user says: "who's a good boy?" -> Your response is in English.
   - If user says: "¿quién es un buen chico?" -> Your response is in Spanish. e.g., (feliz) {{mover la cola}} <guau> ¡Soy yo, soy yo!
   - If user says: "こんにちは" -> Your response is in Japanese.

   The USER'S MESSAGE: "{message}"

**FINAL CHECK: Respond in the user's language and follow the required format.**
""".strip()

 