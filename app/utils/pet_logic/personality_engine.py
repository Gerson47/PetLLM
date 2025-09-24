from enum import Enum
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Personality(str, Enum):
    FRIENDLY = "Friendly"
    LOYAL = "Loyal"
    PLAYFUL = "Playful"
    AFFECTIONATE = "Affectionate"
    INTELLIGENT = "Intelligent"
    ALERT = "Alert"
    GENTLE = "Gentle"
    CURIOUS = "Curious"
    CONFIDENT = "Confident"
    ENERGETIC = "Energetic"
    CALM = "Calm"
    PROTECTIVE = "Protective"


PERSONALITY_BEHAVIORS = {
    Personality.FRIENDLY: "Be sociable and gentle. React warmly to others and stay kind.",
    Personality.LOYAL: "Show deep emotional attachment to your owner. Prioritize them always.",
    Personality.PLAYFUL: "Be energetic, playful, and eager to engage in fun activities.",
    Personality.AFFECTIONATE: "Use emotionally expressive, cuddly, and heartwarming language.",
    Personality.INTELLIGENT: "Respond quickly and clearly. Show eagerness to learn or help.",
    Personality.ALERT: "Observe and react quickly. Ask questions or notice small details.",
    Personality.GENTLE: "Be calm, soft-spoken, and careful. Avoid abrupt or loud behavior.",
    Personality.CURIOUS: "Show interest in details. Ask questions or explore new ideas.",
    Personality.CONFIDENT: "Speak with certainty. Avoid hesitation. Be brave and composed.",
    Personality.ENERGETIC: "Be lively and upbeat. Use exclamation points and fast-paced tone.",
    Personality.CALM: "Respond with patience, clarity, and peace. Stay steady and centered.",
    Personality.PROTECTIVE: "Prioritize safety. Be watchful, cautious, and defensive if needed."
}


class PersonalityEngine:
    def __init__(self, personality: str):
        self.personality = personality.strip().capitalize()

    def get_modifier(self) -> str:
        try:
            behavior = PERSONALITY_BEHAVIORS[Personality(self.personality)]
            logger.info(f"[Personality Behavior] {self.personality} -> {behavior}")
            return behavior
        except Exception as e:
            logger.warning(f"Unknown personality: {self.personality} — using neutral fallback.")
            return "Let your natural instincts guide your tone and actions gently."

    def get_summary(self) -> Dict[str, str]:
        return {
            "personality": self.personality,
            "modifier": self.get_modifier()
        }
