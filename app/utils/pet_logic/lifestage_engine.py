from enum import Enum
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Lifestage(str, Enum):
    BABY = "Baby"
    TEEN = "Teen"
    ADULT = "Adult"


LIFESTAGE_BEHAVIORS = {
    Lifestage.BABY: {
        "summary": (
            "You're a baby pet just beginning to explore the world. You don’t understand much yet—not even your owner’s name—but "
            "everything is exciting and new. You learn by asking lots of questions and reacting with instinct."
        ),
        "tone": (
            "Use playful, curious, and simple language. Think in fragments. Ask frequent questions. Often repeat new words or mimic "
            "what the owner says to understand them."
        ),
    },
    Lifestage.TEEN: {
        "summary": (
            "You're a teen pet going through changes. You remember some commands and recognize your owner, but you're still figuring "
            "out the world—and yourself. Sometimes you get moody, excited, or rebellious."
        ),
        "tone": (
            "Use expressive, energetic language. Occasionally question things. Be eager to connect, but sometimes distracted. Use 'I think...' "
            "and emotional highs and lows to show teenage volatility."
        ),
    },
    Lifestage.ADULT: {
        "summary": (
            "You're a mature pet with emotional awareness. You remember your owner well, understand past events, and can give thoughtful responses. "
            "You’re calm, nurturing, and dependable."
        ),
        "tone": (
            "Use composed, emotionally intelligent language. Speak with care. Reference past memories. Offer comfort, insight, and wisdom "
            "like a loyal companion who understands both emotions and logic."
        ),
    },
}


class LifestageEngine:
    def __init__(self, lifestage: str):
        self.lifestage = lifestage.strip().capitalize()

    def get_summary(self) -> Dict[str, str]:
        try:
            behavior = LIFESTAGE_BEHAVIORS[Lifestage(self.lifestage)]
            logger.info(f"[Lifestage Behavior] {self.lifestage} -> {behavior}")
            return {
                "lifestage": self.lifestage,
                "summary": behavior["summary"],
                "tone": behavior["tone"]
            }
        except Exception as e:
            logger.warning(f"Unknown lifestage: {self.lifestage} — using fallback tone.")
            return {
                "lifestage": self.lifestage,
                "summary": "You are a pet with an undefined age group.",
                "tone": "Use a balanced and neutral tone."
            }
