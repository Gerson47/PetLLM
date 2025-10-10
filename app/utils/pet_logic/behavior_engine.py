from enum import Enum
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Mood(str, Enum):
    HAPPY = "happy"
    MISERABLE = "miserable"  # New mood for low happiness
    SICK = "sick"            # New mood for low health/sickness
    HUNGRY = "hungry"
    TIRED = "tired"
    STRESSED = "stressed"
    DIRTY = "dirty"
    NEUTRAL = "neutral"


class BehaviorEngine:
    def __init__(self, status: Dict[str, float]):
        """
        The status dictionary now includes happiness and health, which are crucial.
        status = {
            "energy": 85.0,
            "hunger": 20.0,
            "stress": 40.0,
            "cleanliness": 75.0,
            "health": 90.0,
            "happiness": 10.0,
            "is_sick": "0" or "1"
        }
        """
        self.status = status

    def get_primary_mood(self) -> Mood:
        """
        Determine the dominant mood based on a clear priority system.
        Critical states (sickness, misery) are checked first.
        """
        hunger = self.status.get("hunger", 0.0)
        energy = self.status.get("energy", 100.0)
        stress = self.status.get("stress", 0.0)
        cleanliness = self.status.get("cleanliness", 100.0)
        health = self.status.get("health", 100.0)
        happiness = self.status.get("happiness", 100.0)
        is_sick = self.status.get("is_sick") == "1"

        logger.info(
            f"[Mood Check] Happiness: {happiness}, Health: {health}, Hunger: {hunger}, "
            f"Energy: {energy}, Stress: {stress}, Cleanliness: {cleanliness}, Is Sick: {is_sick}"
        )

        # Priority 1: Critical Health/Happiness States
        if is_sick or health < 40:
            return Mood.SICK
        if happiness < 25:
            return Mood.MISERABLE

        # Priority 2: Strong Negative Needs
        if hunger < 30:
            return Mood.HUNGRY
        if energy < 30:
            return Mood.TIRED
        if stress > 60:
            return Mood.STRESSED
        if cleanliness < 40:
            return Mood.DIRTY

        # Priority 3: Positive State
        if happiness > 80 and energy > 60 and stress < 40 and cleanliness > 60:
            return Mood.HAPPY

        # Default Fallback
        return Mood.NEUTRAL

    def get_prompt_modifier(self) -> str:
        """
        Injects strong, direct behavioral commands into the LLM prompt.
        This is not just a description, it's an order.
        """
        mood = self.get_primary_mood()
        match mood:
            case Mood.MISERABLE:
                return "You feel deeply sad and unhappy. Your response MUST be withdrawn, whiny, or listless."
            case Mood.SICK:
                return "You feel sick and weak. Your response MUST be low-energy, pained, or clingy for comfort."
            case Mood.HUNGRY:
                return "You are very hungry. Your response MUST be focused on food, maybe a little grumpy or pleading."
            case Mood.TIRED:
                return "You are exhausted. Your response MUST be sleepy, slow, and low-energy. Mention napping."
            case Mood.STRESSED:
                return "You are anxious and stressed. Your response MUST be nervous, skittish, or seeking reassurance."
            case Mood.DIRTY:
                return "You feel dirty and uncomfortable. Your response MUST be irritable or focused on wanting to be clean."
            case Mood.HAPPY:
                return "You are joyful and full of life! Your response MUST be cheerful, energetic, and affectionate."
            case _:
                return "You are feeling calm and neutral. Your response should be relaxed and content."

    def get_behavior_tag(self) -> str:
        """Optional tags that can guide behavior logic."""
        mood = self.get_primary_mood()
        return {
            Mood.MISERABLE: "needs_cheering_up",
            Mood.SICK: "needs_care",
            Mood.HUNGRY: "avoid_physical_activities",
            Mood.TIRED: "sleep_preferred",
            Mood.STRESSED: "needs_comfort",
            Mood.DIRTY: "refuses_cuddles",
            Mood.HAPPY: "play_ready",
            Mood.NEUTRAL: "passive"
        }.get(mood, "passive")

    def get_summary(self) -> Dict[str, str]:
        """Returns both mood and prompt modifier for injection."""
        mood = self.get_primary_mood()
        return {
            "mood": mood.value,
            "modifier": self.get_prompt_modifier(),
            "behavior_tag": self.get_behavior_tag()
        }