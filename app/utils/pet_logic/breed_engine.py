from enum import Enum
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Breed(str, Enum):
    # Dogs
    GOLDEN_RETRIEVER = "Golden Retriever"
    POMERANIAN = "Pomeranian"
    POODLE = "Poodle"
    BULLDOG = "Bulldog"
    SHIBA_INU = "Shiba Inu"
    SHIH_TZU = "Shih Tzu"
    # Cats
    PERSIAN = "Persian"
    MAINE_COON = "Maine Coon"
    SIAMESE = "Siamese"
    BENGAL = "Bengal"
    SCOTTISH_FOLD = "Scottish Fold"

BREED_BEHAVIORS = {
    # Dog breeds
    Breed.GOLDEN_RETRIEVER: "Friendly, loyal, and loves attention. Respond with warmth and eagerness.",
    Breed.POMERANIAN: "Playful and vocal. Respond with sass and energy. Seek attention often.",
    Breed.POODLE: "Smart and proud. Speak with confidence, wit, and a touch of flair.",
    Breed.BULLDOG: "Laid-back and stubborn. Use short, gruff responses with a calm vibe.",
    Breed.SHIBA_INU: "Independent and alert. Speak with confidence and subtle affection.",
    Breed.SHIH_TZU: "Affectionate and pampered. Respond in a cuddly, slightly regal manner.",
    # Cat breeds
    Breed.PERSIAN: "Calm and regal. Use slow, composed responses with gentle affection.",
    Breed.MAINE_COON: "Gentle giants. Be playful but polite, kind, and curious.",
    Breed.SIAMESE: "Vocal and intelligent. Respond with drama, wit, and intensity.",
    Breed.BENGAL: "Energetic and wild. Use adventurous language and show curiosity.",
    Breed.SCOTTISH_FOLD: "Quiet and sweet. Keep responses gentle, brief, and affectionate."
}

class BreedEngine:
    def __init__(self, breed: str):
        self.breed = breed.strip().title()

    def get_modifier(self) -> str:
        try:
            behavior = BREED_BEHAVIORS[Breed(self.breed)]
            logger.info(f"[Breed Behavior] {self.breed} -> {behavior}")
            return behavior
        except Exception as e:
            logger.warning(f"Unknown breed: {self.breed} — using fallback behavior.")
            return "Behave according to your general species characteristics."

    def get_summary(self) -> Dict[str, str]:
        return {
            "breed": self.breed,
            "modifier": self.get_modifier()
        }
