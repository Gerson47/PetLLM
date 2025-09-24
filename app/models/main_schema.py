from pydantic import BaseModel
from fastapi import Form, Request, UploadFile, File
from typing import Literal, List

# User Profile Form
def get_user_profile_form(
    user_id: str = Form(...),
    age: int = Form(...),
):
    return {
        "user_id": user_id,
        "age": age,
    }

# Pet Profile Form 
def get_pet_profile_form(
    user_id: str = Form(...),
    pet_name: str = Form(...),
    pet_type: Literal["cat", "dog"] = Form(...),
    breed: Literal[
        "Golden Retriever", "Shiba Inu", "Husky",  # Dogs
        "Persian", "Siamese", "Bengal"             # Cats
    ] = Form(...),
    pet_age_group: Literal["baby", "teen", "adult"] = Form(...)
):
    return {
        "user_id": user_id,
        "pet_name": pet_name,
        "pet_type": pet_type,
        "breed": breed,
        "age_group": pet_age_group,
        "education_level": 1,
        "known_commands": [],
        "knowledge_base": {
            "owner_name": user_id
        }
    }
# Chat Request 
def get_chat_form(
    user_id: str = Form(...),
    pet_id: str = Form(...),
    message: str = Form(...),
):
    return {
        "user_id": user_id,
        "pet_id": pet_id,
        "message": message,
    }

class ChatFeatures(BaseModel):
    motions: List[str]
    sounds: List[str]
    emotions: List[str]

class ChatResponse(BaseModel):
    response: str
    features: ChatFeatures