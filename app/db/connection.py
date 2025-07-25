# app/db/connection.py

from motor.motor_asyncio import AsyncIOMotorClient
from decouple import config

MONGO_URI = config("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client.petpal_db

# Expose the collections directly
chats_collection = db.chats
user_profiles_collection = db.user_profiles 

def get_db():
    return db