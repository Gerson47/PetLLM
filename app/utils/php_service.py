import httpx
import logging

API_BASE = "http://3.34.198.113/aipet/api/v1"

logger = logging.getLogger(__name__)

async def get_user_by_id(user_id: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{API_BASE}/users/profile", headers=headers)
            response.raise_for_status()
            return response.json().get("user", {})
        except httpx.HTTPStatusError as e:
            logger.error("User API returned %s: %s", e.response.status_code, e)
            raise
        except httpx.RequestError as e:
            logger.error("User API request error: %s", e)
            raise

async def get_pet_by_id(pet_id: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_BASE}/pets", headers=headers)
        response.raise_for_status()
        pets = response.json().get("pets", [])

    for pet in pets:
        if str(pet.get("pet_id")) == str(pet_id):
            return pet

    return None

async def get_pet_status_by_id(pet_id: str, token: str):
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{API_BASE}/pets/{pet_id}/status", headers=headers)

        if response.status_code == 404:
            print(f"⚠ No status found for pet {pet_id}. Returning default.")
            return {}  

        response.raise_for_status()
        return response.json().get("data", {})
