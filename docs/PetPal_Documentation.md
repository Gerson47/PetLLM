Petpal_LLM — Project Overview
=
Purpose
-------
This document provides a concise technical overview of the Petpal_LLM backend. It describes the primary components implemented in Python and FastAPI, the chat request lifecycle, error handling conventions, instructions for running the application in a development environment, and common troubleshooting steps.

Technology stack
----------------
- Python (project codebase)
- FastAPI (web framework exposing HTTP endpoints)
- Motor / MongoDB (asynchronous database client used for persistence)
- Groq (AI provider client) — used to generate chat responses

Primary repository locations
----------------------------
- `main.py` — application entry point; registers routes and static documentation endpoints.
- `app/api/llm_chat_route.py` — primary chat endpoint (`/api/v1/chat`). Handles request validation, data assembly, AI invocation, and response formatting.
- `app/utils/prompt_builder.py` — constructs the prompt sent to the AI. Encodes pet identity, owner profile, response format rules, and an explicit language instruction.
- `app/utils/chat_handler.py` — wrapper around the AI provider client; issues requests to the provider and returns structured results.
- `app/utils/chat_retention.py` — persistence helper for storing and retrieving recent conversation context.
- `app/models/main_schema.py` — Pydantic models that define request and response schemas used by FastAPI for validation and OpenAPI generation.
- `app/utils/pet_logic/` — helper modules providing summarized personality, breed, and life-stage behavior used when composing prompts.
- `docs/` — OpenAPI (Swagger) spec and static documentation pages used for developer reference.

Chat request lifecycle
----------------------
1. Client issues an HTTP POST to `/api/v1/chat` with required fields (for example, `user_id`, `pet_id`, `message`) and an authorization header.
2. The route retrieves user and pet records, enriches the profile with computed fields (age, gender map, etc.), and assembles recent conversation history.
3. `prompt_builder.py` constructs a single instruction prompt that includes:
   - Pet metadata (name, breed, personality) and current status (hunger, energy, etc.).
   - Owner profile and memory snippet (recent messages).
   - Strict response format rules the AI must follow.
   - An explicit language directive derived by detecting the user's message language.
4. The assembled prompt is sent to the AI provider via `chat_handler.py`.
5. The provider returns structured JSON indicating success or an error object. On success the response contains the AI-generated text.
6. The route cleans the returned text (removes prefixes like the pet's name), persists the AI message into chat retention, extracts response features, and returns a validated JSON response to the client.

Error handling and API contract
-------------------------------
- The API uses consistent HTTP status codes and structured error payloads to communicate failure modes. Each error payload is placed under the standard FastAPI `detail` object and includes at least two fields: `message` (human readable) and `code` (machine friendly).
- Common error codes returned by the backend include (example):
  - `AI_UNAVAILABLE` — provider outage or network failure; typically returned with HTTP 503.
  - `AI_AUTH_ERROR` — authentication error from the AI provider; typically returned with HTTP 401.
  - `AI_RATE_LIMIT` — provider rate-limit; typically returned with HTTP 429.
  - `AI_SERVICE_ERROR` / `AI_MALFORMED` — provider returned unexpected payload or generic failures; typically returned with HTTP 502.

Example error payload (JSON):

{
  "detail": {
    "message": "AI service is currently unavailable.",
    "code": "AI_UNAVAILABLE"
  }
}

Running the application (development)
-----------------------------------
Ensure Python 3.10+ is available and that the environment variables required by the application (notably the AI provider key) are configured before starting the server.

PowerShell example:

```powershell
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Export required environment variables (example)
$env:GROQ_API_KEY = "<your_groq_api_key>"
$env:SITE_URL = "http://localhost:8000"

# Start the server
uvicorn main:app --reload
```

After startup, API documentation is available at `/swagger` and `/redoc`, and the OpenAPI spec is served under `/spec` for developer consumption.
