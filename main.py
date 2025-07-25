from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from app.api.llm_chat_route_lstm import router as chat_router


# FastAPI App Initialization
app = FastAPI(
    title="PetPal",
    description="Virtual Pet Simulator.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])

# Health Endpoints
@app.get("/")
async def root():
    return {"message": "Server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8084))
    uvicorn.run(app, host="0.0.0.0", port=port)

