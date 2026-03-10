from fastapi import APIRouter
from app.api.v1.endpoints import websocket

api_router = APIRouter()

# You can add more endpoints here later (e.g., REST endpoints for settings)
api_router.include_router(websocket.router, tags=["Voice Streaming"])