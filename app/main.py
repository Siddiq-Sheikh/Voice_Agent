import torch

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Import Services
from fastapi.staticfiles import StaticFiles
from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService

# Import Router
from app.api.v1.router import api_router

# Global dictionary to hold our heavy AI models in VRAM
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*55 + "\n 🚀 INITIALIZING AI PIPELINE \n" + "="*55)
    
    # Load models
    models["tts"] = TTSService()
    models["stt"] = STTService()
    models["llm"] = LLMService()
    
    # --- THE MISSING LINK ---
    # Attach the loaded dictionary to FastAPI's internal state
    # so the WebSocket router can actually find them!
    app.state.models = models
    
    yield
    
    print("\n🛑 Shutting down AI engines...")
    models.clear()

app = FastAPI(lifespan=lifespan, title="Voice AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. REGISTER THE API ROUTER FIRST
# This ensures WebSocket traffic gets routed to your Python code safely.
app.include_router(api_router, prefix="/api/v1")

# 2. MOUNT STATIC FILES LAST
# This acts as a catch-all for your frontend HTML, CSS, and JS.
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")