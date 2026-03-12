# --- THE FIX: THIS MUST BE LINE 1 ---
import torch  

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.logger import log
from app.api.webrtc import router as webrtc_router

# Import your heavy ML services
from app.services.stt import STTService
from app.services.llm import LLMService
from app.services.tts import TTSService

from fastapi.staticfiles import StaticFiles

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 INITIALIZING AI PIPELINE...")

    # Load the models straight into VRAM
    stt = STTService()
    llm = LLMService()
    tts = TTSService()

    # Store them globally so your WebRTC router can grab them instantly
    app.state.models = {
        "stt": stt,
        "llm": llm,
        "tts": tts
    }

    log.info("✅ Pipeline Ready. Listening for UDP connections.")
    yield

    log.info("🛑 Shutting down AI models...")
    stt.is_listening = False # Safely kill the VAD thread

app = FastAPI(title="WebRTC Voice Agent", lifespan=lifespan)

# Mandatory for WebRTC browser connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the signaling endpoint
app.include_router(webrtc_router, prefix="/api/v1")

# --- THE FIX: SERVE THE FRONTEND ---
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")

@app.get("/health")
async def health_check():
    return {"status": "online", "system": "nominal"}