import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import Config
from app.core.config import settings  # Ensure this path is correct for your project

# Import Services
from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.db_service import DatabaseService 

# Import Router
from app.api.v1.router import api_router

# Global dictionary to hold our heavy AI models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*55 + "\n 🚀 INITIALIZING AI PIPELINE \n" + "="*55)
    
    # 1. Boot up Database with config from .env
    db = DatabaseService(
        db_user=settings.db_user,
        db_pass=settings.db_password,
        db_name=settings.db_name,
        db_host=settings.db_host,
        db_port=settings.db_port
    )
    await db.connect()
    models["db"] = db
    
    # 2. Detect Hardware for AI Services
    # We pass the device settings or let the services auto-detect as we configured
    models["tts"] = TTSService()
    models["stt"] = STTService()
    
    # 3. Inject the live Database connection into the LLM
    models["llm"] = LLMService(db_service=db)
    
    app.state.models = models
    
    yield
    
    print("\n🛑 Shutting down AI engines and disconnecting...")
    
    if "db" in models and models["db"].pool:
        await models["db"].pool.close()
        print(">> [DB] 🔴 PostgreSQL Connection Closed.")
        
    models.clear()

app = FastAPI(lifespan=lifespan, title="Voice AI Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")