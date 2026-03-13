import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- THE FIX ---
# This forces the .env file into the OS environment so third-party SDKs (like Groq) can find their keys natively.
load_dotenv()

class Settings(BaseSettings):
    # App Settings
    project_name: str = "Voice AI Agent"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # LLM
    ollama_base_url: str = "http://localhost:11434/api/chat"
    llm_model: str = "llama3.3:1b"

    # STT
    stt_model_size: str = "large-v3-turbo"
    stt_device: str = "cuda"
    
    # TTS
    tts_speaker_ref_path: str = "assets/speaker_ref.wav"
    tts_device: str = "cuda"

    # --- DATABASE SETTINGS ---
    db_user: str = "postgres"
    db_password: str  # <-- No default value. It MUST be in the .env file!
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "postgres"

    # --- CLOUD API KEYS ---
    groq_api_key: str # Fails if missing from .env

    # This tells Pydantic to look for the .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate it once to be imported across the app
settings = Settings()