from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # This tells Pydantic to look for the .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Instantiate it once to be imported across the app
settings = Settings()