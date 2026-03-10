from fastapi import Request
from typing import Generator

def get_ai_models(request: Request):
    """
    Dependency to safely inject the AI models loaded in the lifespan state.
    This replaces importing the global dictionary directly.
    """
    # FastAPI attaches the lifespan state to the request app
    return getattr(request.app.state, "models", {})

def get_stt_service(request: Request):
    models = get_ai_models(request)
    return models.get("stt")

def get_llm_service(request: Request):
    models = get_ai_models(request)
    return models.get("llm")

def get_tts_service(request: Request):
    models = get_ai_models(request)
    return models.get("tts")