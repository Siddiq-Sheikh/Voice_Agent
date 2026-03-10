from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging

# Import the orchestrator (NO imports from main.py anymore!)
from app.services.agent import VoiceAgent

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    """
    Handles real-time bi-directional audio streaming between the client and the AI.
    """
    await websocket.accept()
    logger.info(">> [WebSocket] Client Connected.")

    # 1. Grab models dynamically from the FastAPI app state!
    # This completely bypasses the circular import problem.
    models = getattr(websocket.app.state, "models", {})
    
    stt_service = models.get("stt")
    llm_service = models.get("llm")
    tts_service = models.get("tts")

    # Guard clause: Ensure the server is actually ready
    if not all([stt_service, llm_service, tts_service]):
        logger.error(">> [WebSocket] Models are not fully loaded yet.")
        await websocket.close(code=1011, reason="AI engines warming up. Try again in a moment.")
        return

    # 2. Instantiate a fresh Agent Orchestrator for this specific session
    agent = VoiceAgent(stt=stt_service, llm=llm_service, tts=tts_service)

    try:
        # 3. Hand over control to the Agent
        await agent.start_session(websocket)

    except WebSocketDisconnect:
        logger.info(">> [WebSocket] Client Disconnected gracefully.")
        
    except Exception as e:
        logger.error(f">> [WebSocket] Unexpected session error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
            
    finally:
        logger.info(">> [WebSocket] Session closed and cleaned up.")