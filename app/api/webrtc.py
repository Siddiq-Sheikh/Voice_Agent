from fastapi import APIRouter, Request
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from app.services.agent import WebRTCAgent
from app.core.logger import log

router = APIRouter()

# Global set to keep track of active connections so Python doesn't garbage collect them
pcs = set()

class WebRTCOffer(BaseModel):
    sdp: str
    type: str

@router.post("/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer, request: Request):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log.info(f"🔗 [WebRTC] Connection State: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            pcs.discard(pc)

    # Pull the pre-loaded ML models from the FastAPI lifespan state
    models = request.app.state.models
    
    # Initialize the brain
    agent = WebRTCAgent(pc, models["stt"], models["llm"], models["tts"])
    
    # Negotiate the UDP connection
    client_offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    await pc.setRemoteDescription(client_offer)
    
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}