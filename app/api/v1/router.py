from fastapi import APIRouter, Request
from pydantic import BaseModel
from aiortc import RTCPeerConnection, RTCSessionDescription
from app.services.agent import WebRTCAgent

api_router = APIRouter()

# Global set to keep track of active peer connections so they aren't garbage collected
pcs = set()

class WebRTCOffer(BaseModel):
    sdp: str
    type: str

@api_router.post("/webrtc/offer")
async def webrtc_offer(offer: WebRTCOffer, request: Request):
    """The Signaling Endpoint: Establishes the UDP P2P connection."""
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"🔗 [WebRTC] Connection State: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            pcs.discard(pc)

    # Grab the loaded AI models from your lifespan state
    models = request.app.state.models
    
    # Initialize your new WebRTC Agent
    agent = WebRTCAgent(pc, models["stt"], models["llm"], models["tts"])
    
    # Process the frontend's offer
    client_offer = RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    await pc.setRemoteDescription(client_offer)
    
    # Generate and return the server's answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}