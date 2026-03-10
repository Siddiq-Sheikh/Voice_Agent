from pydantic import BaseModel
from typing import Optional, Any

class AgentEvent(BaseModel):
    """Defines JSON control messages sent over the WebSocket."""
    type: str  # e.g., "status", "transcript", "error"
    content: str
    metadata: Optional[dict[str, Any]] = None