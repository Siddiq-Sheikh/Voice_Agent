import logging
import sys

def setup_logging():
    """Configures global logging for the FastAPI application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # You can uncomment the line below to save logs to a file
            # logging.FileHandler("agent.log")
        ]
    )
    
    # Silence noisy third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)