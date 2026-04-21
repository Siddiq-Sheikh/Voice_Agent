import os
import base64
import io
import asyncio
from PIL import ImageGrab # pip install pillow
from groq import AsyncGroq # pip install groq
from app.core.config import settings

class VisionService:
    def __init__(self, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
        print(f">> [VISION] Initializing Groq Screen-Aware Agent (Model: {model_name})...")
        self.model = model_name
        
        # Initialize the async Groq client. 
        # It will automatically look for the GROQ_API_KEY in your environment variables.
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

    def _capture_and_encode(self):
        """Synchronous function to capture and encode the screen."""
        # Grab the primary screen
        screenshot = ImageGrab.grab()
        
        # Downscale to keep latency low and fit context windows easily
        screenshot.thumbnail((1280, 720))

        # Convert to Base64 string
        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def describe_screen(self, custom_prompt=None):
        """Asynchronously captures the screen and gets a description from the Groq VLM."""
        print(">> [VISION] Snap! Capturing screen...")
        
        # Run the blocking ImageGrab in a separate thread to protect audio loop
        base64_image = await asyncio.to_thread(self._capture_and_encode)

        print(">> [VISION] Analyzing image with Groq...")
        
        prompt = custom_prompt or (
            "You are a helpful voice assistant's eyes. Describe what is on this screen "
            "in 1 or 2 short, conversational sentences. Do not use markdown."
        )

        try:
            # Send the multimodal request using Groq's async client
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    # Cloud APIs require this data URL format prefix
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.2,
                max_completion_tokens=100 # Keep the response short for TTS
            )
            
            description = response.choices[0].message.content.strip()
            print(f">> [VISION] Output: {description}")
            return description
                
        except Exception as e:
            print(f">> [VISION Error]: {e}")
            return "I'm sorry, I seem to have lost my connection to the cloud and can't see the screen."

# --- Quick Test Block ---
if __name__ == "__main__":
    # Make sure your API key is set before running this!
    # For Windows: setx GROQ_API_KEY "your_key_here"
    # For Mac/Linux: export GROQ_API_KEY="your_key_here"
    
    async def test_vision():
        vision = VisionService()
        result = await vision.describe_screen()
        print("\nFinal Result:", result)

    asyncio.run(test_vision())