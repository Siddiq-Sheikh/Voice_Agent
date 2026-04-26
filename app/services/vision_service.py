import os
import base64
import io
import time
import asyncio
from PIL import ImageGrab 
from rapidocr_onnxruntime import RapidOCR
from groq import AsyncGroq 

class VisionService:
    def __init__(self, vision_model="llama-3.2-11b-vision-preview"):
        print(f">> [VISION] Initializing Hybrid Vision Engine...")
        print(f"   - VLM Engine: Groq ({vision_model})")
        print(f"   - OCR Engine: Local RapidOCR")
        
        self.model = vision_model
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        self.ocr = RapidOCR()

    def _capture_and_encode(self):
        """Captures and base64 encodes for the VLM."""
        screenshot = ImageGrab.grab()
        screenshot.thumbnail((1280, 720))
        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _capture_and_extract(self):
        """Captures and extracts raw text for the OCR."""
        screenshot = ImageGrab.grab()
        result, _ = self.ocr(screenshot)
        if not result:
            return "No readable text or code found on the screen."
        extracted_text = "\n".join([line[1] for line in result])
        return extracted_text[:15000] # Cap context window

    async def describe_screen(self, custom_prompt=None):
        """Uses the Visual LLM to look at graphs, UI, and general screen state."""
        base64_image = await asyncio.to_thread(self._capture_and_encode)
        user_question = custom_prompt if custom_prompt else "Describe what is on this screen."
        
        final_prompt = (
            "You are the visual cortex of a human-like voice assistant. "
            f"USER'S REQUEST: '{user_question}'\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer based ONLY on the provided screen image.\n"
            "2. Keep it natural, casual, and limited to 1 or 2 short sentences.\n"
            "3. DO NOT use markdown formatting. No asterisks, bolding, or lists."
        )

        try:
            content_payload = [
                {"type": "text", "text": final_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]

            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.2,
                max_completion_tokens=100 
            )
            
            description = response.choices[0].message.content.strip().replace("*", "")
            return description
                
        except Exception as e:
            print(f">> [VLM Error]: {e}")
            return "I lost my connection to the visual cortex."

    async def extract_text(self):
        """Uses local OCR to rip dense text/code from the screen."""
        text = await asyncio.to_thread(self._capture_and_extract)
        return text