import os
import base64
import io
import time
import asyncio
import pyautogui # For the PageDown key
from PIL import ImageGrab 
from groq import AsyncGroq 

class VisionService:
    def __init__(self, model_name="meta-llama/llama-4-scout-17b-16e-instruct"):
        print(f">> [VISION] Initializing Full-Page Vision Agent (Model: {model_name})...")
        self.model = model_name
        self.groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

    def _capture_and_encode(self):
        """Synchronous function to capture and encode a single screen."""
        screenshot = ImageGrab.grab()
        screenshot.thumbnail((1280, 720))

        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    async def describe_screen(self, custom_prompt=None, full_page=False):
        """Asynchronously captures the screen. If full_page is True, it scrolls and snaps multiple times."""
        
        base64_images = []
        
        if full_page:
            print(">> [VISION] Full Page Mode: Scrolling and Snapping...")
            # Take 3 screenshots total (Top, Middle, Bottom)
            for i in range(3):
                img = await asyncio.to_thread(self._capture_and_encode)
                base64_images.append(img)
                
                # Press PageDown and wait a moment for the smooth scrolling animation to finish
                if i < 2: 
                    pyautogui.press('pagedown')
                    await asyncio.sleep(0.6) 
            
            # Scroll back to the top so you aren't left at the bottom of the page!
            pyautogui.press('pageup')
            pyautogui.press('pageup')
            
        else:
            print(">> [VISION] Snap! Capturing single screen...")
            img = await asyncio.to_thread(self._capture_and_encode)
            base64_images.append(img)

        print(">> [VISION] Analyzing image(s) with Groq...")
        
        # --- THE FIX: Combine the strict rules WITH the user's specific question ---
        user_question = custom_prompt if custom_prompt else "Describe what is on this screen."
        
        final_prompt = (
            "You are the visual cortex of a human-like voice assistant. "
            f"USER'S REQUEST: '{user_question}'\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer the user's request based ONLY on the provided screen images.\n"
            "2. If there are multiple images, treat them as a continuous scrolling page.\n"
            "3. Keep your answer casual, natural, and limited to 1 or 2 short sentences.\n"
            "4. CRITICAL: Speak exactly as a human would speak out loud. DO NOT use any markdown formatting whatsoever. Absolutely no asterisks, no bold text, and no lists."
        )

        try:
            api_start = time.perf_counter()
            
            # Build the payload. We start with the text prompt...
            content_payload = [{"type": "text", "text": final_prompt}]
            
            # ...and then dynamically attach as many images as we captured!
            for b64_img in base64_images:
                content_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })

            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content_payload}],
                temperature=0.2,
                max_completion_tokens=100 
            )
            
            api_latency = (time.perf_counter() - api_start) * 1000
            print(f"   [Vision Latency] Groq LLM API took: {api_latency:.2f}ms")
            
            description = response.choices[0].message.content.strip()
            description = description.replace("*", "")
            
            print(f">> [VISION] Output: {description}")
            return description
                
        except Exception as e:
            print(f">> [VISION Error]: {e}")
            return "I'm sorry, I seem to have lost my connection to the cloud and can't see the screen."