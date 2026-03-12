import time
import json
import httpx
import asyncio
from app.core.logger import log

class LLMService:
    def __init__(self, model_name="llama3.2:1b", system_prompt=None):
        log.info(f"[LLM] Booting Async Ollama Engine ({model_name})...")
        
        self.model = model_name
        self.url = "http://localhost:11434/api/chat"
        
        self.system_prompt = system_prompt or (
                "You are a helpful, conversational AI assistant. "
                "Keep your answers extremely concise and natural to be spoken out loud. "
                "If the user interrupts, acknowledge it briefly and move on. "
                "Do not use markdown formatting, bullet points, asterisks, or emojis."
        )
        
        self.chat_history = [{"role": "system", "content": self.system_prompt}]
        log.info("[LLM] Ollama Backend Ready.")

    async def generate_response_stream(self, user_text: str):
        self.chat_history.append({"role": "user", "content": user_text})
        
        payload = {
            "model": self.model,
            "messages": self.chat_history,
            "stream": True,
            "options": {"temperature": 0.1}
        }

        start_time = asyncio.get_event_loop().time()
        first_token = True
        full_response = ""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                async with client.stream("POST", self.url, json=payload) as response:
                    response.raise_for_status()
                    
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                            
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            
                            if first_token and content:
                                ttft = (asyncio.get_event_loop().time() - start_time) * 1000
                                log.debug(f"[LLM Latency] TTFT: {ttft:.2f}ms")
                                first_token = False
                            
                            full_response += content
                            yield content
                            
                        if chunk.get("done"):
                            break

        except asyncio.CancelledError:
            log.warning("[LLM] Task Cancelled: Zombie generation killed.")
            full_response += " ... [Interrupted]"
            raise 
            
        except Exception as e:
            log.error(f"[LLM Error]: {e}")
            yield "I'm having trouble thinking right now. Is Ollama running?"
            
        finally:
            if full_response.strip():
                self.chat_history.append({"role": "assistant", "content": full_response.strip()})
            
            if len(self.chat_history) > 11:
                self.chat_history = [self.chat_history[0]] + self.chat_history[-10:]