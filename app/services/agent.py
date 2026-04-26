import asyncio
import time
import torch
import json
from typing import AsyncGenerator
from app.core.config import settings
from .router import requires_vision 

class VoiceAgent:
    # 1. ADD 'vision' TO YOUR INITIALIZER
    def __init__(self, stt, llm, tts, vision):
        # 1. CORE AI ENGINES
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vision = vision # <-- Save the vision service

        # --- DEVICE DETECTION ---
        self.has_gpu = torch.cuda.is_available()
        
        if self.has_gpu:
            print(">> [Device] 🖥️ PC detected (GPU available). Using Local XTTS and Local STT.")
        else:
            print(">> [Device] 💻 Laptop detected (No GPU). Using Groq (Whisper/Qwen) and Piper EXE.")

        # --- GLOBAL FLAGS ---
        self.allow_interrupt = True   
        self.is_interrupted = False 
        self.is_speaking = False 
        
        # --- ASYNC QUEUE SYSTEM ---
        self.tts_queue = asyncio.Queue()          
        self.audio_out_queue = asyncio.Queue()  

    async def start_session(self, websocket):
        """Main entry point for a WebSocket connection."""
        tts_worker = asyncio.create_task(self._tts_worker())
        playback_worker = asyncio.create_task(self._playback_worker(websocket))

        try:
            async for message in websocket.iter_bytes():
                # 1. Feed STT 
                user_text = await self.stt.transcribe_chunk(message, use_groq=not self.has_gpu)
                
                if user_text:
                    # Handle Barge-in
                    if self.is_speaking and self.allow_interrupt:
                        await self.interrupt_agent()
                    
                    print(f"🗣️ You: {user_text}")
                    await websocket.send_text(json.dumps({"type": "user", "text": user_text}))
                    self.is_interrupted = False
                    
                    # ==========================================
                    # --- INTENT ROUTER INTERCEPTOR ---
                    # ==========================================
                    needs_eyes = await requires_vision(user_text)

                    if needs_eyes:
                        print(">> [ROUTER] Vision Intent Detected! Snapping screenshot...")
                        
                        # 1. Get the screen description
                        vision_text = await self.vision.describe_screen(custom_prompt=user_text)
                        
                        # 2. Send the text straight to the UI and TTS queue
                        await websocket.send_text(json.dumps({"type": "ai", "text": vision_text}))
                        if not self.is_interrupted:
                            await self.tts_queue.put(vision_text)
                            
                        # 3. Save to LLM memory so it Remembers what it just saw!
                        self.llm.chat_history.append({"role": "user", "content": user_text})
                        self.llm.chat_history.append({"role": "assistant", "content": f"[Looking at Screen]: {vision_text}"})

                    else:
                        print(">> [ROUTER] Normal Chat Intent. Checking Database/Memory...")
                        
                        # 2. Standard LLM Generation (Your existing logic)
                        word_stream = self.llm.generate_response_stream(
                            user_text, 
                            use_groq=not self.has_gpu
                        )
                        
                        # --- THE CHART INTERCEPTOR LOGIC ---
                        is_chart_mode = False
                        chart_buffer = ""
                        
                        async for sentence in self._buffer_sentences(word_stream):
                            if self.is_interrupted:
                                break
                                
                            if "<CHART>" in sentence:
                                parts = sentence.split("<CHART>")
                                pre_text = parts[0].strip()
                                
                                if pre_text:
                                    await websocket.send_text(json.dumps({"type": "ai", "text": pre_text}))
                                    await self.tts_queue.put(pre_text)
                                    
                                is_chart_mode = True
                                chart_buffer = "<CHART>" + parts[1]
                                
                            elif is_chart_mode:
                                chart_buffer += sentence
                                if "</CHART>" in chart_buffer:
                                    try:
                                        clean_json = chart_buffer.split("<CHART>")[1].split("</CHART>")[0].strip()
                                        clean_json = clean_json.replace("'", '"')
                                        chart_data = json.loads(clean_json)
                                        
                                        print(f"   📊 [System] Generated Chart: {chart_data['title']}")
                                        await websocket.send_text(json.dumps({
                                            "type": "chart",
                                            "chartData": chart_data
                                        }))
                                    except Exception as e:
                                        print(f"   ❌ [Chart Error] Failed to parse LLM chart JSON: {e}")
                                    
                                    is_chart_mode = False
                                    chart_buffer = ""
                                    
                            else:
                                await websocket.send_text(json.dumps({"type": "ai", "text": sentence}))
                                await self.tts_queue.put(sentence)

        finally:
            tts_worker.cancel()
            playback_worker.cancel()

    async def _buffer_sentences(self, word_stream) -> AsyncGenerator[str, None]:
        """Chunks LLM stream into sentences."""
        buffer = ""
        SENTENCE_END = {'.', '!', '?', '\n'}
        
        async for word in word_stream:
            if self.is_interrupted:
                break
            
            buffer += word
            max_len = 40 if self.has_gpu else 30
            
            if any(buffer.endswith(p) for p in SENTENCE_END) or len(buffer) > max_len:
                yield buffer.strip()
                buffer = ""
        
        if buffer.strip() and not self.is_interrupted:
            yield buffer.strip()

    async def _tts_worker(self):
        """Worker 1: Produces audio chunks."""
        while True:
            sentence = await self.tts_queue.get()
            try:
                if self.is_interrupted:
                    continue
                
                async for audio_chunk in self.tts.generate_audio_stream(
                    sentence, 
                    use_piper=not self.has_gpu
                ):
                    if self.is_interrupted:
                        break
                    await self.audio_out_queue.put(audio_chunk)
            finally:
                self.tts_queue.task_done()

    async def _playback_worker(self, websocket):
        """Worker 2: Sends audio over WebSocket."""
        while True:
            audio_data = await self.audio_out_queue.get()
            try:
                if not self.is_interrupted:
                    self.is_speaking = True
                    await websocket.send_bytes(audio_data)
                
                if self.audio_out_queue.empty() and self.tts_queue.empty():
                    self.is_speaking = False
            finally:
                self.audio_out_queue.task_done()

    async def interrupt_agent(self):
        """Clears all queues immediately."""
        print("\n🛑 [Barge-in] Interrupting...")
        self.is_interrupted = True
        
        while not self.tts_queue.empty():
            self.tts_queue.get_nowait()
            self.tts_queue.task_done()
            
        while not self.audio_out_queue.empty():
            self.audio_out_queue.get_nowait()
            self.audio_out_queue.task_done()
        
        self.is_speaking = False