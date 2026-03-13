# import asyncio
# import time
# from typing import AsyncGenerator
# from app.core.config import settings

# class VoiceAgent:
#     def __init__(self, stt, llm, tts):
#         # 1. CORE AI ENGINES (Injected from lifespan)
#         self.stt = stt
#         self.llm = llm
#         self.tts = tts

#         # --- GLOBAL FLAGS ---
#         self.allow_interrupt = True   
#         self.is_interrupted = False 
#         self.is_speaking = False 
        
#         # --- ASYNC QUEUE SYSTEM ---
#         # Replacing Threaded Queues with Asyncio Queues for FastAPI compatibility
#         self.tts_queue = asyncio.Queue()          
#         self.audio_out_queue = asyncio.Queue()  

#     async def start_session(self, websocket):
#         """Main entry point for a WebSocket connection."""
#         # Start the background workers as tasks
#         tts_worker = asyncio.create_task(self._tts_worker())
#         playback_worker = asyncio.create_task(self._playback_worker(websocket))

#         try:
#             # Main Loop: Listen for incoming audio bytes from WebSocket
#             async for message in websocket.iter_bytes():
#                 # 1. Handle "Barge-in" (Interrupt logic)
#                 if self.is_speaking and self.allow_interrupt:
#                     # Logic to detect voice in the incoming stream would go here
#                     # For now, we'll assume the STT service handles VAD
#                     await self.interrupt_agent()

#                 # 2. Feed STT
#                 user_text = await self.stt.transcribe_chunk(message)
                
#                 if user_text:
#                     print(f"🗣️ You: {user_text}")
#                     self.is_interrupted = False
                    
#                     # 3. Stream LLM -> Sentence Buffer -> TTS Queue
#                     word_stream = self.llm.generate_response_stream(user_text)
#                     async for sentence in self._buffer_sentences(word_stream):
#                         if self.is_interrupted:
#                             break
#                         await self.tts_queue.put(sentence)

#         finally:
#             tts_worker.cancel()
#             playback_worker.cancel()

#     async def _buffer_sentences(self, word_stream) -> AsyncGenerator[str, None]:
#         """Chunks LLM stream into clean sentences for XTTS prosody."""
#         buffer = ""
#         SENTENCE_END = {'.', '!', '?', '\n'}
        
#         async for word in word_stream:
#             if self.is_interrupted:
#                 break
            
#             buffer += word
#             if any(buffer.endswith(p) for p in SENTENCE_END) or len(buffer) > 40:
#                 yield buffer.strip()
#                 buffer = ""
        
#         if buffer.strip():
#             yield buffer.strip()

#     async def _tts_worker(self):
#         """Worker 1: Consumes sentences and produces audio chunks via XTTS."""
#         while True:
#             sentence = await self.tts_queue.get()
#             if self.is_interrupted:
#                 self.tts_queue.task_done()
#                 continue
            
#             # XTTS Streaming Inference
#             async for audio_chunk in self.tts.generate_audio_stream(sentence):
#                 if self.is_interrupted:
#                     break
#                 await self.audio_out_queue.put(audio_chunk)
            
#             self.tts_queue.task_done()

#     async def _playback_worker(self, websocket):
#         """Worker 2: Consumes audio chunks and sends them over WebSocket."""
#         while True:
#             audio_data = await self.audio_out_queue.get()
            
#             if not self.is_interrupted:
#                 self.is_speaking = True
#                 # Send raw PCM or encoded Opus bytes to client
#                 await websocket.send_bytes(audio_data)
            
#             self.audio_out_queue.task_done()
            
#             # Check if we are done speaking
#             if self.audio_out_queue.empty() and self.tts_queue.empty():
#                 self.is_speaking = False

#     async def interrupt_agent(self):
#         """Clears all queues and stops current playback immediately."""
#         print("\n🛑 [Barge-in Detected! Killing Audio]")
#         self.is_interrupted = True
#         self.is_speaking = False
        
#         # Drain queues
#         while not self.tts_queue.empty():
#             self.tts_queue.get_nowait()
#             self.tts_queue.task_done()
            
#         while not self.audio_out_queue.empty():
#             self.audio_out_queue.get_nowait()
#             self.audio_out_queue.task_done()

import asyncio
import time
import torch
from typing import AsyncGenerator
from app.core.config import settings

class VoiceAgent:
    def __init__(self, stt, llm, tts):
        # 1. CORE AI ENGINES
        self.stt = stt
        self.llm = llm
        self.tts = tts

        # --- DEVICE DETECTION ---
        # Checks if a CUDA-enabled GPU is present
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
                # On Laptop: Uses Groq Whisper-Large
                # On PC: Uses Local STT model
                user_text = await self.stt.transcribe_chunk(message, use_groq=not self.has_gpu)
                
                if user_text:
                    # Handle Barge-in
                    if self.is_speaking and self.allow_interrupt:
                        await self.interrupt_agent()
                    
                    print(f"🗣️ You: {user_text}")
                    self.is_interrupted = False
                    
                    # 2. LLM Generation
                    # On Laptop: Uses Groq Qwen-7B
                    # On PC: Uses Local LLM
                    word_stream = self.llm.generate_response_stream(
                        user_text, 
                        use_groq=not self.has_gpu
                    )
                    
                    async for sentence in self._buffer_sentences(word_stream):
                        if self.is_interrupted:
                            break
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
            # Laptop tweak: Piper handles shorter sentences faster than XTTS
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
                
                # On Laptop: Uses Piper .exe
                # On PC: Uses Local XTTS
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