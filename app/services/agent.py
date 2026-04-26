import asyncio
import time
import torch
import json
from typing import AsyncGenerator
from app.core.config import settings
from .router import requires_vision 

class VoiceAgent:
    def __init__(self, stt, llm, tts, vision):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.vision = vision 

        self.has_gpu = torch.cuda.is_available()
        self.allow_interrupt = True   
        self.is_interrupted = False 
        self.is_speaking = False 
        
        self.tts_queue = asyncio.Queue()          
        self.audio_out_queue = asyncio.Queue()  

        mode = "Local AI (GPU)" if self.has_gpu else "Cloud AI (Groq)"
        print(f"\n>> [SYSTEM] Agent Initialized | Mode: {mode}")

    async def start_session(self, websocket):
        print(">> [SYSTEM] WebSocket Connected. Listening...\n")
        tts_worker = asyncio.create_task(self._tts_worker())
        playback_worker = asyncio.create_task(self._playback_worker(websocket))

        try:
            async for message in websocket.iter_bytes():
                
                # --- 1. STT SERVICE ---
                stt_start = time.perf_counter()
                user_text = await self.stt.transcribe_chunk(message, use_groq=not self.has_gpu)
                stt_time = (time.perf_counter() - stt_start) * 1000
                
                if user_text:
                    if self.is_speaking and self.allow_interrupt:
                        await self.interrupt_agent()
                    
                    self.is_interrupted = False
                    print(f"\n🎤 [STT] '{user_text}' ({stt_time:.2f}ms)")
                    await websocket.send_text(json.dumps({"type": "user", "text": user_text}))
                    
                    clean_text = user_text.strip().replace("?", "").replace("!", "").replace(".", "")
                    word_count = len(clean_text.split())

                    # --- 2. ROUTER SERVICE ---
                    router_start = time.perf_counter()
                    if word_count >= 3:
                        intent = await requires_vision(clean_text)
                    else:
                        intent = "NO"
                    router_time = (time.perf_counter() - router_start) * 1000
                    print(f"🧠 [ROUTER] Intent: {intent} ({router_time:.2f}ms)")

                    # ----------------------------------------------------
                    # --- PATH A: VISUAL LLM (Graphs, UI, Images) ---
                    # ----------------------------------------------------
                    if intent == "VISION_SINGLE":
                        vlm_start = time.perf_counter()
                        vision_text = await self.vision.describe_screen(custom_prompt=user_text)
                        vlm_time = (time.perf_counter() - vlm_start) * 1000
                        
                        print(f"👁️ [VLM] '{vision_text}' ({vlm_time:.2f}ms)")
                        await websocket.send_text(json.dumps({"type": "ai", "text": vision_text}))
                        
                        if not self.is_interrupted:
                            await self.tts_queue.put(vision_text)
                            
                        self.llm.chat_history.append({"role": "user", "content": user_text})
                        self.llm.chat_history.append({"role": "assistant", "content": f"[Looking at Screen]: {vision_text}"})

                    # ----------------------------------------------------
                    # --- PATH B: OCR + TEXT LLM (Code, Documents) ---
                    # ----------------------------------------------------
                    else:
                        llm_input_prompt = user_text 

                        if intent == "READ_TEXT":
                            ocr_start = time.perf_counter()
                            screen_text = await self.vision.extract_text()
                            ocr_time = (time.perf_counter() - ocr_start) * 1000
                            print(f"📝 [OCR] Extracted {len(screen_text)} characters ({ocr_time:.2f}ms)")
                            
                            llm_input_prompt = (
                                f"You are analyzing the user's screen. Answer their request using the extracted code/text below.\n\n"
                                f"--- EXTRACTED SCREEN TEXT ---\n{screen_text}\n---------------------------\n\n"
                                f"User Request: {user_text}"
                            )
                            
                            self.llm.chat_history.append({"role": "user", "content": user_text})
                            self.llm.chat_history.append({"role": "assistant", "content": "[System: Scanned screen text via OCR]"})

                        # Execute normal streaming LLM (Chat or OCR Fallthrough)
                        word_stream = self.llm.generate_response_stream(
                            llm_input_prompt, 
                            use_groq=not self.has_gpu
                        )
                        
                        is_chart_mode = False
                        chart_buffer = ""
                        llm_full_response = ""
                        llm_start = time.perf_counter()
                        first_chunk_received = False
                        
                        async for sentence in self._buffer_sentences(word_stream):
                            if self.is_interrupted:
                                break
                                
                            if not first_chunk_received:
                                ttft = (time.perf_counter() - llm_start) * 1000
                                first_chunk_received = True
                                
                            if "<CHART>" in sentence:
                                parts = sentence.split("<CHART>")
                                pre_text = parts[0].strip()
                                
                                if pre_text:
                                    llm_full_response += pre_text + " "
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
                                        print(f"📊 [CHART] Generated: '{chart_data.get('title', 'Untitled')}'")
                                        
                                        await websocket.send_text(json.dumps({
                                            "type": "chart",
                                            "chartData": chart_data
                                        }))
                                    except Exception as e:
                                        print(f"❌ [CHART ERROR]: {e}")
                                    
                                    is_chart_mode = False
                                    chart_buffer = ""
                            else:
                                llm_full_response += sentence + " "
                                await websocket.send_text(json.dumps({"type": "ai", "text": sentence}))
                                await self.tts_queue.put(sentence)
                                
                        if not self.is_interrupted and llm_full_response.strip():
                            print(f"💬 [LLM] '{llm_full_response.strip()}' (TTFT: {ttft:.2f}ms)")

        except Exception as e:
            print(f">> [SYSTEM] Connection error: {e}")
        finally:
            print(">> [SYSTEM] WebSocket Closed.")
            tts_worker.cancel()
            playback_worker.cancel()

    async def _buffer_sentences(self, word_stream) -> AsyncGenerator[str, None]:
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
        while True:
            sentence = await self.tts_queue.get()
            try:
                if self.is_interrupted:
                    continue
                
                tts_start = time.perf_counter()
                first_audio_yielded = False
                
                async for audio_chunk in self.tts.generate_audio_stream(
                    sentence, 
                    use_piper=not self.has_gpu
                ):
                    if self.is_interrupted:
                        break
                        
                    if not first_audio_yielded:
                        ttfa = (time.perf_counter() - tts_start) * 1000
                        print(f"🔊 [TTS] Audio ready (TTFA: {ttfa:.2f}ms)")
                        first_audio_yielded = True
                        
                    await self.audio_out_queue.put(audio_chunk)
            finally:
                self.tts_queue.task_done()

    async def _playback_worker(self, websocket):
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
        print("\n🛑 [BARGE-IN] Interrupted.")
        self.is_interrupted = True
        
        while not self.tts_queue.empty():
            self.tts_queue.get_nowait()
            self.tts_queue.task_done()
            
        while not self.audio_out_queue.empty():
            self.audio_out_queue.get_nowait()
            self.audio_out_queue.task_done()
        
        self.is_speaking = False