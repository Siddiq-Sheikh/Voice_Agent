# import os
# import queue
# import time
# import threading
# import collections
# import numpy as np
# import onnxruntime as ort
# from faster_whisper import WhisperModel
# import asyncio

# class STTService:
#     def __init__(self, model_size="large-v3-turbo", device="cuda"):
#         print(f">> [STT] Booting up Bare-Metal STT '{model_size}' engine...")
        
#         # --- THE FIX: Define a local project folder for the model ---
#         whisper_model_path = os.path.join(os.getcwd(), "whisper_model")
#         print(f">> [STT] Model directory set to: {whisper_model_path}")
        
#         # Your Faster-Whisper setup with the local download_root
#         self.model = WhisperModel(
#             model_size, 
#             device=device, 
#             compute_type="float16" if device == "cuda" else "int8",
#             download_root=whisper_model_path  # <-- This forces it to save in your project folder!
#         )
        
#         self.vad_model_path = "assets/silero_vad.onnx"
#         if not os.path.exists(self.vad_model_path):
#             print(">> ❌ [VAD ERROR] 'silero_vad.onnx' not found in assets folder!")
            
#         print(">> [VAD] Loading your official Silero VAD ONNX model...")
#         self.vad_session = ort.InferenceSession(self.vad_model_path, providers=['CPUExecutionProvider'])
        
#         # YOUR EXACT STATES
#         self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
#         self._vad_context = np.zeros((1, 64), dtype=np.float32)
        
#         self.incoming_ws_queue = queue.Queue() 
#         self.audio_queue = queue.Queue()       
        
#         # --- THE MISSING BUFFER FIX ---
#         self.vad_chunk_buffer = bytearray()
        
#         self.is_listening = True
#         self.is_paused = False 
        
#         self.audio_thread = threading.Thread(target=self._vad_worker, daemon=True)
#         self.audio_thread.start()

#     def _vad_worker(self):
#         """Your exact blood-and-tears VAD loop, processing the Web Stream."""
#         pre_speech_buffer = collections.deque(maxlen=15) 
#         recording_buffer = []
        
#         is_recording = False
#         silence_counter = 0
#         sr_tensor = np.array(16000, dtype=np.int64)
        
#         print(">> 🟢 STT Worker Thread Active. Processing WebSocket stream...")
        
#         while self.is_listening:
#             try:
#                 # Get the perfectly sized 1024-byte chunk
#                 raw_bytes = self.incoming_ws_queue.get(timeout=0.5)
#                 audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
#                 # --- YOUR EXACT VOLUME CALC ---
#                 current_volume = float(np.max(np.abs(audio_array)))
                
#                 # --- YOUR EXACT PARAMETERS ---
#                 chunk_with_context = np.concatenate([self._vad_context, audio_array.reshape(1, -1)], axis=1)
                
#                 ort_inputs = {
#                     "input": chunk_with_context, 
#                     "sr": sr_tensor, 
#                     "state": self._vad_state
#                 }
                
#                 out, self._vad_state = self.vad_session.run(None, ort_inputs)
#                 prob_val = float(np.max(out))
#                 self._vad_context = chunk_with_context[:, -64:]
                
#                 # --- YOUR EXACT DUAL-GATE (0.85 Prob / 0.4 Vol) ---
#                 if prob_val > 0.85 and current_volume > 0.2: 
#                     if not is_recording:
#                         is_recording = True
#                         recording_buffer = list(pre_speech_buffer)
                    
#                     recording_buffer.append(raw_bytes)
#                     silence_counter = 0
                    
#                 else: 
#                     if is_recording:
#                         recording_buffer.append(raw_bytes)
#                         silence_counter += 1
                        
#                         if silence_counter > 20: 
#                             is_recording = False
                            
#                             if not self.is_paused:
#                                 full_audio = b"".join(recording_buffer)
#                                 self.audio_queue.put(full_audio)
                                
#                             recording_buffer = []
#                             self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
#                             self._vad_context = np.zeros((1, 64), dtype=np.float32)
#                     else:
#                         pre_speech_buffer.append(raw_bytes)
                        
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 print(f">> [Audio Worker Error]: {e}")

#     async def transcribe_chunk(self, audio_bytes: bytes) -> str:
#         # --- THE FIX: SLICE BROWSER CHUNKS INTO PERFECT 1024-BYTE BITES ---
#         self.vad_chunk_buffer.extend(audio_bytes)
        
#         while len(self.vad_chunk_buffer) >= 1024:
#             exact_chunk = self.vad_chunk_buffer[:1024]
#             self.vad_chunk_buffer = self.vad_chunk_buffer[1024:]
#             self.incoming_ws_queue.put(exact_chunk)
            
#         if not self.audio_queue.empty():
#             full_audio_bytes = self.audio_queue.get()
#             text = await asyncio.to_thread(self._run_transcription, full_audio_bytes)
#             return text
#         return ""

#     def _run_transcription(self, raw_bytes: bytes) -> str:
#         start_total = time.perf_counter()
        
#         t_format_start = time.perf_counter()
#         audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
#         target_length = 16000 
#         if len(audio_array) < target_length:
#             padding = np.zeros(target_length - len(audio_array), dtype=np.float32)
#             audio_array = np.concatenate([audio_array, padding])
#         t_format = (time.perf_counter() - t_format_start) * 1000
        
#         t_transcribe_start = time.perf_counter()
#         segments, _ = self.model.transcribe(audio_array, beam_size=1, language="en")
#         t_transcribe = (time.perf_counter() - t_transcribe_start) * 1000
        
#         t_process_start = time.perf_counter()
#         text = " ".join([seg.text for seg in segments]).strip()
        
#         clean_text = text.lower().strip(" .!?*-_")
#         ghost_phrases = [
#             "you", "bye", "thanks", "subscribe", "thank you", 
#             "amara.org", "i'm going to go to bed now",
#             "blank_audio", "silence", "speaking in foreign language"
#         ]
        
#         is_system_tag = (
#             (clean_text.startswith("[") and clean_text.endswith("]")) or
#             (clean_text.startswith("(") and clean_text.endswith(")")) or
#             (clean_text.startswith("*") and clean_text.endswith("*")) or
#             (clean_text.startswith("-") and clean_text.endswith("-"))
#         )
        
#         if clean_text in ghost_phrases or is_system_tag:
#             text = ""
            
#         t_process = (time.perf_counter() - t_process_start) * 1000
#         total_time = (time.perf_counter() - start_total) * 1000
        
#         if text:
#             print(f"\n   [STT Latency] Format: {t_format:.2f}ms | Transcribe: {t_transcribe:.2f}ms | Filter: {t_process:.2f}ms | Total: {total_time:.2f}ms")
#             print(f">> [STT] Recognized: {text}")
#             return text
            
#         return ""

import os
import queue
import time
import threading
import collections
import numpy as np
import onnxruntime as ort
from faster_whisper import WhisperModel
import asyncio
import io
import wave
from groq import Groq  # Make sure to pip install groq

class STTService:
    def __init__(self, model_size="large-v3-turbo", device="cuda"):
        # Hardware Detection is handled by agent.py passing use_groq flag
        print(f">> [STT] Initializing Engine (Local Device: {device})...")
        
        # 1. Local Model Setup (Still needed for PC mode)
        whisper_model_path = os.path.join(os.getcwd(), "whisper_model")
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type="float16" if device == "cuda" else "int8",
            download_root=whisper_model_path
        )
        
        # 2. Groq Setup (For Laptop mode)
        # Ensure you have GROQ_API_KEY in your environment variables
        self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        # --- VAD SETUP (Remains local to handle "Silence detection" even on Laptop) ---
        self.vad_model_path = "assets/silero_vad.onnx"
        self.vad_session = ort.InferenceSession(self.vad_model_path, providers=['CPUExecutionProvider'])
        
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)
        
        self.incoming_ws_queue = queue.Queue() 
        self.audio_queue = queue.Queue()       
        self.vad_chunk_buffer = bytearray()
        
        self.is_listening = True
        self.is_paused = False 
        
        self.audio_thread = threading.Thread(target=self._vad_worker, daemon=True)
        self.audio_thread.start()

    def _vad_worker(self):
        """Processes the Web Stream and detects when a user stops talking."""
        pre_speech_buffer = collections.deque(maxlen=15) 
        recording_buffer = []
        is_recording = False
        silence_counter = 0
        sr_tensor = np.array(16000, dtype=np.int64)
        
        while self.is_listening:
            try:
                raw_bytes = self.incoming_ws_queue.get(timeout=0.5)
                audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                current_volume = float(np.max(np.abs(audio_array)))
                
                chunk_with_context = np.concatenate([self._vad_context, audio_array.reshape(1, -1)], axis=1)
                ort_inputs = {"input": chunk_with_context, "sr": sr_tensor, "state": self._vad_state}
                out, self._vad_state = self.vad_session.run(None, ort_inputs)
                prob_val = float(np.max(out))
                self._vad_context = chunk_with_context[:, -64:]
                
                if prob_val > 0.85 and current_volume > 0.2: 
                    if not is_recording:
                        is_recording = True
                        recording_buffer = list(pre_speech_buffer)
                    recording_buffer.append(raw_bytes)
                    silence_counter = 0
                else: 
                    if is_recording:
                        recording_buffer.append(raw_bytes)
                        silence_counter += 1
                        if silence_counter > 20: 
                            is_recording = False
                            if not self.is_paused:
                                self.audio_queue.put(b"".join(recording_buffer))
                            recording_buffer = []
                            self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
                            self._vad_context = np.zeros((1, 64), dtype=np.float32)
                    else:
                        pre_speech_buffer.append(raw_bytes)
            except queue.Empty: continue
            except Exception as e: print(f">> [Audio Worker Error]: {e}")

    async def transcribe_chunk(self, audio_bytes: bytes, use_groq: bool = False) -> str:
        """Called by agent.py. Slices audio and returns transcription."""
        self.vad_chunk_buffer.extend(audio_bytes)
        
        while len(self.vad_chunk_buffer) >= 1024:
            exact_chunk = self.vad_chunk_buffer[:1024]
            self.vad_chunk_buffer = self.vad_chunk_buffer[1024:]
            self.incoming_ws_queue.put(exact_chunk)
            
        if not self.audio_queue.empty():
            full_audio_bytes = self.audio_queue.get()
            
            # --- IF/ELSE FOR LAPTOP VS PC ---
            if use_groq:
                # Laptop Mode: Use Groq Cloud
                return await asyncio.to_thread(self._run_groq_transcription, full_audio_bytes)
            else:
                # PC Mode: Use Local Faster-Whisper
                return await asyncio.to_thread(self._run_transcription, full_audio_bytes)
        return ""

    def _run_groq_transcription(self, raw_bytes: bytes) -> str:
        """Sends audio to Groq Whisper-Large-v3."""
        try:
            start_time = time.perf_counter()
            # Wrap raw PCM in a WAV container for Groq
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(raw_bytes)
            buffer.seek(0)

            # Groq API Call
            transcription = self.groq_client.audio.transcriptions.create(
                file=("audio.wav", buffer.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
            
            latency = (time.perf_counter() - start_time) * 1000
            print(f"   [STT Latency] Groq Cloud: {latency:.2f}ms")
            return self._filter_ghosts(transcription)
        except Exception as e:
            print(f">> [Groq STT Error]: {e}")
            return ""

    def _run_transcription(self, raw_bytes: bytes) -> str:
        """Local Faster-Whisper (PC Mode)."""
        audio_array = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_array, beam_size=1, language="en")
        text = " ".join([seg.text for seg in segments]).strip()
        return self._filter_ghosts(text)

    def _filter_ghosts(self, text: str) -> str:
        """Helper to remove Hallucinations/System tags."""
        clean_text = text.lower().strip(" .!?*-_")
        ghost_phrases = ["you", "bye", "thanks", "thank you", "amara.org", "blank_audio"]
        
        if clean_text in ghost_phrases or (clean_text.startswith("[") and clean_text.endswith("]")):
            return ""
        return text if text else ""