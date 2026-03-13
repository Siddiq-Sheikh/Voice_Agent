# import os
# import time
# import torch
# import numpy as np
# import asyncio
# import urllib.request
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# from huggingface_hub import snapshot_download

# class TTSService:
#     def __init__(self, speaker_wav_path: str = "assets/speaker_ref.wav"):
#         print("\n" + "="*55 + "\n 🗣️  BOOTING XTTS V2 ENGINE \n" + "="*55)
#         start_init = time.perf_counter()
        
#         # 1. Download/Locate Model
#         print(">> [TTS] Locating XTTS v2 checkpoint...")
#         model_path = os.path.join(os.getcwd(), "xtts_model")
#         if not os.path.exists(model_path):
#             print(">> [TTS] Downloading XTTS v2 (First time only)...")
#             snapshot_download(repo_id="coqui/XTTS-v2", local_dir=model_path)

#         # 2. Load Config & Model
#         config = XttsConfig()
#         config.load_json(os.path.join(model_path, "config.json"))
#         self.model = Xtts.init_from_config(config)
#         self.model.load_checkpoint(
#             config, 
#             checkpoint_dir=model_path, 
#             eval=True, 
#             use_deepspeed=False
#         )
        
#         # Move to GPU
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model.to(self.device)
            
#         self.sample_rate = 24000

#         # 3. Extract and Cache Speaker Latents
#         print(f">> [TTS] Caching speaker latents from {speaker_wav_path}...")
        
#         if not os.path.exists(speaker_wav_path):
#             print(f">> [TTS] ⚠️ No speaker reference found at {speaker_wav_path}!")
#             print(">> [TTS] Downloading a default female voice (LJSpeech)...")
#             os.makedirs(os.path.dirname(speaker_wav_path), exist_ok=True)
#             sample_url = "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"
#             urllib.request.urlretrieve(sample_url, speaker_wav_path)
#             print(">> [TTS] Default speaker_ref.wav downloaded successfully!")
            
#         # Extract latents natively
#         self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
#             audio_path=[speaker_wav_path]
#         )

#         # NOTE: We removed the .half() precision conversion here!
#         # Your RTX 2060 Super has plenty of VRAM to handle native 32-bit.

#         init_time = (time.perf_counter() - start_init) * 1000
#         print(f">> [TTS Latency] Model Loaded & Optimized in {init_time:.2f}ms")

#         # 4. WARM-UP
#         print(">> [TTS] Warming up engine (CUDA graphs)...")
#         self._warmup()
#         print(">> [TTS] Engine Ready.")

#     def _warmup(self):
#         chunks = self.model.inference_stream(
#             "System online.",
#             "en",
#             self.gpt_cond_latent,
#             self.speaker_embedding,
#             enable_text_splitting=False
#         )
#         for _ in chunks:
#             pass 

#     async def generate_audio_stream(self, text: str):
#         if not text.strip(): return

#         start_gen = time.perf_counter()
#         first_chunk = True

#         try:
#             stream = self.model.inference_stream(
#                 text,
#                 language="en",
#                 gpt_cond_latent=self.gpt_cond_latent,
#                 speaker_embedding=self.speaker_embedding,
#                 enable_text_splitting=False 
#             )

#             for chunk in stream:
#                 if first_chunk:
#                     ttfa = (time.perf_counter() - start_gen) * 1000
#                     print(f"   [TTS Latency] TTFA (Streaming): {ttfa:.2f}ms")
#                     first_chunk = False

#                 audio_np = chunk.cpu().numpy().astype(np.float32)
#                 yield audio_np.tobytes()
                
#                 await asyncio.sleep(0)

#             total_time = (time.perf_counter() - start_gen) * 1000
#             print(f"   [TTS Latency] Synthesis Complete: {total_time:.2f}ms")

#         except Exception as e:
#             print(f">> [TTS Error]: {e}")

import os
import time
import torch
import numpy as np
import asyncio
import urllib.request
import subprocess
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download

class TTSService:
    def __init__(self, speaker_wav_path: str = "assets/speaker_ref.wav"):
        print("\n" + "="*55 + "\n 🗣️  BOOTING DUAL TTS ENGINE (PC: XTTS | Laptop: Piper) \n" + "="*55)
        start_init = time.perf_counter()
        
        # --- XTTS SETUP (PC Mode) ---
        print(">> [TTS] Initializing XTTS v2...")
        model_path = os.path.join(os.getcwd(), "xtts_model")
        if not os.path.exists(model_path):
            snapshot_download(repo_id="coqui/XTTS-v2", local_dir=model_path)

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=False)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Cache XTTS Latents
        if not os.path.exists(speaker_wav_path):
            os.makedirs(os.path.dirname(speaker_wav_path), exist_ok=True)
            sample_url = "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"
            urllib.request.urlretrieve(sample_url, speaker_wav_path)
            
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=[speaker_wav_path])

        # --- PIPER SETUP (Laptop Mode) ---
        # Ensure you have piper.exe and the model in these paths
        self.piper_path = os.path.join(os.getcwd(), "assets", "piper", "piper.exe")
        self.piper_model = os.path.join(os.getcwd(), "assets", "piper", "en_US-lessac-medium.onnx")
        
        init_time = (time.perf_counter() - start_init) * 1000
        print(f">> [TTS Latency] Engines Loaded in {init_time:.2f}ms")

    async def generate_audio_stream(self, text: str, use_piper: bool = False):
        if not text.strip(): return
        start_gen = time.perf_counter()

        if use_piper:
            # --- LAPTOP MODE: Piper EXE ---
            try:
                # Piper outputs raw PCM to stdout, which we capture
                command = [
                    self.piper_path,
                    "--model", self.piper_model,
                    "--output_raw"
                ]
                
                # Run Piper as a subprocess
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )

                # Send text to Piper
                stdout, _ = await process.communicate(input=text.encode())
                
                # Convert raw bytes to Float32 for your browser/websocket
                audio_np = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
                
                print(f"   [TTS Latency] Piper (Laptop): {(time.perf_counter()-start_gen)*1000:.2f}ms")
                yield audio_np.tobytes()

            except Exception as e:
                print(f">> [Piper Error]: {e}")
        
        else:
            # --- PC MODE: XTTS ---
            try:
                # Direct XTTS Inference for stability
                out = self.model.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    enable_text_splitting=True
                )
                
                audio_np = out['wav'].cpu().numpy().astype(np.float32)
                print(f"   [TTS Latency] XTTS (PC): {(time.perf_counter()-start_gen)*1000:.2f}ms")
                yield audio_np.tobytes()
                
            except Exception as e:
                print(f">> [XTTS Error]: {e}")