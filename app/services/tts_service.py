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

# --- FIX FOR PYTORCH 2.6+ SECURITY ---
torch.serialization.add_safe_globals([XttsConfig])

class TTSService:
    def __init__(self, speaker_wav_path: str = "assets/speaker_ref.wav"):
        print("\n" + "="*55 + "\n 🗣️  BOOTING DUAL TTS ENGINE \n" + "="*55)
        
        self.has_gpu = torch.cuda.is_available()
        self.model = None # Placeholder for XTTS
        self.speaker_wav_path = speaker_wav_path
        
        # 1. Setup Piper Paths (Always available as Fallback)
        self.piper_path = os.path.join(os.getcwd(), "assets", "piper", "piper.exe")
        self.piper_model = os.path.join(os.getcwd(), "assets", "piper", "en_US-lessac-medium.onnx")

        # 2. Hardware Routing
        if self.has_gpu:
            print(">> [TTS] GPU Detected. Initializing XTTS v2 Streaming Engine...")
            self._setup_xtts()
        else:
            print(">> [TTS] No GPU. XTTS initialization skipped. Piper active.")
            self._verify_piper_install()

    def _verify_piper_install(self):
        """Checks if Piper is actually installed where it should be."""
        if not os.path.exists(self.piper_path) or not os.path.exists(self.piper_model):
            print(">> ❌ [PIPER ERROR] Missing piper.exe or .onnx model in assets/piper/")

    def _setup_xtts(self):
        """Your exact XTTS initialization with Warmup."""
        start_init = time.perf_counter()
        model_path = os.path.join(os.getcwd(), "xtts_model")
        
        if not os.path.exists(model_path):
            print(">> [TTS] Downloading XTTS v2 (First time only)...")
            snapshot_download(repo_id="coqui/XTTS-v2", local_dir=model_path)

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(
            config, 
            checkpoint_dir=model_path, 
            eval=True, 
            use_deepspeed=False
        )
        
        self.model.to("cuda")
        
        print(f">> [TTS] Caching speaker latents from {self.speaker_wav_path}...")
        if not os.path.exists(self.speaker_wav_path):
            os.makedirs(os.path.dirname(self.speaker_wav_path), exist_ok=True)
            sample_url = "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"
            urllib.request.urlretrieve(sample_url, self.speaker_wav_path)
            
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[self.speaker_wav_path]
        )
        
        init_time = (time.perf_counter() - start_init) * 1000
        print(f">> [TTS Latency] Model Loaded & Optimized in {init_time:.2f}ms")

        print(">> [TTS] Warming up engine (CUDA graphs)...")
        self._warmup()
        print(">> [TTS] XTTS Engine Ready.")

    def _warmup(self):
        chunks = self.model.inference_stream(
            "System online.",
            "en",
            self.gpt_cond_latent,
            self.speaker_embedding,
            enable_text_splitting=False
        )
        for _ in chunks: pass 

    def _run_piper_sync(self, text: str) -> bytes:
        """Synchronous wrapper for Piper to bypass Windows asyncio bugs."""
        piper_dir = os.path.dirname(self.piper_path)
        command = [self.piper_path, "--model", self.piper_model, "--output_raw"]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            
        result = subprocess.run(
            command, input=text.encode('utf-8'),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=piper_dir, startupinfo=startupinfo
        )
        
        if result.returncode != 0:
            print(f">> [Piper Internal Error]: {result.stderr.decode('utf-8', errors='ignore')}")
            return b""
        return result.stdout

    async def generate_audio_stream(self, text: str, use_piper: bool = False, **kwargs):
        if not text.strip(): return
        
        # Route to Piper if forced, OR if XTTS isn't loaded (laptop mode)
        if not use_piper and self.model is None:
            use_piper = True

        if use_piper:
            # --- PIPER LOGIC ---
            try:
                stdout = await asyncio.to_thread(self._run_piper_sync, text)
                if stdout:
                    audio_np = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
                    yield audio_np.tobytes()
            except Exception as e:
                print(f">> [Piper Execution Error]: {e}")
        else:
            # --- XTTS STREAMING LOGIC ---
            start_gen = time.perf_counter()
            first_chunk = True
            try:
                stream = self.model.inference_stream(
                    text,
                    language="en",
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    enable_text_splitting=False 
                )

                for chunk in stream:
                    if first_chunk:
                        ttfa = (time.perf_counter() - start_gen) * 1000
                        print(f"   [TTS Latency] TTFA (Streaming): {ttfa:.2f}ms")
                        first_chunk = False

                    # The safety check so it doesn't crash on standard arrays
                    if isinstance(chunk, torch.Tensor):
                        audio_np = chunk.cpu().numpy().astype(np.float32)
                    else:
                        audio_np = np.array(chunk, dtype=np.float32)
                        
                    yield audio_np.tobytes()
                    await asyncio.sleep(0)

                total_time = (time.perf_counter() - start_gen) * 1000
                print(f"   [TTS Latency] Synthesis Complete: {total_time:.2f}ms")

            except Exception as e:
                print(f">> [XTTS Error]: {e}")