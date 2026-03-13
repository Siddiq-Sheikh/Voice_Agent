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
        
        # 1. Setup Piper Paths (Always available)
        self.piper_path = os.path.join(os.getcwd(), "assets", "piper", "piper.exe")
        self.piper_model = os.path.join(os.getcwd(), "assets", "piper", "en_US-lessac-medium.onnx")

        if self.has_gpu:
            print(">> [TTS] GPU Detected. Initializing XTTS v2...")
            self._setup_xtts()
        else:
            print(">> [TTS] No GPU. XTTS initialization skipped. Piper active.")

    def _setup_xtts(self):
        """Heavy lifting for XTTS - Only runs if GPU is present."""
        start_init = time.perf_counter()
        model_path = os.path.join(os.getcwd(), "xtts_model")
        
        if not os.path.exists(model_path):
            print(">> [TTS] Downloading XTTS v2...")
            snapshot_download(repo_id="coqui/XTTS-v2", local_dir=model_path)

        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        
        # We pass weights_only=False inside the XTTS internal loader via a workaround 
        # or rely on the add_safe_globals we added at the top.
        self.model.load_checkpoint(
            config, 
            checkpoint_dir=model_path, 
            eval=True, 
            use_deepspeed=False
        )
        
        self.model.to("cuda")
        
        # Extract speaker latents
        if not os.path.exists(self.speaker_wav_path):
            os.makedirs(os.path.dirname(self.speaker_wav_path), exist_ok=True)
            sample_url = "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"
            urllib.request.urlretrieve(sample_url, self.speaker_wav_path)
            
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(
            audio_path=[self.speaker_wav_path]
        )
        
        print(f">> [TTS] XTTS Ready ({(time.perf_counter() - start_init):.2f}s)")

    async def generate_audio_stream(self, text: str, use_piper: bool = False):
        if not text.strip(): return
        
        # Safety check: if user wants XTTS but no GPU/Model is loaded, force Piper
        if not use_piper and self.model is None:
            print(">> [TTS Warning] XTTS requested but not loaded. Falling back to Piper.")
            use_piper = True

        if use_piper:
            # --- PIPER LOGIC ---
            try:
                command = [self.piper_path, "--model", self.piper_model, "--output_raw"]
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL
                )
                stdout, _ = await process.communicate(input=text.encode())
                audio_np = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
                yield audio_np.tobytes()
            except Exception as e:
                print(f">> [Piper Error]: {e}")
        else:
            # --- XTTS LOGIC ---
            try:
                # Using standard inference for better quality over WebSocket
                out = self.model.inference(
                    text=text,
                    language="en",
                    gpt_cond_latent=self.gpt_cond_latent,
                    speaker_embedding=self.speaker_embedding,
                    enable_text_splitting=True
                )
                yield out['wav'].cpu().numpy().astype(np.float32).tobytes()
            except Exception as e:
                print(f">> [XTTS Error]: {e}")