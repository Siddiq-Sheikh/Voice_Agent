import asyncio
import numpy as np
from typing import AsyncGenerator
from aiortc import MediaStreamTrack, RTCPeerConnection
from av import AudioFrame
from av.audio.resampler import AudioResampler
from app.core.logger import log

class TTSAudioTrack(MediaStreamTrack):
    """Custom WebRTC Track that pulls PCM audio from the XTTS queue."""
    kind = "audio"

    def __init__(self, audio_queue: asyncio.Queue):
        super().__init__()
        self.audio_queue = audio_queue
        self.sample_rate = 24000 
        self.time = 0

    async def recv(self):
        pcm_data = await self.audio_queue.get()
        
        # Parse the float32 bytes coming from your XTTS engine
        audio_array = np.frombuffer(pcm_data, dtype=np.float32)
        audio_array = audio_array.reshape(1, -1)
        
        # 'flt' tells PyAV to expect float32 data!
        frame = AudioFrame.from_ndarray(audio_array, format='flt', layout='mono')
        frame.sample_rate = self.sample_rate
        
        frame.pts = self.time
        self.time += frame.samples
        frame.time_base = 1 / self.sample_rate
        
        self.audio_queue.task_done()
        return frame


class WebRTCAgent:
    def __init__(self, pc: RTCPeerConnection, stt, llm, tts):
        self.pc = pc
        self.stt = stt
        self.llm = llm
        self.tts = tts
        
        self.is_interrupted = False 
        self.is_speaking = False 
        
        self.tts_queue = asyncio.Queue()          
        self.audio_out_queue = asyncio.Queue()  
        
        self.output_track = TTSAudioTrack(self.audio_out_queue)
        self.pc.addTrack(self.output_track)

        @pc.on("track")
        def on_track(track):
            if track.kind == "audio":
                log.info("🎤 [WebRTC] Microphone Track Received!")
                asyncio.create_task(self.process_incoming_audio(track))

        self.tts_worker_task = asyncio.create_task(self._tts_worker())

    async def process_incoming_audio(self, track):
        """Down-samples WebRTC 48kHz audio and streams to the Silero VAD thread."""
        resampler = AudioResampler(format='s16', layout='mono', rate=16000)
        log.info("🟢 [WebRTC] Audio streaming to STT VAD loop...")

        while True:
            try:
                frame = await track.recv()
                resampled_frames = resampler.resample(frame)
                
                for r_frame in resampled_frames:
                    pcm_bytes = r_frame.to_ndarray().tobytes()
                    
                    # Feed the exact bytes instantly so Silero can slice it!
                    await self.handle_stt_and_llm(pcm_bytes)
                    
            except Exception as e:
                log.warning(f"🛑 [WebRTC] Track closed or disconnected: {e}")
                break

    async def handle_stt_and_llm(self, audio_bytes):
        """Triggers the LLM pipeline the moment the VAD yields a completed sentence."""
        
        user_text = await self.stt.transcribe_chunk(audio_bytes)
        
        if user_text and user_text.strip():
            self.is_interrupted = False
            self.is_speaking = True 
            
            # 1. Start generation
            word_stream = self.llm.generate_response_stream(user_text)
            
            # 2. Chunk it into sentences and push to TTS
            async for sentence in self._buffer_sentences(word_stream):
                if self.is_interrupted:
                    break
                await self.tts_queue.put(sentence)

    async def _buffer_sentences(self, word_stream) -> AsyncGenerator[str, None]:
        buffer = ""
        SENTENCE_END = {'.', '!', '?', '\n'}
        
        async for word in word_stream:
            if self.is_interrupted:
                break
            
            buffer += word
            if any(buffer.endswith(p) for p in SENTENCE_END) or len(buffer) > 40:
                yield buffer.strip()
                buffer = ""
        
        if buffer.strip() and not self.is_interrupted:
            yield buffer.strip()

    async def _tts_worker(self):
        while True:
            sentence = await self.tts_queue.get()
            
            if self.is_interrupted:
                self.tts_queue.task_done()
                continue
            
            # Stream the raw audio bytes directly into the WebRTC track queue
            async for audio_chunk in self.tts.generate_audio_stream(sentence):
                if self.is_interrupted:
                    break
                await self.audio_out_queue.put(audio_chunk)
            
            self.tts_queue.task_done()
            
            if self.tts_queue.empty() and self.audio_out_queue.empty():
                self.is_speaking = False