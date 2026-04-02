"""Kyutai moshi-server ASR streaming client."""

import asyncio
import subprocess

import msgpack
import numpy as np
import websockets


class KyutaiApi:
    SAMPLE_RATE = 24_000
    CHUNK_SAMPLES = 1_920  # 80 ms at 24 kHz

    def __init__(self, server_url: str = "ws://127.0.0.1:8080/api/asr-streaming"):
        self.server_url = server_url

    # ------------------------------------------------------------------
    # Audio loading
    # ------------------------------------------------------------------

    def _load_audio(self, path: str) -> np.ndarray:
        """Load any audio file, resample to 24 kHz mono, return float32 array."""
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-i", path,
            "-ar", str(self.SAMPLE_RATE),
            "-ac", "1",
            "-f", "f32le",
            "-",
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return np.array([], dtype=np.float32)
        return np.frombuffer(result.stdout, dtype=np.float32).copy()

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    _MARKER_ID = 1

    async def _run(self, audio: np.ndarray) -> str:
        """Stream float32 audio to moshi-server, return full transcript string."""
        words: list[str] = []

        async with websockets.connect(
            self.server_url,
            additional_headers={"kyutai-api-key": "public_token"},
        ) as ws:
            # Send Init, then wait for Ready
            await ws.send(msgpack.packb({"type": "Init"}))
            ready = msgpack.unpackb(await ws.recv(), raw=False)
            if ready.get("type") != "Ready":
                raise RuntimeError(f"Expected Ready, got: {ready}")

            # --- send audio + receive words concurrently ---
            async def _send() -> None:
                for i in range(0, len(audio), self.CHUNK_SAMPLES):
                    chunk = audio[i : i + self.CHUNK_SAMPLES]
                    if len(chunk) < self.CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, self.CHUNK_SAMPLES - len(chunk)))
                    await ws.send(msgpack.packb({"type": "Audio", "pcm": chunk.tolist()}))
                # Signal end-of-stream with a Marker.
                # Then send silence: step_idx only advances when a channel has active audio,
                # so the batch loop stalls without it and the Marker echo is never sent.
                # 64 frames of silence (> asr_delay_in_tokens=32) is enough to flush the delay.
                await ws.send(msgpack.packb({"type": "Marker", "id": self._MARKER_ID}))
                silence = [0.0] * (self.CHUNK_SAMPLES * 64)
                await ws.send(msgpack.packb({"type": "Audio", "pcm": silence}))

            async def _recv() -> None:
                async for raw in ws:
                    msg = msgpack.unpackb(raw, raw=False)
                    t = msg.get("type")
                    if t == "Word":
                        words.append(msg.get("text", ""))
                    elif t == "Marker":
                        break

            await asyncio.gather(_send(), _recv())

        return " ".join(words).strip()
