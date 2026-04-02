import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "kyutai"))
from kyutai_api import KyutaiApi  # noqa: E402

KYUTAI_URL = "ws://127.0.0.1:8080/api/asr-streaming"

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=Path)
parser.add_argument("--concurrency", type=int, default=10, help="Max parallel streams")
parser.add_argument("--server-url", default=KYUTAI_URL)
args = parser.parse_args()

audio_dir = args.base_dir / "audio"
transcript_dir = args.base_dir / "transcripts_kyutai"
transcript_dir.mkdir(parents=True, exist_ok=True)

wav_files = sorted(audio_dir.glob("*.wav"))
print(f"Found {len(wav_files)} WAV files in {audio_dir}")

api = KyutaiApi(server_url=args.server_url)


async def _transcribe(wav_file: Path, semaphore: asyncio.Semaphore) -> None:
    transcript_path = transcript_dir / (wav_file.stem + ".txt")
    if transcript_path.exists():
        print(f"Skipping {wav_file.name} (already transcribed)")
        return

    async with semaphore:
        print(f"Transcribing {wav_file.name}...")
        loop = asyncio.get_event_loop()
        # _load_audio uses ffmpeg subprocess — offload to thread to release the GIL
        audio: np.ndarray = await loop.run_in_executor(
            None, api._load_audio, str(wav_file)
        )
        if audio.size == 0:
            print(f"[WARN] Could not load audio for {wav_file.name}, skipping.")
            return

        try:
            text: str = await asyncio.wait_for(api._run(audio), timeout=60.0)
        except asyncio.TimeoutError:
            print(f"[WARN] Timeout for {wav_file.name}, skipping.")
            return

        if not text.strip():
            print(f"[WARN] Empty transcript for {wav_file.name}, skipping.")
            return

        transcript_path.write_text(text)
        print(f"  -> {transcript_path.name}")


async def main() -> None:
    semaphore = asyncio.Semaphore(args.concurrency)
    await asyncio.gather(*[_transcribe(f, semaphore) for f in wav_files])


asyncio.run(main())
print("Done.")
