import argparse
from pathlib import Path

from faster_whisper import WhisperModel

parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=Path)
args = parser.parse_args()

audio_dir = args.base_dir / "audio"
transcript_dir = args.base_dir / "transcripts"
transcript_dir.mkdir(parents=True, exist_ok=True)

model = WhisperModel("tiny.en", device="auto", compute_type="auto")
print("Loaded Whisper model")

wav_files = sorted(audio_dir.glob("*.wav"))
print(f"Found {len(wav_files)} WAV files in {audio_dir}")

for wav_file in wav_files:
    transcript_path = transcript_dir / (wav_file.stem + ".txt")
    print(f"Transcribing {wav_file.name}...")
    segments, _ = model.transcribe(str(wav_file), language="en")
    text = " ".join(seg.text.strip() for seg in segments)

    if not text.strip():
        print(f"[WARN] Empty transcript for {wav_file.name}, skipping.")
        continue

    transcript_path.write_text(text)
    print(f"  -> {transcript_path.name}")

print("Done.")
