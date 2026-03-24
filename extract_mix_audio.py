"""Extract raw mix audio from the dawn_chorus_en dataset into mix/audio/."""

import io
import wave
from pathlib import Path

from datasets import Audio, load_dataset

OUTPUT_DIR = Path("mix/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("ai-coustics/dawn_chorus_en", split="eval")
for col, feat in dataset.features.items():
    if isinstance(feat, Audio):
        dataset = dataset.cast_column(col, Audio(decode=False))
print(f"Found {len(dataset)} examples.")

for i, row in enumerate(dataset):
    mix_data = row.get("mix")
    if not mix_data or "bytes" not in mix_data:
        print(f"  [{i + 1}] Skipping: no mix bytes")
        continue

    file_id = row.get("id") or mix_data.get("path") or f"example_{i}"
    if not str(file_id).endswith(".wav"):
        file_id = f"{file_id}.wav"

    out_path = OUTPUT_DIR / file_id
    if out_path.exists():
        continue

    out_path.write_bytes(mix_data["bytes"])
    print(f"  [{i + 1}/{len(dataset)}] {file_id}")

print("Done.")
