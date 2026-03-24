#!/usr/bin/env python3
"""Batch enhancement of dawn_chorus_en using ai-coustics Python SDK."""

from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
import aic_sdk as aic
from dotenv import load_dotenv

load_dotenv()

# torchcodec (pulled in by datasets) prints FFmpeg load errors to both C-level
# stderr (fd 2) and Python's sys.stderr; silence both for the import.
_saved_stderr_fd = os.dup(2)
_saved_sys_stderr = sys.stderr
with open(os.devnull, "w") as _devnull:
    os.dup2(_devnull.fileno(), 2)
    sys.stderr = _devnull
    try:
        from datasets import Audio, load_dataset
    finally:
        os.dup2(_saved_stderr_fd, 2)
        os.close(_saved_stderr_fd)
        sys.stderr = _saved_sys_stderr

REPO_ID = "ai-coustics/dawn_chorus_en"
SPLIT = "eval"

# Make one copy of this script per model and change these constants.
MODEL_ID = "quail-vf-2.0-l-16khz"
OUTPUT_DIR = Path("quail_vf_2_0_l_16khz_el_80/audio")
MODELS_DIR = Path("models")

# Optional tuning
ENHANCEMENT_LEVEL = 0.8  # range 0.0 .. 1.0
BYPASS = 0.0             # 0.0 = enhance, 1.0 = passthrough with latency
LICENSE_ENV = "AIC_SDK_LICENSE"


def _load_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM, got sample width {sampwidth}")

    audio = np.frombuffer(raw, dtype=np.int16)

    if channels == 2:
        # Match the behavior of the original script: downmix stereo to mono.
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels != 1:
        raise ValueError(f"Expected mono or stereo, got {channels} channels")

    return audio, sr


def _write_wav(path: Path, audio_i16: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.astype(np.int16).tobytes())


def _to_float32_mono(audio_i16: np.ndarray) -> np.ndarray:
    # Shape must be (channels, frames) for aic_sdk.
    audio_f32 = audio_i16.astype(np.float32) / 32768.0
    return audio_f32[np.newaxis, :]


def _to_int16_mono(audio_f32_chxT: np.ndarray) -> np.ndarray:
    if audio_f32_chxT.ndim != 2 or audio_f32_chxT.shape[0] != 1:
        raise ValueError(f"Expected shape (1, T), got {audio_f32_chxT.shape}")
    audio = audio_f32_chxT[0]
    return (audio * 32768.0).clip(-32768, 32767).astype(np.int16)


def _create_processor(model: aic.Model, license_key: str, sr: int) -> tuple[aic.Processor, int]:
    config = aic.ProcessorConfig.optimal(
        model,
        sample_rate=sr,
        num_channels=1,
        allow_variable_frames=False,
    )
    processor = aic.Processor(model, license_key, config)

    proc_ctx = processor.get_processor_context()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, ENHANCEMENT_LEVEL)
    proc_ctx.set_parameter(aic.ProcessorParameter.Bypass, BYPASS)

    frame_len = config.num_frames
    return processor, frame_len


def _enhance_offline(
    processor: aic.Processor,
    audio_i16: np.ndarray,
    frame_len: int,
) -> np.ndarray:
    """
    Offline chunked processing with delay compensation.

    We:
    1. chunk/pad input to frame_len,
    2. append zero frames to flush delayed output,
    3. remove the reported processor delay from the front,
    4. trim back to original signal length.
    """
    proc_ctx = processor.get_processor_context()
    proc_ctx.reset()

    delay_samples = proc_ctx.get_output_delay()

    audio_f32 = _to_float32_mono(audio_i16)
    num_input_frames = audio_f32.shape[1]

    # Add enough trailing zeros to flush delayed output.
    flush_frames = max(1, int(np.ceil(delay_samples / frame_len)))
    padded_length = int(np.ceil(num_input_frames / frame_len) * frame_len)
    total_length = padded_length + flush_frames * frame_len

    padded = np.zeros((1, total_length), dtype=np.float32)
    padded[:, :num_input_frames] = audio_f32

    chunks = []
    for start in range(0, total_length, frame_len):
        chunk = padded[:, start : start + frame_len]
        enhanced = processor.process(chunk)
        chunks.append(enhanced)

    enhanced_full = np.concatenate(chunks, axis=1)

    # Delay compensation: align output to original input timeline.
    start = delay_samples
    end = start + num_input_frames
    enhanced_aligned = enhanced_full[:, start:end]

    return _to_int16_mono(enhanced_aligned)


def main() -> None:
    license_key = os.environ.get(LICENSE_ENV)
    if not license_key:
        raise RuntimeError(
            f"Missing license key. Set environment variable {LICENSE_ENV}."
        )

    print(f"SDK version: {aic.get_sdk_version()}")
    print(f"Compatible model version: {aic.get_compatible_model_version()}")

    print(f"Loading model: {MODEL_ID}")
    model_path = aic.Model.download(MODEL_ID, MODELS_DIR)
    model = aic.Model.from_file(model_path)

    print(f"Loading dataset: {REPO_ID} ({SPLIT})...")
    dataset = load_dataset(REPO_ID, split=SPLIT)
    for col, feat in dataset.features.items():
        if isinstance(feat, Audio):
            dataset = dataset.cast_column(col, Audio(decode=False))
    print(f"Found {len(dataset)} examples.")

    processors_by_sr: dict[int, tuple[aic.Processor, int]] = {}

    print(f"Processing {len(dataset)} file(s)  ->  {OUTPUT_DIR}")

    for i, row in enumerate(dataset):
        mix_data = row.get("mix")
        if not mix_data or "bytes" not in mix_data:
            print(f"  [{i + 1}/{len(dataset)}] Skipping: no 'bytes' in 'mix' column")
            continue

        file_id = row.get("id") or mix_data.get("path") or f"example_{i}"
        if not str(file_id).endswith(".wav"):
            file_id = f"{file_id}.wav"

        audio_i16, sr = _load_wav_bytes(mix_data["bytes"])

        if sr not in processors_by_sr:
            processor, frame_len = _create_processor(model, license_key, sr)
            processors_by_sr[sr] = (processor, frame_len)
            print(f"  initialized processor for sr={sr}, frame_len={frame_len}")

        processor, frame_len = processors_by_sr[sr]
        out_i16 = _enhance_offline(processor, audio_i16, frame_len)

        _write_wav(OUTPUT_DIR / file_id, out_i16, sr)
        print(f"  [{i + 1}/{len(dataset)}] {file_id}")

    print("Done.")


if __name__ == "__main__":
    main()