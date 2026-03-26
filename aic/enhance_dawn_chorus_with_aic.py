#!/usr/bin/env python3
"""Batch enhancement of dawn_chorus_en using ai-coustics Python SDK (async parallel)."""

from __future__ import annotations

import asyncio
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

MODELS_DIR = Path("models")
LICENSE_ENV = "AIC_SDK_LICENSE"

_DEFAULT_MODEL_ID = "quail-vf-2.0-l-16khz"
_DEFAULT_ENHANCEMENT_LEVEL = 0.8
_DEFAULT_BYPASS = 0.0
_DEFAULT_MAX_PARALLEL = 8


def _output_dir(model_id: str, enhancement_level: float) -> Path:
    slug = model_id.replace("-", "_").replace(".", "_")
    el_pct = int(round(enhancement_level * 100))
    return Path(f"aic_{slug}_el{el_pct}") / "audio"


def _load_wav_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Returns float32 (1, T) array and sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM, got sample width {sampwidth}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)

    return audio[np.newaxis, :], sr  # (1, T)


def _write_wav(path: Path, audio_f32: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_i16 = (audio_f32[0] * 32768.0).clip(-32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())


async def _process_one(
    processor: aic.ProcessorAsync,
    model: aic.Model,
    audio_f32: np.ndarray,
    sr: int,
    out_path: Path,
    enhancement_level: float,
    file_id: str,
    idx: int,
    total: int,
) -> None:
    config = aic.ProcessorConfig.optimal(model, sample_rate=sr, num_channels=1)
    await processor.initialize_async(config)

    proc_ctx = processor.get_processor_context()
    proc_ctx.reset()
    proc_ctx.set_parameter(aic.ProcessorParameter.EnhancementLevel, enhancement_level)
    latency = proc_ctx.get_output_delay()

    # Prepend silence to compensate for algorithmic delay
    padded = np.concatenate([np.zeros((1, latency), dtype=np.float32), audio_f32], axis=1)

    frame_len = config.num_frames
    n_frames = padded.shape[1]
    output = np.zeros_like(padded)

    for start in range(0, n_frames, frame_len):
        end = min(start + frame_len, n_frames)
        chunk = np.zeros((1, frame_len), dtype=np.float32)
        chunk[:, : end - start] = padded[:, start:end]
        processed = await processor.process_async(chunk)
        output[:, start : start + processed.shape[1]] = processed

    # Remove latency padding from front, trim to original length
    result = output[:, latency : latency + audio_f32.shape[1]]
    _write_wav(out_path, result, sr)
    print(f"  [{idx}/{total}] {file_id}")


async def _run(
    samples: list[tuple[str, bytes]],
    model: aic.Model,
    license_key: str,
    output_dir: Path,
    enhancement_level: float,
    max_parallel: int,
) -> None:
    processors = [aic.ProcessorAsync(model, license_key) for _ in range(max_parallel)]
    total = len(samples)

    for batch_start in range(0, total, max_parallel):
        batch = samples[batch_start : batch_start + max_parallel]
        tasks = []
        for i, (file_id, wav_bytes) in enumerate(batch):
            audio_f32, sr = _load_wav_bytes(wav_bytes)
            out_path = output_dir / file_id
            if out_path.exists():
                print(f"  [{batch_start + i + 1}/{total}] Skipping {file_id} (exists)")
                continue
            tasks.append(_process_one(
                processors[i], model, audio_f32, sr, out_path,
                enhancement_level, file_id, batch_start + i + 1, total,
            ))
        await asyncio.gather(*tasks)


def main(
    model_id: str = _DEFAULT_MODEL_ID,
    enhancement_level: float = _DEFAULT_ENHANCEMENT_LEVEL,
    bypass: float = _DEFAULT_BYPASS,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
) -> None:
    output_dir = _output_dir(model_id, enhancement_level)
    license_key = os.environ.get(LICENSE_ENV)
    if not license_key:
        raise RuntimeError(f"Missing license key. Set environment variable {LICENSE_ENV}.")

    print(f"SDK version: {aic.get_sdk_version()}")
    print(f"Loading model: {model_id}")
    model_path = aic.Model.download(model_id, MODELS_DIR)
    aic_model = aic.Model.from_file(model_path)

    print(f"Loading dataset: {REPO_ID} ({SPLIT})...")
    dataset = load_dataset(REPO_ID, split=SPLIT)
    for col, feat in dataset.features.items():
        if isinstance(feat, Audio):
            dataset = dataset.cast_column(col, Audio(decode=False))
    print(f"Found {len(dataset)} examples.")

    samples = []
    for i, row in enumerate(dataset):
        mix_data = row.get("mix")
        if not mix_data or "bytes" not in mix_data:
            continue
        file_id = row.get("id") or mix_data.get("path") or f"example_{i}"
        if not str(file_id).endswith(".wav"):
            file_id = f"{file_id}.wav"
        samples.append((file_id, mix_data["bytes"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(samples)} file(s) → {output_dir}  (parallel={max_parallel})")
    asyncio.run(_run(samples, aic_model, license_key, output_dir, enhancement_level, max_parallel))
    print("Done.")


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    _parser.add_argument("--enhancement-level", type=float, default=_DEFAULT_ENHANCEMENT_LEVEL)
    _parser.add_argument("--bypass", type=float, default=_DEFAULT_BYPASS)
    _parser.add_argument("--max-parallel", type=int, default=_DEFAULT_MAX_PARALLEL)
    _args = _parser.parse_args()
    main(
        model_id=_args.model_id,
        enhancement_level=_args.enhancement_level,
        bypass=_args.bypass,
        max_parallel=_args.max_parallel,
    )
