#!/usr/bin/env python3
"""Batch denoising of dawn_chorus_en dataset using the prebuilt libweya_nc library."""

from __future__ import annotations

import ctypes
import io
import platform
import wave
from pathlib import Path

import os

import numpy as np

import sys

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
ATTEN_LIM_DB = 100.0

_HUSH_DIR = Path(__file__).resolve().parent
_LIB_NAME = {"Darwin": "libweya_nc.dylib", "Windows": "weya_nc.dll"}.get(
    platform.system(), "libweya_nc.so"
)
LIB_PATH = _HUSH_DIR / "lib" / _LIB_NAME
MODEL_PATH = _HUSH_DIR.parent / "models" / "advanced_dfnet16k_model_best_onnx.tar.gz"


def _output_dir(atten_lim_db: float) -> Path:
    model_slug = MODEL_PATH.name.split(".")[0]
    return _HUSH_DIR.parent / f"hush_{model_slug}_atten{int(atten_lim_db)}" / "audio"


OUTPUT_DIR = _output_dir(ATTEN_LIM_DB)

def _setup_lib(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path.resolve()))
    lib.weya_nc_model_load_from_path.argtypes = [ctypes.c_char_p]
    lib.weya_nc_model_load_from_path.restype = ctypes.c_void_p
    lib.weya_nc_model_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_model_free.restype = None
    lib.weya_nc_session_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    lib.weya_nc_session_create.restype = ctypes.c_void_p
    lib.weya_nc_session_free.argtypes = [ctypes.c_void_p]
    lib.weya_nc_session_free.restype = None
    lib.weya_nc_reset.argtypes = [ctypes.c_void_p]
    lib.weya_nc_reset.restype = None
    lib.weya_nc_get_frame_length.argtypes = [ctypes.c_void_p]
    lib.weya_nc_get_frame_length.restype = ctypes.c_size_t
    lib.weya_nc_process_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.weya_nc_process_frame.restype = ctypes.c_float
    return lib


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
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif channels != 1:
        raise ValueError(f"Expected mono or stereo, got {channels} channels")
    return audio, sr


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.astype(np.int16).tobytes())


def _denoise(
    lib: ctypes.CDLL, session: int, audio_i16: np.ndarray, frame_len: int
) -> np.ndarray:
    frame_in = np.zeros(frame_len, dtype=np.float32)
    frame_out = np.zeros(frame_len, dtype=np.float32)
    out_i16 = np.zeros_like(audio_i16, dtype=np.int16)
    fp = ctypes.POINTER(ctypes.c_float)

    idx = 0
    while idx < len(audio_i16):
        end = min(idx + frame_len, len(audio_i16))
        frame_in.fill(0.0)
        frame_in[: end - idx] = audio_i16[idx:end].astype(np.float32) / 32768.0
        lib.weya_nc_process_frame(
            session,
            frame_in.ctypes.data_as(fp),
            frame_out.ctypes.data_as(fp),
        )
        out_i16[idx:end] = (
            (frame_out[: end - idx] * 32768.0).clip(-32768, 32767).astype(np.int16)
        )
        idx = end

    return out_i16


def main(output_dir: Path = OUTPUT_DIR, atten_lim_db: float = ATTEN_LIM_DB) -> None:
    print(f"Loading dataset: {REPO_ID} ({SPLIT})...")
    dataset = load_dataset(REPO_ID, split=SPLIT)
    for col, feat in dataset.features.items():
        if isinstance(feat, Audio):
            dataset = dataset.cast_column(col, Audio(decode=False))
    print(f"Found {len(dataset)} examples.")

    lib = _setup_lib(LIB_PATH)
    model = lib.weya_nc_model_load_from_path(str(MODEL_PATH.resolve()).encode())
    if not model:
        raise RuntimeError(f"Could not load model: {MODEL_PATH}")

    session = None
    current_sr = None
    frame_len = None

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(dataset)} file(s)  →  {output_dir}")

    for i, row in enumerate(dataset):
        mix_data = row.get("mix")
        if not mix_data or "bytes" not in mix_data:
            print(f"  [{i}/{len(dataset)}] Skipping: no 'bytes' in 'mix' column")
            continue

        file_id = row.get("id") or mix_data.get("path") or f"example_{i}"
        if not file_id.endswith(".wav"):
            file_id = f"{file_id}.wav"

        audio_i16, sr = _load_wav_bytes(mix_data["bytes"])

        if sr != current_sr:
            if session is not None:
                lib.weya_nc_session_free(session)
            session = lib.weya_nc_session_create(
                model, sr, ctypes.c_float(atten_lim_db)
            )
            if not session:
                raise RuntimeError(f"Could not create session for sr={sr}")
            frame_len = int(lib.weya_nc_get_frame_length(session))
            current_sr = sr
        else:
            lib.weya_nc_reset(session)

        assert session is not None and frame_len is not None
        out_path = output_dir / file_id
        if out_path.exists():
            print(f"  [{i + 1}/{len(dataset)}] Skipping {file_id} (already exists)")
            continue

        out_i16 = _denoise(lib, session, audio_i16, frame_len)
        _write_wav(out_path, out_i16, sr)
        print(f"  [{i + 1}/{len(dataset)}] {file_id}")

    if session is not None:
        lib.weya_nc_session_free(session)
    lib.weya_nc_model_free(model)
    print("Done.")


def enhance_file(input_path: Path, output_path: Path, atten_lim_db: float = ATTEN_LIM_DB) -> None:
    """Enhance a single WAV file — used by tests and CLI."""
    lib = _setup_lib(LIB_PATH)
    model = lib.weya_nc_model_load_from_path(str(MODEL_PATH.resolve()).encode())
    if not model:
        raise RuntimeError(f"Could not load model from {MODEL_PATH}")

    audio_i16, sr = _load_wav_bytes(input_path.read_bytes())
    session = lib.weya_nc_session_create(model, sr, ctypes.c_float(atten_lim_db))
    if not session:
        raise RuntimeError(f"Could not create session for sr={sr}")
    frame_len = int(lib.weya_nc_get_frame_length(session))

    out_i16 = _denoise(lib, session, audio_i16, frame_len)

    lib.weya_nc_session_free(session)
    lib.weya_nc_model_free(model)

    _write_wav(output_path, out_i16, sr)


if __name__ == "__main__":
    import argparse as _argparse
    _parser = _argparse.ArgumentParser()
    _parser.add_argument("--input", type=Path, help="Single WAV file to enhance")
    _parser.add_argument("--output", type=Path, help="Output WAV path")
    _parser.add_argument("--atten-lim-db", type=float, default=ATTEN_LIM_DB,
                         help=f"Attenuation limit in dB (default: {ATTEN_LIM_DB})")
    _args, _ = _parser.parse_known_args()

    if _args.input and _args.output:
        enhance_file(_args.input, _args.output, atten_lim_db=_args.atten_lim_db)
    else:
        main(output_dir=_output_dir(_args.atten_lim_db), atten_lim_db=_args.atten_lim_db)
