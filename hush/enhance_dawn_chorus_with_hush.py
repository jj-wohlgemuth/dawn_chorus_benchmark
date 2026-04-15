#!/usr/bin/env python3
"""Batch denoising of dawn_chorus_en using Hush (PyTorch + libdf).

Uses libdf for feature extraction and synthesis, matching the training pipeline
exactly.  Output is bit-for-bit identical to the reference denoised samples.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Suppress FFmpeg noise from datasets import
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

from libdf import DF, erb, erb_norm, unit_norm
from model.dfnet_se import DfNetSE, as_complex, as_real, get_config, get_norm_alpha

REPO_ID = "ai-coustics/dawn_chorus_en"
SPLIT = "eval"
ATTEN_LIM_DB = 100.0

_HUSH_DIR = Path(__file__).resolve().parent
MODEL_PATH = _HUSH_DIR.parent / "models" / "model_best.ckpt"


def _output_dir(atten_lim_db: float) -> Path:
    model_slug = MODEL_PATH.stem  # "model_best"
    return _HUSH_DIR.parent / f"hush_{model_slug}_atten{int(atten_lim_db)}" / "audio"


OUTPUT_DIR = _output_dir(ATTEN_LIM_DB)


def _load_model(ckpt_path: Path, device: torch.device) -> DfNetSE:
    config = get_config()
    model = DfNetSE(config).to(device)
    state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    try:
        model.model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            if all(k.startswith("model.") for k in state_dict):
                stripped = {k[6:]: v for k, v in state_dict.items()}
                model.model.load_state_dict(stripped, strict=True)
            else:
                raise
    model.eval()
    return model


def _load_wav_bytes(data: bytes, target_sr: int) -> torch.Tensor:
    """Load WAV bytes as mono float32 tensor [1, T], resampled to target_sr."""
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
    elif channels != 1:
        raise ValueError(f"Unexpected channel count: {channels}")
    wav = torch.from_numpy(audio).unsqueeze(0)  # [1, T]
    if sr != target_sr:
        import torchaudio
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


@torch.no_grad()
def _enhance(
    model: DfNetSE,
    audio: torch.Tensor,
    atten_lim_db: float | None = None,
    pad_delay: bool = True,
) -> torch.Tensor:
    """Enhance audio using libdf feature extraction (matches training pipeline)."""
    config = model.config
    device = next(model.parameters()).device

    df_state = DF(
        sr=config.sr,
        fft_size=config.fft_size,
        hop_size=config.hop_size,
        nb_bands=config.nb_erb,
        min_nb_erb_freqs=config.min_nb_freqs,
    )
    orig_len = int(audio.shape[-1])
    n_fft = df_state.fft_size()
    hop = df_state.hop_size()

    if pad_delay:
        audio = F.pad(audio, (0, n_fft))

    alpha = get_norm_alpha(df_state.sr(), df_state.hop_size(), config.norm_tau)
    spec_np = df_state.analysis(audio.numpy(), reset=True)
    erb_fb = df_state.erb_widths()
    spec = torch.as_tensor(spec_np)
    erb_feat_np = erb_norm(erb(spec_np, erb_fb), alpha)
    spec_feat_np = unit_norm(spec_np[..., : config.nb_df], alpha)
    spec_t = as_real(spec).unsqueeze(1).to(device)
    erb_feat_t = torch.as_tensor(erb_feat_np).unsqueeze(1).to(device)
    spec_feat_t = as_real(torch.as_tensor(spec_feat_np)).unsqueeze(1).to(device)

    spec_enh = model.model(spec_t.clone(), erb_feat_t, spec_feat_t)[0]
    spec_enh_c = as_complex(spec_enh.squeeze(1)).cpu()

    if atten_lim_db is not None and abs(float(atten_lim_db)) > 0:
        lim = 10 ** (-abs(float(atten_lim_db)) / 20.0)
        spec_in_c = as_complex(spec.unsqueeze(1).squeeze(1))
        spec_enh_c = spec_in_c * lim + spec_enh_c * (1.0 - lim)

    enh_np = df_state.synthesis(spec_enh_c.numpy(), reset=True)
    enh = torch.from_numpy(np.asarray(enh_np, dtype=np.float32))

    if pad_delay:
        delay = n_fft - hop
        enh = enh[:, delay : orig_len + delay]

    return enh


def _save_wav(path: Path, audio: torch.Tensor, sr: int) -> None:
    """Save float32 tensor as 32-bit float WAV via soundfile."""
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.squeeze().cpu().numpy(), sr, subtype="FLOAT")


def enhance_file(input_path: Path, output_path: Path, atten_lim_db: float = ATTEN_LIM_DB) -> None:
    """Enhance a single WAV file — used by tests and CLI."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(MODEL_PATH, device)
    config = model.config
    audio = _load_wav_bytes(input_path.read_bytes(), config.sr)
    enhanced = _enhance(model, audio, atten_lim_db=atten_lim_db)
    _save_wav(output_path, enhanced, config.sr)


def main(output_dir: Path = OUTPUT_DIR, atten_lim_db: float = ATTEN_LIM_DB) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = _load_model(MODEL_PATH, device)
    config = model.config

    print(f"Loading dataset: {REPO_ID} ({SPLIT})...")
    dataset = load_dataset(REPO_ID, split=SPLIT)
    for col, feat in dataset.features.items():
        if isinstance(feat, Audio):
            dataset = dataset.cast_column(col, Audio(decode=False))
    print(f"Found {len(dataset)} examples.")

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

        out_path = output_dir / file_id
        if out_path.exists():
            print(f"  [{i + 1}/{len(dataset)}] Skipping {file_id} (already exists)")
            continue

        audio = _load_wav_bytes(mix_data["bytes"], config.sr)
        enhanced = _enhance(model, audio, atten_lim_db=atten_lim_db)
        _save_wav(out_path, enhanced, config.sr)
        print(f"  [{i + 1}/{len(dataset)}] {file_id}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="Single WAV file to enhance")
    parser.add_argument("--output", type=Path, help="Output WAV path")
    parser.add_argument("--atten-lim-db", type=float, default=ATTEN_LIM_DB)
    args, _ = parser.parse_known_args()

    if args.input and args.output:
        enhance_file(args.input, args.output, atten_lim_db=args.atten_lim_db)
    else:
        main(output_dir=_output_dir(args.atten_lim_db), atten_lim_db=args.atten_lim_db)
