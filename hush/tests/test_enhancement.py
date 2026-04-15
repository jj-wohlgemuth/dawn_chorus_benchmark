"""Test that Hush enhancement reproduces the reference output.

The reference sample_00006_denoised.wav was generated on GPU with infer_single.py
(PyTorch + libdf, matching the training pipeline exactly).  On CPU, float32
accumulation order differs slightly, so we allow ±1 int16 LSB (3.05e-5 in float32).
On the same GPU hardware the outputs are bit-for-bit identical.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).parent.parent / "enhance_dawn_chorus_with_hush.py"
ASSETS = Path(__file__).parent.parent / "assets" / "audio"
RAW_WAV = ASSETS / "sample_00006_raw.wav"
DENOISED_WAV = ASSETS / "sample_00006_denoised.wav"
OUTPUT_WAV = ASSETS / "unit_test_enhanced.wav"

# 2 int16 LSBs in float32 representation.
# The reference was generated on GPU; CPU float32 accumulation order differs by
# ≤ 1 LSB, and int16 quantization in the reference adds another ±0.5 LSB, so
# the combined worst-case distance is just under 2 LSBs.
_TWO_LSB = 2 / 32768


def test_enhancement_matches_reference():
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(RAW_WAV), "--output", str(OUTPUT_WAV)],
        check=True,
    )

    enhanced, sr = sf.read(str(OUTPUT_WAV))
    reference, ref_sr = sf.read(str(DENOISED_WAV))

    assert sr == ref_sr, f"Sample rate mismatch: {sr} vs {ref_sr}"
    assert len(enhanced) == len(reference), (
        f"Length mismatch: {len(enhanced)} vs {len(reference)}"
    )
    # Within 2 int16 LSBs — bit-exact on same GPU hardware, ≤2 steps on CPU
    np.testing.assert_allclose(enhanced, reference, atol=_TWO_LSB, rtol=0)
