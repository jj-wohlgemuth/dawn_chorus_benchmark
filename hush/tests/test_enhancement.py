"""Test that Hush enhancement reproduces the reference output."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPT = Path(__file__).parent.parent / "enhance_dawn_chorus_with_hush_onnx.py"
ASSETS = Path(__file__).parent.parent / "assets" / "audio"
RAW_WAV = ASSETS / "sample_00006_raw.wav"
DENOISED_WAV = ASSETS / "sample_00006_denoised.wav"
OUTPUT_WAV = ASSETS / "unit_test_enhanced.wav"


def test_enhancement_matches_reference():
    subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(RAW_WAV), "--output", str(OUTPUT_WAV)],
        check=True,
    )

    enhanced, sr = sf.read(OUTPUT_WAV, dtype="int16")
    reference, ref_sr = sf.read(DENOISED_WAV, dtype="int16")

    assert sr == ref_sr
    assert len(enhanced) == len(reference)
    np.testing.assert_array_equal(enhanced, reference)
