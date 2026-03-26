import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate-ref",
        action="store_true",
        help="Regenerate the reference denoised wav file",
    )
    parser.addoption(
        "--atten-lim-db",
        type=float,
        default=None,
        help="Test only this specific attenuation limit value (dB)",
    )
