"""Shared fixtures for quantize tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sr() -> int:
    return 22050


@pytest.fixture
def mono_120bpm(sr: int) -> np.ndarray:
    """~4 seconds of synthetic audio with clear 120 BPM transients.

    120 BPM = 2 beats/sec = beat every 0.5s.
    Creates short click transients at each beat position.
    """
    duration = 4.0
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float64)

    beat_interval = sr // 2  # 0.5s between beats at 120 BPM
    click_len = int(sr * 0.005)  # 5ms click

    for i in range(0, n_samples, beat_interval):
        end = min(i + click_len, n_samples)
        # Short burst of noise as a transient
        audio[i:end] = np.random.default_rng(42).uniform(-0.8, 0.8, end - i)

    return audio


@pytest.fixture
def stereo_120bpm(mono_120bpm: np.ndarray) -> np.ndarray:
    """Stereo version of the 120 BPM signal."""
    return np.column_stack([mono_120bpm, mono_120bpm * 0.8])


@pytest.fixture
def short_audio(sr: int) -> np.ndarray:
    """0.1 second audio -- too short for reliable BPM detection."""
    n_samples = int(sr * 0.1)
    return np.random.default_rng(99).uniform(-0.5, 0.5, n_samples)


@pytest.fixture
def sample_folder(tmp_path: Path, mono_120bpm: np.ndarray, sr: int) -> Path:
    """Folder with 3 wav files for batch testing."""
    for name in ["beat_a.wav", "beat_b.wav", "beat_c.wav"]:
        sf.write(str(tmp_path / name), mono_120bpm, sr)
    return tmp_path


@pytest.fixture
def mixed_folder(
    tmp_path: Path,
    mono_120bpm: np.ndarray,
    short_audio: np.ndarray,
    sr: int,
) -> Path:
    """Folder with valid files, a too-short file, and a non-audio file."""
    sf.write(str(tmp_path / "good.wav"), mono_120bpm, sr)
    sf.write(str(tmp_path / "tiny.wav"), short_audio, sr)
    (tmp_path / "readme.txt").write_text("not audio")
    return tmp_path
