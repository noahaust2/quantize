"""Shared fixtures for quantize tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from PIL import Image


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


# --- onset_video fixtures ---


@pytest.fixture
def mono_with_onsets(sr: int) -> np.ndarray:
    """~4 seconds of audio with clear transient clicks at known positions.

    Clicks at 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5s.
    """
    duration = 4.0
    n_samples = int(sr * duration)
    audio = np.zeros(n_samples, dtype=np.float64)

    click_len = int(sr * 0.005)  # 5ms click
    rng = np.random.default_rng(42)

    for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
        start = int(t * sr)
        end = min(start + click_len, n_samples)
        audio[start:end] = rng.uniform(-0.9, 0.9, end - start)

    return audio


@pytest.fixture
def sample_rgb_images(tmp_path: Path) -> list[Path]:
    """4 small solid-color RGB PNG images."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    paths = []
    for i, color in enumerate(colors, 1):
        p = tmp_path / f"frame_{i:02d}.png"
        img = Image.new("RGB", (100, 100), color)
        img.save(p)
        paths.append(p)
    return paths


@pytest.fixture
def sample_rgba_images(tmp_path: Path) -> list[Path]:
    """3 small RGBA PNG images with alpha channel."""
    colors = [(255, 0, 0, 128), (0, 255, 0, 200), (0, 0, 255, 50)]
    paths = []
    for i, color in enumerate(colors, 1):
        p = tmp_path / f"alpha_{i:02d}.png"
        img = Image.new("RGBA", (100, 100), color)
        img.save(p)
        paths.append(p)
    return paths


@pytest.fixture
def onset_folder(
    tmp_path: Path, mono_with_onsets: np.ndarray, sr: int
) -> Path:
    """Folder with 4 RGB images + 1 audio file (valid onset_video input)."""
    # Write audio
    sf.write(str(tmp_path / "beat.wav"), mono_with_onsets, sr)

    # Write images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, color in enumerate(colors, 1):
        img = Image.new("RGB", (100, 100), color)
        img.save(tmp_path / f"frame_{i:02d}.png")

    return tmp_path


@pytest.fixture
def onset_folder_rgba(
    tmp_path: Path, mono_with_onsets: np.ndarray, sr: int
) -> Path:
    """Folder with 3 RGBA images + 1 audio file."""
    sf.write(str(tmp_path / "beat.wav"), mono_with_onsets, sr)

    colors = [(255, 0, 0, 128), (0, 255, 0, 200), (0, 0, 255, 50)]
    for i, color in enumerate(colors, 1):
        img = Image.new("RGBA", (100, 100), color)
        img.save(tmp_path / f"alpha_{i:02d}.png")

    return tmp_path


@pytest.fixture
def no_audio_folder(tmp_path: Path) -> Path:
    """Folder with images but no audio file."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    img.save(tmp_path / "frame_01.png")
    return tmp_path


@pytest.fixture
def no_images_folder(tmp_path: Path, mono_with_onsets: np.ndarray, sr: int) -> Path:
    """Folder with audio but no images."""
    sf.write(str(tmp_path / "beat.wav"), mono_with_onsets, sr)
    return tmp_path


@pytest.fixture
def multi_audio_folder(
    tmp_path: Path, mono_with_onsets: np.ndarray, sr: int
) -> Path:
    """Folder with images + multiple audio files (ambiguous)."""
    sf.write(str(tmp_path / "beat1.wav"), mono_with_onsets, sr)
    sf.write(str(tmp_path / "beat2.wav"), mono_with_onsets, sr)
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    img.save(tmp_path / "frame_01.png")
    return tmp_path
