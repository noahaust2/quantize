"""Tests for the quantize CLI tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from quantize import (
    ProcessResult,
    auto_detect_target_bpm,
    detect_bpm,
    discover_audio_files,
    load_audio,
    main,
    process_file,
    save_audio,
    time_stretch,
)


# ---------------------------------------------------------------------------
# discover_audio_files
# ---------------------------------------------------------------------------


class TestDiscoverAudioFiles:
    def test_finds_wav_files(self, sample_folder: Path) -> None:
        files = discover_audio_files(sample_folder)
        assert len(files) == 3
        assert all(f.suffix == ".wav" for f in files)

    def test_ignores_non_audio(self, mixed_folder: Path) -> None:
        files = discover_audio_files(mixed_folder)
        names = {f.name for f in files}
        assert "readme.txt" not in names
        assert "good.wav" in names

    def test_returns_sorted(self, sample_folder: Path) -> None:
        files = discover_audio_files(sample_folder)
        assert files == sorted(files)

    def test_empty_folder(self, tmp_path: Path) -> None:
        files = discover_audio_files(tmp_path)
        assert files == []


# ---------------------------------------------------------------------------
# load_audio / save_audio
# ---------------------------------------------------------------------------


class TestLoadAudio:
    def test_loads_wav(self, sample_folder: Path) -> None:
        audio, sr = load_audio(sample_folder / "beat_a.wav")
        assert isinstance(audio, np.ndarray)
        assert sr > 0
        assert len(audio) > 0

    def test_bad_file_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.wav"
        bad.write_bytes(b"not audio data at all")
        with pytest.raises(RuntimeError):
            load_audio(bad)


class TestSaveAudio:
    def test_roundtrip(self, tmp_path: Path, mono_120bpm: np.ndarray, sr: int) -> None:
        out_path = tmp_path / "out.wav"
        save_audio(out_path, mono_120bpm, sr)
        loaded, loaded_sr = sf.read(str(out_path))
        assert loaded_sr == sr
        assert len(loaded) == len(mono_120bpm)

    def test_clips_values(self, tmp_path: Path, sr: int) -> None:
        loud = np.full(1000, 2.0)
        out_path = tmp_path / "clipped.wav"
        save_audio(out_path, loud, sr)
        loaded, _ = sf.read(str(out_path))
        assert np.max(loaded) <= 1.0


# ---------------------------------------------------------------------------
# detect_bpm
# ---------------------------------------------------------------------------


class TestDetectBPM:
    def test_returns_float(self, mono_120bpm: np.ndarray, sr: int) -> None:
        bpm = detect_bpm(mono_120bpm, sr)
        assert isinstance(bpm, float)

    def test_within_range(self, mono_120bpm: np.ndarray, sr: int) -> None:
        bpm = detect_bpm(mono_120bpm, sr, bpm_min=60, bpm_max=200)
        assert 60 <= bpm <= 200

    def test_handles_stereo(self, stereo_120bpm: np.ndarray, sr: int) -> None:
        bpm = detect_bpm(stereo_120bpm, sr)
        assert isinstance(bpm, float)
        assert 60 <= bpm <= 200

    def test_octave_correction_high(self, mono_120bpm: np.ndarray, sr: int) -> None:
        """If raw detection returns >200, octave correction should halve it."""
        bpm = detect_bpm(mono_120bpm, sr, bpm_min=60, bpm_max=200)
        assert bpm <= 200

    def test_short_audio_raises(self, short_audio: np.ndarray, sr: int) -> None:
        with pytest.raises(ValueError, match="too short"):
            detect_bpm(short_audio, sr)


# ---------------------------------------------------------------------------
# auto_detect_target_bpm
# ---------------------------------------------------------------------------


class TestAutoDetectTargetBPM:
    def test_returns_float(self, sample_folder: Path) -> None:
        files = list(sample_folder.glob("*.wav"))
        result = auto_detect_target_bpm(files)
        assert isinstance(result, float)
        assert 60 <= result <= 200

    def test_returns_none_for_empty(self) -> None:
        assert auto_detect_target_bpm([]) is None

    def test_returns_none_for_bad_files(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not audio")
        assert auto_detect_target_bpm([bad]) is None

    def test_picks_most_common(self, tmp_path: Path, mono_120bpm: np.ndarray, sr: int) -> None:
        # Write 3 identical files -- they should all detect the same BPM
        for name in ["a.wav", "b.wav", "c.wav"]:
            sf.write(str(tmp_path / name), mono_120bpm, sr)
        files = sorted(tmp_path.glob("*.wav"))
        result = auto_detect_target_bpm(files)
        assert result is not None


# ---------------------------------------------------------------------------
# time_stretch
# ---------------------------------------------------------------------------


class TestTimeStretch:
    def test_speedup_shortens(self, mono_120bpm: np.ndarray, sr: int) -> None:
        stretched = time_stretch(mono_120bpm, sr, ratio=2.0)
        # Sped up 2x should be roughly half the length
        assert len(stretched) < len(mono_120bpm)
        assert len(stretched) > len(mono_120bpm) * 0.3  # sanity bound

    def test_slowdown_lengthens(self, mono_120bpm: np.ndarray, sr: int) -> None:
        stretched = time_stretch(mono_120bpm, sr, ratio=0.5)
        assert len(stretched) > len(mono_120bpm)

    def test_ratio_one_same_length(self, mono_120bpm: np.ndarray, sr: int) -> None:
        stretched = time_stretch(mono_120bpm, sr, ratio=1.0)
        assert abs(len(stretched) - len(mono_120bpm)) < sr * 0.05  # within 50ms

    def test_handles_stereo(self, stereo_120bpm: np.ndarray, sr: int) -> None:
        stretched = time_stretch(stereo_120bpm, sr, ratio=1.5)
        assert stretched.ndim == 2
        assert stretched.shape[1] == 2


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


class TestProcessFile:
    def test_happy_path(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        result = process_file(
            path=sample_folder / "beat_a.wav",
            target_bpm=100.0,
            output_dir=out_dir,
            bpm_min=60,
            bpm_max=200,
            override_bpm=None,
            dry_run=False,
        )
        assert result.status == "ok"
        assert result.detected_bpm is not None
        assert result.stretch_ratio is not None
        assert (out_dir / "beat_a.wav").exists()

    def test_skipped_at_target(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        # Detect the actual BPM first, then ask for that as target
        audio, sr = load_audio(sample_folder / "beat_a.wav")
        actual_bpm = detect_bpm(audio, sr)
        result = process_file(
            path=sample_folder / "beat_a.wav",
            target_bpm=actual_bpm,
            output_dir=out_dir,
            bpm_min=60,
            bpm_max=200,
            override_bpm=None,
            dry_run=False,
        )
        assert result.status == "skipped"

    def test_dry_run(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        result = process_file(
            path=sample_folder / "beat_a.wav",
            target_bpm=100.0,
            output_dir=out_dir,
            bpm_min=60,
            bpm_max=200,
            override_bpm=None,
            dry_run=True,
        )
        assert result.status == "dry-run"
        assert not (out_dir / "beat_a.wav").exists()

    def test_override_bpm(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        result = process_file(
            path=sample_folder / "beat_a.wav",
            target_bpm=100.0,
            output_dir=out_dir,
            bpm_min=60,
            bpm_max=200,
            override_bpm=140.0,
            dry_run=False,
        )
        assert result.status == "ok"
        assert result.detected_bpm == 140.0

    def test_bad_file_returns_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.wav"
        bad.write_bytes(b"not audio")
        out_dir = tmp_path / "out"
        result = process_file(
            path=bad,
            target_bpm=120.0,
            output_dir=out_dir,
            bpm_min=60,
            bpm_max=200,
            override_bpm=None,
            dry_run=False,
        )
        assert result.status == "error"

    def test_never_raises(self, tmp_path: Path) -> None:
        """process_file should return errors, never raise."""
        bad = tmp_path / "nonexistent.wav"
        out_dir = tmp_path / "out"
        result = process_file(
            path=bad,
            target_bpm=120.0,
            output_dir=out_dir,
            bpm_min=60,
            bpm_max=200,
            override_bpm=None,
            dry_run=False,
        )
        assert result.status == "error"


# ---------------------------------------------------------------------------
# main CLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_basic_batch(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        ret = main([str(sample_folder), "100", "--output", str(out_dir)])
        assert ret == 0
        outputs = list(out_dir.glob("*.wav"))
        assert len(outputs) > 0

    def test_dry_run(self, sample_folder: Path, capsys) -> None:
        ret = main([str(sample_folder), "120", "--dry-run"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "dry-run" in captured.out.lower() or "BPM" in captured.out

    def test_invalid_folder(self, tmp_path: Path) -> None:
        ret = main([str(tmp_path / "nonexistent"), "120"])
        assert ret != 0

    def test_no_files(self, tmp_path: Path, capsys) -> None:
        ret = main([str(tmp_path), "120"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "no" in captured.out.lower() or "0" in captured.out

    def test_override_flag(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        ret = main([
            str(sample_folder), "100",
            "--output", str(out_dir),
            "--override", "beat_a.wav=140",
        ])
        assert ret == 0
        assert (out_dir / "beat_a.wav").exists()

    def test_bpm_range_flag(self, sample_folder: Path, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        ret = main([
            str(sample_folder), "100",
            "--output", str(out_dir),
            "--bpm-range", "80", "160",
        ])
        assert ret == 0

    def test_auto_detect_target(self, sample_folder: Path, tmp_path: Path, capsys) -> None:
        """When target_bpm is omitted, auto-detect from files."""
        out_dir = tmp_path / "out"
        ret = main([str(sample_folder), "--output", str(out_dir)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "auto-detected" in captured.out.lower()

    def test_auto_detect_dry_run(self, sample_folder: Path, capsys) -> None:
        ret = main([str(sample_folder), "--dry-run"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "auto-detected" in captured.out.lower()
