"""Tests for onset_video module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from onset_video import (
    assign_images,
    build_parser,
    compute_sections,
    detect_onsets,
    discover_folder,
    has_alpha,
    main,
    natural_sort_key,
    process_folder,
)


class TestNaturalSort:
    def test_numeric_parts(self, tmp_path: Path) -> None:
        names = ["frame10.png", "frame2.png", "frame1.png", "frame20.png"]
        paths = [tmp_path / n for n in names]
        result = sorted(paths, key=natural_sort_key)
        assert [p.name for p in result] == [
            "frame1.png",
            "frame2.png",
            "frame10.png",
            "frame20.png",
        ]

    def test_mixed_alpha_numeric(self, tmp_path: Path) -> None:
        names = ["b2.png", "a10.png", "a2.png", "b1.png"]
        paths = [tmp_path / n for n in names]
        result = sorted(paths, key=natural_sort_key)
        assert [p.name for p in result] == [
            "a2.png",
            "a10.png",
            "b1.png",
            "b2.png",
        ]

    def test_case_insensitive(self, tmp_path: Path) -> None:
        names = ["Frame2.PNG", "frame1.png"]
        paths = [tmp_path / n for n in names]
        result = sorted(paths, key=natural_sort_key)
        assert [p.name for p in result] == ["frame1.png", "Frame2.PNG"]


class TestDiscoverFolder:
    def test_valid_folder(self, onset_folder: Path) -> None:
        images, audio = discover_folder(onset_folder)
        assert len(images) == 4
        assert audio.suffix == ".wav"
        # Check natural sort order
        assert images[0].name == "frame_01.png"
        assert images[-1].name == "frame_04.png"

    def test_no_images(self, no_images_folder: Path) -> None:
        with pytest.raises(ValueError, match="No image files"):
            discover_folder(no_images_folder)

    def test_no_audio(self, no_audio_folder: Path) -> None:
        with pytest.raises(ValueError, match="No audio file"):
            discover_folder(no_audio_folder)

    def test_multiple_audio(self, multi_audio_folder: Path) -> None:
        with pytest.raises(ValueError, match="Multiple audio files"):
            discover_folder(multi_audio_folder)


class TestDetectOnsets:
    def test_returns_onsets_and_duration(self, onset_folder: Path) -> None:
        audio_path = onset_folder / "beat.wav"
        onset_times, duration = detect_onsets(audio_path, sensitivity=0.5)
        assert isinstance(onset_times, np.ndarray)
        assert duration > 0
        # Should detect some onsets from the synthetic clicks
        assert len(onset_times) > 0

    def test_sensitivity_affects_count(self, onset_folder: Path) -> None:
        audio_path = onset_folder / "beat.wav"
        onsets_low, _ = detect_onsets(audio_path, sensitivity=0.0)
        onsets_high, _ = detect_onsets(audio_path, sensitivity=1.0)
        # Lower sensitivity should detect at least as many onsets
        assert len(onsets_low) >= len(onsets_high)


class TestComputeSections:
    def test_with_pre_onset(self) -> None:
        onsets = np.array([1.0, 2.0, 3.0])
        sections = compute_sections(onsets, 4.0, pre_onset=True)
        assert len(sections) == 4
        assert sections[0] == (0.0, 1.0)
        assert sections[1] == (1.0, 2.0)
        assert sections[2] == (2.0, 3.0)
        assert sections[3] == (3.0, 4.0)

    def test_without_pre_onset(self) -> None:
        onsets = np.array([1.0, 2.0, 3.0])
        sections = compute_sections(onsets, 4.0, pre_onset=False)
        assert len(sections) == 3
        assert sections[0] == (1.0, 2.0)
        assert sections[1] == (2.0, 3.0)
        assert sections[2] == (3.0, 4.0)

    def test_no_onsets(self) -> None:
        onsets = np.array([])
        sections = compute_sections(onsets, 5.0)
        assert sections == [(0.0, 5.0)]

    def test_single_onset(self) -> None:
        onsets = np.array([2.5])
        sections = compute_sections(onsets, 5.0, pre_onset=True)
        assert len(sections) == 2
        assert sections[0] == (0.0, 2.5)
        assert sections[1] == (2.5, 5.0)


class TestAssignImages:
    @pytest.fixture
    def four_images(self, tmp_path: Path) -> list[Path]:
        return [tmp_path / f"img{i}.png" for i in range(4)]

    @pytest.fixture
    def sections_12(self) -> list[tuple[float, float]]:
        return [(float(i), float(i + 1)) for i in range(12)]

    def test_distribute_even(
        self, four_images: list[Path], sections_12: list[tuple[float, float]]
    ) -> None:
        result = assign_images(four_images, sections_12, mode="distribute")
        assert len(result) == 12
        # Each image should get 3 consecutive sections
        for i in range(3):
            assert result[i][0] == four_images[0]
        for i in range(3, 6):
            assert result[i][0] == four_images[1]
        for i in range(6, 9):
            assert result[i][0] == four_images[2]
        for i in range(9, 12):
            assert result[i][0] == four_images[3]

    def test_distribute_uneven(self, tmp_path: Path) -> None:
        images = [tmp_path / f"img{i}.png" for i in range(3)]
        sections = [(float(i), float(i + 1)) for i in range(7)]
        result = assign_images(images, sections, mode="distribute")
        assert len(result) == 7
        # numpy.array_split distributes as evenly as possible
        # 7 sections / 3 images -> chunks of sizes 3, 2, 2
        img_counts = {}
        for img, _, _ in result:
            img_counts[img] = img_counts.get(img, 0) + 1
        counts = sorted(img_counts.values(), reverse=True)
        assert counts == [3, 2, 2]

    def test_distribute_more_images_than_sections(self, tmp_path: Path) -> None:
        images = [tmp_path / f"img{i}.png" for i in range(5)]
        sections = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        result = assign_images(images, sections, mode="distribute")
        assert len(result) == 3
        # Only first 3 images used
        assert result[0][0] == images[0]
        assert result[1][0] == images[1]
        assert result[2][0] == images[2]

    def test_cycle(
        self, four_images: list[Path], sections_12: list[tuple[float, float]]
    ) -> None:
        result = assign_images(four_images, sections_12, mode="cycle")
        assert len(result) == 12
        for i, (img, _, _) in enumerate(result):
            assert img == four_images[i % 4]

    def test_truncate(
        self, four_images: list[Path], sections_12: list[tuple[float, float]]
    ) -> None:
        result = assign_images(four_images, sections_12, mode="truncate")
        assert len(result) == 4

    def test_truncate_fewer_sections(self, tmp_path: Path) -> None:
        images = [tmp_path / f"img{i}.png" for i in range(10)]
        sections = [(0.0, 1.0), (1.0, 2.0)]
        result = assign_images(images, sections, mode="truncate")
        assert len(result) == 2

    def test_invalid_mode(self, four_images: list[Path]) -> None:
        with pytest.raises(ValueError, match="Unknown mismatch mode"):
            assign_images(four_images, [(0.0, 1.0)], mode="invalid")


class TestHasAlpha:
    def test_rgb_no_alpha(self, sample_rgb_images: list[Path]) -> None:
        assert not has_alpha(sample_rgb_images[0])

    def test_rgba_has_alpha(self, sample_rgba_images: list[Path]) -> None:
        assert has_alpha(sample_rgba_images[0])


class TestProcessFolder:
    def test_dry_run(self, onset_folder: Path) -> None:
        result = process_folder(
            folder=onset_folder,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="auto",
            pre_onset=True,
            dry_run=True,
        )
        assert result.status == "dry-run"
        assert result.n_images == 4
        assert result.n_sections > 0

    def test_error_no_images(self, no_images_folder: Path) -> None:
        result = process_folder(
            folder=no_images_folder,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="auto",
            pre_onset=True,
            dry_run=True,
        )
        assert result.status == "error"
        assert "No image files" in result.message

    def test_error_no_audio(self, no_audio_folder: Path) -> None:
        result = process_folder(
            folder=no_audio_folder,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="auto",
            pre_onset=True,
            dry_run=True,
        )
        assert result.status == "error"
        assert "No audio file" in result.message

    def test_error_multi_audio(self, multi_audio_folder: Path) -> None:
        result = process_folder(
            folder=multi_audio_folder,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="auto",
            pre_onset=True,
            dry_run=True,
        )
        assert result.status == "error"
        assert "Multiple audio files" in result.message

    def test_never_raises(self, tmp_path: Path) -> None:
        # Process a completely empty folder -- should return error, not raise
        result = process_folder(
            folder=tmp_path,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="auto",
            pre_onset=True,
            dry_run=True,
        )
        assert result.status == "error"


def _has_ffmpeg() -> bool:
    import subprocess

    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


needs_ffmpeg = pytest.mark.skipif(
    not _has_ffmpeg(),
    reason="ffmpeg not found on PATH",
)


@needs_ffmpeg
class TestBuildVideo:
    """Integration test requiring ffmpeg."""

    def test_renders_h264(self, onset_folder: Path) -> None:
        result = process_folder(
            folder=onset_folder,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="h264",
            pre_onset=True,
            dry_run=False,
        )
        assert result.status == "ok", f"Expected ok, got: {result.message}"
        assert result.output_path is not None
        out = Path(result.output_path)
        assert out.exists()
        assert out.suffix == ".mp4"
        assert out.stat().st_size > 0

    def test_renders_prores_with_alpha(self, onset_folder_rgba: Path) -> None:
        result = process_folder(
            folder=onset_folder_rgba,
            output_dir=None,
            sensitivity=0.5,
            mismatch="distribute",
            fps=30,
            codec="prores4444",
            pre_onset=True,
            dry_run=False,
        )
        assert result.status == "ok", f"Expected ok, got: {result.message}"
        assert result.output_path is not None
        out = Path(result.output_path)
        assert out.exists()
        assert out.suffix == ".mov"
        assert out.stat().st_size > 0


class TestMainCLI:
    def test_dry_run_exit_code(self, onset_folder: Path) -> None:
        code = main([str(onset_folder), "--dry-run"])
        assert code == 0

    def test_invalid_folder(self, tmp_path: Path) -> None:
        code = main([str(tmp_path / "nonexistent")])
        assert code == 1

    def test_invalid_sensitivity(self, onset_folder: Path) -> None:
        code = main([str(onset_folder), "--sensitivity", "2.0", "--dry-run"])
        assert code == 1

    def test_parser_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["./folder1"])
        assert args.sensitivity == 0.5
        assert args.mismatch == "distribute"
        assert args.fps == 30
        assert args.codec == "auto"
        assert not args.no_pre_onset
        assert not args.dry_run

    def test_multiple_folders(self, onset_folder: Path, tmp_path: Path) -> None:
        # Create a second valid folder
        import soundfile as sf
        from PIL import Image as PILImage

        folder2 = tmp_path / "folder2"
        folder2.mkdir()
        # Reuse audio from onset_folder
        import shutil

        shutil.copy(onset_folder / "beat.wav", folder2 / "beat.wav")
        img = PILImage.new("RGB", (100, 100), (255, 0, 0))
        img.save(folder2 / "frame_01.png")

        code = main([str(onset_folder), str(folder2), "--dry-run"])
        assert code == 0
