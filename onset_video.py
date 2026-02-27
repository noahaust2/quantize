"""onset_video -- Create videos from image sequences synced to audio transients.

Takes one or more folders, each containing an image sequence and an audio file.
Detects onset (transient) points in the audio using librosa, divides the audio
timeline into sections, and creates a video where each image is displayed for
the duration of its assigned sections. Supports alpha channel preservation
via ProRes 4444.

Usage:
    python onset_video.py ./folder1 ./folder2
    python onset_video.py ./folder1 --dry-run
    python onset_video.py ./folder1 --sensitivity 0.3 --mismatch cycle
    python onset_video.py ./folder1 --codec prores4444
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def natural_sort_key(path: Path) -> list[str | int]:
    """Sort key that handles numeric parts correctly.

    frame2.png < frame10.png (not lexicographic where '10' < '2').
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def discover_folder(folder: Path) -> tuple[list[Path], Path]:
    """Find image sequence and audio file in a folder.

    Returns (sorted_image_files, audio_file).

    Raises:
        ValueError: If folder structure is invalid (no images, no audio,
                    or multiple audio files).
    """
    images = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]
    images.sort(key=natural_sort_key)

    audio_files = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if not images:
        raise ValueError(f"No image files found in {folder}")
    if not audio_files:
        raise ValueError(f"No audio file found in {folder}")
    if len(audio_files) > 1:
        names = ", ".join(f.name for f in audio_files)
        raise ValueError(
            f"Multiple audio files in {folder} (expected exactly 1): {names}"
        )

    return images, audio_files[0]


def detect_onsets(
    audio_path: Path,
    sensitivity: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Detect onset times in an audio file.

    Args:
        audio_path: Path to audio file.
        sensitivity: 0.0 (many onsets) to 1.0 (few onsets).

    Returns:
        (onset_times_seconds, total_duration_seconds)
    """
    audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = float(len(audio) / sr)

    delta = sensitivity * 0.3
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, delta=delta)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return onset_times, duration


def compute_sections(
    onset_times: np.ndarray,
    total_duration: float,
    pre_onset: bool = True,
) -> list[tuple[float, float]]:
    """Convert onset times to (start, end) section pairs.

    Args:
        onset_times: Array of onset timestamps in seconds.
        total_duration: Total audio duration in seconds.
        pre_onset: If True, first section is [0, first_onset).
                   If False, first section starts at first_onset.

    Returns:
        List of (start_time, end_time) tuples.
    """
    if len(onset_times) == 0:
        return [(0.0, total_duration)]

    boundaries: list[float] = []

    if pre_onset:
        boundaries.append(0.0)

    boundaries.extend(float(t) for t in onset_times)
    boundaries.append(total_duration)

    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end > start:
            sections.append((start, end))

    if not sections:
        return [(0.0, total_duration)]

    return sections


def assign_images(
    images: list[Path],
    sections: list[tuple[float, float]],
    mode: str = "distribute",
) -> list[tuple[Path, float, float]]:
    """Assign images to time sections based on mismatch mode.

    Args:
        images: Sorted list of image paths.
        sections: List of (start, end) time pairs.
        mode: "distribute", "cycle", or "truncate".

    Returns:
        List of (image_path, start_time, end_time) assignments.
    """
    n_images = len(images)
    n_sections = len(sections)

    if mode == "distribute":
        # Divide sections evenly among images.
        # If more images than sections, only use first N images.
        effective_images = min(n_images, n_sections)
        # Split section indices into chunks, one per image
        indices = np.arange(n_sections)
        chunks = np.array_split(indices, effective_images)

        assignments = []
        for img_idx, chunk in enumerate(chunks):
            for sec_idx in chunk:
                start, end = sections[sec_idx]
                assignments.append((images[img_idx], start, end))
        return assignments

    elif mode == "cycle":
        assignments = []
        for i, (start, end) in enumerate(sections):
            assignments.append((images[i % n_images], start, end))
        return assignments

    elif mode == "truncate":
        count = min(n_images, n_sections)
        return [
            (images[i], sections[i][0], sections[i][1]) for i in range(count)
        ]

    else:
        raise ValueError(f"Unknown mismatch mode: {mode!r}")


def has_alpha(image_path: Path) -> bool:
    """Check if an image has an alpha channel."""
    with Image.open(image_path) as img:
        return img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info


@dataclass
class FolderResult:
    """Result of processing a single folder."""

    folder: str
    n_images: int = 0
    n_sections: int = 0
    onset_times: list[float] = field(default_factory=list)
    output_path: str | None = None
    status: str = "ok"  # "ok", "error", "dry-run"
    message: str = ""


def build_video(
    assignments: list[tuple[Path, float, float]],
    audio_path: Path,
    output_path: Path,
    fps: int = 30,
    codec: str = "auto",
    alpha: bool = False,
) -> None:
    """Compose a video from image assignments with audio.

    Uses moviepy 2.0 API. Each image is displayed for its assigned duration.
    """
    from moviepy import AudioFileClip, ImageClip, concatenate_videoclips

    clips = []
    for image_path, start, end in assignments:
        duration = end - start
        if duration <= 0:
            continue
        clip = ImageClip(str(image_path)).with_duration(duration)
        clips.append(clip)

    if not clips:
        raise RuntimeError("No clips to render (all sections have zero duration)")

    video = concatenate_videoclips(clips, method="compose")

    audio_clip = AudioFileClip(str(audio_path))
    # Trim audio if video is shorter (truncate mode)
    if video.duration < audio_clip.duration:
        audio_clip = audio_clip.subclipped(0, video.duration)
    video = video.with_audio(audio_clip)

    use_prores = codec == "prores4444" or (codec == "auto" and alpha)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_prores:
        actual_path = output_path.with_suffix(".mov")
        video.write_videofile(
            str(actual_path),
            fps=fps,
            codec="prores_ks",
            ffmpeg_params=["-profile:v", "4444", "-pix_fmt", "yuva444p10le"],
            audio_codec="pcm_s16le",
            logger="bar",
        )
    else:
        actual_path = output_path.with_suffix(".mp4")
        video.write_videofile(
            str(actual_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )

    video.close()
    audio_clip.close()
    for clip in clips:
        clip.close()


def format_dry_run(
    folder: Path,
    audio_path: Path,
    images: list[Path],
    assignments: list[tuple[Path, float, float]],
    onset_times: np.ndarray,
    duration: float,
    sensitivity: float,
    mode: str,
    codec: str,
    alpha: bool,
) -> str:
    """Format a dry-run report for a folder."""
    codec_label = "ProRes 4444 (alpha)" if (codec == "prores4444" or (codec == "auto" and alpha)) else "H.264"
    alpha_str = " (RGBA — alpha detected)" if alpha else ""

    lines = [
        f"Folder: {folder}",
        f"  Audio: {audio_path.name} ({duration:.2f}s)",
        f"  Images: {len(images)} files{alpha_str}",
        f"  Onsets: {len(onset_times)} (sensitivity: {sensitivity:.2f})",
        f"  Codec: {codec_label}",
        f"  Mode: {mode}",
        "",
        f"  {'#':<4}  {'Start':<10}  {'End':<10}  {'Duration':<10}  Image",
    ]

    for i, (img_path, start, end) in enumerate(assignments, 1):
        dur = end - start
        # Check if this image is being reused (cycle/distribute indicator)
        img_idx = images.index(img_path) if img_path in images else -1
        suffix = ""
        if mode == "cycle" and i > len(images):
            suffix = "  (cycle)"
        elif mode == "distribute":
            # Check if same image as previous
            if i > 1 and assignments[i - 2][0] == img_path:
                suffix = ""  # same chunk, no label needed

        lines.append(
            f"  {i:<4}  {start:<10.3f}  {end:<10.3f}  {dur:<10.3f}  {img_path.name}{suffix}"
        )

    return "\n".join(lines)


def process_folder(
    folder: Path,
    output_dir: Path | None,
    sensitivity: float,
    mismatch: str,
    fps: int,
    codec: str,
    pre_onset: bool,
    dry_run: bool,
    verbose: bool = False,
) -> FolderResult:
    """Process a single folder end-to-end. Never raises."""
    result = FolderResult(folder=str(folder))

    try:
        images, audio_path = discover_folder(folder)
        result.n_images = len(images)
    except ValueError as e:
        result.status = "error"
        result.message = str(e)
        return result

    try:
        onset_times, duration = detect_onsets(audio_path, sensitivity)
        result.onset_times = onset_times.tolist()
    except Exception as e:
        result.status = "error"
        result.message = f"Onset detection failed: {e}"
        return result

    sections = compute_sections(onset_times, duration, pre_onset=pre_onset)
    result.n_sections = len(sections)

    assignments = assign_images(images, sections, mode=mismatch)

    alpha = has_alpha(images[0])

    if dry_run:
        report = format_dry_run(
            folder, audio_path, images, assignments, onset_times,
            duration, sensitivity, mismatch, codec, alpha,
        )
        print(report)
        result.status = "dry-run"
        return result

    # Determine output path
    if output_dir:
        out_path = output_dir / f"{folder.name}_onset"
    else:
        out_path = folder / "output"
    # Extension is set by build_video based on codec

    try:
        build_video(assignments, audio_path, out_path, fps, codec, alpha)
        use_prores = codec == "prores4444" or (codec == "auto" and alpha)
        ext = ".mov" if use_prores else ".mp4"
        result.output_path = str(out_path.with_suffix(ext))
        result.status = "ok"
    except Exception as e:
        result.status = "error"
        result.message = f"Video build failed: {e}"

    return result


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onset_video",
        description="Create videos from image sequences synced to audio transients.",
        epilog=(
            "Examples:\n"
            "  python onset_video.py ./seq1 ./seq2        # process two folders\n"
            "  python onset_video.py ./seq1 --dry-run     # preview onset mapping\n"
            "  python onset_video.py ./seq1 --sensitivity 0.3\n"
            "  python onset_video.py ./seq1 --codec prores4444\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "folders",
        nargs="+",
        help="One or more folders to process (each containing images + audio)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory (default: output file in each input folder)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Onset detection sensitivity 0.0-1.0 (default: 0.5). Lower = more onsets",
    )
    parser.add_argument(
        "--mismatch",
        choices=["distribute", "cycle", "truncate"],
        default="distribute",
        help="How to handle image/section count mismatch (default: distribute)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video framerate (default: 30)",
    )
    parser.add_argument(
        "--codec",
        choices=["auto", "h264", "prores4444"],
        default="auto",
        help="Video codec (default: auto — detects alpha for ProRes 4444)",
    )
    parser.add_argument(
        "--no-pre-onset",
        action="store_true",
        help="First image starts AT the first onset, not before it",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print onset/image mapping without rendering",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on errors."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate sensitivity range
    if not 0.0 <= args.sensitivity <= 1.0:
        print("Error: --sensitivity must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    # Check ffmpeg
    if not args.dry_run and not check_ffmpeg():
        print(
            "Error: ffmpeg not found on PATH. Install it: https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output) if args.output else None
    pre_onset = not args.no_pre_onset

    folders = [Path(f) for f in args.folders]
    for folder in folders:
        if not folder.is_dir():
            print(f"Error: {folder} is not a directory", file=sys.stderr)
            return 1

    results: list[FolderResult] = []
    for i, folder in enumerate(folders, 1):
        print(f"\n[{i}/{len(folders)}] Processing {folder}...")
        result = process_folder(
            folder=folder,
            output_dir=output_dir,
            sensitivity=args.sensitivity,
            mismatch=args.mismatch,
            fps=args.fps,
            codec=args.codec,
            pre_onset=pre_onset,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        results.append(result)

        if result.status == "ok":
            print(f"  -> {result.output_path}")
        elif result.status == "error":
            print(f"  ERROR: {result.message}", file=sys.stderr)

    # Summary
    print()
    ok = sum(1 for r in results if r.status == "ok")
    dry = sum(1 for r in results if r.status == "dry-run")
    errors = sum(1 for r in results if r.status == "error")

    parts = []
    if ok:
        parts.append(f"{ok} ok")
    if dry:
        parts.append(f"{dry} dry-run")
    if errors:
        parts.append(f"{errors} errors")
    print(f"Done: {', '.join(parts)}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
