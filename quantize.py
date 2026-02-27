"""quantize -- Batch BPM detection and time-stretching CLI tool.

Takes a folder of audio clips, detects their BPM, and time-stretches
them all to a target BPM without changing pitch.

Usage:
    python quantize.py ./samples                          # auto-detect target BPM
    python quantize.py ./samples 120                      # explicit target BPM
    python quantize.py ./samples --output ./output
    python quantize.py ./samples --dry-run
    python quantize.py ./samples --bpm-range 80 160
    python quantize.py ./samples 120 --override kick.wav=140
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
DEFAULT_BPM_MIN = 60.0
DEFAULT_BPM_MAX = 200.0
BPM_TOLERANCE = 0.5  # Skip if within this many BPM of target
MIN_DURATION_SEC = 1.0  # Minimum duration for reliable BPM detection


@dataclass
class ProcessResult:
    """Result of processing a single audio file."""

    filename: str
    detected_bpm: float | None
    target_bpm: float
    stretch_ratio: float | None
    status: str  # "ok", "skipped", "error", "dry-run"
    message: str


def discover_audio_files(folder: Path) -> list[Path]:
    """Find all supported audio files in a folder (non-recursive, sorted)."""
    files = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file, returning (samples, sample_rate).

    Uses soundfile as primary loader. Falls back to librosa for
    formats soundfile can't handle (e.g. some mp3 files).

    Raises:
        RuntimeError: If the file cannot be loaded.
    """
    try:
        audio, sr = sf.read(str(path), dtype="float64", always_2d=False)
        return audio, sr
    except Exception:
        pass

    try:
        audio, sr = librosa.load(str(path), sr=None, mono=False)
        if audio.ndim == 2:
            audio = audio.T  # librosa returns (channels, samples), we want (samples, channels)
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Cannot load {path.name}: {e}") from e


def detect_bpm(
    audio: np.ndarray,
    sr: int,
    bpm_min: float = DEFAULT_BPM_MIN,
    bpm_max: float = DEFAULT_BPM_MAX,
) -> float:
    """Detect BPM of audio, constrained to a plausible range.

    Applies octave correction: if the raw result is outside
    [bpm_min, bpm_max], doubles or halves it until it fits.

    Raises:
        ValueError: If audio is too short for BPM detection.
    """
    # Convert stereo to mono
    if audio.ndim > 1:
        mono = librosa.to_mono(audio.T)
    else:
        mono = audio

    duration = len(mono) / sr
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short for BPM detection ({duration:.2f}s, need {MIN_DURATION_SEC}s)"
        )

    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    # librosa may return an array in some versions
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    # Octave correction
    while tempo > bpm_max and tempo > 0:
        tempo /= 2.0
    while tempo < bpm_min and tempo > 0:
        tempo *= 2.0

    return max(bpm_min, min(bpm_max, tempo))


def auto_detect_target_bpm(
    files: list[Path],
    bpm_min: float = DEFAULT_BPM_MIN,
    bpm_max: float = DEFAULT_BPM_MAX,
) -> float | None:
    """Scan files and return the most common BPM (rounded to nearest integer).

    Returns None if no files could be analyzed.
    """
    from collections import Counter

    bpms: list[int] = []
    for path in files:
        try:
            audio, sr = load_audio(path)
            bpm = detect_bpm(audio, sr, bpm_min, bpm_max)
            bpms.append(round(bpm))
        except Exception:
            continue

    if not bpms:
        return None

    # Most common rounded BPM
    counter = Counter(bpms)
    most_common_bpm, _ = counter.most_common(1)[0]
    return float(most_common_bpm)


_rubberband_available: bool | None = None


def has_rubberband() -> bool:
    """Check if pyrubberband and the rubberband binary are available. Cached."""
    global _rubberband_available
    if _rubberband_available is None:
        try:
            import pyrubberband as pyrb

            test = np.zeros(2048, dtype=np.float64)
            pyrb.time_stretch(test, sr=22050, rate=1.5)
            _rubberband_available = True
        except Exception:
            _rubberband_available = False
    return _rubberband_available


def time_stretch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Time-stretch audio by the given ratio. Pitch is preserved.

    Uses Rubber Band if available (higher quality), otherwise falls back
    to librosa's phase vocoder.

    ratio > 1.0 = speed up, ratio < 1.0 = slow down.
    """
    if has_rubberband():
        import pyrubberband as pyrb

        if audio.ndim == 1:
            return pyrb.time_stretch(audio, sr=sr, rate=ratio)
        channels = [
            pyrb.time_stretch(audio[:, ch], sr=sr, rate=ratio)
            for ch in range(audio.shape[1])
        ]
        min_len = min(len(ch) for ch in channels)
        return np.column_stack([ch[:min_len] for ch in channels])

    # Fallback: librosa phase vocoder
    if audio.ndim == 1:
        return librosa.effects.time_stretch(audio, rate=ratio)
    channels = [
        librosa.effects.time_stretch(audio[:, ch], rate=ratio)
        for ch in range(audio.shape[1])
    ]
    min_len = min(len(ch) for ch in channels)
    return np.column_stack([ch[:min_len] for ch in channels])


def save_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    """Save audio to wav. Clips to [-1.0, 1.0] to prevent distortion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), clipped, sr)


def process_file(
    path: Path,
    target_bpm: float,
    output_dir: Path,
    bpm_min: float,
    bpm_max: float,
    override_bpm: float | None,
    dry_run: bool,
    tolerance: float = BPM_TOLERANCE,
) -> ProcessResult:
    """Process a single audio file. Never raises -- errors become ProcessResult."""
    filename = path.name

    try:
        audio, sr = load_audio(path)
    except RuntimeError as e:
        return ProcessResult(
            filename=filename,
            detected_bpm=None,
            target_bpm=target_bpm,
            stretch_ratio=None,
            status="error",
            message=str(e),
        )

    if override_bpm is not None:
        detected_bpm = override_bpm
    else:
        try:
            detected_bpm = detect_bpm(audio, sr, bpm_min, bpm_max)
        except ValueError as e:
            return ProcessResult(
                filename=filename,
                detected_bpm=None,
                target_bpm=target_bpm,
                stretch_ratio=None,
                status="error",
                message=str(e),
            )

    ratio = target_bpm / detected_bpm

    if abs(detected_bpm - target_bpm) < tolerance:
        return ProcessResult(
            filename=filename,
            detected_bpm=detected_bpm,
            target_bpm=target_bpm,
            stretch_ratio=ratio,
            status="skipped",
            message="Already at target BPM",
        )

    if dry_run:
        return ProcessResult(
            filename=filename,
            detected_bpm=detected_bpm,
            target_bpm=target_bpm,
            stretch_ratio=ratio,
            status="dry-run",
            message="",
        )

    try:
        stretched = time_stretch(audio, sr, ratio)
    except Exception as e:
        return ProcessResult(
            filename=filename,
            detected_bpm=detected_bpm,
            target_bpm=target_bpm,
            stretch_ratio=ratio,
            status="error",
            message=f"Time-stretch failed: {e}",
        )

    out_path = output_dir / (path.stem + ".wav")
    try:
        save_audio(out_path, stretched, sr)
    except Exception as e:
        return ProcessResult(
            filename=filename,
            detected_bpm=detected_bpm,
            target_bpm=target_bpm,
            stretch_ratio=ratio,
            status="error",
            message=f"Save failed: {e}",
        )

    return ProcessResult(
        filename=filename,
        detected_bpm=detected_bpm,
        target_bpm=target_bpm,
        stretch_ratio=ratio,
        status="ok",
        message="",
    )


def print_summary(results: list[ProcessResult]) -> None:
    """Print a formatted table of processing results."""
    if not results:
        print("No files processed.")
        return

    # Column widths
    max_name = max(len(r.filename) for r in results)
    name_w = max(max_name, 4)

    header = (
        f"  {'File':<{name_w}}   {'Detected':>8}   {'Target':>6}   {'Ratio':>6}   Status"
    )
    sep = "  " + "-" * (len(header) - 2)

    print()
    print(header)
    print(sep)

    for r in results:
        bpm_str = f"{r.detected_bpm:.1f}" if r.detected_bpm is not None else "--"
        ratio_str = f"{r.stretch_ratio:.3f}" if r.stretch_ratio is not None else "--"
        status_str = r.status
        if r.message:
            status_str += f" ({r.message})"

        print(
            f"  {r.filename:<{name_w}}   {bpm_str:>8}   {r.target_bpm:>6.1f}   {ratio_str:>6}   {status_str}"
        )

    print()

    ok = sum(1 for r in results if r.status == "ok")
    skipped = sum(1 for r in results if r.status == "skipped")
    dry = sum(1 for r in results if r.status == "dry-run")
    errors = sum(1 for r in results if r.status == "error")

    parts = []
    if ok:
        parts.append(f"{ok} ok")
    if skipped:
        parts.append(f"{skipped} skipped")
    if dry:
        parts.append(f"{dry} dry-run")
    if errors:
        parts.append(f"{errors} errors")
    print(f"  Done: {', '.join(parts)}")
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantize",
        description="Batch-detect BPM and time-stretch audio files to a target BPM.",
        epilog=(
            "Examples:\n"
            "  python quantize.py ./samples                          # auto-detect target\n"
            "  python quantize.py ./samples 120                      # explicit target\n"
            "  python quantize.py ./samples --output ./output\n"
            "  python quantize.py ./samples --dry-run\n"
            "  python quantize.py ./samples --bpm-range 80 160\n"
            '  python quantize.py ./samples 120 --override kick.wav=140\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input_folder", help="Folder containing audio files")
    parser.add_argument(
        "target_bpm", type=float, nargs="?", default=None,
        help="Target BPM (omit to auto-detect from most common BPM in folder)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output folder (default: <input_folder>/quantized/)",
    )
    parser.add_argument(
        "--bpm-range",
        nargs=2,
        type=float,
        default=[DEFAULT_BPM_MIN, DEFAULT_BPM_MAX],
        metavar=("MIN", "MAX"),
        help=f"BPM detection range (default: {DEFAULT_BPM_MIN} {DEFAULT_BPM_MAX})",
    )
    parser.add_argument(
        "--override",
        action="append",
        metavar="FILE=BPM",
        help="Manual BPM override for a specific file (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show detected BPMs without processing",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=BPM_TOLERANCE,
        help=f"BPM tolerance for skipping (default: {BPM_TOLERANCE})",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on errors."""
    parser = build_parser()
    args = parser.parse_args(argv)

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        print(f"Error: {args.input_folder} is not a directory", file=sys.stderr)
        return 1

    bpm_min, bpm_max = args.bpm_range
    if bpm_min >= bpm_max:
        print("Error: BPM range min must be less than max", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_folder / "quantized"

    # Parse overrides
    overrides: dict[str, float] = {}
    for entry in args.override or []:
        if "=" not in entry:
            print(f"Error: invalid override format: {entry!r} (expected FILE=BPM)", file=sys.stderr)
            return 1
        name, bpm_str = entry.rsplit("=", 1)
        try:
            overrides[name] = float(bpm_str)
        except ValueError:
            print(f"Error: invalid BPM in override: {bpm_str!r}", file=sys.stderr)
            return 1

    # Discover files
    files = discover_audio_files(input_folder)
    if not files:
        print(f"No audio files found in {input_folder}")
        return 0

    # Determine target BPM
    target_bpm = args.target_bpm
    if target_bpm is not None and target_bpm <= 0:
        print("Error: target BPM must be positive", file=sys.stderr)
        return 1

    if target_bpm is None:
        print(f"\n  Scanning {len(files)} files to detect target BPM...")
        target_bpm = auto_detect_target_bpm(files, bpm_min, bpm_max)
        if target_bpm is None:
            print("Error: could not detect BPM from any file", file=sys.stderr)
            return 1
        print(f"  Auto-detected target: {target_bpm:.0f} BPM")

    # Report time-stretch engine
    if not args.dry_run:
        if has_rubberband():
            print("  Engine: Rubber Band (high quality)")
        else:
            print(
                "  Engine: librosa phase vocoder (install rubberband for better quality)\n"
                "    pip install pyrubberband + https://breakfastquay.com/rubberband/"
            )

    mode = "DRY RUN" if args.dry_run else "processing"
    print(f"\n  {mode}: {len(files)} files -> target {target_bpm} BPM\n")

    # Process
    tolerance = args.tolerance

    results: list[ProcessResult] = []
    for i, path in enumerate(files, 1):
        override = overrides.get(path.name)
        result = process_file(
            path=path,
            target_bpm=target_bpm,
            output_dir=output_dir,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            override_bpm=override,
            dry_run=args.dry_run,
            tolerance=tolerance,
        )
        results.append(result)

        # Progress line
        bpm_str = f"{result.detected_bpm:.1f}" if result.detected_bpm else "?"
        ratio_str = f"{result.stretch_ratio:.3f}" if result.stretch_ratio else "--"
        print(f"  [{i}/{len(files)}] {path.name} ... {bpm_str} BPM (ratio {ratio_str}) [{result.status}]")

    print_summary(results)

    has_errors = any(r.status == "error" for r in results)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
