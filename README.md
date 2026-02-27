# quantize

Two audio/video CLI tools:

- **quantize.py** — Batch-detect BPM and time-stretch audio files to a common tempo
- **onset_video.py** — Create videos from image sequences synced to audio transients

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) on PATH (required for video rendering)
- Optional: [Rubber Band](https://breakfastquay.com/rubberband/) for higher-quality time-stretching

## Install

```bash
pip install -r requirements.txt
```

## onset_video.py

Takes folders containing an image sequence + audio file. Detects transient points (onsets) in the audio and creates a video where each image is displayed in sync with those transients.

```bash
# Preview the onset mapping before rendering
python onset_video.py ./my_sequence --dry-run

# Render video (auto-detects alpha PNGs → ProRes 4444)
python onset_video.py ./seq1 ./seq2

# Adjust onset sensitivity (0.0 = many onsets, 1.0 = few)
python onset_video.py ./seq1 --sensitivity 0.3

# Force ProRes 4444 for alpha channel preservation
python onset_video.py ./seq1 --codec prores4444
```

Each input folder should contain:
- Image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`)
- Exactly one audio file (`.wav`, `.mp3`, `.flac`, `.ogg`)

## quantize.py

```bash
python quantize.py ./samples              # auto-detect target BPM
python quantize.py ./samples 120          # explicit target BPM
python quantize.py ./samples --dry-run    # preview without processing
```

## Tests

```bash
python -m pytest
```

Note: Video rendering tests require ffmpeg and are automatically skipped if it's not available.
