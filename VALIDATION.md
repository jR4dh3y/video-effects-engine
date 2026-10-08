# Validation

Checked on Linux on 2026-10-08 using the locally installed Python, NumPy, SciPy, OpenCV and FFmpeg.

## Passed

- Python compilation of `engine.py`, `mix.py`, and `assemble.py`.
- End-to-end rendering with the committed `example.json` and generated three-second test footage with source audio. The test used CPU libx264 encoding, normal-speed time mapping, an impact marker, and fade transitions.
- Audio mixing and two-pass loudness normalization using generated music and source audio.
- Final assembly, including its expected frame-count assertion.
- FFprobe confirmed H.264 video at 1920x1080, 60 fps, 120 frames, AAC audio, and a 2.000-second output.
- Full FFmpeg decode of the final output completed without errors.
- Reviewed the extraction diff against the original engine and assembly scripts. The original montage workspace was not changed.
- Reviewed publishable files for original machine paths and credentials. Generated test media and caches are excluded by `.gitignore`.

## Not tested in this extraction

- RIFE interpolation, GPU selection, NVIDIA encoding, or installation on another machine.
- Multiple-play concatenation and cross-dissolves, stepped fast-forward, or every optional effect combination.
- Real-time viewing/listening or subjective quality of a real montage.
- A fresh dependency install in an isolated environment, Windows, or macOS.
- Integration into the recipient's engine, interactive preview, or any general editor functionality.

The supplied example verifies the portable rendering/mixing/export path. It does not establish that every inherited montage feature works with arbitrary media.
