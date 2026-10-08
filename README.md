# Video effects engine

A Python footage renderer extracted from Radhey's CS2 montage project. Includes smooth time remapping, RIFE slow motion, optical-flow motion blur, stepped long-exposure fast-forward, grading, zoom, shake, flash, chromatic aberration, and transitions. The companion scripts mix audio and assemble rendered pieces.

This is an experimental render backend, not an interactive editor. See [HANDOFF.md](HANDOFF.md) for the discussion and integration direction.

## Setup

Linux is the supported starting point. Install Python 3.11 or newer and FFmpeg with ffprobe and libx264 available, then:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
mkdir -p clips audio
```

Supply your own media. Put a video with an audio track, at least three seconds long, at `clips/example.mp4`, and music at `audio/song.wav`. No footage, soundtrack, third-party binaries, or model weights are included.

```sh
python engine.py example.json 0
EDIT=example python mix.py
EDIT=example OUTNAME=example-final.mp4 python assemble.py
```

The result is `render/example/example-final.mp4`. Output is fixed at 1920x1080, 60 fps. The example runs at normal speed and does not require RIFE. Other aspect ratios are stretched by the current decoder.

## Slow motion and encoding

When a speed curve falls below 0.92x, the engine requires an external `rife-ncnn-vulkan` executable and model. Obtain a compatible build from the upstream [RIFE ncnn Vulkan project](https://github.com/nihui/rife-ncnn-vulkan) or its [TNTwise fork](https://github.com/TNTwise/rife-ncnn-vulkan). The original project used the latter's `rife-v4.26` model. Configure the model actually available in your installation; builds and model formats must match.

```sh
export RIFE_BIN=/absolute/path/to/rife-ncnn-vulkan
export RIFE_MODEL=/absolute/path/to/rife-v4.26
export RIFE_GPU=0
```

RIFE runs from the executable's directory. Its output must use one-based `%06d.png` numbering, as expected by `dense_frame`. It uses a Linux file lock to serialize interpolation. Windows support is untested and requires replacing `fcntl` locking.

Encoding defaults to CPU `libx264`. To use NVIDIA encoding, set `VIDEO_CODEC=h264_nvenc` for both rendering and assembly. This requires a compatible GPU, driver, and FFmpeg build.

## Edit format and integration

`index.txt` maps clip IDs to filenames as `id|filename`. IDs currently need to be numeric because they seed deterministic shake. Set `CLIPS_DIR` and `CLIP_INDEX` to use other media directories and index files.

Each play has:

- `out`: start and end in global output seconds.
- `anchors`: strictly increasing `[output_seconds, source_seconds]` pairs. PCHIP interpolates output time to source time.
- `kills`: `[output_seconds, source_seconds]` impact markers. The name reflects the original montage; these can represent other events.
- `kill_strength`: one strength per marker, and `audio_lag`: source audio offset in seconds.
- `tin` / `tout`: `cut`, `whip`, `flash`, `fade`, or `blend`. Blend is applied by assembly and requires overlapping adjacent plays.
- Optional `bw`: output-time monochrome intervals, `zoomk`: `[output_seconds, scale]` pairs, `zoom`, and `fade_frames`.

To use the frame renderer directly:

```python
import engine

engine.render_play(spec, start_frame, frame_count, output_path, work_directory)
```

Create output and `render/` directories before using this API with RIFE. `spec` uses the play format above. The CLI creates its directories automatically. Rendering only emits video; mixing and assembly are separate processes. The mixer reads `<EDIT>.json` from the repository directory and expects `audio/song.wav`.

## Limits

- Paths are configurable, but resolution, frame rate, grading and many effect settings remain global constants.
- Raw RGB caches can consume several GB per play. RIFE also writes intermediate PNGs.
- `.done` cache markers do not validate media or settings. Use a new work directory after changing inputs. The CLI removes a play's work directory after success.
- RIFE and motion blur can produce artifacts around occlusions, HUD elements and cuts.
- The audio mixer requires source audio and enough samples around impact markers. It is montage-specific and uses varispeed audio rather than pitch-preserving stretching.
- No timeline schema validation, interactive preview, undo, layers, captions, cancellation, or robust job recovery yet.
- Original CS2 watermark boxes are disabled. `CS2_DELOGO=1` restores them only for the original clip IDs; do not use this for arbitrary footage.

The MIT license covers this repository's code. External tools, models, and media retain their own licenses.
