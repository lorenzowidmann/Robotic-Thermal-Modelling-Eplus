# TimeSyncCheck

Post-processing tool that synchronizes FLIR thermal, ZED RGB, and LiDAR pose
streams into a triplet manifest. No live capture.

## Pipeline

**Stage 1 — manual FLIR<->ZED event sync**
FLIR and ZED are not hardware-synced. A steppable viewer shows each stream so
you pick the one frame in each where a shared heat-source event (lighter /
heat gun / hot object) is visible. The FLIR<->ZED time offset is the
difference of the two selected frames' own timestamps, saved to
`<session>/flir_zed_offset.json` for reuse.

Because the offset is derived from a shared physical event, it absorbs any
constant clock/timezone difference between the two cameras -- FLIR timestamps
only need to be internally consistent, not absolutely correct.

**Stage 2 — triplet manifest**
Using the Stage-1 offset and the LiDAR<->ZED relationship
(`--lidar-zed-offset`), each FLIR frame (reference stream) is matched to the
nearest ZED PNG and the nearest LiDAR `/Odometry` pose on a common clock (the
ZED clock). Each triplet records the three paths/timestamps, the LiDAR pose,
the pairwise time deltas after correction, and a match-confidence flag when
any delta exceeds `--max-delta`. Output is JSON in the session folder for
downstream tools (e.g. EmissivityCalculation) to consume.

This tool only produces the manifest: no emissivity estimation, radiometric
temperature conversion, or point-cloud fusion/coloring happens here.

> **Note:** the LiDAR<->ZED clock relationship is assumed to be a shared host
> clock (offset 0) until verified on the rig -- see `--lidar-zed-offset`.

## Pre-step -- `retime_fullrate_frames.py`

Run this **before** `sync_manifest.py` on any session whose ZED frames came
from `DataAcquisition/extract_fullrate_frames.py`.

That extractor re-times the mp4 on a uniform grid pinned to
`session.started_utc`, and both ends of the model are biased: the first grab
lands only after the UVC pipeline warms up, and `fps = (n-1)/duration_s` is
derived from a wall clock that also covers camera open/close -- so the error
grows over the session. The real grab loop jitters on top of that.

The recorder already measured the truth: every subsampled `frames/` PNG carries
its real `t_offset_s`, and every one of them is also a frame of the mp4. Matching
each PNG back to its mp4 frame index (NCC on a downscaled descriptor) gives
~one ground-truth anchor every `--frame-interval` seconds; the rest follows by
interpolation. Ambiguous anchors (static scene, many identical frames) are
dropped, not guessed.

Without this, the ZED clock wanders by several hundred ms against the LiDAR
clock -- the projected cloud lands on the wrong part of the RGB frame, by a
different amount at the start and at the end of the session (session 9: mean
error 0.35 s, worst 0.57 s -> 0.09 s / 0.10 s after retiming).

```
py retime_fullrate_frames.py --session-dir <ZED session dir>           # dry run
py retime_fullrate_frames.py --session-dir <ZED session dir> --apply
```

`--apply` backs the uniform version up as `fullrate/metadata.uniform.json`.
The ZED clock moves, so **regenerate `sync_manifest.json`** afterwards -- the
Stage-1 FLIR<->ZED event offset was calibrated on the old timestamps and has to
be recomputed (`--recompute-offset`).

## Setup

```
pip install -r requirements.txt
```

## Usage

```
py sync_manifest.py --session-dir recordings/20260726_140311 \
    --flir-dir "C:\...\FlyrCamera\20250823_211855" --bag path/to/rosbag2

# skip the Stage-1 viewer
py sync_manifest.py --session-dir <dir> --flir-dir <dir> --bag <dir> \
    --flir-event-frame 12 --zed-event-frame 3

py sync_manifest.py --session-dir <dir> --flir-dir <dir> --bag <dir> \
    --recompute-offset --max-delta 0.05
```

## Inputs

- `--session-dir` — ZED session folder from `zed_record.py` (holds
  `metadata.json` + `frames/`). The offset config and manifest are written
  here.
- `--flir-dir` — folder of FLIR radiometric `*_R.jpg` frames (as read by
  `RadiometricCalibration/ThermalData.py`).
- `--bag` — rosbag2 folder (`metadata.yaml` + `.db3`/`.mcap`) with the LiDAR
  odometry topic.
- `--odom-topic` (default `/Odometry`)
- `--store` (default `ROS2_HUMBLE`) — rosbags typestore for bags without
  embedded type defs.
- `--max-delta` (default `0.1`s) — max allowed time delta between any two
  streams in a triplet after correction; beyond it the triplet is flagged
  `low-confidence`.
- `--lidar-zed-offset` (default `0.0`) — seconds added to a LiDAR timestamp to
  put it on the ZED clock. UNVERIFIED assumption; measure on the rig.
- `--recompute-offset` — force Stage 1 even if `flir_zed_offset.json` exists.
- `--flir-event-frame` / `--zed-event-frame` — skip the Stage-1 viewer by
  giving event-frame indices directly.
- `--output` (default `sync_manifest.json`) — manifest filename, written
  inside `--session-dir`.

## Output

`<session-dir>/sync_manifest.json`: schema `sync_manifest/v1` with inputs,
sync offsets, matching summary, and one triplet per FLIR frame (flir/zed/lidar
timestamps + paths, LiDAR pose, deltas_s, match_status).

`<session-dir>/flir_zed_offset.json`: schema `flir_zed_offset/v1`, the Stage-1
result, reused on subsequent runs unless `--recompute-offset` is passed.
