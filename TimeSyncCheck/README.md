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
