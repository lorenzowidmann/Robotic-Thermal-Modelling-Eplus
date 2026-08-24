# Data Acquisition

ZED 2i capture for a rover session, plus a post-processing helper to recover
full-framerate frames from the recorded video. Pure recording tools: no
emissivity estimation, CLIP inference, or radiometric math lives here.

## Setup

```powershell
pip install -r requirements.txt   # opencv-python (uvc backend)
```

`--backend sdk` additionally needs the ZED SDK's `pyzed` (see
`requirements.txt`); `extract_fullrate_frames.py` needs `ffprobe` (ffmpeg) on
PATH.

## `zed_record.py` — ZED 2i recorder

Two capture backends (`--backend`):
- `sdk` — ZED SDK (pyzed). Full stereo master: native SVO2 + mp4 + PNG frames.
  Requires an NVIDIA GPU with CUDA.
- `uvc` — plain USB video (OpenCV), no SDK/GPU. RGB only, no SVO2/depth.
- `auto` (default) — try `sdk`, fall back to `uvc` if `pyzed` isn't importable.

```powershell
python zed_record.py --backend uvc --resolution HD1080 --frame-interval 0.4
```

Stops on Ctrl+C, or after `--duration SEC`. Key flags: `--resolution`
(HD2K/HD1080/HD720/VGA), `--fps`, `--eye left|right` (default right, to match
`EmissivityCalculation`'s CLIP classification), `--frame-interval` (seconds
between dumped PNGs; `0.0` default dumps every captured frame),
`--svo-compression h264|h265|lossless` (sdk only), `--no-mp4`/`--no-frames` to
skip an export.

Output: one timestamped session folder under `--output-dir` (default
`recordings/`) with `session.svo2` (sdk only), `session_<eye>.mp4`,
`frames/<eye>_NNNNNN.png`, and `metadata.json` (camera info, recording
settings, start/stop timestamps, per-frame offsets).

## `extract_fullrate_frames.py` — recover every frame, post-session

Extracts every frame of a session's `session_right.mp4` at the video's real
measured fps into a `fullrate/frames/` subfolder alongside the session's
existing subsampled `frames/` (left untouched), with a matching
`fullrate/metadata.json` (same `zed_record/v1` schema). Real fps is measured
from the decoded frame count (`ffprobe -count_frames`) rather than assumed
from the nominal `--fps` requested at record time, since the OpenCV/UVC
capture loop can jitter.

```powershell
python extract_fullrate_frames.py --session-dir <ZED session dir>
```

## Full recording session (LiDAR + ZED + FLIR)

Sequence to record one environment/facade scan session with all three sensors
and SLAM active. For the extrinsic-calibration session (9-pose board) the
procedure differs — FAST-LIO isn't needed there.

### Before starting

- **Turn the FLIR on manually** and start its own SD-card recording. It's the
  only sensor with no terminal command, so the easiest one to forget. Start it
  **before** the others and stop it **after**, so it covers the whole session.
- Check free disk space on the rover — at HD1080 the ZED PNGs are heavy.
- Optional but recommended: visually check the LiDAR framing with LivoxViewer2
  **before** starting the driver (the two can't run together, they contend for
  the sensor connection):
  ```bash
  sudo ifconfig enp89s0 192.168.1.50
  cd ~/LivoxViewer2
  ./LivoxViewer2.sh
  ```
  Close it completely before proceeding to T1.

### T1 — LiDAR driver

```bash
sudo ifconfig enp89s0 192.168.1.50
cd ~/ros2_ws
source install/setup.bash
ros2 launch livox_ros_driver2 msg_HAP_launch.py
```

`sudo ifconfig` sets the point-to-point subnet dedicated to the HAP. Only
needs re-running after a machine reboot.

### T2 — FAST-LIO (SLAM)

```bash
cd ~/ros2_ws
source install/setup.bash
ros2 launch fast_lio mapping.launch.py config_file:=haplidar.yaml
```

Opens RViz automatically — watch the map build up, so you notice immediately
if tracking loses lock during the scan.

### T3 — Bag recording

```bash
cd ~/Desktop/LorenzoWidmann/SLAM
source ~/ros2_ws/install/setup.bash
ros2 bag record --qos-profile-overrides-path qos.yaml /cloud_registered /Odometry /livox/lidar /livox/imu
```

Four topics, two categories:
- `/cloud_registered`, `/Odometry` — **FAST-LIO output** (registered cloud +
  estimated pose). The reason T2 runs.
- `/livox/lidar`, `/livox/imu` — **raw data**. Keep regardless — needed to
  reprocess offline with different parameters, or to run FAST-LIO-SAM (loop
  closure) later.

⚠️ `qos.yaml` with `reliability: best_effort` is **mandatory** for the Livox
topics: with the default `RELIABLE`, messages get silently dropped and the bag
comes out empty.

No `-o` — folder name is auto-generated from the timestamp.

### 🔴 T4 — LiDAR stream monitoring (MANDATORY)

```bash
ros2 topic hz /livox/lidar
```

⚠️ **Not optional.** Leave it open and watch it for the whole session. Expected
value: **~4.7 Hz**. If it stops printing values or says "no new messages", the
driver has died and the recording is continuing **empty**.

**Why** (lesson from `ExtCalibration_try4`, 2026-07-29): a session recorded for
90 seconds by the clock produced a bag with only **31 seconds of actual data**.
The LiDAR driver had restarted/duplicated mid-session (`new node registered
with same name` warning), the publisher died, but `ros2 bag record` kept
running without flagging anything. `ros2 bag record` **records silence without
complaint** — the stopwatch is not a check, `topic hz` is.

Also worth watching the SLAM output:
```bash
ros2 topic hz /Odometry
```

If `hz` behaves oddly, check the nodes are alive:
```bash
ros2 node list
```

### T5 — ZED recording

```bash
cd ~/Desktop/LorenzoWidmann/Code/Thesis/SensorFusion
source .venv/bin/activate
python3 zed_record.py --backend uvc --resolution HD1080 --frame-interval 0.4
```

Stops with **Ctrl+C** (no `--duration`).

⚠️ **`--resolution HD1080` must match the resolution of the intrinsics in
use.** The intrinsics redone on 2026-07-30 are at 1920x1080, so HD1080 is
correct. Recording at a resolution different from the calibration one skews
the K matrix (and therefore the fusion) **without raising any error**.

**On `--frame-interval`**: `0.4` = 2.5 fps. That was generous for calibration
(board held still 90s per pose), but during a scan **the rover is moving** and
each frame frames a different slice of the scene — at 2.5 fps you risk
coverage gaps between consecutive frames. Consider `0.1` (10 fps) if motion
isn't very slow — coverage can't be recovered in post-processing, disk space
can.

### After the recording

**Verify the bag before trusting it:**
```bash
ros2 bag info <bag_folder_name>
```
Check every topic has a message count > 0 (appearing in the list isn't
enough) and that the duration matches the actual capture time — if it's much
shorter, the stream dropped out somewhere (see T4).

**Copy the data to Windows**: use `rsync` instead of `cp` — shows progress and
resumes interrupted transfers:
```bash
rsync -ah --progress /source/path/ /destination/path/
```
⚠️ An interrupted copy leaves truncated files that *look* complete. That's
exactly what corrupted a `.db3` on 2026-07-29 (`database disk image is
malformed`), wasting time before realizing the problem was the transfer, not
the data.

**Verify the copy is complete**, comparing source and destination:
```bash
du -sh /source/path
du -sh /destination/path
find /source/path -type f | wc -l
find /destination/path -type f | wc -l
```

### Terminal summary

| # | What | Stops with |
|---|---|---|
| — | FLIR (manual power-on on the camera) | manually, last |
| T1 | LiDAR driver | Ctrl+C |
| T2 | FAST-LIO + RViz | Ctrl+C |
| T3 | `ros2 bag record` | Ctrl+C |
| T4 | `ros2 topic hz` (monitoring) | Ctrl+C |
| T5 | `zed_record.py` | Ctrl+C |

Start order: FLIR -> T1 -> T2 -> T3 -> T4 -> T5.
Stop order: T5 -> T3 -> T4 -> T2 -> T1 -> FLIR.

## Structure

```
DataAcquisition/
├── zed_record.py               # ZED 2i record-only CLI (SVO2/UVC + mp4 + PNG frames)
├── extract_fullrate_frames.py  # post-session: recover every frame at measured real fps
└── requirements.txt
```
