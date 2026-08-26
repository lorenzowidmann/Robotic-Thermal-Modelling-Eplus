# MATLAB_SensorFusionValidation

Visual check that the LiDAR↔camera **extrinsics actually line up** on real scene
geometry, not just on the four calibration-target holes they were fitted to.

For a chosen synced triplet it projects the same LiDAR scan **twice** — once into
the FLIR, once into the ZED — then paints each ZED pixel with the FLIR value
sampled at the corresponding FLIR pixel. If the calibration is right, the thermal
colours land on the matching structures in the RGB image. If it drifts, you see
it immediately as thermal bleeding across an edge.

This is a **sanity check on the projection chain**, not the radiometric fusion
itself — that lives in `../SensorFusionLoader/`.

<p align="center">
  <img src="output/flir_on_zed_session9_pose09.png" width="720" alt="FLIR radiometric values projected onto the ZED frame, session 9 pose 09"><br>
  <em>Session 9, pose 09 — FLIR values (colormap <code>hot</code>) sampled through
  the LiDAR cloud and scattered onto the ZED frame.</em>
</p>

| file | what it does |
|---|---|
| `FlirLidarZedViewer.m` | the viewer. Loads a triplet, projects into both cameras, draws the overlay. Arrow keys step through poses without reopening the bag. |
| `output/` | saved PNGs, `flir_on_zed_session9_pose<NN>.png`. Created automatically. |

## Usage

Needs an **interactive MATLAB desktop session**. It will not work under
`matlab -batch`: in batch the figure closes as soon as the function returns and
no event loop is left to listen for keys.

```matlab
cd 'C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\RTM-EPlus\MATLAB_SensorFusionValidation'
FlirLidarZedViewer        % starts at pose 9
FlirLidarZedViewer(30)    % starts at pose 30
```

Keys (the figure window must have focus):

| key | action |
|---|---|
| `→` / `n` | next pose |
| `←` / `p` | previous pose |
| `s` | save the current frame to `output/` |
| `q` / close window | quit |

## What it does, step by step

1. Reads `sync_manifest.json` and picks the triplet at `S.idx` — FLIR file, ZED
   file, LiDAR timestamp and `/Odometry` pose, all already time-matched by
   `../TimeSyncCheck/`.
2. Pulls every `/cloud_registered` message within `±0.4 s` of that timestamp from
   the ROS2 bag and merges them (see *Scan accumulation* below).
3. Transforms the cloud **world → body**. `/cloud_registered` is published by
   FAST-LIO2 already in the world frame (`camera_init`), so it has to be brought
   back with the triplet's own pose: `p_body = R_wb' * (p_world - t_wb)`.
4. Applies the two extrinsics and projects with a pinhole + Brown–Conrady model
   (`projectPinhole`), keeping only points in front of the camera and inside the
   image bounds.
5. Runs a per-camera **z-buffer** to drop occluded points (`zBufferMask`).
6. Loads the raw radiometric FLIR frame from `.npy` (`readNpyFloat32`, a minimal
   built-in reader — no external dependency), maps it through `hot`, samples it
   at the projected FLIR pixels and scatters those colours over the ZED image.

## Baked-in calibration

Both extrinsics are hard-coded results, adopted from the LVT2Calib sessions:

| transform | poses used | min3D RMSE |
|---|---|---|
| LiDAR → FLIR | 6 clean (01,02,03,05,07,08) | 5.8 cm |
| LiDAR → ZED (right eye) | 8 (01,02,03,05,07,08,09,10) | 6.8 cm |

Intrinsics are the **no-skew** MATLAB models: FLIR Vue Pro R 336×256, ZED 2i
right eye 1080p. Same values as `../SensorFusionLoader/rig_calibration.yaml` —
if you recalibrate, update both.

> On that RMSE: it is dominated by the **depth** component. The alignment error
> in LVT2Calib is reported in the `stereo` frame (ROS REP-103, x = forward), so
> `RMSE_x` *is* the depth error — the expected weak axis of a monocular pose
> estimate from a planar target. Lateral error is ~6 mm against ~67 mm in depth.

## Gotchas worth knowing

**FLIR is mounted upside down.** The images were rotated 180° *before* corner
detection during extrinsic calibration, and the original K (fitted on
un-rotated images) was reused as-is, **without** re-centring `cx`/`cy` on the new
grid. This script deliberately replicates that convention — same `*_rot180*`
folder, same un-recentred K — because that is the convention `Tr_laser_to_cam`
was actually estimated under. Re-centring here would introduce an inconsistency
with the extrinsics, not fix one. Verified empirically: >99 % of LiDAR points
project inside the FLIR bounds this way.

**Scan accumulation.** The Livox HAP scans non-repetitively, so a single
`/cloud_registered` message (~0.2 s) only covers partial bands of the scene —
hence the wide stripes on the overlay. `S.lidarAccumHalfWindow_s = 0.4` merges
the neighbouring scans to densify it. All of them are transformed with the
*same* `/Odometry` pose, which assumes the rig is near-stationary over that
window: widen it for more points, at the cost of motion blur.

**Occlusion filter is not optional.** FLIR and ZED sit ~13 cm apart on the rig,
so near an edge they see "around the corner" differently. Without the z-buffer a
wall point hidden from the FLIR but geometrically inside its frustum still gets
sampled, picking up the colour of the foreground edge — clearly visible at
session 9 poses 71 and 106. `S.zBufferTol_m = 0.08` is deliberately the same
order as the calibration RMSE.

**Body frame is assumed equal to the LiDAR frame.** No separate IMU–LiDAR
extrinsic is known for this rig. The resulting error is expected to be small but
has not been quantified.

## Requires

MATLAB with:

| function | toolbox |
|---|---|
| `ros2bagreader`, `readMessages`, `rosReadXYZ` | ROS Toolbox |
| `quat2rotm` | Navigation / Robotics System Toolbox |
| `mat2gray`, `ind2rgb`, `imshow`, `imread` | Image Processing Toolbox |

## Data it reads

Paths are hard-coded at the top of the file (`sessionRoot` and below), all under
`Dati_vfinal\SLAM\` — not tracked in this repo:

```
ZED/20260730_161223/fullrate/sync_manifest.json   triplets + /Odometry poses
ZED/20260730_161223/fullrate/frames/              ZED RGB frames
Flir/session9_only_rot180/                        FLIR .npy, already rotated 180°
Lidar/rosbag2_2026_07_30-18_12_20/                ROS2 bag, /cloud_registered
```
