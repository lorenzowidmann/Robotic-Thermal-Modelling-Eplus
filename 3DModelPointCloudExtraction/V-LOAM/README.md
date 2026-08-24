# V-LOAM — visual-LiDAR depth association (side experiment)

> **Status: not part of the building-model pipeline.** Nothing outside this
> folder reads anything it produces, and the parent module's README does not
> list it as a step. The `.osm` model is built from FAST-LIO / Livox SLAM poses,
> not from V-LOAM. Kept because the experiment is finished and its results
> (`depth_assoc_out/`) are thesis material.

Gives visual features metric depth by associating each one with nearby LiDAR
range points projected into the camera frame, instead of letting monocular
visual odometry carry an arbitrary scale — Zhang & Singh 2015, *"Visual-lidar
Odometry and Mapping: low-drift, robust and fast"*, Sec. IV/V.

The paper's own depthmap step (2-D KD-tree in spherical coordinates, 3 nearest
points → local planar patch → ray/plane intersection) is **not** reproduced.
MATLAB does a simpler nearest-projected-point-within-pixel-radius association.
The Python here only produces real 3-D LiDAR points, in the camera frame,
matched in time to each sampled ZED frame.

## Files

| file | what it does |
|---|---|
| `lidar_zed_depth_sync.py` | syncs ZED **keyframes** to LiDAR poses/scans and dumps what MATLAB needs. Frame timestamps come from the sparse PNG dump in `metadata.json`. |
| `lidar_zed_video_depth_sync.py` | same, for the **full video**: syncs to `session_right.mp4`'s own frame timestamps (continuous 30 fps), video frame index → epoch via constant fps. A separate script because the timestamp *source* genuinely differs; everything else (LiDAR point source, quaternion inversion, rig calibration, the `--lidar-zed-offset` caveat) is imported from the script above, not reimplemented. |
| `lidar_depth_association_test.m` | MATLAB: the projection + ORB association itself. |
| `vloam_depth_vo.m` | MATLAB: the visual odometry run on top of it. |
| `depth_assoc_out/` | results — trajectories in TUM format, ablation runs (`ablate_ang*_epi*`), before/after tuning figures, overlay QA frames, `depth_assoc_summary.csv`. |

## Usage

```powershell
py lidar_zed_depth_sync.py --session-dir <ZED session> --bag <rosbag2 folder>
py lidar_zed_video_depth_sync.py --session-dir <ZED session>\fullrate --bag <rosbag2 folder>
```

Then run the two `.m` files in MATLAB against `depth_assoc_out/`.

Reuses `nearest_index` / `load_lidar_poses` from `../../TimeSyncCheck/sync_manifest.py`
and the rig extrinsics from `../../SensorFusionLoader/`.
