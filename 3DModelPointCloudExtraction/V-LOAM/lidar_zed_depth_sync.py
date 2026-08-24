#!/usr/bin/env python3
"""V-LOAM-style depth association, prep step: sync ZED keyframes to LiDAR
poses/scans and dump everything a MATLAB script needs to do the actual
projection + ORB association (lidar_depth_association_test.m, same folder).

Concept (Zhang & Singh 2015, "Visual-lidar Odometry and Mapping: low-drift,
robust and fast", Sec. IV/V): give visual features metric depth by
associating each one with nearby LiDAR range points projected into the
camera frame, instead of letting monocular VO carry an arbitrary scale. The
paper builds a depthmap (2D KD-tree in spherical coords, 3 nearest points ->
local planar patch -> ray/plane intersection). Per the task, this script
does NOT reproduce that KD-tree/patch step -- MATLAB does a much simpler
nearest-projected-point-within-pixel-radius association. This script only
gets real 3D LiDAR points, in the camera frame, matched in time to each
sampled ZED frame.

Why this file exists in Python and not MATLAB: reading a rosbag2 (.db3,
custom Livox message types) only works in this project via the `rosbags`
library (see Thesis/SensorFusion/sync_manifest.py, load_lidar_poses), which
has no MATLAB equivalent set up here. Run with the `sensorfusion` venv
(has rosbags + opencv + numpy):
    C:\\venvs\\sensorfusion\\Scripts\\python.exe lidar_zed_depth_sync.py

Reused, NOT reimplemented (per task instructions):
  - load_zed_frames, nearest_index, load_lidar_poses  <- sync_manifest.py
  - read_poses_tum_with_ts                              <- verify_loops_appearance.py
  - load_rig_calibration                                <- Thesis/Calibration/rig_calibration.py
New code here is only: reading /cloud_registered (PointCloud2, not covered
by any existing loader) and the LiDAR-pose-source and point-cloud-source
choice documented below.

LiDAR point source (chosen with the user, see conversation -- not guessed):
  /cloud_registered (sensor_msgs/PointCloud2, world/"camera_init" frame) is
  read for the scan whose OWN header timestamp is nearest (after
  --lidar-zed-offset) to each sampled ZED frame's timestamp. It is then
  brought back into the LiDAR/laser sensor frame by inverting the /Odometry
  pose whose timestamp is nearest to THAT SPECIFIC SCAN's own timestamp
  (self-consistent pair: both come from the same FAST-LIO front-end
  instant, so the inversion is exact regardless of what happens later
  during pose-graph/loop-closure optimization).

  poses_tum_matlab.txt (the pose-graph-optimized, "already elaborated"
  keyframe trajectory the task points at) is used ONLY to report which
  optimized LiDAR pose is nearest each ZED sample (satisfies "trova il
  LiDAR pose piu' vicino nel tempo" literally) -- it is NOT used for the
  scan-to-sensor-frame inversion, to avoid mixing an optimized/corrected
  pose with a raw-odometry-frame point cloud.

  KNOWN SIMPLIFICATION: /Odometry (FAST-LIO) is the pose of the estimator's
  body/IMU frame, not necessarily the LiDAR's own optical origin. For a
  Livox unit with an integrated IMU (mm-level offset from the laser origin)
  this is treated as negligible here, same implicit assumption as the rest
  of this project's LiDAR<->camera extrinsics (rig_calibration.yaml calls
  it the "laser frame" throughout). Not separately corrected.

lidar_zed_offset: the LiDAR<->ZED clock relationship is UNVERIFIED (shared
host clock assumed, default 0.0) -- identical caveat to sync_manifest.py
and verify_loops_appearance.py. This run found the two streams' wall-clock
ranges overlap almost exactly (LiDAR /Odometry 15:50:43.90-15:57:21.10 UTC,
ZED session 15:50:47.60-15:57:21.09 UTC) which is consistent with, but does
NOT prove, a shared/zero-offset clock. Override --lidar-zed-offset once
actually measured on the rig.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- Cross-repo reuse (same sys.path.insert pattern as verify_loops_appearance.py) ---
_SENSORFUSION_DIR = Path(r"C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis\SensorFusion")
_LOOPCLOSURE_ANALYSIS_DIR = Path(
    r"C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\FAST-LIO-SAM-SC-QN\fast_lio_sam_sc_qn\loop_closure_analysis"
)
_CALIBRATION_DIR = Path(r"C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis\Calibration")
for _d in (_SENSORFUSION_DIR, _LOOPCLOSURE_ANALYSIS_DIR, _CALIBRATION_DIR):
    sys.path.insert(0, str(_d))
try:
    from sync_manifest import load_zed_frames, nearest_index, load_lidar_poses
except ImportError as e:
    sys.exit(f"Could not import from {_SENSORFUSION_DIR / 'sync_manifest.py'}: {e}")
try:
    from verify_loops_appearance import read_poses_tum_with_ts
except ImportError as e:
    sys.exit(f"Could not import read_poses_tum_with_ts from {_LOOPCLOSURE_ANALYSIS_DIR}: {e}")
try:
    from rig_calibration import load_rig_calibration
except ImportError as e:
    sys.exit(f"Could not import load_rig_calibration from {_CALIBRATION_DIR}: {e}")

DEFAULT_ZED_SESSION = Path(r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_155047")
DEFAULT_BAG = Path(r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45")
DEFAULT_POSES_TUM = DEFAULT_BAG / "poses_tum_matlab.txt"
DEFAULT_RIG_CALIB = _CALIBRATION_DIR / "rig_calibration.yaml"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "depth_assoc_out"

MAX_POINTS_PER_SAMPLE = 3000  # deterministic stride-subsample cap, see --max-points


# --------------------------------------------------------------------------- #
# /cloud_registered (PointCloud2) reading -- not covered by sync_manifest.py
# --------------------------------------------------------------------------- #

def _pointcloud2_xyz(msg) -> np.ndarray:
    """Decode x,y,z (float32) from a sensor_msgs/PointCloud2 message using its
    own `fields` (offset-based) and `point_step` -- robust to the field
    order/extra channels actually present (this bag's /cloud_registered also
    carries intensity/normals/curvature, which we don't need), via a
    structured dtype rather than a hand-rolled strided view."""
    offsets = {f.name: f.offset for f in msg.fields}
    for name in ("x", "y", "z"):
        if name not in offsets:
            raise ValueError(f"/cloud_registered PointCloud2 missing field {name!r}")
    n = msg.width * msg.height
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [np.float32, np.float32, np.float32],
        "offsets": [offsets["x"], offsets["y"], offsets["z"]],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    pts = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def load_cloud_epochs(bag: Path, topic: str, store: str):
    """Pass A: header timestamps only, for every message on `topic`, in bag
    order. Cheap (keeps no point payload)."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores[store])
    epochs = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not found in bag. Available: {topics}")
        for connection, _bag_ts, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            epochs.append(float(stamp.sec) + float(stamp.nanosec) * 1e-9)
    return np.array(epochs, dtype=float)


def load_cloud_points_at_ordinals(bag: Path, topic: str, store: str, wanted_ordinals: set):
    """Pass B: full point decode, only for messages whose 0-based ordinal
    (in bag/topic order, matching load_cloud_epochs's order) is in
    `wanted_ordinals`. Returns {ordinal: (epoch, Nx3 float32 array)}."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores[store])
    out = {}
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for i, (connection, _bag_ts, rawdata) in enumerate(reader.messages(connections=conns)):
            if i not in wanted_ordinals:
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            epoch = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            out[i] = (epoch, _pointcloud2_xyz(msg))
            if len(out) == len(wanted_ordinals):
                break
    return out


# --------------------------------------------------------------------------- #
# Quaternion -> rotation matrix (x,y,z,w), no scipy available in this venv
# --------------------------------------------------------------------------- #

def quat_xyzw_to_rotmat(q) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def subsample(points: np.ndarray, cap: int) -> np.ndarray:
    n = points.shape[0]
    if n <= cap:
        return points
    k = int(np.ceil(n / cap))
    return points[::k]


# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zed-session", default=str(DEFAULT_ZED_SESSION), metavar="DIR")
    p.add_argument("--bag", default=str(DEFAULT_BAG), metavar="DIR")
    p.add_argument("--poses-tum", default=str(DEFAULT_POSES_TUM), metavar="PATH")
    p.add_argument("--rig-calib", default=str(DEFAULT_RIG_CALIB), metavar="PATH")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), metavar="DIR")
    p.add_argument("--stride", type=int, default=10, metavar="N",
                    help="use every Nth ZED frame from metadata.json's frame list (default 10)")
    p.add_argument("--cloud-topic", default="/cloud_registered", metavar="TOPIC")
    p.add_argument("--odom-topic", default="/Odometry", metavar="TOPIC")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--lidar-zed-offset", type=float, default=0.0, metavar="SEC",
                    help="seconds added to a LiDAR timestamp to put it on the ZED clock. "
                    "Default 0.0 is an UNVERIFIED shared-host-clock assumption, same caveat "
                    "as sync_manifest.py / verify_loops_appearance.py -- see module docstring.")
    p.add_argument("--max-points", type=int, default=MAX_POINTS_PER_SAMPLE, metavar="N",
                    help=f"deterministic stride-subsample cap per sampled scan (default {MAX_POINTS_PER_SAMPLE})")
    return p.parse_args()


def main():
    args = parse_args()
    zed_session = Path(args.zed_session)
    bag = Path(args.bag)
    poses_tum_path = Path(args.poses_tum)
    rig_calib_path = Path(args.rig_calib)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lidar_zed_offset == 0.0:
        print("[lidar_zed_depth_sync] WARNING: --lidar-zed-offset is 0.0 (default), an "
              "UNVERIFIED shared-host-clock assumption between LiDAR and ZED, not a measured "
              "value. Same caveat as sync_manifest.py/verify_loops_appearance.py. Pass "
              "--lidar-zed-offset once measured on the rig.", file=sys.stderr)

    # --- rig calibration (canonical source, reused as-is) -------------------
    rig = load_rig_calibration(rig_calib_path)
    K1080 = rig.zed_K_for(1920, 1080)
    calib_out = {
        "schema": "lidar_depth_assoc_calib/v1",
        "source_rig_calibration": str(rig_calib_path),
        "zed_image_size": {"width": 1920, "height": 1080},
        "zed_K_1920x1080": {
            "fx": float(K1080[0, 0]), "fy": float(K1080[1, 1]),
            "cx": float(K1080[0, 2]), "cy": float(K1080[1, 2]),
        },
        # dist_coeffs order in rig_calibration.yaml is [k1,k2,p1,p2,k3]; split
        # explicitly here so MATLAB's cameraIntrinsics(RadialDistortion=,
        # TangentialDistortion=) can't be fed the wrong order by mistake.
        "zed_dist": {
            "k1": float(rig.zed_calib.dist[0]), "k2": float(rig.zed_calib.dist[1]),
            "p1": float(rig.zed_calib.dist[2]), "p2": float(rig.zed_calib.dist[3]),
            "k3": float(rig.zed_calib.dist[4]),
        },
        "T_lidar_to_zed": rig.T_lidar_to_zed.tolist(),  # 4x4, laser frame -> ZED camera frame
    }
    (out_dir / "rig_calibration_resolved.json").write_text(json.dumps(calib_out, indent=2))
    print(f"[lidar_zed_depth_sync] wrote {out_dir / 'rig_calibration_resolved.json'}")

    # --- streams --------------------------------------------------------------
    zed_frames = load_zed_frames(zed_session)  # reused
    print(f"[lidar_zed_depth_sync] {len(zed_frames)} ZED frame(s) in metadata.json")
    sampled = zed_frames[:: args.stride]
    print(f"[lidar_zed_depth_sync] sampling every {args.stride}: {len(sampled)} keyframe(s)")

    posetum = read_poses_tum_with_ts(poses_tum_path)  # reused; row idx == keyframe idx
    posetum_epochs = np.array([t for t, _, _ in posetum])
    print(f"[lidar_zed_depth_sync] {len(posetum)} keyframe pose(s) in {poses_tum_path.name}")

    odom_poses = load_lidar_poses(bag, args.odom_topic, args.store)  # reused
    odom_epochs = np.array([p["epoch"] for p in odom_poses])
    print(f"[lidar_zed_depth_sync] {len(odom_poses)} raw {args.odom_topic} pose(s)")

    cloud_epochs = load_cloud_epochs(bag, args.cloud_topic, args.store)
    print(f"[lidar_zed_depth_sync] {len(cloud_epochs)} {args.cloud_topic} scan(s)")

    # --- per-sample nearest matches (Pass A: timestamps only) ---------------
    manifest_rows = []
    wanted_cloud_ordinals = set()
    for sample_idx, zf in enumerate(sampled):
        zed_epoch = zf["epoch"]

        # 1) nearest ALREADY-ELABORATED LiDAR pose (poses_tum_matlab.txt), as
        #    literally requested by the task -- reporting only, see docstring.
        posetum_on_zed_clock = posetum_epochs + args.lidar_zed_offset
        pt_i = nearest_index(posetum_on_zed_clock, zed_epoch)
        posetum_delta = float(posetum_on_zed_clock[pt_i] - zed_epoch)

        # 2) nearest raw /cloud_registered scan (used for the actual points)
        cloud_on_zed_clock = cloud_epochs + args.lidar_zed_offset
        c_i = nearest_index(cloud_on_zed_clock, zed_epoch)
        cloud_delta = float(cloud_on_zed_clock[c_i] - zed_epoch)
        wanted_cloud_ordinals.add(c_i)

        manifest_rows.append({
            "sample_idx": sample_idx, "zed_file": zf["file"], "zed_epoch": zed_epoch,
            "posetum_row_idx": int(pt_i), "posetum_epoch": float(posetum_epochs[pt_i]),
            "posetum_delta_s": posetum_delta,
            "cloud_ordinal": int(c_i), "cloud_delta_s": cloud_delta,
        })

    print(f"[lidar_zed_depth_sync] decoding {len(wanted_cloud_ordinals)} unique "
          f"{args.cloud_topic} scan(s) (Pass B) ...")
    clouds = load_cloud_points_at_ordinals(bag, args.cloud_topic, args.store, wanted_cloud_ordinals)

    # --- invert each wanted scan into the LiDAR/laser sensor frame using the
    #     /Odometry pose nearest to THAT SCAN's own timestamp (self-consistent) --
    points_rows = []
    n_exported_by_ordinal = {}
    for c_i in sorted(wanted_cloud_ordinals):
        cloud_epoch, pts_world = clouds[c_i]
        o_i = nearest_index(odom_epochs, cloud_epoch)
        pose = odom_poses[o_i]
        R = quat_xyzw_to_rotmat(pose["orientation"])
        t = np.array(pose["position"])
        pts_lidar = (pts_world - t) @ R  # R^T @ (p - t), row-vector form
        pts_lidar = subsample(pts_lidar, args.max_points)
        n_exported_by_ordinal[c_i] = pts_lidar.shape[0]
        for x, y, z in pts_lidar:
            points_rows.append((c_i, float(x), float(y), float(z)))

    for row in manifest_rows:
        row["n_points_exported"] = n_exported_by_ordinal.get(row["cloud_ordinal"], 0)

    # --- write outputs --------------------------------------------------------
    import csv
    manifest_path = out_dir / "lidar_depth_assoc_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        fields = ["sample_idx", "zed_file", "zed_epoch", "posetum_row_idx", "posetum_epoch",
                   "posetum_delta_s", "cloud_ordinal", "cloud_delta_s", "n_points_exported"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)
    print(f"[lidar_zed_depth_sync] wrote {manifest_path} ({len(manifest_rows)} row(s))")

    points_path = out_dir / "lidar_depth_assoc_points.csv"
    with open(points_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cloud_ordinal", "x_lidar", "y_lidar", "z_lidar"])
        w.writerows(points_rows)
    print(f"[lidar_zed_depth_sync] wrote {points_path} ({len(points_rows)} point row(s), "
          f"cap {args.max_points}/scan)")

    n_low_sync = sum(1 for r in manifest_rows if abs(r["cloud_delta_s"]) > 0.5)
    print(f"[lidar_zed_depth_sync] sync summary: {len(manifest_rows)} keyframe(s), "
          f"{n_low_sync} with |cloud_delta_s| > 0.5s (loose sync warning threshold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
