#!/usr/bin/env python3
"""Full-video counterpart of lidar_zed_depth_sync.py: sync LiDAR scans to
session_right.mp4's own frame timestamps (continuous 30fps video), not the
sparse 984-PNG dump in metadata.json, for vloam_depth_vo.m (same folder).

Why a separate script rather than extending lidar_zed_depth_sync.py: the
frame-timestamp SOURCE is genuinely different (video frame index -> epoch
via constant fps, vs. metadata.json's per-PNG t_offset_s list) and nothing
existing covers it -- read its docstring for the full reuse map (sync +
rosbag reading) which is unchanged here. Everything else (LiDAR point
source decision, quaternion inversion, rig calibration resolution,
--lidar-zed-offset caveat) is identical and reused directly, not
reimplemented:
    load_cloud_epochs, load_cloud_points_at_ordinals, quat_xyzw_to_rotmat,
    subsample                        <- lidar_zed_depth_sync.py (this folder)
    nearest_index, load_lidar_poses  <- Thesis/SensorFusion/sync_manifest.py
    load_rig_calibration             <- Thesis/Calibration/rig_calibration.py

Decode cost is decoupled from --stride: only ~1973 unique /cloud_registered
scans exist regardless of how many video frames are sampled (Pass A/B dedup,
unchanged from lidar_zed_depth_sync.py) -- --stride only changes how many
VIDEO frames get a manifest row, not how much of the bag gets decoded.

Run with the `sensorfusion` venv (rosbags + opencv + numpy):
    C:\\venvs\\sensorfusion\\Scripts\\python.exe lidar_zed_video_depth_sync.py
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_SENSORFUSION_DIR = Path(r"C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis\SensorFusion")
_CALIBRATION_DIR = Path(r"C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis\Calibration")
for _d in (_THIS_DIR, _SENSORFUSION_DIR, _CALIBRATION_DIR):
    sys.path.insert(0, str(_d))
try:
    from sync_manifest import nearest_index, load_lidar_poses
except ImportError as e:
    sys.exit(f"Could not import from {_SENSORFUSION_DIR / 'sync_manifest.py'}: {e}")
try:
    from rig_calibration import load_rig_calibration
except ImportError as e:
    sys.exit(f"Could not import load_rig_calibration from {_CALIBRATION_DIR}: {e}")
try:
    from lidar_zed_depth_sync import (
        load_cloud_epochs, load_cloud_points_at_ordinals, quat_xyzw_to_rotmat, subsample,
        MAX_POINTS_PER_SAMPLE,
    )
except ImportError as e:
    sys.exit(f"Could not import from {_THIS_DIR / 'lidar_zed_depth_sync.py'}: {e}")

DEFAULT_ZED_SESSION = Path(r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_155047")
DEFAULT_BAG = Path(r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45")
DEFAULT_RIG_CALIB = _CALIBRATION_DIR / "rig_calibration.yaml"
DEFAULT_OUT_DIR = _THIS_DIR / "depth_assoc_out"


def video_frame_epochs(zed_session: Path, video_name: str, stride: int):
    """Frame index -> absolute epoch for session_right.mp4, sampled every
    `stride`-th frame. metadata.json has no per-video-frame timestamp list
    (only the sparse PNG dump) -- frame i's timestamp is started_utc +
    i / fps, the same constant-fps assumption zed_record.py's own recording
    loop is built on (see metadata.json's camera.fps)."""
    import cv2

    meta = json.loads((zed_session / "metadata.json").read_text(encoding="utf-8"))
    started = meta["session"]["started_utc"]
    started_epoch = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
    fps = float(meta["camera"]["fps"])

    cap = cv2.VideoCapture(str(zed_session / video_name))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {zed_session / video_name}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if abs(video_fps - fps) > 0.5:
        print(f"[lidar_zed_video_depth_sync] WARNING: metadata fps={fps} but video "
              f"reports {video_fps} -- using metadata fps for timestamps.", file=sys.stderr)

    idx = np.arange(0, n_frames, stride)
    epochs = started_epoch + idx / fps
    return idx, epochs, n_frames, fps


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zed-session", default=str(DEFAULT_ZED_SESSION), metavar="DIR")
    p.add_argument("--video-name", default="session_right.mp4", metavar="NAME")
    p.add_argument("--bag", default=str(DEFAULT_BAG), metavar="DIR")
    p.add_argument("--rig-calib", default=str(DEFAULT_RIG_CALIB), metavar="PATH")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), metavar="DIR")
    p.add_argument("--stride", type=int, default=3, metavar="N",
                    help="use every Nth video frame (default 3, ~10fps effective)")
    p.add_argument("--cloud-topic", default="/cloud_registered", metavar="TOPIC")
    p.add_argument("--odom-topic", default="/Odometry", metavar="TOPIC")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--lidar-zed-offset", type=float, default=0.0, metavar="SEC",
                    help="seconds added to a LiDAR timestamp to put it on the ZED clock. "
                    "Default 0.0 is an UNVERIFIED shared-host-clock assumption, same caveat "
                    "as sync_manifest.py / lidar_zed_depth_sync.py -- see their docstrings.")
    p.add_argument("--max-points", type=int, default=MAX_POINTS_PER_SAMPLE, metavar="N",
                    help=f"deterministic stride-subsample cap per unique scan (default {MAX_POINTS_PER_SAMPLE})")
    return p.parse_args()


def main():
    args = parse_args()
    zed_session = Path(args.zed_session)
    bag = Path(args.bag)
    rig_calib_path = Path(args.rig_calib)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.lidar_zed_offset == 0.0:
        print("[lidar_zed_video_depth_sync] WARNING: --lidar-zed-offset is 0.0 (default), an "
              "UNVERIFIED shared-host-clock assumption between LiDAR and ZED, not a measured "
              "value. Pass --lidar-zed-offset once measured on the rig.", file=sys.stderr)

    # --- rig calibration (canonical source, reused as-is; same resolved.json
    #     shape lidar_zed_depth_sync.py already writes, kept in sync here) ---
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
        "zed_dist": {
            "k1": float(rig.zed_calib.dist[0]), "k2": float(rig.zed_calib.dist[1]),
            "p1": float(rig.zed_calib.dist[2]), "p2": float(rig.zed_calib.dist[3]),
            "k3": float(rig.zed_calib.dist[4]),
        },
        "T_lidar_to_zed": rig.T_lidar_to_zed.tolist(),
    }
    (out_dir / "rig_calibration_resolved.json").write_text(json.dumps(calib_out, indent=2))
    print(f"[lidar_zed_video_depth_sync] wrote {out_dir / 'rig_calibration_resolved.json'}")

    # --- video frame timestamps (new; not covered by any existing loader) ---
    frame_idx, frame_epochs, n_frames_total, fps = video_frame_epochs(zed_session, args.video_name, args.stride)
    print(f"[lidar_zed_video_depth_sync] {args.video_name}: {n_frames_total} frame(s) @ {fps}fps, "
          f"sampling every {args.stride} -> {len(frame_idx)} frame(s)")

    odom_poses = load_lidar_poses(bag, args.odom_topic, args.store)  # reused
    odom_epochs = np.array([p["epoch"] for p in odom_poses])
    print(f"[lidar_zed_video_depth_sync] {len(odom_poses)} raw {args.odom_topic} pose(s)")

    cloud_epochs = load_cloud_epochs(bag, args.cloud_topic, args.store)  # reused
    print(f"[lidar_zed_video_depth_sync] {len(cloud_epochs)} {args.cloud_topic} scan(s)")

    # --- per-sampled-frame nearest scan (reused nearest_index, same pattern
    #     as lidar_zed_depth_sync.py) ---
    manifest_rows = []
    wanted_cloud_ordinals = set()
    cloud_on_zed_clock = cloud_epochs + args.lidar_zed_offset
    for fi, epoch in zip(frame_idx, frame_epochs):
        c_i = nearest_index(cloud_on_zed_clock, epoch)
        cloud_delta = float(cloud_on_zed_clock[c_i] - epoch)
        wanted_cloud_ordinals.add(c_i)
        manifest_rows.append({
            "frame_idx": int(fi), "video_epoch": float(epoch),
            "cloud_ordinal": int(c_i), "cloud_delta_s": cloud_delta,
        })

    print(f"[lidar_zed_video_depth_sync] decoding {len(wanted_cloud_ordinals)} unique "
          f"{args.cloud_topic} scan(s) (Pass B) ...")
    clouds = load_cloud_points_at_ordinals(bag, args.cloud_topic, args.store, wanted_cloud_ordinals)

    points_rows = []
    n_exported_by_ordinal = {}
    for c_i in sorted(wanted_cloud_ordinals):
        cloud_epoch, pts_world = clouds[c_i]
        o_i = nearest_index(odom_epochs, cloud_epoch)
        pose = odom_poses[o_i]
        R = quat_xyzw_to_rotmat(pose["orientation"])
        t = np.array(pose["position"])
        pts_lidar = (pts_world - t) @ R
        pts_lidar = subsample(pts_lidar, args.max_points)
        n_exported_by_ordinal[c_i] = pts_lidar.shape[0]
        for x, y, z in pts_lidar:
            points_rows.append((c_i, float(x), float(y), float(z)))

    for row in manifest_rows:
        row["n_points_exported"] = n_exported_by_ordinal.get(row["cloud_ordinal"], 0)

    import csv
    manifest_path = out_dir / "video_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        fields = ["frame_idx", "video_epoch", "cloud_ordinal", "cloud_delta_s", "n_points_exported"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)
    print(f"[lidar_zed_video_depth_sync] wrote {manifest_path} ({len(manifest_rows)} row(s))")

    points_path = out_dir / "video_lidar_points.csv"
    with open(points_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cloud_ordinal", "x_lidar", "y_lidar", "z_lidar"])
        w.writerows(points_rows)
    print(f"[lidar_zed_video_depth_sync] wrote {points_path} ({len(points_rows)} point row(s), "
          f"cap {args.max_points}/scan)")

    n_low_sync = sum(1 for r in manifest_rows if abs(r["cloud_delta_s"]) > 0.5)
    print(f"[lidar_zed_video_depth_sync] sync summary: {len(manifest_rows)} frame(s), "
          f"{n_low_sync} with |cloud_delta_s| > 0.5s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
