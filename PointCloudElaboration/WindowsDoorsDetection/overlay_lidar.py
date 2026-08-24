"""QA view for the step where a LiDAR point becomes a door/window/other point.

paint_openings.py labels a point by the mask its pixel lands in. That is one
projection and one array lookup, and if either is wrong every downstream number
is wrong in a way no 3-D view makes obvious -- a systematic pixel offset just
moves the whole wall. This draws the projection back onto the frame it was done
in, so the two can be compared where the decision is actually taken.

Points are drawn over stage 1's `overlay.png` (which already carries the mask
outlines and their labels), coloured by the class each point was GIVEN:

    red    door       blue   window       grey   other / unlabelled

A point sitting inside a blue outline but drawn grey, or vice versa, is a
projection error. Points that hug the outlines are the projection working.

Usage:
    C:\\venvs\\planefit\\Scripts\\python.exe overlay_lidar.py ^
        --session-dir ...\\fullrate --bag ...\\rosbag2_2026_07_30-18_12_20 ^
        --opening-map-dir ...\\fullrate\\opening_map_m2f_full ^
        --frame 20250906_233214_R --out overlay_lidar.png

    ... --every-n 12 --limit 4      :: a contact sheet instead of one frame
    ... --point-radius 2            :: fatter dots on a downsampled scan
    ... --depth-band 1.5            :: same gate paint_openings.py applies

Venv: C:\\venvs\\planefit.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paint_openings as po  # noqa: E402  (its bootstrap sets up the imports below)
from projection import project_lidar_to_camera  # noqa: E402
from rig_calibration import load_rig_calibration  # noqa: E402

# BGR, cv2's order -- the same colours the 3-D views use.
CLASS_BGR = {"door": (40, 40, 220), "window": (230, 110, 40)}
OTHER_BGR = (150, 150, 150)


def parse_args():
    p = argparse.ArgumentParser(
        description="Draw the projected LiDAR back onto the frame it was labelled in")
    p.add_argument("--session-dir", required=True, metavar="DIR")
    p.add_argument("--bag", required=True, metavar="DIR")
    p.add_argument("--opening-map-dir", default=None, metavar="DIR")
    p.add_argument("--odom-bag", default=None, metavar="DIR",
                   help="Poses from this bag instead of --bag's own, as in "
                        "paint_openings.py --odom-bag.")
    p.add_argument("--frame", default=None, metavar="STEM",
                   help="One frame stem. Without it, --every-n/--limit pick a set.")
    p.add_argument("--every-n", type=int, default=12, metavar="N")
    p.add_argument("--limit", type=int, default=4, metavar="N")
    p.add_argument("--depth-band", type=float, default=0.0, metavar="M",
                   help="Same per-segment depth gate paint_openings.py applies; 0 = off, "
                        "which shows the raw assignment including the through-the-mask "
                        "smear.")
    p.add_argument("--point-radius", type=int, default=1, metavar="PX")
    p.add_argument("--stride", type=int, default=1, metavar="N",
                   help="Draw every Nth projected point (default 1, all of them).")
    p.add_argument("--no-base", action="store_true",
                   help="Draw on black instead of stage 1's overlay.png.")
    p.add_argument("--out", default="overlay_lidar.png", metavar="PNG")
    p.add_argument("--store", default="ROS2_HUMBLE")
    p.add_argument("--calibration", default=None, metavar="YAML")
    return p.parse_args()


def draw_frame(stem, triplet, cloud, opening_dir, cal, args):
    """One frame: the overlay with its own projected points on top."""
    labels = np.load(opening_dir / stem / "labels.npy")
    doc = json.loads((opening_dir / stem / "segments.json").read_text(encoding="utf-8"))
    info = {int(s["id"]): s["top_class"] for s in doc["segments"]}
    zh, zw = labels.shape

    base_path = opening_dir / stem / "overlay.png"
    if args.no_base or not base_path.is_file():
        img = np.zeros((zh, zw, 3), np.uint8)
    else:
        img = cv2.imread(str(base_path))
        if img is None or img.shape[:2] != (zh, zw):
            img = cv2.resize(img, (zw, zh)) if img is not None else \
                np.zeros((zh, zw, 3), np.uint8)

    _t, points_world = cloud
    uv, depth, valid = project_lidar_to_camera(
        points_world, np.array(triplet["lidar"]["position"]),
        np.array(triplet["lidar"]["orientation"]), cal.T_lidar_to_zed,
        cal.zed_K_for(zw, zh), cal.zed_calib.dist, zw, zh)
    px = np.round(uv[valid]).astype(int)
    px[:, 0] = np.clip(px[:, 0], 0, zw - 1)
    px[:, 1] = np.clip(px[:, 1], 0, zh - 1)
    sids = labels[px[:, 1], px[:, 0]]
    dep = depth[valid]

    counts = {"door": 0, "window": 0, "other": 0}
    for k in range(0, len(px), max(1, args.stride)):
        sid = int(sids[k])
        cls = info.get(sid, "other") if sid >= 0 else "other"
        if cls in CLASS_BGR and args.depth_band > 0:
            m = sids == sid
            if abs(float(dep[k]) - float(np.median(dep[m]))) > args.depth_band:
                cls = "other"
        counts[cls if cls in counts else "other"] += 1
        cv2.circle(img, (int(px[k, 0]), int(px[k, 1])),
                   args.point_radius, CLASS_BGR.get(cls, OTHER_BGR), -1)

    txt = (f"{stem}   {len(px)} pts in frame   "
           f"door={counts['door']} window={counts['window']} other={counts['other']}")
    cv2.rectangle(img, (0, 0), (zw, 34), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    print("  " + txt)
    return img


def main():
    args = parse_args()
    session = Path(args.session_dir)
    opening_dir = (Path(args.opening_map_dir) if args.opening_map_dir
                   else session / "opening_map")
    cal = load_rig_calibration(args.calibration
                               or (po._ROOT / "SensorFusionLoader" / "rig_calibration.yaml"))
    manifest = json.loads((session / "sync_manifest.json").read_text(encoding="utf-8"))

    trips = [t for t in manifest["triplets"]
             if (opening_dir / Path(t["flir"]["file"]).stem).is_dir()]
    if args.frame:
        trips = [t for t in trips if Path(t["flir"]["file"]).stem == args.frame]
        if not trips:
            sys.exit(f"{args.frame} has no stage 1 output under {opening_dir}.")
    else:
        trips = trips[::args.every_n][:args.limit]

    traj = None
    if args.odom_bag:
        sys.path.insert(0, str(po._LOADER_DIR))
        import livox_odometry_loader as lol
        traj = lol.read_trajectory(Path(args.odom_bag), store=args.store)

    clouds = po.lm.load_clouds(
        Path(args.bag), [t["lidar"]["timestamp_zedclock"] for t in trips],
        store=args.store, loader_dir=po._LOADER_DIR, traj=traj)

    tiles = []
    for triplet, cloud in zip(trips, clouds):
        stem = Path(triplet["flir"]["file"]).stem
        if cloud is None:
            print(f"skip {stem}: no LiDAR scan near that instant", file=sys.stderr)
            continue
        tiles.append(draw_frame(stem, triplet, cloud, opening_dir, cal, args))

    if not tiles:
        sys.exit("Nothing drawn.")
    out = tiles[0] if len(tiles) == 1 else np.vstack(
        [cv2.resize(t, (t.shape[1] // 2, t.shape[0] // 2)) for t in tiles])
    cv2.imwrite(args.out, out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
