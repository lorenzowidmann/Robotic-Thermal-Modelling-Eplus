"""
Apply a fixed filled circle (from pick_hole_patch.py) to every frame of a
split ZED pose folder, to remove a background object seen through a hole
(e.g. a door handle) that breaks the RGB blob detector's circularity test.

The board is static during a pose, so a single (cx, cy, radius) found on one
frame (pick_hole_patch.py) is valid for every frame of that pose. This only
edits the RGB copy used by the ZED detector -- it never touches the physical
board, the LiDAR point cloud, or the FLIR frames, which are separate data
streams and are unaffected either way.

Input is a pose folder already produced by split_zed_poses.py (frames/ +
metadata.json). Output is a sibling folder with the same structure, patched
frames, and the SAME metadata.json copied unchanged (frame list/timing don't
change) -- ready to use directly as zed_frame_publisher.py --session-dir.

Usage:
    py patch_hole_pose.py --pose-dir <NewCalibration>\\Zed\\Poses\\pose_01 \\
        --outdir <NewCalibration>\\Zed\\Poses\\pose_01_patched \\
        --cx 430 --cy 210 --radius 65 --color 210,205,200
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-dir", required=True,
                    help="Source pose folder (from split_zed_poses.py: frames/ + metadata.json)")
    ap.add_argument("--outdir", required=True,
                    help="Destination pose folder (created fresh)")
    ap.add_argument("--cx", type=int, required=True)
    ap.add_argument("--cy", type=int, required=True)
    ap.add_argument("--radius", type=int, required=True)
    ap.add_argument("--color", required=True,
                    help="Fill colour as R,G,B (printed by pick_hole_patch.py)")
    args = ap.parse_args()

    r, g, b = (int(v) for v in args.color.split(","))
    color_bgr = (b, g, r)

    src = Path(args.pose_dir)
    src_frames = src / "frames"
    src_meta = src / "metadata.json"
    if not src_frames.is_dir():
        sys.exit(f"frames/ not found in {src}")
    if not src_meta.is_file():
        sys.exit(f"metadata.json not found in {src}")

    dst = Path(args.outdir)
    dst_frames = dst / "frames"
    dst_frames.mkdir(parents=True, exist_ok=True)

    files = sorted(src_frames.glob("*.png"))
    if not files:
        sys.exit(f"No PNG frames in {src_frames}")

    n = 0
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARNING: unreadable, skipped: {f.name}")
            continue
        cv2.circle(img, (args.cx, args.cy), args.radius, color_bgr, -1)
        cv2.imwrite(str(dst_frames / f.name), img)
        n += 1

    shutil.copy2(src_meta, dst / "metadata.json")
    print(f"Patched {n}/{len(files)} frames -> {dst_frames}")
    print(f"metadata.json copied unchanged -> {dst / 'metadata.json'}")
    print(f"\nUse with: --session-dir {dst}")


if __name__ == "__main__":
    main()
