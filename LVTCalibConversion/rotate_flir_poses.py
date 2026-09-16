"""
Rotate FLIR (RJPG) frames of one or more poses by 180 degrees.

Why: the FLIR is mounted upside down on the rig (see flir_frame_publisher.py
--rotate180), so the frames saved to disk are "upside down". This script
applies the same rotation but to the FILES, for pipelines that read the JPEGs
directly instead of going through flir_frame_publisher.

By default only rotates the visible JPEG image (cv2.imread/imwrite); if the
file is an RJPG with RawThermalImage in its EXIF (16-bit radiometric data),
that blob is NOT rotated and is lost on save (cv2 does not preserve EXIF).
Fine for the LVT2Calib flow (uses only the palette image, --image-mode
embedded), not for downstream use of the raw radiometric data.

--npy additionally extracts the actual apparent-temperature map (deg C) via
`flyr` (same reader as RadiometricCalibration/ThermalData.py), rotates that
array 180 degrees and saves it as <name>.npy next to the rotated JPEG. This
is the "*_rot180/*.npy" convention expected by
RadiometricCalibration/correct_session.py (--flir-dir), SensorFusionLoader's
rig_calibration.yaml (flir.rotated_180_before_calibration) and
MATLAB_SensorFusionValidation/FlirLidarZedViewer.m.

Originals are never touched: output goes to a sibling folder
"<pose>_rot180" (rotated frames, same file names).

Usage:
    py rotate_flir_poses.py --poses-root C:\\...\\NewCalibration\\Flir\\Poses
    py rotate_flir_poses.py --pose-dir   C:\\...\\Flir\\Poses\\pose_01
    py rotate_flir_poses.py --poses-root <...>\\Poses --dry-run
    py rotate_flir_poses.py --pose-dir   <...>\\session9_only --npy
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("Requires opencv-python:  py -m pip install opencv-python")

IMG_EXTS = (".jpg", ".jpeg", ".rjpg")


def rotate_pose_dir(pose_dir: Path, suffix: str, dry_run: bool, save_npy: bool) -> int:
    files = sorted(
        f for f in pose_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    if not files:
        return 0

    out_dir = pose_dir.parent / (pose_dir.name + suffix)
    print(f"{pose_dir} -> {out_dir}  ({len(files)} frames{', +npy' if save_npy else ''})")
    if dry_run:
        return len(files)

    if save_npy:
        import numpy as np
        try:
            import flyr
        except ImportError:
            sys.exit("--npy requires the 'flyr' package:  py -m pip install flyr")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  SKIP (unreadable): {f.name}")
            continue
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        cv2.imwrite(str(out_dir / f.name), rotated)

        if save_npy:
            temp = flyr.unpack(str(f)).celsius
            temp_rot = np.rot90(temp, 2)
            # float32 ('<f4'): flyr returns float64, but the *_rot180/*.npy
            # convention is float32 -- MATLAB_SensorFusionValidation's minimal
            # readNpyFloat32 reader rejects anything else outright.
            np.save(out_dir / (f.stem + ".npy"), temp_rot.astype(np.float32))

        n_ok += 1
    return n_ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--poses-root", help="Folder containing pose_XX subfolders.")
    g.add_argument("--pose-dir", help="Single pose_XX folder.")
    p.add_argument("--suffix", default="_rot180",
                    help="Output folder suffix (default: _rot180).")
    p.add_argument("--dry-run", action="store_true",
                    help="Show what would be done without writing anything.")
    p.add_argument("--npy", action="store_true",
                    help="Also save each frame's rotated apparent-temperature map "
                         "(deg C, via flyr) as <name>.npy next to the rotated JPEG.")
    args = p.parse_args()

    if args.pose_dir:
        pose_dirs = [Path(args.pose_dir)]
    else:
        root = Path(args.poses_root)
        pose_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.endswith(args.suffix))

    total = 0
    for pd in pose_dirs:
        total += rotate_pose_dir(pd, args.suffix, args.dry_run, args.npy)

    verb = "to rotate" if args.dry_run else "rotated"
    print(f"Total frames {verb}: {total}")


if __name__ == "__main__":
    main()
