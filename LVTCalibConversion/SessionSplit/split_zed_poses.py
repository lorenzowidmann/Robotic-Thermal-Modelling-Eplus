"""
Extracts, from a zed_record.py session, one subfolder per pose ready for
zed_frame_publisher.py --session-dir, using the ranges already computed by
zed_pose_check.py (zed_check.csv: columns posa, primo_file, ultimo_file,
verdetto).

zed_frame_publisher.py ALWAYS reads the whole session passed to it (no time
window flag): to isolate a single pose in the LVT2Calib pipeline you
therefore need a physical subfolder with only that pose's frames plus a
filtered metadata.json (same zed_record/v1 schema, subset of "frames").
This script does exactly that, for every pose with a given verdict
(default: INSIDE only).

The script reads the original session read-only and WRITES only inside
--outdir (never into the source session).

Typical usage:
    py split_zed_poses.py --session <zed_session> --check-csv zed_check.csv --outdir <NewCalibration>\\Zed\\Poses
    py split_zed_poses.py --session <zed_session> --check-csv zed_check.csv --outdir <out> --verdict INSIDE CLIPPED
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path


def load_metadata(session_dir):
    meta_path = session_dir / "metadata.json"
    if not meta_path.is_file():
        sys.exit(f"metadata.json not found in {session_dir}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_check_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(
        description="Extracts per-pose subfolders from a ZED session, using "
                    "the ranges from zed_pose_check.py (zed_check.csv)."
    )
    ap.add_argument("--session", required=True,
                    help="zed_record.py session folder (with frames/ and metadata.json)")
    ap.add_argument("--check-csv", required=True,
                    help="zed_check.csv produced by zed_pose_check.py")
    ap.add_argument("--outdir", required=True,
                    help="Destination folder; each pose goes into <outdir>/pose_NN/")
    ap.add_argument("--verdict", nargs="+", default=["INSIDE"],
                    help="Verdicts to include, exact string match (default: INSIDE). "
                         "Passing 'CLIPPED' only matches when the verdict is exactly "
                         "'CLIPPED' with no sides listed in brackets; "
                         "to include those too use --verdict-prefix CLIPPED")
    ap.add_argument("--verdict-prefix", nargs="*", default=[],
                    help="Verdicts to include by prefix (e.g. CLIPPED to also pick up "
                         "'CLIPPED [L]')")
    args = ap.parse_args()

    session_dir = Path(args.session)
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"frames/ not found in {session_dir}")
    meta = load_metadata(session_dir)
    manifest = meta.get("frames") or []
    by_file = {fr["file"]: fr for fr in manifest if "file" in fr}

    rows = load_check_rows(args.check_csv)
    wanted = []
    for r in rows:
        v = r["verdetto"]
        if v in args.verdict or any(v.startswith(p) for p in args.verdict_prefix):
            wanted.append(r)

    if not wanted:
        sys.exit("No pose matches the requested verdicts.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    on_disk = sorted(f.name for f in frames_dir.glob("*.png"))

    for r in wanted:
        pose = int(r["posa"])
        f0, f1 = r["primo_file"], r["ultimo_file"]
        if f0 not in on_disk or f1 not in on_disk:
            print(f"WARNING pose {pose}: {f0} or {f1} not found in {frames_dir}, skipping.")
            continue
        i0, i1 = on_disk.index(f0), on_disk.index(f1)
        subset_names = on_disk[i0:i1 + 1]

        pose_dir = outdir / f"pose_{pose:02d}"
        pose_frames_dir = pose_dir / "frames"
        pose_frames_dir.mkdir(parents=True, exist_ok=True)

        n_copied = 0
        for name in subset_names:
            dst = pose_frames_dir / name
            if not dst.exists():
                shutil.copy2(frames_dir / name, dst)
            n_copied += 1

        subset_manifest = [by_file[n] for n in subset_names if n in by_file]
        if len(subset_manifest) != len(subset_names):
            print(f"WARNING pose {pose}: {len(subset_names)} files copied but only "
                  f"{len(subset_manifest)} present in the original manifest "
                  "(zed_frame_publisher will ignore them).")

        pose_meta = dict(meta)
        pose_meta["frames"] = subset_manifest
        (pose_dir / "metadata.json").write_text(
            json.dumps(pose_meta, indent=2), encoding="utf-8")

        print(f"pose {pose:>2}  [{r['verdetto']:<10}]  {f0} .. {f1}  "
              f"-> {pose_dir}  ({n_copied} frames)")

    print(f"\nDone. {len(wanted)} poses written under {outdir}")
    print("Use with zed_frame_publisher.py: --session-dir /data/bags/.../Poses/pose_NN")


if __name__ == "__main__":
    main()
