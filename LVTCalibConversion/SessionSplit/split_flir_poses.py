"""
Extracts, from one or more FLIR raw folders, one subfolder per pose ready for
flir_frame_publisher.py --image-dir, using the ranges already computed by
detect_board_poses.py --check-inside (flir_check.csv: columns posa,
primo_file, ultimo_file, verdetto -- see LVTCalibConversion/RecordCheck).

flir_frame_publisher.py reads a plain folder (glob over jpg/jpeg/rjpg), no
manifest: unlike split_zed_poses.py this script only needs to copy the raw
*_R.jpg files into <outdir>/pose_NN/, no metadata.json to filter/rewrite.

FLIR filenames (YYYYMMDD_HHMMSS_R.jpg) sort chronologically as plain strings,
so no timestamp parsing is needed here either -- same reasoning split_zed_poses.py
relies on for ZED's zero-padded frame index.

A long session may be split across several SD-card folders (100_FLIR,
101_FLIR, ...): pass them all to --raw-dir, they are pooled and sorted as one
stream before slicing out each pose's range (same convention as
detect_board_poses.py, which is what produced --check-csv from these files).

The script reads the source folders read-only and WRITES only inside
--outdir (never into the source folders).

Typical usage:
    py split_flir_poses.py --raw-dir <flir_dir> --check-csv flir_check.csv --outdir <NewCalibration>\\Flir\\Poses
    py split_flir_poses.py --raw-dir D:\\DCIM\\100_FLIR D:\\DCIM\\101_FLIR --check-csv flir_check.csv --outdir <out>
    py split_flir_poses.py --raw-dir <flir_dir> --check-csv flir_check.csv --outdir <out> --verdict INSIDE CLIPPED
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path


def load_check_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser(
        description="Extracts per-pose subfolders from FLIR raw folder(s), using "
                    "the ranges from detect_board_poses.py --check-inside (flir_check.csv)."
    )
    ap.add_argument("--raw-dir", nargs="+", required=True,
                    help="One or more FLIR raw folders (in order), e.g. a session "
                         "split by the SD card into 100_FLIR, 101_FLIR, ...")
    ap.add_argument("--check-csv", required=True,
                    help="flir_check.csv produced by "
                         "detect_board_poses.py <flir_dir> --check-inside --csv-out")
    ap.add_argument("--outdir", required=True,
                    help="Destination folder; each pose goes into <outdir>/pose_NN/")
    ap.add_argument("--pattern", default="*_R.jpg",
                    help="Pattern of the radiometric files, case-insensitive "
                         "(default *_R.jpg, same as detect_board_poses.py)")
    ap.add_argument("--verdict", nargs="+", default=["INSIDE"],
                    help="Verdicts to include, exact string match (default: INSIDE). "
                         "Passing 'CLIPPED' only matches when the verdict is exactly "
                         "'CLIPPED' with no sides listed in brackets; "
                         "to include those too use --verdict-prefix CLIPPED")
    ap.add_argument("--verdict-prefix", nargs="*", default=[],
                    help="Verdicts to include by prefix (e.g. CLIPPED to also pick up "
                         "'CLIPPED [L]')")
    args = ap.parse_args()

    on_disk = []
    for d in args.raw_dir:
        p = Path(d)
        if not p.is_dir():
            sys.exit(f"Folder not found: {p}")
        found = list(p.glob(args.pattern))
        if not found:
            print(f"WARNING: no {args.pattern} file in {p}")
        on_disk.extend(found)
    if not on_disk:
        sys.exit(f"No {args.pattern} file found in any --raw-dir.")

    by_name = {}
    for f in on_disk:
        if f.name in by_name:
            sys.exit(f"Duplicate filename across --raw-dir folders: {f.name} "
                     f"({by_name[f.name]} and {f})")
        by_name[f.name] = f
    on_disk_names = sorted(by_name)  # YYYYMMDD_HHMMSS_R.jpg sorts chronologically as text

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

    for r in wanted:
        pose = int(r["posa"])
        f0, f1 = r["primo_file"], r["ultimo_file"]
        if f0 not in by_name or f1 not in by_name:
            print(f"WARNING pose {pose}: {f0} or {f1} not found under --raw-dir, skipping.")
            continue
        i0, i1 = on_disk_names.index(f0), on_disk_names.index(f1)
        subset_names = on_disk_names[i0:i1 + 1]

        pose_dir = outdir / f"pose_{pose:02d}"
        pose_dir.mkdir(parents=True, exist_ok=True)

        n_copied = 0
        for name in subset_names:
            dst = pose_dir / name
            if not dst.exists():
                shutil.copy2(by_name[name], dst)
            n_copied += 1

        print(f"pose {pose:>2}  [{r['verdetto']:<10}]  {f0} .. {f1}  "
              f"-> {pose_dir}  ({n_copied} frames)")

    print(f"\nDone. {len(wanted)} poses written under {outdir}")
    print("Use with flir_frame_publisher.py: --image-dir /data/.../Poses/pose_NN")


if __name__ == "__main__":
    main()
