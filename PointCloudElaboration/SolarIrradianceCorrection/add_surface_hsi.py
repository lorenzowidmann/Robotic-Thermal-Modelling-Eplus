"""Add surface_type + hsi_used columns to a thermal_voxels.csv (voxel_consensus.py
--stage thermal output), via nearest-plane geometry.

Why this exists
----------------
Session 6's original thermal_voxels_u.csv (20260730_154343) has 'surface_type'
(floor/wall/ceiling) and 'hsi_used' (5.9/7.7/10.0, the ISO 6946 internal
surface heat transfer coefficients for downward/horizontal/upward heat flow)
columns that voxel_u_value.py in this repo does not know how to produce --
that file was made by a since-lost script version. This reconstructs the
same per-surface-type hsi assignment geometrically, so a rerun (e.g. after
the zones.py ceiling-classification fix) stays comparable to the original
run instead of silently falling back to one --hsi for every voxel.

Assignment: for each voxel, distance to --floor-plane-id and --ceiling-plane-id
(both usually orientation="floor_ceiling", same normal direction, distinguished
by their own d). Whichever is closer AND within --plane-threshold wins that
surface_type; otherwise "wall" (every voxel not close to a floor/ceiling plane
is a wall in a rectangular room -- consistent with session 6's own partition,
1542 floor + 1436 ceiling + 2482 wall = 5460, no leftover "other" bucket).

voxel_u_value.py's --hsi becomes redundant once this has run: it can read
hsi_used per-voxel the same way voxel_solar_ns.py already does. Run this
BEFORE voxel_u_value.py.

Usage:
    py add_surface_hsi.py --in thermal_voxels.csv --planes planes_session6.json \\
        --floor-plane-id 0 --ceiling-plane-id 4
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--planes", type=Path, required=True)
    ap.add_argument("--floor-plane-id", type=int, required=True)
    ap.add_argument("--ceiling-plane-id", type=int, required=True)
    ap.add_argument("--plane-threshold", type=float, default=0.15, metavar="M")
    ap.add_argument("--hsi-floor", type=float, default=5.9, metavar="W_M2K",
                    help="ISO 6946 downward heat flow (default 5.9)")
    ap.add_argument("--hsi-wall", type=float, default=7.7, metavar="W_M2K",
                    help="ISO 6946 horizontal heat flow (default 7.7)")
    ap.add_argument("--hsi-ceiling", type=float, default=10.0, metavar="W_M2K",
                    help="ISO 6946 upward heat flow (default 10.0)")
    ap.add_argument("--out", type=Path, default=None, help="default: <in>_hsi.csv")
    args = ap.parse_args()

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")

    plane_data = json.loads(args.planes.read_text())
    by_id = {p["id"]: p for p in plane_data["planes"]}
    for pid in (args.floor_plane_id, args.ceiling_plane_id):
        if pid not in by_id:
            raise SystemExit(f"no plane with id {pid} in {args.planes}")

    def dist_to(pid):
        p = by_id[pid]
        n = np.asarray(p["normal"], dtype=float)
        d = float(p["d"])
        return np.abs(xyz @ n + d) / np.linalg.norm(n)

    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    d_floor = dist_to(args.floor_plane_id)
    d_ceil = dist_to(args.ceiling_plane_id)

    surface_type = np.full(len(rows), "wall", dtype=object)
    hsi_used = np.full(len(rows), args.hsi_wall)

    is_floor = (d_floor <= args.plane_threshold) & (d_floor <= d_ceil)
    is_ceiling = (d_ceil <= args.plane_threshold) & (d_ceil < d_floor)
    surface_type[is_floor] = "floor"
    surface_type[is_ceiling] = "ceiling"
    hsi_used[is_floor] = args.hsi_floor
    hsi_used[is_ceiling] = args.hsi_ceiling

    for r, st, h in zip(rows, surface_type, hsi_used):
        r["surface_type"] = st
        r["hsi_used"] = round(float(h), 3)

    out = args.out or args.inp.with_name(args.inp.stem + "_hsi.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    print(f"{len(rows)} voxel(s): {dict(Counter(surface_type.tolist()))}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
