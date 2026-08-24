"""
Generate the LVT2Calib PCD templates with the real calibration board geometry.

Board: 100 x 70 cm, 4 circular holes of diameter 13 cm (radius 0.065 m),
centred at the four points (+-0.15, +-0.15) m relative to the board centre.

Writes two files, using the same names LVT2Calib expects:
  - four_circle_boundary.pcd : outlines only (outer rectangle + 4 circles).
        This is the one `model_path` points at in livox_pattern.launch, and
        the one used for the PCA + ICP registration in isCalibBoard().
  - four_circle_dense.pcd    : the filled board surface with the 4 holes cut
        out (used for visualisation / other comparisons).

Convention: planar cloud on the XY plane (z = 0), centred on the origin.
Registration is PCA-based, so what matters is the relative geometry, not the
absolute orientation.

IMPORTANT (fix 2026-07-29): the repo's original PCD
(four_circle_boundary_ORIGINAL.pcd) uses the 5-field schema
"x y z intensity range". The first version of this script wrote only 4 fields
("x y z intensity"); that schema mismatch can cause a silently wrong load in
pcl::io::loadPCDFile (no explicit error, but data read/aligned incorrectly).
This version writes the same 5-field schema as the original.

Usage:
    py generate_board_template.py --outdir <output_folder>

Then copy the two .pcd files into  lvt2calib/data/template_pcl/  (overwriting
the originals, after making a backup copy of them).
"""

import argparse
import math
from pathlib import Path


# ----------------------------- board geometry ------------------------------
BOARD_W = 1.00          # board width [m]
BOARD_H = 0.70          # board height [m]
HOLE_DIAMETER = 0.13    # hole diameter [m]
HOLE_OFFSET = 0.15      # offset of the hole centres from the board centre [m]


def hole_centers(offset: float):
    """The 4 centres, at the corners of a square of side 2*offset."""
    return [
        (-offset, +offset),
        (+offset, +offset),
        (+offset, -offset),
        (-offset, -offset),
    ]


def sample_rectangle_outline(w: float, h: float, spacing: float):
    """Points along the rectangle perimeter, step `spacing`."""
    pts = []
    hw, hh = w / 2.0, h / 2.0

    n_horiz = max(2, int(round(w / spacing)) + 1)
    for i in range(n_horiz):
        x = -hw + w * i / (n_horiz - 1)
        pts.append((x, +hh))
        pts.append((x, -hh))

    n_vert = max(2, int(round(h / spacing)) + 1)
    for i in range(1, n_vert - 1):     # corners already added above
        y = -hh + h * i / (n_vert - 1)
        pts.append((+hw, y))
        pts.append((-hw, y))

    return pts


def sample_circle_outline(cx: float, cy: float, radius: float, spacing: float):
    """Points along the circumference, step `spacing` measured along the arc."""
    circumference = 2.0 * math.pi * radius
    n = max(8, int(round(circumference / spacing)))
    return [
        (
            cx + radius * math.cos(2.0 * math.pi * i / n),
            cy + radius * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def sample_board_surface(w: float, h: float, radius: float,
                         centers, spacing: float):
    """Grid over the board surface, excluding the interior of the holes."""
    pts = []
    hw, hh = w / 2.0, h / 2.0
    nx = int(round(w / spacing)) + 1
    ny = int(round(h / spacing)) + 1

    r2 = radius * radius
    for ix in range(nx):
        x = -hw + w * ix / (nx - 1)
        for iy in range(ny):
            y = -hh + h * iy / (ny - 1)
            inside_hole = any(
                (x - cx) ** 2 + (y - cy) ** 2 <= r2 for cx, cy in centers
            )
            if not inside_hole:
                pts.append((x, y))
    return pts


def write_pcd(path: Path, points, intensity: float = 0.0) -> None:
    """Write an ASCII PCD with fields x y z intensity range (5 fields),
    the same schema as the original four_circle_boundary_ORIGINAL.pcd.
    """
    n = len(points)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity range\n"
        "SIZE 4 4 4 4 4\n"
        "TYPE F F F F F\n"
        "COUNT 1 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with path.open("w", encoding="ascii") as fh:
        fh.write(header)
        for x, y in points:
            fh.write(f"{x:.6f} {y:.6f} 0.000000 {intensity:.6f} 0.000000\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the LVT2Calib calibration-board PCD templates."
    )
    parser.add_argument("--outdir", required=True,
                        help="Output folder for the two .pcd files")
    parser.add_argument("--board-width", type=float, default=BOARD_W,
                        help=f"Board width in metres (default {BOARD_W})")
    parser.add_argument("--board-height", type=float, default=BOARD_H,
                        help=f"Board height in metres (default {BOARD_H})")
    parser.add_argument("--hole-diameter", type=float, default=HOLE_DIAMETER,
                        help=f"Hole diameter in metres (default {HOLE_DIAMETER}). "
                             "MEASURE IT WITH CALLIPERS after cutting and update this.")
    parser.add_argument("--hole-offset", type=float, default=HOLE_OFFSET,
                        help=f"Offset of the centres from the board centre (default {HOLE_OFFSET})")
    parser.add_argument("--boundary-spacing", type=float, default=0.004,
                        help="Sampling step along the outlines [m] (default 0.004)")
    parser.add_argument("--dense-spacing", type=float, default=0.004,
                        help="Sampling step over the surface [m] (default 0.004)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    radius = args.hole_diameter / 2.0
    centers = hole_centers(args.hole_offset)

    # --- geometric sanity checks ---
    hw, hh = args.board_width / 2.0, args.board_height / 2.0
    for cx, cy in centers:
        if abs(cx) + radius > hw or abs(cy) + radius > hh:
            raise SystemExit(
                f"Hole at ({cx:+.3f}, {cy:+.3f}) with radius {radius:.3f} m "
                f"falls outside the {args.board_width}x{args.board_height} m board edges."
            )
    gap = 2.0 * args.hole_offset - args.hole_diameter
    if gap <= 0:
        raise SystemExit("The holes overlap: increase --hole-offset "
                         "or reduce --hole-diameter.")

    # --- boundary: outer rectangle + 4 circles ---
    boundary = sample_rectangle_outline(args.board_width, args.board_height,
                                        args.boundary_spacing)
    for cx, cy in centers:
        boundary.extend(sample_circle_outline(cx, cy, radius,
                                              args.boundary_spacing))

    # --- dense: surface with the holes cut out ---
    dense = sample_board_surface(args.board_width, args.board_height,
                                 radius, centers, args.dense_spacing)

    boundary_path = outdir / "four_circle_boundary.pcd"
    dense_path = outdir / "four_circle_dense.pcd"
    write_pcd(boundary_path, boundary)
    write_pcd(dense_path, dense)

    dist_centro = math.hypot(args.hole_offset, args.hole_offset)

    print("Geometry used:")
    print(f"  board                : {args.board_width:.3f} x {args.board_height:.3f} m")
    print(f"  hole                 : diameter {args.hole_diameter:.3f} m "
          f"(radius {radius:.4f} m)")
    print(f"  hole centres         : (+-{args.hole_offset:.3f}, +-{args.hole_offset:.3f}) m")
    print(f"  centre-to-board dist : {dist_centro:.4f} m")
    print(f"  gap between holes    : {gap:.3f} m")
    print()
    print("Matching parameters for config/lidar_pattern_param.yaml:")
    print(f"  circle_radius: {radius:.4f}")
    print(f"  centroid_dis_min: {dist_centro - 0.06:.2f}   # {dist_centro:.4f} with margin")
    print(f"  centroid_dis_max: {dist_centro + 0.06:.2f}")
    print()
    print(f"Wrote {boundary_path}  ({len(boundary)} points)")
    print(f"Wrote {dense_path}  ({len(dense)} points)")


if __name__ == "__main__":
    main()
