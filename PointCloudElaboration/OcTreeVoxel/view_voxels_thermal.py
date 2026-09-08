"""PyVista viewer for aligned_octree.py's voxel output, colored by the mean
temperature of nearby thermal-CSV voxels instead of point count.

Two different pipelines, two different grids, joined here for the first time:
- OcTreeVoxel_out/voxels.npz -- this folder's own geometry-only octree/uniform
  grid (voxel_size from aligned_octree.py, e.g. 0.15 m), in the ALIGNED
  (building-frame) coordinates aligned_octree.py produced.
- thermal_voxels.csv (EmissivityCalculation/voxel_consensus.py --stage
  thermal) -- a *different*, 0.20 m uniform grid, in the RAW/pre-alignment
  SLAM (camera_init) frame. It is never rotated/translated by
  aligned_octree.py -- nothing in that pipeline knows this file exists.

Because the two grids are different sizes and don't share a lattice origin,
a voxel center in one essentially never lands exactly on a voxel center in
the other, even after both are expressed in the same frame. So the join here
is a radius search (scipy.spatial.cKDTree.query_ball_point), not an index
lookup: each OcTreeVoxel cell gets the mean t_mean_c of every thermal-CSV
voxel within --match-radius of it (0, one, or several).

Frame join: thermal_voxels.csv's x,y,z are transformed into voxels.npz's
aligned frame with transform.json's *forward* rotation/translation --
`aligned = raw @ rotation.T + translation` -- the same direction
aligned_octree.py applied to the raw LiDAR cloud, and the same direction
view_voxels.load_aligned_points() uses to re-align the raw bag for its
`--points` overlay. NOT rotation_inv/translation_inv: those map the already-
aligned frame back to raw (transform.json's own documented convention, see
OcTreeVoxel/README.md's transform.json section), the opposite of what's
needed for a raw-frame CSV.

Before rendering anything, print_diagnostics() reports the match rate and
both point sets' aligned-frame bounding boxes, plus a match-radius
sensitivity sweep -- if the two bounding boxes don't overlap, or almost
nothing matches, that means the frame join is wrong (transform direction,
wrong bag/session pairing, ...), not that --match-radius needs nudging, and
main() prints a loud warning rather than silently rendering a mostly-grey
result.

Rendering reuses view_voxels.py's cube-glyph approach unchanged (same
_cube_glyphs, drop_floating_voxels, ]/[/0/d live keys, closed-box overlay).
The only rendering difference: color is temperature (deg C, real colorbar),
not point count, and any voxel with no thermal match within --match-radius
renders in a distinct neutral color (--unmatched-color, default lightgray)
rather than being hidden -- so it's visually obvious which cells are
measured vs. simply absent from the thermal data.

Usage:
    python view_voxels_thermal.py --thermal-csv <path/to/thermal_voxels.csv>
    python view_voxels_thermal.py --thermal-csv ... --match-radius 0.25
    python view_voxels_thermal.py --thermal-csv ... --diagnostics-only
    python view_voxels_thermal.py --thermal-csv ... --screenshot out.png
    python view_voxels_thermal.py --thermal-csv ... --orbit-gif out.gif

No default --thermal-csv: unlike voxels.npz/transform.json (this folder's
own build products, always in OcTreeVoxel_out/), thermal_voxels.csv is a
different pipeline's output living elsewhere on disk (see
EmissivityCalculation/voxel_consensus.py), so the path is always explicit.

Venv: C:\\venvs\\planefit (same as the rest of this folder; pyvista/scipy
already in it, see requirements.txt).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from view_voxels import (
    BOX_COLOR,
    BOX_LINE_WIDTH,
    OUT_DIR,
    _cube_glyphs,
    drop_floating_voxels,
    load_box_wireframe,
    load_voxels,
    make_orbit_gif,
)

# Same key bindings as view_voxels.py, unchanged (see that module's docstring
# for why these keys specifically -- unclaimed by pyvista/VTK defaults).
INCREASE_KEY = "bracketright"  # ']'
DECREASE_KEY = "bracketleft"   # '['
RESET_KEY = "0"
DECLUTTER_KEY = "d"

# One thermal-voxel edge length. Neither grid's lattice origin lines up with
# the other's (0.15 m OcTree vs 0.20 m thermal), so a cell's center in one
# grid generally sits somewhere *inside* the corresponding cell of the other,
# not on top of its center -- a radius of about one cell width catches that
# without also pulling in unrelated neighbouring cells. Sensitivity to this
# choice is printed by print_diagnostics(); tune with --match-radius if the
# match rate looks off.
DEFAULT_MATCH_RADIUS = 0.20  # metres

# Sequential (one hue, light -> dark) for a magnitude quantity with no
# meaningful zero-crossing in typical indoor ranges -- not MATLAB's 'jet'
# (ViewThermalCSV.m), which is a non-perceptually-uniform rainbow map.
# Different hue from view_voxels.py's viridis (point count) so the two
# scripts' colorbars are never visually confused for the same quantity.
DEFAULT_TEMP_COLORMAP = "plasma"
DEFAULT_UNMATCHED_COLOR = "lightgray"

# If fewer than this fraction of OcTreeVoxel cells get a thermal match,
# that's a sign of a wrong transform direction / wrong bag-session pairing,
# not a radius that merely needs nudging -- main() warns loudly instead of
# quietly rendering an almost-all-grey scene.
LOW_MATCH_RATE_WARN = 0.10

SENSITIVITY_FACTORS = (0.5, 1.0, 1.5, 2.0, 3.0)


def load_thermal_csv(path):
    """Minimal loader for voxel_consensus.py --stage thermal's
    thermal_voxels.csv (columns x,y,z,t_mean_c,t_std_c,n_obs,material,
    solar_absorptance) -- only x,y,z,t_mean_c are used here. Plain
    csv.DictReader, not pandas: this folder has no pandas dependency (see
    requirements.txt) and doesn't need one for 4 columns actually read.

    Coordinates are in the RAW (pre-alignment) SLAM frame, same as the raw
    LiDAR bag -- see the module docstring's frame-join note. Returns
    (xyz_raw (N,3), t_mean_c (N,)).
    """
    xyz, t_mean = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xyz.append((float(row["x"]), float(row["y"]), float(row["z"])))
            t_mean.append(float(row["t_mean_c"]))
    return np.asarray(xyz, dtype=float), np.asarray(t_mean, dtype=float)


def load_transform(path):
    t = json.loads(Path(path).read_text())
    return np.array(t["rotation"], dtype=float), np.array(t["translation"], dtype=float)


def to_aligned_frame(xyz_raw, rotation, translation):
    """Raw SLAM-frame -> voxels.npz's aligned frame. `aligned = raw @
    rotation.T + translation` -- transform.json's own documented forward
    convention (row-vector points), the same direction aligned_octree.py
    applied to the whole raw LiDAR cloud. See the module docstring for why
    this is the forward transform, not rotation_inv/translation_inv."""
    return xyz_raw @ rotation.T + translation


def match_temperatures(centers, thermal_xyz, thermal_t, radius):
    """For each row of `centers` (OcTreeVoxel cell centers, aligned frame),
    the mean t_mean_c of every thermal-CSV voxel within `radius` metres
    (aligned frame) -- a radius search, not an exact index match, since the
    two grids (0.15 m vs 0.20 m) don't share a lattice. Returns (temps,
    n_matches): temps[i] is NaN and n_matches[i] is 0 where nothing thermal
    fell within radius of centers[i]."""
    from scipy.spatial import cKDTree

    tree = cKDTree(thermal_xyz)
    neighbor_lists = tree.query_ball_point(centers, r=radius)
    temps = np.full(len(centers), np.nan, dtype=float)
    n_matches = np.zeros(len(centers), dtype=np.int64)
    for i, idxs in enumerate(neighbor_lists):
        if idxs:
            temps[i] = thermal_t[idxs].mean()
            n_matches[i] = len(idxs)
    return temps, n_matches


def print_diagnostics(centers, thermal_xyz, radius):
    """Bounding boxes of both aligned-frame point sets + a match-radius
    sensitivity sweep, printed BEFORE any matching/rendering -- see module
    docstring: a low match rate or non-overlapping bboxes here means the
    frame join is wrong, not that --match-radius needs tuning."""
    from scipy.spatial import cKDTree

    lo_v, hi_v = centers.min(axis=0), centers.max(axis=0)
    lo_t, hi_t = thermal_xyz.min(axis=0), thermal_xyz.max(axis=0)
    print(f"OcTreeVoxel centers (aligned frame): {len(centers)} pts, bbox "
          f"x=[{lo_v[0]:.2f},{hi_v[0]:.2f}] y=[{lo_v[1]:.2f},{hi_v[1]:.2f}] "
          f"z=[{lo_v[2]:.2f},{hi_v[2]:.2f}]")
    print(f"thermal CSV (raw->aligned):          {len(thermal_xyz)} pts, bbox "
          f"x=[{lo_t[0]:.2f},{hi_t[0]:.2f}] y=[{lo_t[1]:.2f},{hi_t[1]:.2f}] "
          f"z=[{lo_t[2]:.2f},{hi_t[2]:.2f}]")

    overlap_lo = np.maximum(lo_v, lo_t)
    overlap_hi = np.minimum(hi_v, hi_t)
    overlaps = bool(np.all(overlap_hi > overlap_lo))
    if overlaps:
        print(f"bbox overlap: yes, x=[{overlap_lo[0]:.2f},{overlap_hi[0]:.2f}] "
              f"y=[{overlap_lo[1]:.2f},{overlap_hi[1]:.2f}] "
              f"z=[{overlap_lo[2]:.2f},{overlap_hi[2]:.2f}]")
    else:
        print("bbox overlap: NO -- the two point sets do not occupy the same "
              "region of space in the aligned frame. This means the frame "
              "join is wrong (transform direction, or a mismatched bag/"
              "thermal-csv pairing), not that --match-radius needs tuning. "
              "Stopping short of rendering a meaningless result.",
              file=sys.stderr)

    tree = cKDTree(thermal_xyz)
    print("match-radius sensitivity (OcTreeVoxel cells with >=1 thermal "
          "voxel within radius):")
    for factor in SENSITIVITY_FACTORS:
        r = radius * factor
        counts = tree.query_ball_point(centers, r=r, return_length=True)
        n_matched = int((counts > 0).sum())
        print(f"  r={r:6.3f} m ({factor:g}x default): "
              f"{n_matched}/{len(centers)} matched "
              f"({100.0 * n_matched / max(1, len(centers)):.1f}%)")

    return overlaps


def build_plotter(centers, counts, temps, voxel_size, min_count, colormap,
                   unmatched_color, box_mesh, off_screen, clim=None,
                   min_count_step=1, origin=None, declutter=True):
    import pyvista as pv

    pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
    pl.set_background("white")

    grid_origin = origin if origin is not None else centers.min(axis=0)
    state = {"min_count": int(min_count), "declutter": bool(declutter)}

    # Fixed color range from the FULL matched-temperature distribution (all
    # voxels, min-count=1, declutter off), not recomputed as the live filter
    # changes -- same reasoning as view_voxels.py's clim: otherwise raising
    # min-count would keep rescaling what a color means underfoot.
    if clim is None:
        finite = temps[np.isfinite(temps)]
        clim = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)

    def render_voxels():
        keep = counts >= state["min_count"]
        shown_centers, shown_temps = centers[keep], temps[keep]
        n_after_count = len(shown_centers)

        if state["declutter"] and n_after_count > 1:
            main_mask = drop_floating_voxels(shown_centers, grid_origin, voxel_size)
            shown_centers, shown_temps = shown_centers[main_mask], shown_temps[main_mask]

        n_dropped_floating = n_after_count - len(shown_centers)
        n_matched_shown = int(np.isfinite(shown_temps).sum())
        print(f"voxels: {len(centers)} total, {n_after_count} pass min-count="
              f"{state['min_count']}, {n_dropped_floating} dropped as floating "
              f"(declutter={'on' if state['declutter'] else 'off'}), "
              f"{len(shown_centers)} shown ({n_matched_shown} with a "
              f"temperature, {len(shown_centers) - n_matched_shown} unmatched/"
              f"{unmatched_color})")

        if len(shown_centers) == 0:
            if "voxels" in pl.actors:
                pl.remove_actor("voxels", render=False)
        else:
            glyphs = _cube_glyphs(shown_centers, voxel_size)
            glyphs["temp_c"] = np.repeat(shown_temps, glyphs.n_cells // len(shown_centers))
            pl.add_mesh(glyphs, scalars="temp_c", cmap=colormap, clim=clim,
                        nan_color=unmatched_color, show_scalar_bar=True,
                        scalar_bar_args={"title": "mean temperature (deg C)"},
                        name="voxels")

        # Plain words, not the literal bracket characters: VTK's default HUD
        # font doesn't render '[' ']' (they show up as empty parens).
        pl.add_text(
            f"min-count: {state['min_count']}   "
            f"(right/left bracket key: change by {min_count_step}, 0: reset)\n"
            f"declutter (drop floating voxels): {'on' if state['declutter'] else 'off'}   "
            f"(d: toggle)   grey = no thermal match within radius",
            position="upper_left", font_size=10, name="mincount_label")
        pl.render()

    def bump(delta):
        state["min_count"] = max(1, state["min_count"] + delta)
        render_voxels()

    def reset():
        state["min_count"] = 1
        render_voxels()

    def toggle_declutter():
        state["declutter"] = not state["declutter"]
        render_voxels()

    pl.add_key_event(INCREASE_KEY, lambda: bump(min_count_step))
    pl.add_key_event(DECREASE_KEY, lambda: bump(-min_count_step))
    pl.add_key_event(RESET_KEY, reset)
    pl.add_key_event(DECLUTTER_KEY, toggle_declutter)

    render_voxels()

    if box_mesh is not None:
        pl.add_mesh(box_mesh, color=BOX_COLOR, line_width=BOX_LINE_WIDTH, name="closed_box")

    pl.add_axes()
    return pl


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", type=Path, default=OUT_DIR / "voxels.npz")
    ap.add_argument("--transform", type=Path, default=OUT_DIR / "transform.json")
    ap.add_argument("--thermal-csv", type=Path, required=True,
                    help="thermal_voxels.csv from EmissivityCalculation/"
                         "voxel_consensus.py --stage thermal (raw SLAM frame; "
                         "lives outside this folder, no default path)")
    ap.add_argument("--match-radius", type=float, default=DEFAULT_MATCH_RADIUS,
                    help=f"radius (m) to search for thermal-CSV voxels around "
                         f"each OcTreeVoxel cell center, aligned frame "
                         f"(default {DEFAULT_MATCH_RADIUS} m -- see module "
                         f"docstring; print_diagnostics() prints a sensitivity "
                         f"sweep around this value)")
    ap.add_argument("--planes-aligned", type=Path, default=OUT_DIR / "planes_aligned.json",
                    help="closed box overlay, output of aligned_octree.py")
    ap.add_argument("--no-planes", action="store_true", help="don't overlay the closed box")
    ap.add_argument("--min-count", type=int, default=1,
                    help="hide voxels with fewer than this many LiDAR points "
                         "(starting value -- live-adjustable, see ]/[/0 keys)")
    ap.add_argument("--min-count-step", type=int, default=1,
                    help="how much ]/[ change min-count by per press")
    ap.add_argument("--colormap", default=DEFAULT_TEMP_COLORMAP)
    ap.add_argument("--clim", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                    help="fixed temperature color range in deg C (default: "
                         "data's own min/max over all matched voxels)")
    ap.add_argument("--unmatched-color", default=DEFAULT_UNMATCHED_COLOR,
                    help="color for voxels with no thermal match within "
                         "--match-radius (default: %(default)s)")
    ap.add_argument("--no-declutter", action="store_true",
                    help="don't drop floating voxels (see view_voxels.py's "
                         "drop_floating_voxels)")
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="print load/match/overlap diagnostics and exit, "
                         "don't build any visualization")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="headless render to this image file instead of opening a window")
    ap.add_argument("--orbit-gif", type=Path, default=None,
                    help="headless: write a 360-degree orbit animation to this "
                         ".gif instead of opening a window")
    ap.add_argument("--orbit-frames", type=int, default=36)
    args = ap.parse_args()

    centers, counts, voxel_size, origin, depth = load_voxels(args.voxels)
    depth_str = f"depth={depth}" if depth >= 0 else "depth=n/a (uniform grid, --voxel-size)"
    print(f"loaded {args.voxels}: {len(centers)} voxels, {depth_str}, "
          f"voxel_size={voxel_size:.4f} m")

    thermal_xyz_raw, thermal_t = load_thermal_csv(args.thermal_csv)
    rotation, translation = load_transform(args.transform)
    thermal_xyz = to_aligned_frame(thermal_xyz_raw, rotation, translation)
    print(f"loaded {args.thermal_csv}: {len(thermal_xyz)} thermal voxels "
          f"(raw frame -> aligned via {args.transform})")

    overlaps = print_diagnostics(centers, thermal_xyz, args.match_radius)

    temps, n_matches = match_temperatures(centers, thermal_xyz, thermal_t, args.match_radius)
    n_matched = int((n_matches > 0).sum())
    match_rate = n_matched / max(1, len(centers))
    print(f"temperature match: {n_matched}/{len(centers)} OcTreeVoxel cells "
          f"matched within --match-radius {args.match_radius:.3f} m "
          f"({100.0 * match_rate:.1f}%), {len(centers) - n_matched} unmatched")

    if not overlaps or match_rate < LOW_MATCH_RATE_WARN:
        print(f"WARNING: match rate is very low (<{LOW_MATCH_RATE_WARN:.0%}) "
              f"and/or the bounding boxes don't overlap -- this points at a "
              f"wrong transform direction or a mismatched bag/thermal-csv "
              f"pairing, not a --match-radius that merely needs tuning. "
              f"Check the diagnostics above before trusting the render.",
              file=sys.stderr)

    if args.diagnostics_only:
        return

    box_mesh = None
    if not args.no_planes and args.planes_aligned.exists():
        box_mesh = load_box_wireframe(args.planes_aligned)

    off_screen = args.screenshot is not None or args.orbit_gif is not None
    clim = tuple(args.clim) if args.clim is not None else None
    pl = build_plotter(
        centers, counts, temps, voxel_size, args.min_count, args.colormap,
        args.unmatched_color, box_mesh, off_screen=off_screen, clim=clim,
        min_count_step=args.min_count_step, origin=origin,
        declutter=not args.no_declutter)

    if args.screenshot:
        pl.screenshot(str(args.screenshot))
        print(f"wrote {args.screenshot}")
    elif args.orbit_gif:
        make_orbit_gif(pl, args.orbit_gif, n_frames=args.orbit_frames)
        print(f"wrote {args.orbit_gif} ({args.orbit_frames} frames)")
    else:
        pl.show()


if __name__ == "__main__":
    main()
