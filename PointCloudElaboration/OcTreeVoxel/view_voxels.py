"""PyVista viewer for aligned_octree.py's voxel output.

Shows the occupied voxels as class-free cubes colored by point count (this
pipeline has no semantic labels, geometry/alignment only) -- the fastest way
to see whether the leveling in aligned_octree.py actually came out
Manhattan-aligned: a correctly aligned corridor renders with its walls/floor/
ceiling square to the view, not diagonal.

Also overlays, when available:
- the closed box from planes_aligned.json (already in the aligned frame --
  see aligned_octree.align_and_reclose_planes), should hug the voxel walls
  at an exact 90 degrees if the alignment is right.
- the raw aligned points (re-loaded from the bag and re-transformed, same as
  aligned_octree.py does) -- off by default, expensive at full resolution, so
  --points-stride subsamples for display only.

Color is linear point-count by default, clipped to the 98th percentile
(--clim-percentile) so a handful of very dense voxels don't crush the rest of
the (heavily right-skewed) count distribution into one end of the colormap --
--log-color switches to log10(count) instead.

In the interactive window, the min-count filter (--min-count) is live: the
scene rebuilds in place, no relaunch needed.
    ]     raise min-count by --min-count-step (default 1)
    [     lower min-count by --min-count-step
    0     reset min-count to 1 (show every voxel)
    d     toggle declutter (drop floating voxels, see drop_floating_voxels) --
          on by default; raising min-count can strand a voxel that used to
          touch a since-hidden neighbour, and declutter drops it
(Not bound to +/-/Up/Down: pyvista's own defaults already use those for
camera zoom and point size, so reusing them would fire both at once.)
The current threshold, declutter state, and voxel/point counts print to the
console (and show as on-screen text) on every change.

Usage:
    python view_voxels.py                              # interactive window
    python view_voxels.py --min-count 3                 # hide sparse voxels
    python view_voxels.py --min-count-step 5             # bigger steps per key press
    python view_voxels.py --no-declutter                 # keep floating voxels
    python view_voxels.py --points --points-stride 20    # overlay raw points
    python view_voxels.py --no-planes                    # hide the box overlay
    python view_voxels.py --screenshot out.png            # headless render

Venv: C:\\venvs\\planefit (same as the rest of this folder; pyvista is
already in it, see requirements.txt).
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "OcTreeVoxel_out"  # where aligned_octree.py writes its results

# Voxels are colored by point count on a perceptually-uniform colormap,
# linear by default (--log-color switches to log10(count) -- useful when one
# voxel holds orders of magnitude more points than another, e.g. a wall seen
# face-on vs. grazing incidence, and the linear map looks flat).
DEFAULT_COLORMAP = "viridis"
POINT_COLOR = "black"
POINT_SIZE = 2.0
BOX_COLOR = "red"
BOX_LINE_WIDTH = 3


def load_voxels(path):
    d = np.load(path)
    return d["centers"], d["counts"], float(d["voxel_size"]), d["origin"], int(d["depth"])


def drop_floating_voxels(centers, origin, voxel_size):
    """Keep only the largest 26-connected component of occupied voxels.

    Raising the min-count filter can strand a few voxels that used to touch
    a since-hidden neighbour: they survive the count threshold but no longer
    touch anything else shown, so they render as specks floating in empty
    space. This labels connected components on the actual voxel lattice
    (recovering each voxel's integer (i,j,k) cell index from its center,
    inverting voxelizer.py's `center = (idx + 0.5) * voxel_size + origin`)
    and drops every component except the largest -- the same "declutter"
    idea fit_closed_planes.py already applies to raw points (its
    cluster_labels/largest_cluster_mask), just done on the voxel grid
    instead. 26-connected
    (any shared face, edge, or corner counts as touching) matches how
    adjacent cubes visually read as "connected" in the render.

    Returns a boolean mask into `centers` (True = keep).
    """
    from scipy import ndimage

    if len(centers) <= 1:
        return np.ones(len(centers), dtype=bool)

    idx = np.round((centers - origin) / voxel_size - 0.5).astype(np.int64)
    lo = idx.min(axis=0)
    shape = tuple((idx.max(axis=0) - lo + 1).tolist())
    local = idx - lo

    grid = np.zeros(shape, dtype=bool)
    grid[local[:, 0], local[:, 1], local[:, 2]] = True

    labels, _n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=np.int8))
    comp_id = labels[local[:, 0], local[:, 1], local[:, 2]]
    sizes = np.bincount(comp_id)
    largest = int(np.argmax(sizes[1:]) + 1) if len(sizes) > 1 else 0
    return comp_id == largest


def _cube_glyphs(centers, size):
    import pyvista as pv

    pdata = pv.PolyData(centers)
    cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
    return pdata.glyph(geom=cube, scale=False, orient=False, factor=size)


def load_box_wireframe(planes_aligned_path):
    """The closed box's rectangle edges (planes_aligned.json's corners_3d,
    already in the aligned frame -- see aligned_octree.align_and_reclose_planes)
    as a single pyvista line-segment mesh for overlay.

    Deliberately NOT planes.json's corners rotated by transform.json here:
    planes.json's box was axis-snapped against the *original* (pre-alignment)
    frame, which doesn't exactly match aligned_octree.py's more precise
    rotation -- rotating it naively leaves a small residual tilt (a couple
    degrees), which is exactly the "voxels aren't at 90 degrees to the
    planes" symptom. planes_aligned.json re-closes the box *after* rotating,
    so it's exactly axis-aligned, consistent with the (always axis-aligned by
    construction) voxel grid."""
    import pyvista as pv

    planes = json.loads(Path(planes_aligned_path).read_text())["planes"]

    pts = []
    for p in planes:
        corners = np.asarray(p["corners_3d"], dtype=float)
        for i in range(4):
            pts.append(corners[i])
            pts.append(corners[(i + 1) % 4])
    pts = np.asarray(pts)

    mesh = pv.PolyData(pts)
    n_lines = len(pts) // 2
    cells = np.empty(n_lines * 3, dtype=np.int64)
    cells[0::3] = 2
    cells[1::3] = np.arange(0, len(pts), 2)
    cells[2::3] = np.arange(1, len(pts), 2)
    mesh.lines = cells
    return mesh


def load_aligned_points(transform_path, stride=1):
    """Re-load the raw bag (fit_closed_planes.load_merged_cloud) and
    re-apply the same rigid transform aligned_octree.py used -- for display
    only, not recomputed from the voxels."""
    from fit_closed_planes import load_merged_cloud

    t = json.loads(Path(transform_path).read_text())
    xyz = load_merged_cloud(Path(t["bag"]), t["topic"], t["store"])
    R = np.array(t["rotation"])
    translation = np.array(t["translation"])
    aligned = xyz @ R.T + translation
    return aligned[::stride] if stride > 1 else aligned


# Keysyms bound to raise/lower the live min-count threshold: ']' / '['.
# Deliberately NOT Up/Down/plus/minus -- pyvista's own defaults already bind
# those to camera zoom and point-size (BasePlotter.reset_key_events); reusing
# them would fire both behaviors on one keypress. Brackets are unclaimed by
# both pyvista's defaults and VTK's interactor-style bindings (w/s/r/3/e/f/p/j/t).
INCREASE_KEY = "bracketright"  # ']'
DECREASE_KEY = "bracketleft"   # '['
RESET_KEY = "0"
DECLUTTER_KEY = "d"  # toggle drop_floating_voxels; 'd' is unclaimed by pyvista/VTK defaults


def build_plotter(centers, counts, voxel_size, min_count, colormap, log_color,
                   box_mesh, points, off_screen, clim_percentile=98.0, min_count_step=1,
                   origin=None, declutter=True):
    import pyvista as pv

    scalar_bar_title = "log10(points/voxel)" if log_color else "points/voxel"

    def scalars_for(shown_counts):
        return np.log10(shown_counts.astype(float)) if log_color else shown_counts.astype(float)

    # Fixed color range computed once from the *full* (min-count=1)
    # distribution, not recomputed as the live filter changes -- otherwise
    # raising min-count would keep rescaling what each color means underfoot.
    # See the module docstring for why a plain min/max range crushes the
    # (heavily right-skewed) count distribution into one end of the colormap.
    clim = None
    if clim_percentile > 0 and len(counts):
        all_scalars = scalars_for(counts)
        lo = float(all_scalars.min())
        hi = float(np.percentile(all_scalars, clim_percentile))
        if hi > lo:
            clim = (lo, hi)

    pl = pv.Plotter(window_size=(1400, 900), off_screen=off_screen)
    pl.set_background("white")

    grid_origin = origin if origin is not None else centers.min(axis=0)
    state = {"min_count": int(min_count), "declutter": bool(declutter)}

    def render_voxels():
        keep = counts >= state["min_count"]
        shown_centers, shown_counts = centers[keep], counts[keep]
        n_after_count = len(shown_centers)

        if state["declutter"] and n_after_count > 1:
            main_mask = drop_floating_voxels(shown_centers, grid_origin, voxel_size)
            shown_centers, shown_counts = shown_centers[main_mask], shown_counts[main_mask]

        n_dropped_floating = n_after_count - len(shown_centers)
        print(f"voxels: {len(centers)} total, {n_after_count} pass min-count="
              f"{state['min_count']}, {n_dropped_floating} dropped as floating "
              f"(declutter={'on' if state['declutter'] else 'off'}), "
              f"{len(shown_centers)} shown; count range [{counts.min()}, {counts.max()}]")

        if len(shown_centers) == 0:
            if "voxels" in pl.actors:
                pl.remove_actor("voxels", render=False)
        else:
            glyphs = _cube_glyphs(shown_centers, voxel_size)
            glyphs["count"] = np.repeat(scalars_for(shown_counts),
                                         glyphs.n_cells // len(shown_centers))
            pl.add_mesh(glyphs, scalars="count", cmap=colormap, clim=clim,
                        show_scalar_bar=True, scalar_bar_args={"title": scalar_bar_title},
                        name="voxels")

        # Plain words, not the literal bracket characters: VTK's default HUD
        # font doesn't render '[' ']' (they show up as empty parens).
        pl.add_text(
            f"min-count: {state['min_count']}   "
            f"(right/left bracket key: change by {min_count_step}, 0: reset)\n"
            f"declutter (drop floating voxels): {'on' if state['declutter'] else 'off'}   "
            f"(d: toggle)",
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

    if points is not None:
        pl.add_points(points, color=POINT_COLOR, point_size=POINT_SIZE,
                      render_points_as_spheres=False, name="points")

    pl.add_axes()
    return pl


def make_orbit_gif(pl, path, n_frames=36, factor=2.5, viewup=(0.0, 0.0, 1.0)):
    """Write a 360-degree orbit around the current scene to an animated GIF
    (pyvista's generate_orbital_path/orbit_on_path -- a fixed circular path
    around the bounds already added to `pl`, not a live rotate-with-the-mouse
    session). `factor` scales the orbit radius relative to the scene bounds."""
    pl.open_gif(str(path))
    orbit_path = pl.generate_orbital_path(factor=factor, n_points=n_frames, viewup=list(viewup))
    pl.orbit_on_path(orbit_path, write_frames=True, viewup=list(viewup))
    pl.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voxels", type=Path, default=OUT_DIR / "voxels.npz")
    ap.add_argument("--transform", type=Path, default=OUT_DIR / "transform.json")
    ap.add_argument("--planes-aligned", type=Path, default=OUT_DIR / "planes_aligned.json",
                    help="closed box overlay, output of aligned_octree.py "
                         "(already in the aligned frame -- see "
                         "align_and_reclose_planes for why this isn't just "
                         "planes.json rotated by --transform)")
    ap.add_argument("--no-planes", action="store_true", help="don't overlay the closed box")
    ap.add_argument("--min-count", type=int, default=1,
                    help="hide voxels with fewer than this many points (starting "
                         "value -- live-adjustable in the interactive window, see "
                         "--min-count-step / key bindings in the module docstring)")
    ap.add_argument("--min-count-step", type=int, default=1,
                    help="how much ]/[ (or Up/Down) change min-count by per press")
    ap.add_argument("--colormap", default=DEFAULT_COLORMAP)
    ap.add_argument("--log-color", action="store_true",
                    help="color by log10(count) instead of raw point count "
                         "(useful when a few voxels' counts dwarf the rest)")
    ap.add_argument("--clim-percentile", type=float, default=98.0,
                    help="clip the color range to this percentile of shown voxels' "
                         "counts, so a few outlier voxels don't crush the rest into "
                         "one end of the colormap (0 = off, use full min/max range)")
    ap.add_argument("--points", action="store_true",
                    help="overlay the raw aligned points (re-reads the bag)")
    ap.add_argument("--points-stride", type=int, default=20,
                    help="keep every Nth raw point for display (--points only)")
    ap.add_argument("--screenshot", type=Path, default=None,
                    help="headless render to this image file instead of opening a window")
    ap.add_argument("--orbit-gif", type=Path, default=None,
                    help="headless: write a 360-degree orbit animation to this .gif "
                         "instead of opening a window")
    ap.add_argument("--orbit-frames", type=int, default=36)
    ap.add_argument("--no-declutter", action="store_true",
                    help="don't drop floating voxels (islands disconnected from the "
                         "largest connected component) -- on by default, live-togglable "
                         "with the 'd' key in the interactive window")
    args = ap.parse_args()

    centers, counts, voxel_size, origin, depth = load_voxels(args.voxels)
    depth_str = f"depth={depth}" if depth >= 0 else "depth=n/a (uniform grid, --voxel-size)"
    print(f"loaded {args.voxels}: {len(centers)} voxels, {depth_str}, "
          f"voxel_size={voxel_size:.4f} m")

    box_mesh = None
    if not args.no_planes and args.planes_aligned.exists():
        box_mesh = load_box_wireframe(args.planes_aligned)

    points = None
    if args.points:
        points = load_aligned_points(args.transform, stride=args.points_stride)
        print(f"overlaying {len(points)} raw points (stride {args.points_stride})")

    off_screen = args.screenshot is not None or args.orbit_gif is not None
    pl = build_plotter(
        centers, counts, voxel_size, args.min_count, args.colormap,
        args.log_color, box_mesh, points, off_screen=off_screen,
        clim_percentile=args.clim_percentile, min_count_step=args.min_count_step,
        origin=origin, declutter=not args.no_declutter)

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
