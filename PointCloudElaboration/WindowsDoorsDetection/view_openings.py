"""Viewer for stage 2's output: the consensus opening voxels drawn as coloured
cubes (door red / window blue, the same PLY_COLORS opening_voxel_consensus.py
writes), optionally on top of the scene cloud so the openings sit in the room
rather than floating in an empty axis box.

The .ply stage 2 writes holds *only* the opening voxels -- by design, it is the
input to the polygon fit, not a picture. This script is the picture.

Three ways to get the scene behind them, in order of how much they cost:

  (none)        just the voxels. Fastest, and enough to check the two glazed
                walls came out as two planes.
  --bag         rebuilds the world clouds from the raw /livox/lidar topic
                through ../LivoxLidarOdometryLoader -- the same cloud stage 2
                voted on, so what you see is what voted. Needs --session-dir.
  --scene-ply   any .ply/.pcd already on disk (e.g. LivoxLidarOdometryLoader's
                --export-ply, or 3DModelPointCloudExtraction/SavedBag/*.pcd).

Both scene sources are display-only: --scene-stride and --point-filter-num
decimate them, and neither touches the voxels.

Usage:
    :: voxels only, interactive
    C:\\venvs\\planefit\\Scripts\\python.exe view_openings.py ^
        --consensus-dir ...\\fullrate\\opening_map_m2f_full_consensus

    :: voxels in the room, every 5th frame of the session rebuilt behind them
    C:\\venvs\\planefit\\Scripts\\python.exe view_openings.py ^
        --consensus-dir ...\\opening_map_m2f_full_consensus ^
        --session-dir ...\\fullrate --bag ...\\rosbag2_2026_07_30-18_12_20 ^
        --every-n 5 --scene-max-range 12

    :: no cubes -- colour the scene points themselves by the voxel they sit in
    ... --bag ... --session-dir ... --paint-cloud

    :: headless
    ... --screenshot openings.png

Those are cmd.exe -- `^` continues a line there, and NOT in PowerShell, where it
is no continuation at all and the next line fails on `--`. From PowerShell use a
backtick or one line.

Venv: C:\\venvs\\planefit (pyvista + open3d + rosbags all present there;
sensorfusion and emissivity have no pyvista).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent

# Same colours opening_voxel_consensus.PLY_COLORS writes into the .ply, so a
# screenshot from here and the .ply opened in CloudCompare read identically.
CLASS_COLORS = {"door": (220, 40, 40), "window": (40, 110, 230)}
SCENE_COLOR = (170, 170, 175)


def parse_args():
    p = argparse.ArgumentParser(
        description="Show stage 2's consensus opening voxels, optionally over the scene cloud")
    p.add_argument("--consensus-dir", required=True, metavar="DIR",
                   help="Directory holding door_window_voxels.csv (stage 2's --out-dir).")
    p.add_argument("--voxel", type=float, default=0.20, metavar="M",
                   help="Voxel edge, must match the --voxel stage 2 ran with (default 0.20).")
    p.add_argument("--min-agreement", type=float, default=0.0, metavar="F",
                   help="Hide voxels whose consensus agreement is below this. Stage 2's own "
                        "default gate is 0.0, so the csv holds everything; raise it here to "
                        "see how much of the map survives a stricter threshold without "
                        "re-running the vote.")
    p.add_argument("--min-observations", type=int, default=0, metavar="N",
                   help="Hide voxels seen by fewer than N distinct frames. This is the honest "
                        "multi-view filter -- n_votes counts LiDAR points, not views.")
    p.add_argument("--classes", default="door,window", metavar="A,B",
                   help="Which classes to draw (default door,window).")
    p.add_argument("--points", action="store_true",
                   help="Draw the voxels as points instead of cubes -- one point per voxel "
                        "centre. Faster on huge maps. Not --paint-cloud.")
    p.add_argument("--paint-cloud", action="store_true",
                   help="Draw no voxel geometry at all: colour the SCENE points by the opening "
                        "voxel each one falls in (door red / window blue, everything else grey). "
                        "Needs a scene, so --bag or --scene-ply. This shows the openings at the "
                        "cloud's own resolution rather than the vote's 0.20 m -- a cube hides "
                        "whether the points inside it are a flat pane or a smear.")
    p.add_argument("--paint-point-size", type=float, default=4.0, metavar="PX",
                   help="Point size for the painted points with --paint-cloud (default 4); the "
                        "unpainted rest stays at the scene's own size.")

    p.add_argument("--session-dir", default=None, metavar="DIR",
                   help="Session whose sync_manifest.json picks the scene frames, with --bag.")
    p.add_argument("--bag", default=None, metavar="DIR",
                   help="Rebuild the scene cloud from this rosbag's raw lidar topic.")
    p.add_argument("--every-n", type=int, default=5, metavar="N",
                   help="Take every Nth manifest triplet for the scene (default 5). The scene "
                        "is only backdrop -- consecutive frames overlap almost completely.")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--point-filter-num", type=int, default=4, metavar="N",
                   help="Decimate each rebuilt scan by N at read time (default 4).")
    p.add_argument("--scene-max-range", type=float, default=0.0, metavar="M",
                   help="Drop scene points beyond this range from the sensor, 0 = keep all. "
                        "Unrelated to stage 2's --max-range, which gated the votes.")
    p.add_argument("--scene-stride", type=int, default=1, metavar="N",
                   help="Extra display-only decimation of the assembled scene cloud.")
    p.add_argument("--scene-ply", default=None, metavar="PLY",
                   help="Load the scene from a .ply/.pcd instead of a bag.")
    p.add_argument("--scene-opacity", type=float, default=0.35, metavar="A")
    p.add_argument("--lidar-topic", default="/livox/lidar", metavar="TOPIC")
    p.add_argument("--pose-topic", default="/Odometry", metavar="TOPIC")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")

    p.add_argument("--screenshot", default=None, metavar="PNG",
                   help="Render off-screen to this PNG instead of opening a window.")
    p.add_argument("--window-size", default="1600,1000", metavar="W,H")
    return p.parse_args()


def load_voxels(csv_path: Path):
    """stdlib csv, no pandas -- same as openings/table.py."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("x", "y", "z", "agreement", "w_door", "w_other", "w_window"):
            if r.get(k):
                r[k] = float(r[k])
        for k in ("n_observations", "n_votes"):
            if r.get(k):
                r[k] = int(r[k])
    return rows


def scene_from_bag(args):
    """The cloud stage 2 actually voted on, rebuilt through the sibling loader."""
    loader_dir = HERE.parent / "LivoxLidarOdometryLoader"
    if not loader_dir.is_dir():
        sys.exit(f"--bag needs {loader_dir}, which is not there.")
    sys.path.insert(0, str(loader_dir))
    import livox_odometry_loader as lol

    session_dir = Path(args.session_dir)
    targets, _trips = lol.targets_from_session(session_dir, args.limit, args.every_n)
    print(f"scene: {len(targets)} frame(s) from {session_dir.name}/sync_manifest.json")
    clouds = lol.nearest_clouds_for_targets(
        Path(args.bag), targets, args.lidar_topic, args.store,
        odom_topic=args.pose_topic, max_range=args.scene_max_range,
        point_filter_num=args.point_filter_num)
    parts = [c[1] for c in clouds if c is not None]
    if not parts:
        sys.exit("scene: no scan matched any target -- check --bag and --lidar-topic.")
    pts = np.vstack(parts)
    print(f"scene: {len(pts)} points from {len(parts)} scan(s)")
    return pts


def scene_from_file(path: Path):
    if path.suffix.lower() == ".pcd":
        import open3d as o3d
        pts = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    else:
        pts = np.asarray(pv.read(str(path)).points)
    print(f"scene: {len(pts)} points from {path.name}")
    return pts


def paint_cloud(scene: np.ndarray, kept, voxel: float):
    """Split the scene points by which opening voxel each falls in.

    Inverts write_voxel_map's key -> centre mapping (centre = (key + 0.5) * voxel)
    to recover stage 2's integer keys, then bins the scene the same way stage 2
    binned the votes. Same voxel edge in, same cells out -- pass the --voxel
    stage 2 ran with or this silently paints the wrong points.

    Returns (by_class, rest): the points inside a door/window voxel, per class,
    and everything else.
    """
    lut = {}
    for r in kept:
        key = tuple(int(np.rint(r[a] / voxel - 0.5)) for a in ("x", "y", "z"))
        lut[key] = r["opening_class"]

    keys = np.floor(scene / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    # One dict lookup per occupied cell, not per point -- the cloud has orders
    # of magnitude more points than cells.
    cls_of_cell = np.array([lut.get(tuple(k), "") for k in uniq], dtype=object)
    cls_of_point = cls_of_cell[inv]

    by_class, hit = {}, np.zeros(len(scene), dtype=bool)
    for cls in sorted({c for c in cls_of_cell if c}):
        m = cls_of_point == cls
        by_class[cls] = scene[m]
        hit |= m
    return by_class, scene[~hit]


def main():
    args = parse_args()
    csv_path = Path(args.consensus_dir) / "door_window_voxels.csv"
    if not csv_path.is_file():
        sys.exit(f"{csv_path} not found -- run opening_voxel_consensus.py first.")

    rows = load_voxels(csv_path)
    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    kept = [r for r in rows
            if r["opening_class"] in wanted
            and r.get("agreement", 1.0) >= args.min_agreement
            and r.get("n_observations", 0) >= args.min_observations]
    if not kept:
        sys.exit(f"{len(rows)} voxel(s) in the csv, none passed the filters.")

    by_class = {}
    for r in kept:
        by_class.setdefault(r["opening_class"], []).append((r["x"], r["y"], r["z"]))
    print(f"{csv_path.name}: {len(rows)} voxel(s), {len(kept)} drawn -- " +
          ", ".join(f"{c}={len(v)}" for c, v in sorted(by_class.items())))

    w, h = (int(v) for v in args.window_size.split(","))
    pl = pv.Plotter(off_screen=bool(args.screenshot), window_size=(w, h))
    pl.set_background("white")

    scene = None
    if args.scene_ply:
        scene = scene_from_file(Path(args.scene_ply))
    elif args.bag:
        if not args.session_dir:
            sys.exit("--bag needs --session-dir (the manifest picks which scans to load).")
        scene = scene_from_bag(args)
    if scene is not None and len(scene):
        if args.scene_stride > 1:
            scene = scene[::args.scene_stride]
    if args.paint_cloud and (scene is None or not len(scene)):
        sys.exit("--paint-cloud has nothing to paint: pass --bag (with --session-dir) "
                 "or --scene-ply.")

    if args.paint_cloud:
        painted, rest = paint_cloud(scene, kept, args.voxel)
        n_hit = sum(len(v) for v in painted.values())
        print(f"painted {n_hit} of {len(scene)} scene point(s) "
              f"({100 * n_hit / len(scene):.1f}%) -- " +
              ", ".join(f"{c}={len(v)}" for c, v in sorted(painted.items())))
        if len(rest):
            pl.add_mesh(pv.PolyData(rest), color=SCENE_COLOR, point_size=1.5,
                        opacity=args.scene_opacity, render_points_as_spheres=False,
                        label=f"scene ({len(rest)})")
        for cls, pts in sorted(painted.items()):
            pl.add_mesh(pv.PolyData(pts), color=CLASS_COLORS.get(cls, (120, 120, 120)),
                        point_size=args.paint_point_size, render_points_as_spheres=True,
                        label=f"{cls} ({len(pts)} pts)")
        by_class = {}   # nothing left to glyph
    elif scene is not None and len(scene):
        pl.add_mesh(pv.PolyData(scene), color=SCENE_COLOR, point_size=1.5,
                    opacity=args.scene_opacity, render_points_as_spheres=False,
                    label=f"scene ({len(scene)})")

    # One pass per class: two solid colours read better than a 2-entry colormap,
    # and it keeps the legend honest about what is actually drawn.
    cube = pv.Cube(x_length=args.voxel, y_length=args.voxel, z_length=args.voxel)
    for cls, xyz in sorted(by_class.items()):
        poly = pv.PolyData(np.asarray(xyz, dtype=float))
        colour = CLASS_COLORS.get(cls, (120, 120, 120))
        if args.points:
            pl.add_mesh(poly, color=colour, point_size=9,
                        render_points_as_spheres=True, label=f"{cls} ({len(xyz)})")
        else:
            pl.add_mesh(poly.glyph(geom=cube, scale=False, orient=False),
                        color=colour, label=f"{cls} ({len(xyz)})")

    # size drives the legend font too -- bigger and the labels run off the frame.
    pl.add_legend(bcolor="white", border=True, loc="upper right", size=(0.14, 0.10))
    pl.show_axes()
    pl.add_text(f"{Path(args.consensus_dir).name}  voxel {args.voxel:.2f} m",
                font_size=9, color="black")

    if args.screenshot:
        pl.show(screenshot=args.screenshot)
        print(f"Wrote {args.screenshot}")
    else:
        pl.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
