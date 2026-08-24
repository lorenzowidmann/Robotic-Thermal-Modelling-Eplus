"""Stage 1 only: colour the LiDAR points by the segment they project into, with
NO voxel vote, NO consensus, and every gate off by default.

view_openings.py shows what survived stage 2. This shows what stage 2 was given,
so the two together say how much the vote removes and where. Nothing here is a
better answer than the consensus -- it is deliberately the unfiltered one.

What it does per frame: project that frame's world cloud into the ZED image
(same `project_lidar_to_camera` stage 2 uses), read `labels.npy` at each pixel,
look the segment id up in `segments.json`, and keep the point under that
segment's `top_class`. Points from different frames are different points and are
all kept -- there is nothing to reconcile, which is the whole difference from
stage 2.

The gates stage 2 applies, all OFF here unless asked for:

  --min-confidence   stage 2 defaults to 0.5, dropping every weak segment's votes
  --max-range        stage 2 defaults to 8 m, dropping every distant point
  --voxel/vote       there is none: no 0.20 m binning, no argmax over classes

And one gate stage 2 does NOT have:

  --depth-band M     per segment, keep only points within M metres of that
                     segment's median depth. The projection has no visibility
                     test -- any point whose pixel lands in a mask is labelled
                     by it, including points on near clutter that merely sit in
                     front of a far door. That is what puts "door" points in mid
                     air ahead of the end wall. This is the cheap fix; a real
                     one needs a depth buffer.

Usage:
    C:\\venvs\\planefit\\Scripts\\python.exe paint_openings.py ^
        --session-dir ...\\fullrate --bag ...\\rosbag2_2026_07_30-18_12_20 ^
        --opening-map-dir ...\\fullrate\\opening_map_m2f_full

    ... --depth-band 1.5              :: drop the through-the-mask smear
    ... --out-ply raw_openings.ply    :: coloured points for CloudCompare
    ... --screenshot raw.png          :: headless

(cmd.exe above -- `^` is not a PowerShell continuation, use a backtick or one
line there.)

Venv: C:\\venvs\\planefit -- it is the only one with pyvista, and it has cv2,
yaml and rosbags too, so the calibration stack imports.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _find_root(start: Path) -> Path:
    """Walk up until the directory holding SensorFusionLoader/ is found."""
    for d in [start, *start.parents]:
        if (d / "SensorFusionLoader").is_dir():
            return d
    raise RuntimeError(
        f"SensorFusionLoader/ not found in any parent of {start} -- it holds "
        "rig_calibration.py/.yaml and projection.py, which this script needs.")


_ROOT = _find_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(_ROOT / "SensorFusionLoader"))
from rig_calibration import load_rig_calibration  # noqa: E402
from projection import project_lidar_to_camera  # noqa: E402

# Same reason and same ordering constraint as opening_voxel_consensus.py: this
# insert only exists so lidar_metrics can reach project_to_flir, whose own
# broken sys.path.insert is harmless once rig_calibration/projection are already
# in sys.modules. Do not move it above the SensorFusionLoader import.
sys.path.insert(0, str(_ROOT / "EmissivityCalculation"))
_LOADER_DIR = _ROOT / "PointCloudElaboration" / "LivoxLidarOdometryLoader"

sys.path.insert(0, str(Path(__file__).resolve().parent / "openings"))
import lidar_metrics as lm  # noqa: E402

# The colours opening_voxel_consensus.PLY_COLORS writes, so this and
# view_openings.py read identically.
CLASS_COLORS = {"door": (220, 40, 40), "window": (40, 110, 230)}
OTHER_COLOR = (170, 170, 175)


def parse_args():
    p = argparse.ArgumentParser(
        description="Colour the LiDAR by stage 1's per-frame segments, with no consensus")
    p.add_argument("--session-dir", required=True, metavar="DIR")
    p.add_argument("--bag", required=True, metavar="DIR")
    p.add_argument("--opening-map-dir", default=None, metavar="DIR",
                   help="Stage 1 output (default <session>/opening_map).")
    p.add_argument("--classes", default="door,window", metavar="A,B",
                   help="Which classes to colour (default door,window). Everything else, "
                        "including unlabelled pixels, is drawn as grey scene.")

    p.add_argument("--min-confidence", type=float, default=0.0, metavar="C",
                   help="Drop points whose segment scored below this. 0 = keep everything "
                        "(default). Stage 2 uses 0.5 -- pass that to see what the gate costs.")
    p.add_argument("--max-range", type=float, default=0.0, metavar="M",
                   help="Drop points farther than this from the sensor. 0 = keep everything "
                        "(default). Stage 2 uses 8.")
    p.add_argument("--depth-band", type=float, default=0.0, metavar="M",
                   help="Per segment, keep only points within M metres of that segment's "
                        "median depth. 0 = off (default). This is the occlusion proxy: it "
                        "removes points that merely sit in front of the masked surface.")

    p.add_argument("--every-n", type=int, default=5, metavar="N",
                   help="Take every Nth manifest triplet (default 5).")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--point-filter-num", type=int, default=2, metavar="N",
                   help="Decimate each scan by N at read time (default 2).")
    p.add_argument("--dedup", type=float, default=0.0, metavar="M",
                   help="Collapse points to one per M-metre cell before drawing, per class. "
                        "0 = off (default). Display only -- it does not vote.")

    p.add_argument("--odom-bag", default=None, metavar="DIR",
                   help="Take the POSES from this bag instead of --bag's own /Odometry, "
                        "keeping the scans from --bag. This is how the drift-corrected "
                        "trajectory written by 3DModelPointCloudExtraction/BagFilter*.m "
                        "(SavedBag_Odometry/<stamp>) gets used: the correction only ever "
                        "touched the poses, so the points still have to come from the "
                        "original recording. Run --fix-digests on that bag once first, or "
                        "it will not open.")
    p.add_argument("--lidar-topic", default=lm.RAW_LIDAR_TOPIC, metavar="TOPIC")
    p.add_argument("--pose-topic", default=lm.POSE_TOPIC, metavar="TOPIC")
    p.add_argument("--registered-topic", default=lm.REGISTERED_TOPIC, metavar="TOPIC")
    p.add_argument("--cloud-source", choices=("raw", "registered"), default="raw")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--calibration", default=None, metavar="YAML")

    p.add_argument("--out-ply", default=None, metavar="PLY",
                   help="Write the coloured points out for CloudCompare.")
    p.add_argument("--out-csv", default=None, metavar="CSV",
                   help="Write the opening points as x,y,z,opening_class -- the input "
                        "fit_openings.py wants. Only the painted points; the grey scene is "
                        "not an opening and is left out.")
    p.add_argument("--out-scene-csv", default=None, metavar="CSV",
                   help="Write the NON-opening points as x,y,z. These are the wall, floor "
                        "and clutter returns, and fit_openings.py --void-evidence needs "
                        "them: glazing returns nothing, so a hole in the wall's coverage is "
                        "evidence FOR a window -- but only if the wall around that hole was "
                        "scanned, which is what these points establish.")
    p.add_argument("--no-show", action="store_true",
                   help="Skip the window entirely (pairs with --out-ply).")
    p.add_argument("--screenshot", default=None, metavar="PNG")
    p.add_argument("--scene-opacity", type=float, default=0.30, metavar="A")
    p.add_argument("--clip-percentile", type=float, default=1.0, metavar="P",
                   help="Drop scene points outside the P..100-P percentile box before drawing, "
                        "so stray far returns do not decide the camera (default 1, 0 = off). "
                        "Display only, and never applied to the painted points.")
    p.add_argument("--point-size", type=float, default=4.0, metavar="PX")
    p.add_argument("--window-size", default="1600,1000", metavar="W,H")
    return p.parse_args()


def frames_with(needed: Path, triplets: list) -> list:
    out = []
    for t in triplets:
        if (needed / Path(t["flir"]["file"]).stem).is_dir():
            out.append(t)
    return out


def dedup_indices(pts: np.ndarray, cell: float) -> np.ndarray:
    """Index of one surviving point per cell. Which one survives does not matter.

    Returned as indices rather than points so the per-point provenance
    (which frame, which segment) can be subset the same way -- losing that
    alignment is what would make the mask projection in fit_openings.py
    impossible to trace back.
    """
    if cell <= 0 or not len(pts):
        return np.arange(len(pts))
    _keys, idx = np.unique(np.floor(pts / cell).astype(np.int64), axis=0, return_index=True)
    return idx


def dedup_cells(pts: np.ndarray, cell: float) -> np.ndarray:
    """One point per cell."""
    return pts[dedup_indices(pts, cell)]


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    cal = load_rig_calibration(args.calibration
                               or (_ROOT / "SensorFusionLoader" / "rig_calibration.yaml"))

    manifest = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    opening_dir = (Path(args.opening_map_dir) if args.opening_map_dir
                   else session_dir / "opening_map")
    work = frames_with(opening_dir, triplets)
    if not work:
        sys.exit(f"No frame has stage 1 output under {opening_dir} -- run "
                 "classify_openings.py first.")
    wanted = {c.strip() for c in args.classes.split(",") if c.strip()}
    print(f"{len(work)} frame(s) with opening maps under {opening_dir.name}")

    traj = None
    if args.odom_bag:
        sys.path.insert(0, str(_LOADER_DIR))
        import livox_odometry_loader as lol
        traj = lol.read_trajectory(Path(args.odom_bag), odom_topic=args.pose_topic,
                                   store=args.store)
        print(f"poses from {Path(args.odom_bag).name}: {len(traj.times)} over "
              f"{traj.times[-1] - traj.times[0]:.1f} s")

    clouds = lm.load_clouds(
        Path(args.bag), [t["lidar"]["timestamp_zedclock"] for t in work],
        store=args.store, source=args.cloud_source, lidar_topic=args.lidar_topic,
        registered_topic=args.registered_topic, pose_topic=args.pose_topic,
        loader_dir=_LOADER_DIR, traj=traj)

    by_class = defaultdict(list)
    prov = defaultdict(list)        # class -> (frame index, segment id) per point
    other = []
    n_seen = n_conf = n_far = n_band = 0
    seg_frames = Counter()          # class -> frames that contributed any point

    for frame_idx, (triplet, cloud) in enumerate(zip(work, clouds)):
        stem = Path(triplet["flir"]["file"]).stem
        if cloud is None:
            print(f"skip {stem}: no LiDAR scan near that instant", file=sys.stderr)
            continue
        _t, points_world = cloud
        # load_clouds has no decimation of its own, so stride here -- before the
        # projection, so pixel lookups and points stay index-aligned.
        if args.point_filter_num > 1:
            points_world = points_world[::args.point_filter_num]
        labels = np.load(opening_dir / stem / "labels.npy")
        doc = json.loads((opening_dir / stem / "segments.json").read_text(encoding="utf-8"))
        info = {int(s["id"]): (s["top_class"], float(s["confidence"])) for s in doc["segments"]}
        zh, zw = labels.shape

        uv, depth, valid = project_lidar_to_camera(
            points_world, np.array(triplet["lidar"]["position"]),
            np.array(triplet["lidar"]["orientation"]), cal.T_lidar_to_zed,
            cal.zed_K_for(zw, zh), cal.zed_calib.dist, zw, zh)
        if not valid.any():
            continue
        pts = points_world[valid]
        dep = depth[valid]
        px = np.round(uv[valid]).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, zw - 1)
        px[:, 1] = np.clip(px[:, 1], 0, zh - 1)
        sids = labels[px[:, 1], px[:, 0]]
        n_seen += len(pts)

        claimed = np.zeros(len(pts), dtype=bool)
        frame_classes = set()
        for sid in np.unique(sids):
            sid = int(sid)
            if sid < 0 or sid not in info:
                continue
            cls, conf = info[sid]
            if cls not in wanted:
                continue
            m = sids == sid
            if args.min_confidence > 0 and conf < args.min_confidence:
                n_conf += int(m.sum())
                continue
            if args.max_range > 0:
                far = m & (dep > args.max_range)
                n_far += int(far.sum())
                m = m & ~far
            if args.depth_band > 0 and m.any():
                # The mask is one surface; points far off its median depth are
                # in front of it, not on it.
                off = m & (np.abs(dep - float(np.median(dep[m]))) > args.depth_band)
                n_band += int(off.sum())
                m = m & ~off
            if not m.any():
                continue
            by_class[cls].append(pts[m])
            # Provenance, one row per point: which frame and which segment of it
            # this point came from. fit_openings.py --masks needs it to go back
            # to the mask that produced the point; without it a point is only a
            # class, and the mask -- the better measurement of the two -- cannot
            # be found again.
            prov[cls].append(np.column_stack([
                np.full(int(m.sum()), frame_idx, dtype=np.int64),
                np.full(int(m.sum()), sid, dtype=np.int64)]))
            claimed |= m
            frame_classes.add(cls)
        for cls in frame_classes:
            seg_frames[cls] += 1
        other.append(pts[~claimed])

    if not by_class:
        sys.exit("No point landed in a door/window segment -- nothing to draw.")

    painted = {c: np.vstack(v) for c, v in by_class.items()}
    painted_prov = {c: np.vstack(v) for c, v in prov.items()}
    rest = np.vstack(other) if other else np.empty((0, 3))
    n_hit = sum(len(v) for v in painted.values())
    print(f"{n_seen} projected point(s), {n_hit} in an opening segment "
          f"({100.0 * n_hit / max(1, n_seen):.1f}%) -- " +
          ", ".join(f"{c}={len(v)} over {seg_frames[c]} frame(s)"
                    for c, v in sorted(painted.items())))
    dropped = [f"{n} {why}" for n, why in
               ((n_conf, f"below confidence {args.min_confidence}"),
                (n_far, f"beyond {args.max_range:g} m"),
                (n_band, f"off-band by more than {args.depth_band:g} m")) if n]
    print("dropped: " + (", ".join(dropped) if dropped else "nothing, every gate is off"))

    if args.dedup > 0:
        keep = {c: dedup_indices(v, args.dedup) for c, v in painted.items()}
        painted = {c: v[keep[c]] for c, v in painted.items()}
        painted_prov = {c: v[keep[c]] for c, v in painted_prov.items()}
        rest = dedup_cells(rest, args.dedup)
        print(f"dedup at {args.dedup:g} m -> " +
              ", ".join(f"{c}={len(v)}" for c, v in sorted(painted.items())))

    if args.out_ply:
        write_ply(Path(args.out_ply), painted, rest)
    if args.out_csv:
        path = Path(args.out_csv)
        stems = [Path(t["flir"]["file"]).stem for t in work]
        with path.open("w", encoding="utf-8", newline="") as f:
            # `frame` and `segment_id` are the provenance fit_openings.py --masks
            # reads back; everything else ignores the two extra columns.
            f.write("x,y,z,opening_class,frame,segment_id\n")
            for cls, pts in sorted(painted.items()):
                pv = painted_prov[cls]
                for p, (fi, sid) in zip(pts, pv):
                    f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{cls},"
                            f"{stems[fi]},{sid}\n")
        print(f"Wrote {sum(len(v) for v in painted.values())} labelled point(s) to {path}")
    if args.out_scene_csv:
        path = Path(args.out_scene_csv)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("x,y,z\n")
            for p in rest:
                f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n")
        print(f"Wrote {len(rest)} scene point(s) to {path}")

    if args.no_show:
        return 0

    # A few stray grey returns hundreds of metres out otherwise fit the camera to
    # them and shrink the corridor to a smudge. Trimmed on the SCENE only --
    # painted points are the answer and are never dropped for looks.
    if args.clip_percentile > 0 and len(rest):
        p = args.clip_percentile
        lo, hi = np.percentile(rest, [p, 100.0 - p], axis=0)
        inside = np.all((rest >= lo) & (rest <= hi), axis=1)
        if inside.sum() < len(rest):
            print(f"scene clipped to the {p:g}-{100 - p:g} percentile box: "
                  f"{len(rest)} -> {int(inside.sum())} point(s)")
        rest = rest[inside]

    import pyvista as pv
    w, h = (int(v) for v in args.window_size.split(","))
    pl = pv.Plotter(off_screen=bool(args.screenshot), window_size=(w, h))
    pl.set_background("white")
    if len(rest):
        pl.add_mesh(pv.PolyData(rest), color=OTHER_COLOR, point_size=1.5,
                    opacity=args.scene_opacity, label=f"scene ({len(rest)})")
    for cls, pts in sorted(painted.items()):
        pl.add_mesh(pv.PolyData(pts), color=CLASS_COLORS.get(cls, (120, 120, 120)),
                    point_size=args.point_size, render_points_as_spheres=True,
                    label=f"{cls} ({len(pts)})")
    pl.add_legend(bcolor="white", border=True, loc="upper right", size=(0.14, 0.10))
    pl.show_axes()
    pl.add_text(f"{opening_dir.name}  stage 1 only, no consensus", font_size=9, color="black")
    if args.screenshot:
        pl.show(screenshot=args.screenshot)
        print(f"Wrote {args.screenshot}")
    else:
        pl.show()
    return 0


def write_ply(path: Path, painted, rest):
    """Ascii PLY, same colour convention as opening_voxel_consensus.py's."""
    chunks = [(rest, OTHER_COLOR)] + [(v, CLASS_COLORS.get(c, (120, 120, 120)))
                                      for c, v in sorted(painted.items())]
    n = sum(len(p) for p, _ in chunks)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for pts, c in chunks:
            for p in pts:
                f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
    print(f"Wrote {n} points to {path}")


if __name__ == "__main__":
    raise SystemExit(main())
