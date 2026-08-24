"""Door / window / other per region, for every frame of a recorded ZED session
(driven by SensorFusion/sync_manifest.py's sync_manifest.json).

Stage 1 of two. Segmentation AND classification come from one Mask2Former
forward pass on the ADE20K-150 taxonomy, which already contains `windowpane`
and `door`; `openings/segmentation_m2f.py` explains the mechanism and the
per-pixel confidence it derives for stage 2's vote gate.

What this replaced, and why
---------------------------
Until now stage 1 ran SAM in everything-mode and asked CLIP to label each mask
door/window/other. Measured on session 9, that pair had three problems the new
one does not:

  * SAM's masks were decoded at 256x256 and resized to 1920x1080 with
    INTER_NEAREST, so boundaries arrived as ~7 px staircases, and the gap-fill
    then painted the leftover pixels with the nearest id -- inventing outlines
    across the floor and the radiators.
  * `other` was one CLIP prompt covering walls, floors, ceilings, radiators,
    pillars, people and clutter. Now every one of the other 148 ADE classes
    lands in `other` by omission, and each keeps its own region.
  * ~27 s/frame against 4.7-5.2 s/frame here, on the same CPU.

It also retires zone_prior.py's known failure. That module's README section
"The ceiling rule eats high windows" describes zone_of() calling a wide,
high window `ceiling` and forcing it to `other` -- undetectable by
construction. Zones now come from the ADE class (segmentation_m2f.zone_from_ade),
so a clerestory stays a windowpane, and `floor` for stage 1B's floor-contact
test is the real floor mask rather than a bbox guess.

The LiDAR is now consulted at stage 1, for doors only
----------------------------------------------------
Pass --bag and every door candidate is measured in metres before any pixel
rule is allowed to discard it. A door-sized measurement rescues a candidate a
size rule wanted to kill; a definitively wrong one rejects a candidate the
size rules would have kept; no measurement changes nothing. Windows are
untouched by this -- glazing returns no LiDAR, so a metric window gate would
fail hardest on the class it is meant to validate. See openings/geometry.py's
header and openings/lidar_metrics.py.

Without --bag the run is exactly the old pixel-only behaviour, and needs no
rosbags in the venv.

Output per frame, under <out-dir>/<flir_frame_stem>/:
    labels.npy    -- int32 HxW region-id raster on the full ZED pixel grid,
                     -1 where no region reached --min-area
    segments.json -- schema "opening_map/v1": per-segment id, bbox, centroid,
                     area, top_class, confidence, top_k, zone, ade
    overlay.png   -- optional (--overlay): kept openings outlined, plus every
                     REJECTED candidate outlined in yellow with the rule that
                     killed it, which is what makes the image a QA artefact
                     rather than a picture

Venv: torch + transformers + torchvision + opencv + scipy. With --bag, also
rosbags.

Usage:
    py classify_openings.py --session-dir ...\\ZED\\20260730_161223\\fullrate --limit 5 --overlay
    py classify_openings.py --session-dir ...\\fullrate --bag ...\\rosbag2_2026_07_30-18_12_20 --overlay
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from openings import OpeningTable, OTHER_CLASS
from openings import geometry as geom
from openings import lidar_metrics as lm
from openings import segmentation_m2f as m2f


# --- sibling modules, imported by path -------------------------------------
# The Thesis-final-wt2 root is located by searching upwards, see _find_root.
# NOTE that the calibration loader lives in SensorFusionLoader/, NOT Calibration/ --
# EmissivityCalculation's own scripts still hardcode "../Calibration" and are
# broken in this repo as a result; that is a pre-existing gap, not something
# this module fixes.
def _find_root(start: Path) -> Path:
    """Walk up until the directory holding SensorFusionLoader/ is found.

    Searched rather than counted with a fixed number of .parent hops: this
    module has already been moved one level once (WindowsDetection/
    OpeningClassification/ -> WindowsDoorsDetection/), which silently broke a
    hardcoded count. Searching survives the next move too.
    """
    for d in [start, *start.parents]:
        if (d / "SensorFusionLoader").is_dir():
            return d
    raise RuntimeError(
        f"SensorFusionLoader/ not found in any parent of {start} -- it holds "
        "rig_calibration.py/.yaml and projection.py, which this script needs.")


_ROOT = _find_root(Path(__file__).resolve().parent)
# ../LivoxLidarOdometryLoader -- the raw-cloud reader. Sibling of this module
# under PointCloudElaboration/, and located from _ROOT rather than by counting
# .parent hops, for the same reason _find_root exists.
_LOADER_DIR = _ROOT / "PointCloudElaboration" / "LivoxLidarOdometryLoader"


def _load_lidar_stack():
    """Import the calibration + bag-reading side, in the order that works.

    SensorFusionLoader FIRST: project_to_flir.py runs its own
    `sys.path.insert(0, "../Calibration")` at import time and then does
    `from rig_calibration import ...`. That directory does not exist in this
    repo, so the insert is a no-op -- but if rig_calibration and projection are
    already in sys.modules its top-level imports resolve to
    SensorFusionLoader's copies and it loads cleanly. Do not reorder.

    Deferred into a function so a run without --bag never touches rosbags.
    """
    sys.path.insert(0, str(_ROOT / "SensorFusionLoader"))
    from rig_calibration import load_rig_calibration
    from projection import project_lidar_to_camera
    sys.path.insert(0, str(_ROOT / "EmissivityCalculation"))
    return load_rig_calibration, project_lidar_to_camera


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_zed_frames_dir(session_dir: Path) -> Path:
    """Same convention as SensorFusion/sync_manifest.py::load_zed_frames --
    frames live in metadata.json's recording.frames_dir (default "frames")."""
    meta_path = session_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return session_dir / (meta.get("recording", {}).get("frames_dir") or "frames")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-region door/window/other classification for a synced ZED session")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder holding metadata.json + frames/ + sync_manifest.json "
             "(the output of SensorFusion/sync_manifest.py).")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Output root (default: <session-dir>/opening_map).")
    # --- segmenter ---------------------------------------------------------
    p.add_argument("--model", default=m2f.DEFAULT_MODEL, metavar="REPO",
                   help="Mask2Former semantic checkpoint on the ADE20K-150 taxonomy "
                        f"(default {m2f.DEFAULT_MODEL}). A panoptic or COCO checkpoint "
                        "will load but has different labels and will fail the ade lookup "
                        "in opening_table.csv.")
    p.add_argument("--min-area", type=int, default=m2f.MIN_COMPONENT_AREA, metavar="PX",
                   help="Connected components below this never become a region: they stay "
                        "at -1 in labels.npy and never vote (default "
                        f"{m2f.MIN_COMPONENT_AREA}, the number SAM used, kept so frames "
                        "stay comparable across the switch).")
    p.add_argument("--table", default=None, metavar="CSV",
                   help="Alternative opening_table.csv (class,ade,prompt,notes).")
    # --- stage 1B: merge + geometric plausibility --------------------------
    p.add_argument("--geometry-filter", action=argparse.BooleanOptionalAction, default=True,
                   help="Merge touching same-class regions and reject implausible ones "
                        "(default on). --no-geometry-filter writes the raw per-ADE-region "
                        "result, which is what to compare against when a rule is suspect.")
    p.add_argument("--merge-dilate-px", type=int, default=geom.DILATE_PX, metavar="PX",
                   help="Dilation used ONLY to decide connectivity when merging touching "
                        f"same-class regions (default {geom.DILATE_PX}). A merge never "
                        "absorbs another class's pixels.")
    p.add_argument("--floor-tol-px", type=int, default=geom.FLOOR_TOL_PX, metavar="PX",
                   help="Mask-adjacency gap to the ADE floor region that still counts as "
                        f"standing on it (default {geom.FLOOR_TOL_PX}).")
    p.add_argument("--window-filter", action=argparse.BooleanOptionalAction, default=False,
                   help="Reject merged windows that stand on the floor but are shorter "
                        "than --glass-wall-h-ratio. OFF by default: it was rejecting real "
                        "glazing (session 9 frame 5's bay measures 0.577 against a 0.60 "
                        "threshold), and unlike a door a window has no metric check to "
                        "arbitrate, because glazing returns no LiDAR. Windows are still "
                        "MERGED either way -- only the rejection is off.")
    p.add_argument("--glass-wall-h-ratio", type=float, default=geom.GLASS_WALL_H_RATIO,
                   metavar="F",
                   help="Fraction of frame height a floor-standing window must reach to be "
                        f"a glazed wall rather than a sliver (default "
                        f"{geom.GLASS_WALL_H_RATIO}). Also the ceiling above which a door "
                        "candidate is too tall to be a door.")
    p.add_argument("--door-edge-window-frac", type=float, default=geom.DOOR_EDGE_WINDOW_FRAC,
                   metavar="F",
                   help="A door candidate with a window abutting this fraction of one "
                        f"vertical side is a bay-edge reveal (default "
                        f"{geom.DOOR_EDGE_WINDOW_FRAC}). Never overruled by the LiDAR.")
    p.add_argument("--door-edge-band-px", type=int, default=geom.DOOR_EDGE_BAND_PX, metavar="PX")
    p.add_argument("--min-door-width-px", type=int, default=geom.MIN_DOOR_WIDTH_PX, metavar="PX",
                   help=f"Sliver guard (default {geom.MIN_DOOR_WIDTH_PX}). Depth-dependent, "
                        "so a door-sized LiDAR measurement overrules it.")
    # --- the LiDAR metric check on doors -----------------------------------
    p.add_argument("--bag", default=None, metavar="DIR",
                   help="rosbag2 directory holding the registered LiDAR cloud. Supplied, "
                        "every door candidate is measured in metres before any size rule "
                        "may discard it. Omitted, stage 1B is pixel-only and rosbags is "
                        "not imported.")
    p.add_argument("--cloud-source", choices=("raw", "registered"),
                   default=lm.DEFAULT_CLOUD_SOURCE,
                   help="Where the LiDAR comes from. 'raw' (default) rebuilds world "
                        "clouds from /livox/lidar + /Odometry via "
                        "../LivoxLidarOdometryLoader. 'registered' reads FAST-LIO's "
                        "/cloud_registered, which on session 9 is cropped to a 35-deg "
                        "cone from 4 m and misses both window bays entirely -- kept for "
                        "comparison, not for use.")
    p.add_argument("--lidar-topic", default=lm.RAW_LIDAR_TOPIC, metavar="TOPIC",
                   help="Raw Livox CustomMsg topic, --cloud-source raw.")
    p.add_argument("--pose-topic", default=lm.POSE_TOPIC, metavar="TOPIC",
                   help="Odometry topic, --cloud-source raw.")
    p.add_argument("--odom-topic", dest="registered_topic", default=lm.REGISTERED_TOPIC,
                   metavar="TOPIC",
                   help="Registered-cloud topic, --cloud-source registered. Named "
                        "--odom-topic for backwards compatibility; it is a misnomer, it "
                        "has never been the odometry.")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--calibration", default=None, metavar="YAML",
                   help="rig_calibration.yaml (default: SensorFusionLoader/rig_calibration.yaml).")
    p.add_argument("--door-h-m", type=float, nargs=2, default=[lm.MIN_DOOR_H_M, lm.MAX_DOOR_H_M],
                   metavar=("MIN", "MAX"),
                   help=f"Metric door height band (default {lm.MIN_DOOR_H_M} {lm.MAX_DOOR_H_M}). "
                        "Wide on purpose: the mask is the opening, leaf plus frame plus "
                        "reveal, and the estimate is a pinhole projection, not a fit.")
    p.add_argument("--door-w-m", type=float, nargs=2, default=[lm.MIN_DOOR_W_M, lm.MAX_DOOR_W_M],
                   metavar=("MIN", "MAX"),
                   help=f"Metric door width band (default {lm.MIN_DOOR_W_M} {lm.MAX_DOOR_W_M}).")
    p.add_argument("--metric-min-points", type=int, default=lm.MIN_POINTS, metavar="N",
                   help="Below this many LiDAR returns inside the mask there is no "
                        f"measurement and the pixel rules decide alone (default {lm.MIN_POINTS}).")
    p.add_argument("--metric-max-depth-ratio", type=float, default=lm.MAX_DEPTH_RATIO,
                   metavar="R",
                   help="p95/p05 depth ratio above which the mask is not one surface, so "
                        f"its median depth describes nothing (default {lm.MAX_DEPTH_RATIO}).")
    # --- run control -------------------------------------------------------
    p.add_argument("--every-n", type=int, default=1, metavar="N",
                   help="Process every Nth triplet (default 1).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Stop after N frames.")
    p.add_argument("--overlay", action="store_true",
                   help="Also write overlay.png per frame.")
    return p.parse_args()


def draw_overlay(image: np.ndarray, labels: np.ndarray, segments: list[dict]) -> np.ndarray:
    """Kept openings in colour, REJECTED candidates in yellow with their rule.

    Deliberately NOT cv2.rectangle, and deliberately not mark_boundaries over
    every region. A bounding box misrepresents what was classified -- a thin
    reveal strip beside a window bay looks contained in it on boxes and
    separate on masks, and it was exactly that illusion that made the first QA
    pass misread the false positives. Drawing all 100-odd `other` regions
    buries the handful the overlay exists to show.

    Rejections are drawn because they are the thing to argue with: an image
    where a real door is outlined in yellow and named
    `door_not_floor_standing` says what to change, where a blank frame does
    not.
    """
    import cv2

    colors = {"door": (0, 0, 255), "window": (255, 128, 0)}      # BGR
    rejected_color = (0, 215, 255)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()

    def outline(seg, color, thickness, text):
        mask = (labels == int(seg["id"])).astype(np.uint8)
        if not mask.any():
            return
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(bgr, contours, -1, color, thickness)
        x0, y0 = seg["bbox"][0], seg["bbox"][1]
        cv2.putText(bgr, text, (int(x0), max(16, int(y0) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def lidar_note(lidar):
        """What the LiDAR said, short enough to sit on a contour."""
        if not lidar:
            return ""
        if not lidar.get("usable"):
            return f" [no-metric: {lidar.get('reason')} n={lidar.get('n_points', 0)}]"
        return f" [{lidar['height_m']}x{lidar['width_m']}m @{lidar['depth_m']}m]"

    for seg in segments:
        rej = seg.get("rejected")
        if rej:
            outline(seg, rejected_color, 2,
                    f"X {rej['was']}: {rej['rule']}{lidar_note(rej.get('lidar'))}")
    for seg in segments:
        if seg["top_class"] == OTHER_CLASS or seg.get("rejected"):
            continue
        g = seg.get("geometry") or {}
        tag = " (lidar-rescued)" if g.get("rescued_by_lidar") else ""
        outline(seg, colors.get(seg["top_class"], (0, 255, 255)), 3,
                f"{seg['top_class']} {seg['confidence']:.2f}{lidar_note(g.get('lidar'))}{tag}")
    return bgr


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)

    manifest_path = session_dir / "sync_manifest.json"
    if not manifest_path.exists():
        print(f"No sync_manifest.json in {session_dir} -- run SensorFusion/sync_manifest.py first.",
              file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_dir = load_zed_frames_dir(session_dir)

    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "opening_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = OpeningTable(args.table) if args.table else OpeningTable()
    ade_to_class = table.ade_to_class

    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    print(f"Loading {args.model} ...")
    t_load = time.time()
    proc, model = m2f.load_model(args.model)
    names = set(m2f.ade_name_map(model).values())
    missing = sorted(set(ade_to_class) - names)
    if missing:
        print(f"opening_table.csv maps ADE labels this checkpoint does not have: {missing}. "
              f"Those classes can never be predicted.", file=sys.stderr)
        return 1
    print(f"  loaded in {time.time() - t_load:.1f}s; ade mapping " +
          ", ".join(f"{a} -> {c}" for a, c in sorted(ade_to_class.items())) +
          f", everything else -> {OTHER_CLASS}")

    # --- LiDAR, only if asked ----------------------------------------------
    clouds = cal = project_fn = None
    if args.bag:
        load_rig_calibration, project_fn = _load_lidar_stack()
        cal = load_rig_calibration(
            args.calibration or (_ROOT / "SensorFusionLoader" / "rig_calibration.yaml"))
        src = (f"{args.lidar_topic} + {args.pose_topic}" if args.cloud_source == "raw"
               else args.registered_topic)
        print(f"Reading LiDAR scans from {Path(args.bag).name} [{args.cloud_source}: {src}] "
              f"(door metric check ON: h {args.door_h_m[0]}-{args.door_h_m[1]} m, "
              f"w {args.door_w_m[0]}-{args.door_w_m[1]} m) ...")
        clouds = lm.load_clouds(
            Path(args.bag), [t["lidar"]["timestamp_zedclock"] for t in triplets],
            store=args.store, source=args.cloud_source,
            lidar_topic=args.lidar_topic, registered_topic=args.registered_topic,
            pose_topic=args.pose_topic, loader_dir=_LOADER_DIR)
        npts = [len(c[1]) for c in clouds if c is not None]
        if npts:
            print(f"  {len(npts)} scan(s), {min(npts)}-{max(npts)} points each")
    else:
        print("No --bag: stage 1B is pixel-only, door sizes are never measured.")

    # Bound once, so the CLI bands reach geometry.py without anything mutating
    # lidar_metrics' module globals mid-run.
    def door_verdict_fn(metric):
        return lm.door_verdict(metric, min_h=args.door_h_m[0], max_h=args.door_h_m[1],
                               min_w=args.door_w_m[0], max_w=args.door_w_m[1])

    print(f"Processing {len(triplets)} frame(s) from {manifest_path.name}, "
          f"full frame (no FLIR-FOV crop)")

    total_segments = 0
    total_class = Counter()         # per-region classes, BEFORE stage 1B
    total_rejected = Counter()      # stage 1B rule -> merged detections it killed
    total_rescued = Counter()       # rule -> detections the LiDAR saved from it
    total_kept = Counter()          # class -> merged detections that survived 1B
    total_verdicts = Counter()      # ok / bad / unknown, over door candidates
    total_abstain = Counter()       # why a door candidate had no measurement
    n_no_cloud = 0

    for i, triplet in enumerate(triplets):
        t0 = time.time()
        zed_file = triplet["zed"]["file"]
        flir_file = triplet["flir"]["file"]
        flir_stem = Path(flir_file).stem

        image = np.asarray(Image.open(frames_dir / zed_file).convert("RGB"))
        ih, iw = image.shape[:2]

        labels, segments, _sem, _conf = m2f.m2f_segments(
            image, proc, model, ade_to_class, OTHER_CLASS, min_area=args.min_area)
        for s in segments:
            total_class[s["top_class"]] += 1

        # --- the per-frame LiDAR probe, if there is a scan for this instant --
        door_metric_fn = None
        frame_depth = None
        if clouds is not None:
            cloud = clouds[i]
            if cloud is None:
                n_no_cloud += 1
            else:
                _t_scan, points_world = cloud
                K = cal.zed_K_for(iw, ih)
                uv, depth, valid = project_fn(
                    points_world, np.array(triplet["lidar"]["position"]),
                    np.array(triplet["lidar"]["orientation"]), cal.T_lidar_to_zed,
                    K, cal.zed_calib.dist, iw, ih)
                frame_depth = lm.FrameDepth(uv, depth, valid, (ih, iw))

                def door_metric_fn(region, _fd=frame_depth, _K=K):
                    return lm.measure_region(
                        region, _fd, _K,
                        min_points=args.metric_min_points,
                        max_depth_ratio=args.metric_max_depth_ratio)

        # --- STAGE 1B: merge + geometric plausibility -----------------------
        # Runs on the raster, so labels.npy is rewritten with the merged region
        # ids. Windows are resolved before doors: the door edge-veto tests
        # against merged windows, so a door can never veto a window.
        geom_report = None
        if args.geometry_filter:
            labels, segments, geom_report = geom.apply_geometry_filter(
                labels, segments, m2f.zone_for_merged,
                dilate_px=args.merge_dilate_px,
                floor_tol_px=args.floor_tol_px,
                glass_wall_h_ratio=args.glass_wall_h_ratio,
                door_edge_window_frac=args.door_edge_window_frac,
                door_edge_band_px=args.door_edge_band_px,
                min_door_width_px=args.min_door_width_px,
                min_region_area_px=args.min_area,
                door_metric_fn=door_metric_fn,
                door_verdict_fn=door_verdict_fn,
                window_filter=args.window_filter)
            for rule, n in geom_report["rejected"].items():
                total_rejected[rule] += n
            for cls, n in geom_report["kept"].items():
                total_kept[cls] += n
            for rule, n in geom_report["rescued"].items():
                total_rescued[rule] += n
            for v, n in geom_report["metric_verdicts"].items():
                total_verdicts[v] += n
            for why, n in geom_report["metric_abstentions"].items():
                total_abstain[why] += n

        frame_dir = out_dir / flir_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "labels.npy", labels.astype(np.int32))
        (frame_dir / "segments.json").write_text(json.dumps({
            "schema": "opening_map/v1",
            "generated_by": "classify_openings.py",
            "generated_utc": utc_now_iso(),
            "source_zed_frame": zed_file,
            "source_flir_frame": flir_file,
            "classes": table.classes,
            "segmenter": {"model": args.model, "min_area_px": args.min_area,
                          "ade_to_class": ade_to_class},
            "n_segments": len(segments),
            # Always null: this stage never crops. The key is kept so the file
            # stays readable by anything written against material_map/v1.
            "flir_fov_crop": None,
            # Zones now come from the ADE class, not from bbox shape, so the
            # old "forced to other" bookkeeping has nothing to count: a floor
            # region is `other` because `floor` is not in ade_to_class, not
            # because a prior overrode a classifier. Kept as a stub so a reader
            # written against the old schema does not KeyError.
            "zone_constraint": {"enabled": False, "forced_to_other": {},
                                "note": "superseded by segmentation_m2f.zone_from_ade"},
            # Stage 1B. Segments here are MERGED detections: "merged_from" lists
            # the region ids a detection absorbed, a rejected detection carries
            # a "rejected" block naming the rule, and a door carries its LiDAR
            # measurement under geometry.lidar when --bag was given.
            "geometry_filter": None if geom_report is None else {
                "enabled": True,
                "merged": geom_report["merged"],
                "kept": geom_report["kept"],
                "rejected": geom_report["rejected"],
                "rescued": geom_report["rescued"],
                "metric_verdicts": geom_report["metric_verdicts"],
                "metric_abstentions": geom_report["metric_abstentions"],
                "params": {
                    "merge_dilate_px": args.merge_dilate_px,
                    "floor_tol_px": args.floor_tol_px,
                    "glass_wall_h_ratio": args.glass_wall_h_ratio,
                    "door_edge_window_frac": args.door_edge_window_frac,
                    "door_edge_band_px": args.door_edge_band_px,
                    "min_door_width_px": args.min_door_width_px,
                    "window_filter": args.window_filter,
                    "lidar": None if door_metric_fn is None else {
                        "cloud_source": args.cloud_source,
                        "door_h_m": args.door_h_m, "door_w_m": args.door_w_m,
                        "min_points": args.metric_min_points,
                        "max_depth_ratio": args.metric_max_depth_ratio,
                        "n_returns_in_frame": int(frame_depth.depth.size),
                    },
                },
            },
            "segments": segments,
        }, indent=2), encoding="utf-8")

        if args.overlay:
            import cv2
            cv2.imwrite(str(frame_dir / "overlay.png"), draw_overlay(image, labels, segments))

        total_segments += len(segments)
        dt = time.time() - t0
        n_open = sum(1 for s in segments if s["top_class"] != OTHER_CLASS)
        geom_note = ""
        if geom_report is not None:
            nrej = sum(geom_report["rejected"].values())
            nres = sum(geom_report["rescued"].values())
            geom_note = (f", 1B: {geom_report['kept'].get('window', 0)}W/"
                         f"{geom_report['kept'].get('door', 0)}D kept"
                         + (f", {nrej} rejected" if nrej else "")
                         + (f", {nres} lidar-rescued" if nres else ""))
        print(f"[{i + 1}/{len(triplets)}] {flir_stem}: {len(segments)} regions, "
              f"{n_open} opening(s) in {dt:.1f}s{geom_note}")

    if total_segments:
        print("\nper-region class counts (before 1B): " +
              ", ".join(f"{c}={n}" for c, n in total_class.most_common()))
        if args.geometry_filter:
            print("stage 1B kept: " +
                  (", ".join(f"{c}={n}" for c, n in total_kept.most_common()) or "nothing"))
            if total_rejected:
                print("stage 1B rejected, by rule:")
                for rule, n in total_rejected.most_common():
                    print(f"  {n:4d}  {rule}")
            if total_rescued:
                print("LiDAR rescued from, by rule:")
                for rule, n in total_rescued.most_common():
                    print(f"  {n:4d}  {rule}")
            if args.bag:
                print(f"door metric verdicts: " +
                      ", ".join(f"{v}={total_verdicts.get(v, 0)}"
                                for v in ("ok", "bad", "unknown")))
                if total_verdicts.get("unknown"):
                    print("  no measurement, by cause: " +
                          (", ".join(f"{w}={n}" for w, n in total_abstain.most_common())
                           or "none recorded"))
                    print("  NOTE those candidates were decided by the pixel rules alone. "
                          "few_points means the surface returned nothing (glazing, "
                          "distance, occlusion) and the fix is --metric-min-points or "
                          "accepting the pixel rules there; multi_depth means the mask is "
                          "not one surface and the fix is upstream, in what got merged "
                          "into it.")
        if n_no_cloud:
            print(f"{n_no_cloud} frame(s) had no LiDAR scan near that instant.")
    print(f"Done. Output in {out_dir}")
    print("Next: opening_voxel_consensus.py --session-dir ... --bag ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
