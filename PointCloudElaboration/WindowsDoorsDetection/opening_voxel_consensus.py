"""Multi-view door/window consensus by voxel voting, and the 3-D opening map.

Stage 2 of two. Same mechanism as
EmissivityCalculation/voxel_consensus.py --stage vote, with door/window/other
substituted for materials -- including its measured defaults, which are NOT
re-derived here (--min-vote-confidence 0.5, --min-agreement 0.0,
--depth-power 0.0, --voxel 0.20).

Why
---
classify_openings.py decides door/window/other from ONE view. A door seen
edge-on, half-occluded, or in motion blur is easily called "other"; a bright
picture frame or a wall-mounted panel is easily called "window". Every LiDAR
point already carries a world-frame position and, once projected into the ZED
frame, the segment it fell in -- hence that segment's class and confidence.
Pooling those votes per world voxel turns the ~31 independent looks a session
gets at each surface into one answer, and a physical opening keeps whichever
class the majority of the views that actually saw it agree on.

"other" competes in the vote on equal terms. That is the point: a wall voxel
resolves definitively to "other", and a door voxel has to beat the wall votes
to win, so one confident bad view is not enough to invent an opening. If only
door/window votes were pooled, nothing would ever vote against a false
positive.

Two outputs, neither overwriting anything:

  <session>/opening_map_consensus/<stem>/   per-frame labels.npy +
      segments.json with the consensus class substituted per segment.
      Segments no LiDAR point reached keep their own per-frame call.

  <session>/opening_map_consensus/door_window_voxels.csv and .ply
      the session-level 3-D opening map -- only the voxels whose consensus is
      an opening, with the winning class, vote agreement and n_observations.
      This is what the downstream plane/polygon fit consumes. Mirrors
      voxel_consensus.py's thermal_voxels.csv/.ply.

Venv: this reads the LiDAR bag, so run it with the SensorFusion / rosbags venv
(same convention as project_to_flir.py). It deliberately avoids torch and
pandas: openings/table.py and openings/zone_prior.py are imported as bare
modules, not through the package __init__.

Usage:
    py opening_voxel_consensus.py --session-dir ...\\fullrate
        --bag ...\\rosbag2_2026_07_30-18_12_20
"""

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# --- sibling modules, imported by path -------------------------------------
# The Thesis-final-wt2 root is located by searching upwards, see _find_root.
# The calibration loader lives in SensorFusionLoader/, NOT Calibration/.
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
sys.path.insert(0, str(_ROOT / "SensorFusionLoader"))
from rig_calibration import load_rig_calibration  # noqa: E402
from projection import project_lidar_to_camera  # noqa: E402

# EmissivityCalculation is on the path only so that lidar_metrics can fall back
# to project_to_flir.nearest_clouds_for_targets under --cloud-source registered.
# Import order matters: project_to_flir runs its own
# sys.path.insert(0, "../Calibration") at import time and then does
# `from rig_calibration import ...`. That directory does not exist in this
# repo, so the insert is a no-op -- but the import above has already put
# rig_calibration and projection in sys.modules, so its top-level imports
# resolve to SensorFusionLoader's copies and it loads cleanly. Do not move
# this above the SensorFusionLoader import.
sys.path.insert(0, str(_ROOT / "EmissivityCalculation"))
# ../LivoxLidarOdometryLoader, the raw-cloud reader (numpy + rosbags only, no
# torch), reached through lidar_metrics.load_clouds.
_LOADER_DIR = _ROOT / "PointCloudElaboration" / "LivoxLidarOdometryLoader"

# Bare-module imports, not `from openings import ...`: the package is
# torch-free now that openings/classifier.py (the unused CLIP path) is gone,
# but this stage still avoids the package __init__ on principle -- same policy
# EmissivityCalculation's consensus stage uses, so a torch-needing addition to
# the package later cannot silently break the rosbags venv here.
sys.path.insert(0, str(Path(__file__).resolve().parent / "openings"))
from table import OTHER_CLASS  # noqa: E402
from zone_prior import restrict_opening_ranking  # noqa: E402
import lidar_metrics as lm  # noqa: E402

# Fixed colours for the PLY, so an opening map always reads the same way in
# CloudCompare/Meshlab regardless of what the session contains.
PLY_COLORS = {"door": (220, 40, 40), "window": (40, 110, 230)}
PLY_FALLBACK = (160, 160, 160)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-view door/window consensus by voxel voting + 3-D opening map")
    p.add_argument("--session-dir", required=True, metavar="DIR")
    p.add_argument("--bag", required=True, metavar="DIR",
                   help="rosbag2 folder with the LiDAR odometry topic")
    p.add_argument("--voxel", type=float, default=0.20, metavar="M",
                   help="Voxel edge in metres (default 0.20, same as voxel_consensus.py). Must "
                        "exceed the registration error: composing the two LiDAR<->camera "
                        "extrinsics is ~9 cm RMSE before SLAM drift, so do not go below ~0.15. "
                        "It is also the resolution the downstream polygon fit inherits -- at "
                        "0.20 a 0.9 m door leaf is ~5 voxels wide; going much coarser blurs an "
                        "opening's outline into the wall around it.")
    p.add_argument("--respect-zones", action=argparse.BooleanOptionalAction, default=True,
                   help="Re-apply classify_openings.py's geometric prior to the pooled votes "
                        "(default on). A voxel pools views whose segments had different zones "
                        "-- one straddling the wall/floor junction collects floor votes -- so "
                        "without this the consensus can hand a floor segment a door class.")
    p.add_argument("--min-agreement", type=float, default=0.0, metavar="F",
                   help="Only override a segment's own class when at least this fraction of "
                        "its pooled vote mass backs the winner. Default 0.0 = always take the "
                        "session's answer over a single view, however thin the majority "
                        "(voxel_consensus.py's default and rationale). Raising it keeps the "
                        "per-frame call where the session is genuinely split.")
    p.add_argument("--min-vote-confidence", type=float, default=0.5, metavar="C",
                   help="Drop individual votes below this confidence before pooling (default "
                        "0.5, voxel_consensus.py's measured value -- split-half reproducibility "
                        "rose 69.3%% -> 73.1%% while voxel coverage only fell to 96%%). "
                        "0 disables.")
    p.add_argument("--max-range", type=float, default=8.0, metavar="M",
                   help="Ignore LiDAR points farther than this from the camera when voting "
                        "(default 8.0 m; 0 disables). A surface classified from 20 m away is "
                        "a few pixels of an oblique, blurred region, and SAM merges a whole "
                        "corridor run of glazing into regions spanning 4.8-17.5 m -- measured "
                        "on session 9. Gating at the VOTE is why this uses exact per-point "
                        "depth: no densification is involved, so there is no fail-open case "
                        "for glazing, which simply returns no points and casts no vote either "
                        "way. Note this changes the 3-D product only; per-frame overlays from "
                        "stage 1 still show the full-length regions.")
    p.add_argument("--depth-power", type=float, default=0.0, metavar="P",
                   help="Vote weight is confidence * (1/depth)**P (default 0.0 = ignore "
                        "distance). Measured on the material pipeline: P=0 reproduces better "
                        "than P=0.5 at every confidence floor, so weighting by proximity only "
                        "added variance.")
    p.add_argument("--opening-map-dir", default=None, metavar="DIR",
                   help="classify_openings.py output root (default: <session-dir>/opening_map).")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Default <session-dir>/opening_map_consensus.")
    p.add_argument("--calibration", default=None, metavar="YAML",
                   help="Default: SensorFusionLoader/rig_calibration.yaml.")
    p.add_argument("--cloud-source", choices=("raw", "registered"),
                   default=lm.DEFAULT_CLOUD_SOURCE,
                   help="Where the LiDAR comes from, and it decides whether this stage "
                        "can produce anything at all. 'raw' (default) rebuilds world "
                        "clouds from /livox/lidar + /Odometry via "
                        "../LivoxLidarOdometryLoader. 'registered' reads FAST-LIO's "
                        "/cloud_registered, which on session 9 is cropped to a 35-deg "
                        "cone from 4 m: both window bays fall outside it, get zero "
                        "returns, and no window voxel can exist no matter what stage 1 "
                        "found.")
    p.add_argument("--lidar-topic", default=lm.RAW_LIDAR_TOPIC, metavar="TOPIC")
    p.add_argument("--pose-topic", default=lm.POSE_TOPIC, metavar="TOPIC")
    p.add_argument("--odom-topic", dest="registered_topic", default=lm.REGISTERED_TOPIC,
                   help="Registered-cloud topic, --cloud-source registered. Named "
                        "--odom-topic for backwards compatibility; it is a misnomer.")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--every-n", type=int, default=1, metavar="N")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    return p.parse_args()


def frames_with(needed: Path, triplets: list) -> list:
    """Triplets whose per-frame folder under `needed` exists."""
    out = []
    for t in triplets:
        stem = Path(t["flir"]["file"]).stem
        if (needed / stem).is_dir():
            out.append(t)
    return out


def project_points_zed(cal, triplet, points_world, width, height):
    """LiDAR points -> ZED pixels. Only the ZED is needed here: unlike the
    thermal pipeline there is no FLIR side to this question."""
    return project_lidar_to_camera(
        points_world, np.array(triplet["lidar"]["position"]),
        np.array(triplet["lidar"]["orientation"]), cal.T_lidar_to_zed,
        cal.zed_K_for(width, height), cal.zed_calib.dist, width, height)


def write_voxel_map(out_dir: Path, votes, obs_frames, n_votes_per_voxel, consensus,
                    voxel_m, classes):
    """The session-level 3-D opening map: one row per voxel whose consensus is
    an actual opening. This is the hand-off to the downstream plane/polygon
    fit, which is out of scope here.

    Every class's pooled weight is written out, not just the winner's, so that
    step can apply its own threshold without re-running the vote.
    """
    rows = []
    for key, cls in consensus.items():
        if cls == OTHER_CLASS:
            continue
        counter = votes[key]
        total = sum(counter.values()) or 1.0
        row = {
            "x": round((key[0] + 0.5) * voxel_m, 4),
            "y": round((key[1] + 0.5) * voxel_m, 4),
            "z": round((key[2] + 0.5) * voxel_m, 4),
            "opening_class": cls,
            # Fraction of the pooled vote mass the winner holds. 1.0 means
            # every view that saw this voxel agreed.
            "agreement": round(counter[cls] / total, 4),
            # Distinct frames that saw the voxel -- the real "how many views",
            # unlike n_votes which counts LiDAR points and so grows with how
            # densely the scan happened to hit the surface.
            "n_observations": len(obs_frames[key]),
            # LiDAR points that voted here. Grows with how densely the scan hit
            # the surface, so it is a sampling statistic, not a view count.
            "n_votes": n_votes_per_voxel[key],
        }
        for c in classes:
            row[f"w_{c}"] = round(counter.get(c, 0.0), 4)
        rows.append(row)
    rows.sort(key=lambda r: (r["x"], r["y"], r["z"]))

    csv_path = out_dir / "door_window_voxels.csv"
    ply_path = out_dir / "door_window_voxels.ply"
    if not rows:
        print("No voxel resolved to an opening -- door_window_voxels.csv/.ply not written.",
              file=sys.stderr)
        return None, None, rows

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for r in rows:
            c = PLY_COLORS.get(r["opening_class"], PLY_FALLBACK)
            f.write(f"{r['x']:.3f} {r['y']:.3f} {r['z']:.3f} {c[0]} {c[1]} {c[2]}\n")

    return csv_path, ply_path, rows


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    cal_path = args.calibration or (_ROOT / "SensorFusionLoader" / "rig_calibration.yaml")
    cal = load_rig_calibration(cal_path)

    manifest = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    opening_dir = (Path(args.opening_map_dir) if args.opening_map_dir
                   else session_dir / "opening_map")
    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "opening_map_consensus"

    work = frames_with(opening_dir, triplets)
    if not work:
        print(f"No frame has opening_map output under {opening_dir} -- run "
              "classify_openings.py first.", file=sys.stderr)
        return 1
    print(f"{len(work)} frame(s) with opening maps, voxel {args.voxel * 100:.0f} cm")

    src = (f"{args.lidar_topic} + {args.pose_topic}" if args.cloud_source == "raw"
           else args.registered_topic)
    print(f"Reading LiDAR scans from {Path(args.bag).name} [{args.cloud_source}: {src}] ...")
    clouds = lm.load_clouds(
        Path(args.bag), [t["lidar"]["timestamp_zedclock"] for t in work],
        store=args.store, source=args.cloud_source, lidar_topic=args.lidar_topic,
        registered_topic=args.registered_topic, pose_topic=args.pose_topic,
        loader_dir=_LOADER_DIR)
    npts = [len(c[1]) for c in clouds if c is not None]
    if npts:
        print(f"  {len(npts)} scan(s), {min(npts)}-{max(npts)} points each")

    votes = defaultdict(Counter)        # voxel -> class -> pooled weight
    obs_frames = defaultdict(set)       # voxel -> {frame stem}
    n_votes_per_voxel = Counter()       # voxel -> LiDAR points that voted
    hits = {}                           # stem -> {segment id -> Counter(voxel)}
    classes = set()
    n_votes = n_weak_votes = n_far_votes = 0

    for triplet, cloud in zip(work, clouds):
        stem = Path(triplet["flir"]["file"]).stem
        if cloud is None:
            print(f"skip {stem}: no LiDAR scan near that instant", file=sys.stderr)
            continue
        _t_scan, points_world = cloud
        labels = np.load(opening_dir / stem / "labels.npy")
        doc = json.loads((opening_dir / stem / "segments.json").read_text(encoding="utf-8"))
        classes.update(doc.get("classes") or [])
        info = {int(s["id"]): (s["top_class"], float(s["confidence"])) for s in doc["segments"]}
        zh, zw = labels.shape

        uv, depth, valid = project_points_zed(cal, triplet, points_world, zw, zh)
        if not valid.any():
            continue
        px = np.round(uv[valid]).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, zw - 1)
        px[:, 1] = np.clip(px[:, 1], 0, zh - 1)
        sids = labels[px[:, 1], px[:, 0]]
        vox = np.floor(points_world[valid] / args.voxel).astype(np.int64)
        dep = depth[valid]

        per_seg = defaultdict(Counter)
        for k in range(len(sids)):
            sid = int(sids[k])
            if sid < 0 or sid not in info:
                continue
            cls, conf = info[sid]
            # Too far to have been classified reliably. Checked before the
            # confidence gate because distance is a property of the
            # observation, not of the call.
            if args.max_range > 0 and float(dep[k]) > args.max_range:
                n_far_votes += 1
                continue
            # A near-tie CLIP call is a coin flip and only adds noise; the
            # material pipeline measured split-half reproducibility rising
            # from 69.3% to 73.1% by dropping these.
            if conf < args.min_vote_confidence:
                n_weak_votes += 1
                continue
            weight = conf * (1.0 / max(1.0, float(dep[k]))) ** args.depth_power
            key = (int(vox[k, 0]), int(vox[k, 1]), int(vox[k, 2]))
            votes[key][cls] += weight
            obs_frames[key].add(stem)
            n_votes_per_voxel[key] += 1
            per_seg[sid][key] += 1
            n_votes += 1
        hits[stem] = per_seg

    if not votes:
        print("No votes -- no LiDAR point landed in a labelled segment.", file=sys.stderr)
        return 1

    consensus = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    classes = sorted(classes) or sorted({c for counter in votes.values() for c in counter})
    n_classes = np.array([len(c) for c in votes.values()])
    print(f"{n_votes} votes -> {len(votes)} voxels "
          f"({n_weak_votes} dropped below confidence {args.min_vote_confidence}"
          + (f", {n_far_votes} dropped beyond {args.max_range:g} m" if args.max_range > 0 else "")
          + ")")
    print(f"distinct classes proposed per voxel: mean {n_classes.mean():.2f}, "
          f"max {n_classes.max()}; {100.0 * (n_classes > 1).mean():.1f}% of voxels "
          f"got more than one")
    print("voxel consensus: " +
          ", ".join(f"{c}={n}" for c, n in Counter(consensus.values()).most_common()))

    # Rewrite each frame's segments.json with the consensus of the voxels its
    # own points fell in. Segments no LiDAR point reached keep their original
    # call -- there is nothing better to say about them.
    out_dir.mkdir(parents=True, exist_ok=True)
    n_changed = n_total = n_orphan = n_weak = 0
    changes = Counter()
    for triplet in work:
        stem = Path(triplet["flir"]["file"]).stem
        src = opening_dir / stem
        dst = out_dir / stem
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "labels.npy", dst / "labels.npy")

        doc = json.loads((src / "segments.json").read_text(encoding="utf-8"))
        per_seg = hits.get(stem, {})
        for seg in doc["segments"]:
            sid = int(seg["id"])
            voxels = per_seg.get(sid)
            n_total += 1
            if not voxels:
                seg["consensus"] = {"status": "no_lidar_sample"}
                n_orphan += 1
                continue
            # Pool the vote MASS of each voxel, not the voxel's hard winner.
            # Summing winners is a second argmax on top of the per-voxel one:
            # a voxel where other beat window 51-to-49 would contribute a full
            # vote and discard the 49, which systematically amplifies whichever
            # class is already most common -- and here that is "other", by a
            # wide margin, so it would erase openings outright.
            pooled = Counter()
            for key, count in voxels.items():
                total = sum(votes[key].values()) or 1.0
                for cls, weight in votes[key].items():
                    pooled[cls] += count * weight / total

            # Re-apply the geometric prior: a voxel pools votes from segments
            # that had DIFFERENT zones, so a voxel straddling the wall/floor
            # junction collects floor votes and could hand a floor segment a
            # door class.
            ranked = pooled.most_common()
            if args.respect_zones:
                ranked = restrict_opening_ranking(ranked, seg.get("zone", "any"))
            cls, best = ranked[0]
            agree = best / (sum(p for _c, p in ranked) or 1.0)
            seg["consensus"] = {
                "status": "ok",
                "from_frame": seg["top_class"],
                "n_voxels": len(voxels),
                "agreement": round(agree, 3),
            }
            if agree < args.min_agreement:
                seg["consensus"]["status"] = "below_min_agreement"
                seg["consensus"]["would_be"] = cls
                n_weak += 1
                continue
            if cls != seg["top_class"]:
                n_changed += 1
                changes[f"{seg['top_class']} -> {cls}"] += 1
            seg["top_class"] = cls
            # The consensus confidence IS the vote agreement -- the per-frame
            # CLIP probability no longer describes this label. `best` is
            # normalised against the same ranking `agree` used, restricted or
            # not, so the two are the same number by construction.
            seg["confidence"] = round(agree, 4)
        doc["generated_by"] = "opening_voxel_consensus.py"
        doc["consensus"] = {"voxel_m": args.voxel, "n_voxels": len(votes),
                            "n_votes": n_votes, "max_range_m": args.max_range}
        (dst / "segments.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"\nclasses replaced on {n_changed}/{n_total} segments "
          f"({100.0 * n_changed / max(1, n_total):.1f}%)")
    if changes:
        print("  " + ", ".join(f"{k}: {v}" for k, v in changes.most_common()))
    print(f"kept their per-frame call: {n_orphan} without LiDAR samples, "
          f"{n_weak} below --min-agreement {args.min_agreement}")

    csv_path, ply_path, rows = write_voxel_map(
        out_dir, votes, obs_frames, n_votes_per_voxel, consensus, args.voxel, classes)

    if rows:
        n_obs = np.array([r["n_observations"] for r in rows])
        agree = np.array([r["agreement"] for r in rows])
        print(f"\n{len(rows)} opening voxel(s): " +
              ", ".join(f"{c}={n}" for c, n in
                        Counter(r["opening_class"] for r in rows).most_common()))
        print(f"observations per voxel: median {np.median(n_obs):.0f}, mean {n_obs.mean():.1f}")
        print(f"agreement: median {np.median(agree):.2f}, "
              f"{100.0 * (agree >= 0.5).mean():.0f}% at or above 0.5")
        print(f"\nDone. {csv_path}\n      {ply_path}")
    print(f"Consensus opening map in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
