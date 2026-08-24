"""Multi-view material consensus, and a 3-D thermal map, by voxel voting.

Why
---
classify_session.py decides a material from ONE view of a surface. Measured on
this rig, the same physical 20 cm patch gets called ~3.9 different materials
over a session, and 49% of per-frame labels disagree with what the majority of
views say. That is not an accuracy problem -- every indoor material sits at
e=0.90-0.95, so the mean temperature barely moves -- it is a CONSISTENCY
problem: emissivity flickers across a wall from frame to frame.

Every LiDAR point already carries a world-frame position and, once projected
into the ZED frame, the superpixel it fell in -- hence that superpixel's
material and confidence. Pooling those votes per world voxel turns ~31
independent looks at each surface into one answer. Measured effect: the spatial
spread of emissivity drops 3.1x (std 0.0457 -> 0.0145).

Two stages, because the second needs the correction to have run:

  --stage vote     after project_to_flir.py. Votes materials per voxel and
                   writes a parallel material_map with the consensus material
                   substituted per segment. Feed it to correct_session.py with
                   --material-map-dir.

  --stage thermal  after correct_session.py. Samples corrected_temperature.npy
                   at every LiDAR point and averages per voxel, giving a 3-D
                   map of corrected temperature + consensus material.

Nothing is overwritten: the consensus goes to new directories.

Venv: this reads the LiDAR bag, so run it with the SensorFusion venv (same
convention as project_to_flir.py -- see requirements.txt). It deliberately
avoids emissivity.table, which imports pandas.

Usage:
    py voxel_consensus.py --stage vote --session-dir ...\\fullrate
        --bag ...\\rosbag2_2026_07_30-18_12_20
    py correct_session.py --session-dir ...\\fullrate --flir-dir ...\\session9_only_rot180
        --humidity 50 --air-temp 20 --material-map-dir ...\\fullrate\\material_map_consensus
    py voxel_consensus.py --stage thermal --session-dir ...\\fullrate
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Calibration"))
from projection import project_lidar_to_camera
from rig_calibration import load_rig_calibration
from project_to_flir import nearest_clouds_for_targets

# zones.py is imported by path rather than as emissivity.zones: the package
# __init__ pulls in table.py (pandas) and classifier.py (torch), neither of
# which exists in the rosbags venv this script runs under. zones.py itself has
# no imports at all.
sys.path.insert(0, str(Path(__file__).resolve().parent / "emissivity"))
from zones import ZONE_CANDIDATES, restrict_ranking  # noqa: E402

DEFAULT_TABLE = Path(__file__).resolve().parent / "emissivity_table.csv"


def load_emissivity(path=DEFAULT_TABLE) -> dict:
    """material -> emissivity. Read with stdlib csv on purpose: emissivity.table
    pulls in pandas, which the rosbags venv does not have."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["material"]: float(r["emissivity"]) for r in csv.DictReader(f)}


def load_solar_absorptance(path=DEFAULT_TABLE) -> dict:
    """material -> solar_absorptance. Same stdlib-csv rationale as load_emissivity.
    Fraction of incident shortwave (sunlight) the material absorbs -- what
    lets a sun-exposed voxel run hotter than a passive conduction model
    predicts (see voxel_u_value.py's solar_suspected/solar_possible flags)."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["material"]: float(r["solar_absorptance"]) for r in csv.DictReader(f)}


def parse_args():
    p = argparse.ArgumentParser(description="Multi-view material consensus / 3-D thermal map")
    p.add_argument("--stage", choices=("vote", "thermal"), required=True)
    p.add_argument("--session-dir", required=True, metavar="DIR")
    p.add_argument("--bag", required=True, metavar="DIR", help="rosbag2 folder with the LiDAR odometry topic")
    p.add_argument("--voxel", type=float, default=0.20, metavar="M",
                    help="Voxel edge in metres (default 0.20). Must exceed the registration "
                         "error: composing the two LiDAR<->camera extrinsics is ~9 cm RMSE "
                         "before SLAM drift, so do not go below ~0.15.")
    p.add_argument("--respect-zones", action=argparse.BooleanOptionalAction, default=True,
                    help="Re-apply classify_session.py's geometric prior to the pooled votes "
                         "(default on). A voxel pools views whose segments had different zones, "
                         "so without this the consensus can hand a wall a floor material.")
    p.add_argument("--min-emissivity", type=float, default=0.5, metavar="E",
                    help="Floor on emissivity for segments whose zone carries no categorical "
                         "list (default 0.5). Same role as classify_session.py's "
                         "--low-emissivity-max.")
    p.add_argument("--min-agreement", type=float, default=0.0, metavar="F",
                    help="Only override a segment's own material when at least this fraction "
                         "of its pooled vote mass backs the winner. Default 0.0 = always take "
                         "the session's answer over a single view, however thin the majority. "
                         "Raising it keeps the per-frame call where the session is genuinely "
                         "split: 0.5 cuts reassignments from 31.2%% to 11.3%% of votable "
                         "segments and holds back the drift toward the most common material.")
    p.add_argument("--min-vote-confidence", type=float, default=0.5, metavar="C",
                    help="Drop individual votes below this confidence before pooling "
                         "(default 0.5). Measured by split-half reproducibility -- consensus "
                         "built on even vs odd frames -- agreement rises 69.3%% -> 73.1%% while "
                         "voxel coverage only falls to 96%%. 0 disables.")
    p.add_argument("--depth-power", type=float, default=0.0, metavar="P",
                    help="Vote weight is confidence * (1/depth)**P (default 0.0 = ignore "
                         "distance). Measured: P=0 reproduces better than P=0.5 at every "
                         "confidence floor, so weighting by proximity only added variance.")
    p.add_argument("--material-map-dir", default=None, metavar="DIR")
    p.add_argument("--emissivity-map-dir", default=None, metavar="DIR")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                    help="vote: default <session>/material_map_consensus. "
                         "thermal: default <session>/voxel_map.")
    p.add_argument("--corrected-name", default="corrected_temperature.npy",
                    help="Filename written by correct_session.py (default corrected_temperature.npy)")
    p.add_argument("--calibration", default=None, metavar="YAML")
    p.add_argument("--odom-topic", default="/cloud_registered")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--every-n", type=int, default=1, metavar="N")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    return p.parse_args()


def frames_with(session_dir: Path, needed: Path, triplets: list) -> list:
    """Triplets whose per-frame folder under `needed` exists."""
    out = []
    for t in triplets:
        stem = Path(t["flir"]["file"]).stem
        if (needed / stem).is_dir():
            out.append(t)
    return out


def project_points(cal, triplet, points_world, target, width, height):
    """LiDAR points -> pixels in the ZED or FLIR frame."""
    T = cal.T_lidar_to_zed if target == "zed" else cal.T_lidar_to_flir
    K = cal.zed_K_for(width, height) if target == "zed" else cal.flir.K
    dist = cal.zed_calib.dist if target == "zed" else cal.flir.dist
    return project_lidar_to_camera(
        points_world, np.array(triplet["lidar"]["position"]),
        np.array(triplet["lidar"]["orientation"]), T, K, dist, width, height)


def stage_vote(args, cal, session_dir, triplets):
    material_dir = Path(args.material_map_dir) if args.material_map_dir else session_dir / "material_map"
    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "material_map_consensus"
    eps_of = load_emissivity()

    work = frames_with(session_dir, material_dir, triplets)
    if not work:
        print(f"No frame has material_map output under {material_dir}", file=sys.stderr)
        return 1
    print(f"{len(work)} frame(s) with material maps, voxel {args.voxel * 100:.0f} cm")

    print(f"Reading LiDAR scans from {Path(args.bag).name} ...")
    clouds = nearest_clouds_for_targets(
        Path(args.bag), [t["lidar"]["timestamp_zedclock"] for t in work],
        args.odom_topic, args.store)

    votes = defaultdict(Counter)        # voxel -> material -> weight
    hits = {}                           # stem -> {segment id -> Counter(voxel)}
    n_votes = 0
    n_weak_votes = 0

    for triplet, cloud in zip(work, clouds):
        stem = Path(triplet["flir"]["file"]).stem
        if cloud is None:
            print(f"skip {stem}: no LiDAR scan near that instant", file=sys.stderr)
            continue
        _t_scan, points_world = cloud
        labels = np.load(material_dir / stem / "labels.npy")
        segs = json.loads((material_dir / stem / "segments.json").read_text(encoding="utf-8"))["segments"]
        info = {int(s["id"]): (s["top_material"], float(s["confidence"])) for s in segs}
        zh, zw = labels.shape

        uv, depth, valid = project_points(cal, triplet, points_world, "zed", zw, zh)
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
            material, conf = info[sid]
            # A near-tie CLIP call is a coin flip, and letting it vote only adds
            # noise: dropping votes below --min-vote-confidence raises split-half
            # reproducibility from 69.3% to 73.1% while keeping 96% of voxels.
            if conf < args.min_vote_confidence:
                n_weak_votes += 1
                continue
            weight = conf * (1.0 / max(1.0, float(dep[k]))) ** args.depth_power
            key = (int(vox[k, 0]), int(vox[k, 1]), int(vox[k, 2]))
            votes[key][material] += weight
            per_seg[sid][key] += 1
            n_votes += 1
        hits[stem] = per_seg

    if not votes:
        print("No votes -- no LiDAR point landed in a labelled superpixel.", file=sys.stderr)
        return 1

    consensus = {k: c.most_common(1)[0][0] for k, c in votes.items()}
    n_materials = np.array([len(c) for c in votes.values()])
    print(f"{n_votes} votes -> {len(votes)} voxels "
          f"({n_weak_votes} votes dropped below confidence {args.min_vote_confidence})")
    print(f"distinct materials proposed per voxel: mean {n_materials.mean():.2f}, "
          f"max {n_materials.max()}; {100.0 * (n_materials > 1).mean():.1f}% of voxels "
          f"got more than one")

    # Rewrite each frame's segments.json with the consensus of the voxels its
    # own points fell in. Segments no LiDAR point reached keep their original
    # call -- there is nothing better to say about them.
    out_dir.mkdir(parents=True, exist_ok=True)
    n_changed = n_total = n_orphan = n_weak = 0
    for triplet in work:
        stem = Path(triplet["flir"]["file"]).stem
        src = material_dir / stem
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
            # Summing winners is a second argmax on top of the per-voxel one: a
            # voxel where painted_metal beat plaster 51-to-49 would contribute a
            # full vote and discard the 49, which systematically amplifies
            # whichever class is already most common. Measured before this fix:
            # painted_metal +276 segments, concrete -178, and plaster / brick /
            # wood / cardboard wiped out entirely.
            pooled = Counter()
            for key, count in voxels.items():
                total = sum(votes[key].values()) or 1.0
                for mat, weight in votes[key].items():
                    pooled[mat] += count * weight / total

            # Re-apply the geometric prior. classify_session.py enforces it
            # per frame, but a voxel pools votes from segments that had
            # DIFFERENT zones -- a voxel straddling the wall/floor junction
            # collects floor votes, a voxel spanning a radiator and the ceiling
            # above it collects painted_metal. Without this, 5.1% of overrides
            # landed on a material the segment's own zone forbids (mostly
            # "vertical -> rubber" and "ceiling -> painted_metal").
            ranked = pooled.most_common()
            if args.respect_zones:
                ranked = restrict_ranking(ranked, seg.get("zone", "any"),
                                          eps_of=eps_of, min_eps=args.min_emissivity)
            material, best = ranked[0]
            agree = best / (sum(p for _m, p in ranked) or 1.0)
            seg["consensus"] = {
                "status": "ok",
                "from_frame": seg["top_material"],
                "n_voxels": len(voxels),
                "agreement": round(agree, 3),
            }
            # Below --min-agreement the segment keeps what its own view decided.
            # Off by default: the session's pooled answer is taken over a single
            # view even on a thin majority. Raise it to hold back near-ties --
            # they are where the drift toward the most common material lives.
            if agree < args.min_agreement:
                seg["consensus"]["status"] = "below_min_agreement"
                seg["consensus"]["would_be"] = material
                n_weak += 1
                continue
            if material != seg["top_material"]:
                n_changed += 1
            seg["top_material"] = material
            seg["emissivity"] = eps_of[material]
        doc["generated_by"] = "voxel_consensus.py --stage vote"
        doc["consensus"] = {"voxel_m": args.voxel, "n_voxels": len(votes), "n_votes": n_votes}
        (dst / "segments.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print(f"materials replaced on {n_changed}/{n_total} segments "
          f"({100.0 * n_changed / max(1, n_total):.1f}%)")
    print(f"kept their per-frame call: {n_orphan} without LiDAR samples, "
          f"{n_weak} below --min-agreement {args.min_agreement}")
    print(f"\nDone. Consensus material map in {out_dir}")
    print("Next: correct_session.py ... --material-map-dir "
          f"\"{out_dir}\"")
    return 0


def stage_thermal(args, cal, session_dir, triplets):
    emis_dir = Path(args.emissivity_map_dir) if args.emissivity_map_dir else session_dir / "emissivity_map"
    material_dir = Path(args.material_map_dir) if args.material_map_dir else session_dir / "material_map_consensus"
    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "voxel_map"
    alpha_of = load_solar_absorptance()

    work = [t for t in frames_with(session_dir, emis_dir, triplets)
            if (emis_dir / Path(t["flir"]["file"]).stem / args.corrected_name).exists()]
    if not work:
        print(f"No frame has {args.corrected_name} under {emis_dir} -- run correct_session.py first",
              file=sys.stderr)
        return 1
    print(f"{len(work)} corrected frame(s), voxel {args.voxel * 100:.0f} cm")

    print(f"Reading LiDAR scans from {Path(args.bag).name} ...")
    clouds = nearest_clouds_for_targets(
        Path(args.bag), [t["lidar"]["timestamp_zedclock"] for t in work],
        args.odom_topic, args.store)

    acc = defaultdict(lambda: [0.0, 0.0, 0])      # voxel -> [sum T, sum T^2, n]
    mat = defaultdict(Counter)
    for triplet, cloud in zip(work, clouds):
        stem = Path(triplet["flir"]["file"]).stem
        if cloud is None:
            continue
        _t_scan, points_world = cloud
        corrected = np.load(emis_dir / stem / args.corrected_name)
        fh, fw = corrected.shape

        uv, _depth, valid = project_points(cal, triplet, points_world, "flir", fw, fh)
        if not valid.any():
            continue
        px = np.round(uv[valid]).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, fw - 1)
        px[:, 1] = np.clip(px[:, 1], 0, fh - 1)
        temps = corrected[px[:, 1], px[:, 0]]
        vox = np.floor(points_world[valid] / args.voxel).astype(np.int64)

        seg_material = {}
        seg_path = material_dir / stem / "segments.json"
        seg_id_path = emis_dir / stem / "segment_id.npy"
        segment_id = np.load(seg_id_path) if seg_id_path.exists() else None
        if seg_path.exists():
            seg_material = {int(s["id"]): s["top_material"] for s in
                            json.loads(seg_path.read_text(encoding="utf-8"))["segments"]}

        for k in range(len(temps)):
            t = float(temps[k])
            if not np.isfinite(t):        # correct_session writes NaN where no
                continue                  # candidate was physically plausible
            key = (int(vox[k, 0]), int(vox[k, 1]), int(vox[k, 2]))
            a = acc[key]
            a[0] += t
            a[1] += t * t
            a[2] += 1
            if segment_id is not None and seg_material:
                sid = int(segment_id[px[k, 1], px[k, 0]])
                if sid in seg_material:
                    mat[key][seg_material[sid]] += 1

    if not acc:
        print("No finite corrected temperature landed on a LiDAR point.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, (s1, s2, n) in acc.items():
        mean = s1 / n
        var = max(0.0, s2 / n - mean * mean)
        material = mat[key].most_common(1)[0][0] if mat.get(key) else ""
        rows.append({
            "x": (key[0] + 0.5) * args.voxel,
            "y": (key[1] + 0.5) * args.voxel,
            "z": (key[2] + 0.5) * args.voxel,
            "t_mean_c": round(mean, 3),
            "t_std_c": round(var ** 0.5, 3),
            "n_obs": n,
            "material": material,
            "solar_absorptance": alpha_of.get(material, "") if material else "",
        })
    rows.sort(key=lambda r: (r["x"], r["y"], r["z"]))

    csv_path = out_dir / "thermal_voxels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Same data as a PLY so it can be dropped straight into CloudCompare/Meshlab,
    # coloured by temperature over the 5-95 percentile range.
    t = np.array([r["t_mean_c"] for r in rows])
    lo, hi = np.percentile(t, [5, 95])
    norm = np.clip((t - lo) / max(1e-6, hi - lo), 0, 1)
    rgb = np.stack([norm, np.zeros_like(norm), 1.0 - norm], 1) * 255
    ply_path = out_dir / "thermal_voxels.ply"
    with open(ply_path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for r, c in zip(rows, rgb.astype(int)):
            f.write(f"{r['x']:.3f} {r['y']:.3f} {r['z']:.3f} {c[0]} {c[1]} {c[2]}\n")

    n_obs = np.array([r["n_obs"] for r in rows])
    std = np.array([r["t_std_c"] for r in rows])
    print(f"\n{len(rows)} voxels, {int(n_obs.sum())} samples")
    print(f"observations per voxel: median {np.median(n_obs):.0f}, mean {n_obs.mean():.1f}")
    print(f"corrected T: {t.min():.1f} .. {t.max():.1f} degC, mean {t.mean():.2f}")
    print(f"within-voxel spread: median std {np.median(std):.2f} degC, mean {std.mean():.2f}")
    if any(r["material"] for r in rows):
        print("materials:", Counter(r["material"] for r in rows if r["material"]).most_common(6))
    alpha = np.array([r["solar_absorptance"] for r in rows if r["solar_absorptance"] != ""])
    if alpha.size:
        print(f"solar absorptance: {alpha.min():.2f} .. {alpha.max():.2f}, mean {alpha.mean():.2f} "
              f"({alpha.size}/{len(rows)} voxels with a consensus material)")
    print(f"\nDone. {csv_path}\n      {ply_path}")
    return 0


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    cal_path = args.calibration or (
        Path(__file__).resolve().parent.parent / "Calibration" / "rig_calibration.yaml")
    cal = load_rig_calibration(cal_path)

    manifest = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    if args.stage == "vote":
        return stage_vote(args, cal, session_dir, triplets)
    return stage_thermal(args, cal, session_dir, triplets)


if __name__ == "__main__":
    sys.exit(main())
