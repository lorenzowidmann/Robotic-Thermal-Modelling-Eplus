"""Fuse ZED-classified material/emissivity (classify_session.py) onto real
FLIR pixels, via LiDAR as the common bridge -- no direct FLIR<->ZED
calibration exists or is needed.

For each synced triplet: take the LiDAR scan nearest that instant, project
its points into ZED pixels (T_lidar_to_zed) to pick up a material/emissivity
value from that frame's classify_session.py output, and into FLIR pixels
(T_lidar_to_flir, same points) to place that value in FLIR's own pixel grid.
Points visible in both cameras give a direct sample; everything else is a
brute-force nearest-neighbor fill, and every output pixel is flagged
`sampled: true/false` so filled regions are never presented as measured.

Output per frame, under <out-dir>/<flir_stem>/:
    emissivity.npy    -- float32 HxW (FLIR grid), ready for
                          RadiometricCalibration/main.py --emissivity-map
    distance.npy       -- float32 HxW (FLIR grid), camera-frame depth in
                          metres from the same LiDAR samples, ready for
                          RadiometricCalibration/main.py --distance-map
    segment_id.npy     -- int32 HxW, which ZED superpixel each FLIR pixel came
                          from (-1 where unknown); lets correct_session.py
                          revisit the material choice per segment
    sampled_mask.npy   -- bool HxW, True where a LiDAR point actually landed
                          (not interpolated)
    stats.json         -- n_points_in_scan, n_direct_samples, coverage_pct

Usage:
    py project_to_flir.py --session-dir ...\\fullrate --bag ...\\rosbag2_2026_07_30-18_12_20 --limit 3 --overlay
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Calibration"))
from rig_calibration import load_rig_calibration
from projection import project_lidar_to_camera


def read_pointcloud2(msg) -> np.ndarray:
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def nearest_clouds_for_targets(bag: Path, target_epochs: list[float], topic: str, store: str) -> list[tuple[float, np.ndarray] | None]:
    """One pass over the bag; for each target epoch, return the (timestamp,
    points) of the /cloud_registered message closest to it."""
    n = len(target_epochs)
    best = [None] * n       # (dt, t, points) per still-open target
    finalized = [None] * n  # (t, points) once we've passed the minimum

    typestore = get_typestore(Stores[store])
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not found in bag. Available: {topics}")

        for connection, _bagts, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            pts = None
            for i in range(n):
                if finalized[i] is not None:
                    continue
                dt = abs(t - target_epochs[i])
                cur = best[i]
                if cur is None or dt < cur[0]:
                    if pts is None:
                        pts = read_pointcloud2(msg)
                    best[i] = (dt, t, pts)
                elif t > target_epochs[i]:
                    # timestamps are increasing; once we're past the target
                    # and the distance stopped improving, that minimum is final
                    finalized[i] = (cur[1], cur[2])

    for i in range(n):
        if finalized[i] is None and best[i] is not None:
            finalized[i] = (best[i][1], best[i][2])
    return finalized


def nearest_fill(values: np.ndarray, sampled: np.ndarray, chunk_rows: int = 32) -> np.ndarray:
    """Brute-force nearest-neighbor fill: every unsampled pixel gets the
    value of the closest sampled pixel (Euclidean, in pixel space). No
    scipy/cv2 special functions needed -- chunked to bound memory."""
    h, w = sampled.shape
    ys, xs = np.nonzero(sampled)
    if len(ys) == 0:
        return values.copy()
    sample_coords = np.stack([ys, xs], axis=1).astype(np.float32)  # (S, 2)
    sample_values = values[ys, xs]

    out = values.copy()
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        yy, xx = np.mgrid[r0:r1, 0:w]
        pix_coords = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float32)  # (P, 2)
        d2 = ((pix_coords[:, None, :] - sample_coords[None, :, :]) ** 2).sum(axis=2)  # (P, S)
        nearest = d2.argmin(axis=1)
        out[r0:r1, :] = sample_values[nearest].reshape(r1 - r0, w)
    out[sampled] = values[sampled]
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Project classify_session.py's ZED material map onto FLIR pixels via LiDAR")
    p.add_argument("--session-dir", required=True, metavar="DIR",
                    help="ZED session folder with sync_manifest.json + material_map/ (classify_session.py output)")
    p.add_argument("--bag", required=True, metavar="DIR", help="rosbag2 folder with /cloud_registered")
    p.add_argument("--calibration", default=None, metavar="FILE",
                    help="rig_calibration.yaml path (default: Calibration/rig_calibration.yaml)")
    p.add_argument("--material-map-dir", default=None, metavar="DIR",
                    help="classify_session.py output root (default: <session-dir>/material_map)")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                    help="Output root (default: <session-dir>/emissivity_map)")
    p.add_argument("--odom-topic", default="/cloud_registered", metavar="NAME")
    p.add_argument("--store", default="ROS2_HUMBLE", metavar="NAME")
    p.add_argument("--every-n", type=int, default=1, metavar="N")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--overlay", action="store_true", help="Save a QA PNG per frame")
    return p.parse_args()


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    material_map_dir = Path(args.material_map_dir) if args.material_map_dir else session_dir / "material_map"
    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "emissivity_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_path = args.calibration or (Path(__file__).resolve().parent.parent / "Calibration" / "rig_calibration.yaml")
    cal = load_rig_calibration(cal_path)

    manifest = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    # Only frames classify_session.py has already produced a material map for.
    work = []
    for tr in triplets:
        stem = Path(tr["flir"]["file"]).stem
        seg_path = material_map_dir / stem / "segments.json"
        lab_path = material_map_dir / stem / "labels.npy"
        if seg_path.exists() and lab_path.exists():
            work.append(tr)
        else:
            print(f"skip {stem}: no material_map output (run classify_session.py first)", file=sys.stderr)
    if not work:
        print("Nothing to do -- no triplet has a matching material_map/ output.", file=sys.stderr)
        return 1

    print(f"Fetching {len(work)} LiDAR scan(s) from {Path(args.bag).name} ...")
    target_epochs = [tr["lidar"]["timestamp_zedclock"] for tr in work]
    clouds = nearest_clouds_for_targets(Path(args.bag), target_epochs, args.odom_topic, args.store)

    fh, fw = cal.flir.height, cal.flir.width

    for tr, cloud in zip(work, clouds):
        stem = Path(tr["flir"]["file"]).stem
        if cloud is None:
            print(f"skip {stem}: no LiDAR scan found near t={tr['lidar']['timestamp_zedclock']}", file=sys.stderr)
            continue
        t_scan, points_world = cloud

        seg_data = json.loads((material_map_dir / stem / "segments.json").read_text(encoding="utf-8"))
        labels = np.load(material_map_dir / stem / "labels.npy")
        zh, zw = labels.shape
        emissivity_by_id = {seg["id"]: seg["emissivity"] for seg in seg_data["segments"]}

        lidar_pos = np.array(tr["lidar"]["position"])
        lidar_quat = np.array(tr["lidar"]["orientation"])
        zed_K = cal.zed_K_for(zw, zh)

        uv_zed, _depth_zed, valid_zed = project_lidar_to_camera(
            points_world, lidar_pos, lidar_quat, cal.T_lidar_to_zed,
            zed_K, cal.zed_calib.dist, zw, zh,
        )
        uv_flir, depth_flir, valid_flir = project_lidar_to_camera(
            points_world, lidar_pos, lidar_quat, cal.T_lidar_to_flir,
            cal.flir.K, cal.flir.dist, fw, fh,
        )

        valid = valid_zed & valid_flir
        n_in_scan = len(points_world)
        if not valid.any():
            print(f"skip {stem}: 0/{n_in_scan} points landed in both cameras", file=sys.stderr)
            continue

        zed_px = np.round(uv_zed[valid]).astype(int)
        zed_px[:, 0] = np.clip(zed_px[:, 0], 0, zw - 1)
        zed_px[:, 1] = np.clip(zed_px[:, 1], 0, zh - 1)
        seg_ids = labels[zed_px[:, 1], zed_px[:, 0]]
        point_emissivity = np.array([emissivity_by_id.get(int(s), np.nan) for s in seg_ids])
        point_depth = depth_flir[valid]

        flir_px = np.round(uv_flir[valid]).astype(int)
        flir_px[:, 0] = np.clip(flir_px[:, 0], 0, fw - 1)
        flir_px[:, 1] = np.clip(flir_px[:, 1], 0, fh - 1)

        keep = np.isfinite(point_emissivity)
        emissivity_sparse = np.full((fh, fw), np.nan, dtype=np.float32)
        distance_sparse = np.full((fh, fw), np.nan, dtype=np.float32)
        # Which ZED superpixel each FLIR pixel came from. Carried through so a
        # later step (correct_session.py) can revisit the material decision
        # per segment -- emissivity.npy alone is just floats, the link back to
        # segments.json's alternative candidates would otherwise be lost.
        segment_sparse = np.full((fh, fw), -1.0, dtype=np.float32)
        sampled_mask = np.zeros((fh, fw), dtype=bool)
        emissivity_sparse[flir_px[keep, 1], flir_px[keep, 0]] = point_emissivity[keep]
        distance_sparse[flir_px[keep, 1], flir_px[keep, 0]] = point_depth[keep]
        segment_sparse[flir_px[keep, 1], flir_px[keep, 0]] = seg_ids[keep].astype(np.float32)
        sampled_mask[flir_px[keep, 1], flir_px[keep, 0]] = True

        n_direct = int(sampled_mask.sum())
        if n_direct == 0:
            print(f"skip {stem}: 0 valid (material-labeled) samples land in FLIR frame", file=sys.stderr)
            continue

        emissivity_full = nearest_fill(np.nan_to_num(emissivity_sparse), sampled_mask)
        distance_full = nearest_fill(np.nan_to_num(distance_sparse), sampled_mask)
        # Nearest-fill on ids uses the same routine, then rounds back to int:
        # ids are only ever copied from a sampled pixel, never averaged.
        segment_full = np.rint(nearest_fill(segment_sparse, sampled_mask)).astype(np.int32)

        frame_dir = out_dir / stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "emissivity.npy", emissivity_full.astype(np.float32))
        np.save(frame_dir / "distance.npy", distance_full.astype(np.float32))
        np.save(frame_dir / "segment_id.npy", segment_full)
        np.save(frame_dir / "sampled_mask.npy", sampled_mask)
        (frame_dir / "stats.json").write_text(json.dumps({
            "schema": "flir_emissivity_map/v1",
            "generated_by": "project_to_flir.py",
            "source_flir_frame": tr["flir"]["file"],
            "source_zed_frame": tr["zed"]["file"],
            "lidar_scan_epoch": t_scan,
            "n_points_in_scan": n_in_scan,
            "n_points_in_both_cameras": int(valid.sum()),
            "n_direct_samples": n_direct,
            "coverage_pct": round(100.0 * n_direct / (fh * fw), 3),
        }, indent=2), encoding="utf-8")

        if args.overlay:
            norm = ((emissivity_full - emissivity_full.min()) /
                    max(1e-6, emissivity_full.max() - emissivity_full.min()) * 255).astype(np.uint8)
            heat = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
            heat[sampled_mask] = (0, 0, 255)  # mark direct samples in red (BGR)
            cv2.imwrite(str(frame_dir / "overlay.png"), heat)

        print(f"{stem}: {n_in_scan} scan pts -> {int(valid.sum())} in both cams -> "
              f"{n_direct} direct samples ({100.0 * n_direct / (fh * fw):.1f}% of FLIR frame)")

    print(f"Done. Output in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
