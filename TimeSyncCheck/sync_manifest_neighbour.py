"""Synchronize FLIR thermal, ZED RGB, and LiDAR pose streams into a triplet manifest.

Post-processing tool (no live capture). Two stages:

STAGE 1 — manual FLIR<->ZED event sync
    FLIR and ZED are not hardware-synced. Given a FLIR radiometric-JPEG sequence
    and a ZED session folder (from zed_record.py), a steppable viewer shows each
    stream so you pick the one frame in each where a shared heat-source event
    (lighter / heat gun / hot object) is visible. The FLIR<->ZED time offset is
    then the difference of the two selected frames' own timestamps, saved to
    <session>/flir_zed_offset.json for reuse.

    Because the offset is derived from a shared physical event, it absorbs any
    constant clock/timezone difference between the two cameras -- the FLIR
    timestamps only need to be internally consistent, not absolutely correct.

STAGE 2 — triplet manifest
    Using the Stage-1 FLIR<->ZED offset and the LiDAR<->ZED relationship
    (--lidar-zed-offset), each FLIR frame (the reference stream) is matched to
    the nearest ZED PNG and the nearest LiDAR /Odometry pose in a common clock
    (the ZED clock). Each triplet records the three paths/timestamps, the LiDAR
    pose, the pairwise time deltas after correction, and a match-confidence flag
    when any delta exceeds --max-delta. Output is JSON in the session folder for
    downstream tools (e.g. EmissivityCalculation) to consume.

This file only produces the manifest: no emissivity estimation, radiometric
temperature conversion, or point-cloud fusion/coloring happens here.

NOTE: the LiDAR<->ZED clock relationship is assumed to be a shared host clock
(offset 0) until verified on the rig -- see --lidar-zed-offset. This mirrors the
'timestamp + nearest-match on one common host clock' design note in
RadiometricCalibration/README.md (Synchronization). Override once measured.

Usage:
    py sync_manifest.py --session-dir recordings/20260726_140311 \
        --flir-dir "C:\\...\\FlyrCamera\\20250823_211855" --bag path/to/rosbag2
    py sync_manifest.py --session-dir <dir> --flir-dir <dir> --bag <dir> \
        --flir-event-frame 12 --zed-event-frame 3      # skip the Stage-1 viewer
    py sync_manifest.py --session-dir <dir> --flir-dir <dir> --bag <dir> \
        --recompute-offset --max-delta 0.05
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Timestamp extraction (each stream on its own clock; offsets bridge them)
# --------------------------------------------------------------------------- #

def list_flir_frames(folder: Path) -> list[Path]:
    """Radiometric *_R.jpg frames in a FLIR session folder, in capture order
    (same convention as RadiometricCalibration/ThermalData.py)."""
    frames = sorted(folder.glob("*_R.jpg"))
    if not frames:
        raise FileNotFoundError(f"No *_R.jpg radiometric files in {folder}")
    return frames


def flir_timestamp(path: Path) -> tuple[float, str]:
    """Epoch seconds (float) for one FLIR radiometric JPEG, plus the source used.

    Prefers EXIF DateTimeOriginal (+ SubSecTimeOriginal), then EXIF DateTime,
    then digits in the filename, then file mtime. Naive EXIF times are read as
    UTC to get a *consistent* epoch -- the Stage-1 event offset absorbs the real
    (constant) FLIR<->ZED clock/timezone difference, so only internal
    consistency across FLIR frames matters here.
    """
    from PIL import Image

    exif = Image.open(path).getexif()
    dto = subsec = None
    try:
        ifd = exif.get_ifd(0x8769)  # Exif sub-IFD
        dto = ifd.get(36867)        # DateTimeOriginal
        subsec = ifd.get(37521)     # SubSecTimeOriginal
    except Exception:
        pass
    dto = dto or exif.get(306)      # DateTime (fallback)

    if dto:
        dt = datetime.strptime(str(dto).strip(), "%Y:%m:%d %H:%M:%S")
        ts = dt.replace(tzinfo=timezone.utc).timestamp()
        if subsec:
            ts += float(f"0.{str(subsec).strip()}")
        return ts, "exif"

    digits = "".join(c for c in path.stem if c.isdigit())
    if len(digits) >= 14:  # e.g. YYYYMMDDHHMMSS[...]
        try:
            dt = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            return dt.replace(tzinfo=timezone.utc).timestamp(), "filename"
        except ValueError:
            pass

    return path.stat().st_mtime, "mtime"


def load_zed_frames(session_dir: Path) -> list[dict]:
    """ZED PNG frames + absolute (UTC epoch) timestamps from a zed_record.py
    session's metadata.json (started_utc + per-frame t_offset_s)."""
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json in {session_dir} -- expected a zed_record.py "
            "session folder (its frame manifest carries the ZED timestamps)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    started = meta.get("session", {}).get("started_utc")
    if not started:
        raise ValueError(f"metadata.json has no session.started_utc: {meta_path}")
    started_epoch = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()

    frames_dir = session_dir / (meta.get("recording", {}).get("frames_dir") or "frames")
    manifest = meta.get("frames", [])
    if not manifest:
        raise ValueError(
            f"metadata.json lists no frames (was zed_record.py run with "
            f"--no-frames?): {meta_path}"
        )
    out = []
    for fr in manifest:
        out.append({
            "path": frames_dir / fr["file"],
            "file": fr["file"],
            "epoch": started_epoch + float(fr["t_offset_s"]),
        })
    out.sort(key=lambda f: f["epoch"])
    return out


def load_lidar_poses(bag: Path, topic: str, store: str) -> list[dict]:
    """LiDAR poses from a rosbag2 /Odometry topic: epoch timestamp (header
    stamp), position (x,y,z), orientation quaternion (x,y,z,w). Uses the same
    rosbags AnyReader pattern as PointCloudView/view_pointcloud.py."""
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores[store])
    poses = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not found in bag. Available: {topics}")
        for connection, _bag_ts, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            stamp = msg.header.stamp
            t = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            poses.append({
                "epoch": t,
                "position": [float(p.x), float(p.y), float(p.z)],
                "orientation": [float(q.x), float(q.y), float(q.z), float(q.w)],
            })
    if not poses:
        raise SystemExit(f"No messages on {topic!r} in {bag}")
    poses.sort(key=lambda e: e["epoch"])
    return poses


# --------------------------------------------------------------------------- #
# Stage 1 -- manual event sync
# --------------------------------------------------------------------------- #

def pick_event_frame(n: int, title: str, render) -> int | None:
    """Steppable single-frame viewer. render(ax, i) draws frame i. Left/Right
    (or a/d) step, Enter/Space selects, Esc cancels. Returns the selected index
    or None."""
    import matplotlib.pyplot as plt

    state = {"i": 0, "sel": None}
    fig, ax = plt.subplots()

    def show():
        ax.clear()
        render(ax, state["i"])
        ax.set_title(
            f"{title}   frame {state['i'] + 1}/{n}\n"
            "<- / -> step   Enter=select   Esc=cancel"
        )
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["i"] = min(state["i"] + 1, n - 1); show()
        elif event.key in ("left", "a"):
            state["i"] = max(state["i"] - 1, 0); show()
        elif event.key in ("enter", "return", " "):
            state["sel"] = state["i"]; plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    return state["sel"]


def select_flir_event(frames: list[Path]) -> int | None:
    import flyr

    def render(ax, i):
        ax.imshow(flyr.unpack(str(frames[i])).celsius, cmap="inferno")
    return pick_event_frame(len(frames), "FLIR thermal -- find the heat event", render)


def select_zed_event(zed_frames: list[dict]) -> int | None:
    from PIL import Image

    def render(ax, i):
        ax.imshow(np.asarray(Image.open(zed_frames[i]["path"]).convert("RGB")))
    return pick_event_frame(len(zed_frames), "ZED RGB -- find the heat event", render)


def compute_offset(flir_frames, flir_ts, zed_frames, flir_idx, zed_idx) -> dict:
    """FLIR<->ZED offset doc: adding flir_zed_offset_s to a FLIR timestamp puts
    it on the ZED clock (offset = zed_event_epoch - flir_event_epoch)."""
    flir_epoch = flir_ts[flir_idx]
    zed_epoch = zed_frames[zed_idx]["epoch"]
    return {
        "schema": "flir_zed_offset/v1",
        "generated_by": "sync_manifest.py",
        "flir_zed_offset_s": round(zed_epoch - flir_epoch, 6),
        "flir_event_frame": {
            "index": flir_idx, "file": flir_frames[flir_idx].name,
            "timestamp_flir": flir_epoch,
        },
        "zed_event_frame": {
            "index": zed_idx, "file": zed_frames[zed_idx]["file"],
            "timestamp_zed": zed_epoch,
        },
        "calibrated_utc": utc_now_iso(),
    }


def resolve_offset(args, session_dir, flir_frames, flir_ts, zed_frames) -> dict:
    """Load an existing flir_zed_offset.json (asking whether to reuse) or run
    Stage 1 to compute and save a new one."""
    offset_path = session_dir / "flir_zed_offset.json"

    if offset_path.exists() and not args.recompute_offset:
        doc = json.loads(offset_path.read_text(encoding="utf-8"))
        print(
            f"Existing FLIR<->ZED offset in {offset_path.name}: "
            f"{doc['flir_zed_offset_s']:+.3f} s "
            f"(FLIR frame {doc['flir_event_frame']['file']} <-> "
            f"ZED {doc['zed_event_frame']['file']}, calibrated "
            f"{doc.get('calibrated_utc', '?')})."
        )
        reply = input("Reuse this offset? [Y/n] ").strip().lower()
        if reply in ("", "y", "yes"):
            return doc
        print("Recomputing offset ...")

    # Stage 1: get the two event-frame indices, from flags or the viewer.
    if args.flir_event_frame is not None and args.zed_event_frame is not None:
        flir_idx, zed_idx = args.flir_event_frame, args.zed_event_frame
        print(f"Using event frames from flags: FLIR[{flir_idx}], ZED[{zed_idx}].")
    else:
        print("Stage 1: select the heat-source event frame in each viewer.")
        flir_idx = select_flir_event(flir_frames)
        if flir_idx is None:
            sys.exit("FLIR event selection cancelled.")
        zed_idx = select_zed_event(zed_frames)
        if zed_idx is None:
            sys.exit("ZED event selection cancelled.")

    if not (0 <= flir_idx < len(flir_frames)):
        sys.exit(f"FLIR event frame {flir_idx} out of range (0..{len(flir_frames) - 1}).")
    if not (0 <= zed_idx < len(zed_frames)):
        sys.exit(f"ZED event frame {zed_idx} out of range (0..{len(zed_frames) - 1}).")

    doc = compute_offset(flir_frames, flir_ts, zed_frames, flir_idx, zed_idx)
    offset_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"FLIR<->ZED offset {doc['flir_zed_offset_s']:+.3f} s -> {offset_path}")
    return doc


# --------------------------------------------------------------------------- #
# Stage 2 -- triplet matching
# --------------------------------------------------------------------------- #

def nearest_index(sorted_epochs: np.ndarray, t: float) -> int:
    """Index of the entry in `sorted_epochs` closest to t."""
    i = int(np.searchsorted(sorted_epochs, t))
    cands = [j for j in (i - 1, i) if 0 <= j < len(sorted_epochs)]
    return min(cands, key=lambda j: abs(sorted_epochs[j] - t))


def build_triplets(flir_frames, flir_ts, flir_src, zed_frames, poses,
                   flir_zed_offset, lidar_zed_offset, max_delta) -> tuple[list, dict]:
    """One triplet per FLIR frame (reference). Everything is compared on the ZED
    clock: flir_zed = flir + flir_zed_offset; lidar_zed = lidar + lidar_zed_offset."""
    zed_epochs = np.array([f["epoch"] for f in zed_frames])
    lidar_zed_epochs = np.array([p["epoch"] + lidar_zed_offset for p in poses])

    triplets = []
    n_low = 0
    for i, path in enumerate(flir_frames):
        flir_zed = flir_ts[i] + flir_zed_offset

        zi = nearest_index(zed_epochs, flir_zed)
        li = nearest_index(lidar_zed_epochs, flir_zed)
        zed = zed_frames[zi]
        pose = poses[li]
        lidar_zed = lidar_zed_epochs[li]

        d_flir_zed = flir_zed - zed["epoch"]
        d_flir_lidar = flir_zed - lidar_zed
        d_zed_lidar = zed["epoch"] - lidar_zed

        exceeds = [
            name for name, d in (
                ("flir_zed", d_flir_zed),
                ("flir_lidar", d_flir_lidar),
                ("zed_lidar", d_zed_lidar),
            ) if abs(d) > max_delta
        ]
        status = "low-confidence" if exceeds else "matched"
        if exceeds:
            n_low += 1

        triplets.append({
            "flir": {
                "file": path.name,
                "timestamp_flir": flir_ts[i],
                "timestamp_zedclock": round(flir_zed, 6),
                "timestamp_source": flir_src[i],
            },
            "zed": {
                "file": zed["file"],
                "timestamp_zed": round(zed["epoch"], 6),
            },
            "lidar": {
                "timestamp_lidar": round(pose["epoch"], 6),
                "timestamp_zedclock": round(float(lidar_zed), 6),
                "position": pose["position"],
                "orientation": pose["orientation"],
            },
            "deltas_s": {
                "flir_zed": round(float(d_flir_zed), 6),
                "flir_lidar": round(float(d_flir_lidar), 6),
                "zed_lidar": round(float(d_zed_lidar), 6),
            },
            "match_status": status,
            "exceeds_max_delta": exceeds,
        })

    summary = {"n_triplets": len(triplets), "n_matched": len(triplets) - n_low,
               "n_low_confidence": n_low}
    return triplets, summary


# --------------------------------------------------------------------------- #

def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp, seconds resolution (matches zed_record.py)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def parse_args():
    p = argparse.ArgumentParser(
        description="Synchronize FLIR/ZED/LiDAR streams into a triplet manifest")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder from zed_record.py (holds metadata.json + "
        "frames/). The offset config and manifest are written here.",
    )
    p.add_argument(
        "--flir-dir", required=True, metavar="DIR",
        help="Folder of FLIR radiometric *_R.jpg frames (as read by "
        "RadiometricCalibration/ThermalData.py).",
    )
    p.add_argument(
        "--bag", required=True, metavar="DIR",
        help="rosbag2 folder (metadata.yaml + .db3/.mcap) with the LiDAR "
        "odometry topic.",
    )
    p.add_argument(
        "--odom-topic", default="/Odometry", metavar="NAME",
        help="Odometry topic providing LiDAR pose (default /Odometry).",
    )
    p.add_argument(
        "--store", default="ROS2_HUMBLE", metavar="NAME",
        help="rosbags typestore for bags without embedded type defs "
        "(default ROS2_HUMBLE; matches view_pointcloud.py).",
    )
    p.add_argument(
        "--max-delta", type=float, default=0.1, metavar="SEC",
        help="Max allowed time delta (s) between any two streams in a triplet "
        "after correction; beyond it the triplet is flagged 'low-confidence' "
        "rather than silently paired (default 0.1).",
    )
    p.add_argument(
        "--lidar-zed-offset", type=float, default=0.0, metavar="SEC",
        help="Seconds added to a LiDAR timestamp to put it on the ZED clock. "
        "Default 0.0 assumes a shared host clock -- UNVERIFIED; measure on the "
        "rig and set this once known (see module docstring).",
    )
    p.add_argument(
        "--recompute-offset", action="store_true",
        help="Force Stage 1 (re-select event frames) even if "
        "flir_zed_offset.json already exists, instead of asking.",
    )
    p.add_argument(
        "--flir-event-frame", type=int, default=None, metavar="N",
        help="Stage 1 without the viewer: FLIR event-frame index (use with "
        "--zed-event-frame).",
    )
    p.add_argument(
        "--zed-event-frame", type=int, default=None, metavar="N",
        help="Stage 1 without the viewer: ZED event-frame index (use with "
        "--flir-event-frame).",
    )
    p.add_argument(
        "--output-format", choices=("json",), default="json",
        help="Manifest format (default json). CSV export is a TODO.",
    )
    p.add_argument(
        "--output", default="sync_manifest.json", metavar="NAME",
        help="Manifest filename, written inside --session-dir "
        "(default sync_manifest.json).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    session_dir = Path(args.session_dir)
    flir_dir = Path(args.flir_dir)
    bag = Path(args.bag)
    for label, path in (("--session-dir", session_dir), ("--flir-dir", flir_dir),
                        ("--bag", bag)):
        if not path.exists():
            print(f"{label} not found: {path}", file=sys.stderr)
            return 1

    # --- load the three streams (each on its own clock) ---------------------
    try:
        flir_frames = list_flir_frames(flir_dir)
        zed_frames = load_zed_frames(session_dir)
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        import flyr  # noqa: F401  (fail early with guidance if missing)
    except ImportError:
        print("flyr not installed (reads FLIR radiometric JPEGs). "
              "pip install flyr", file=sys.stderr)
        return 1

    ts_src = [flir_timestamp(f) for f in flir_frames]
    flir_ts = [t for t, _ in ts_src]
    flir_src = [s for _, s in ts_src]
    print(f"FLIR: {len(flir_frames)} frame(s) ({flir_src[0]} timestamps)  |  "
          f"ZED: {len(zed_frames)} frame(s)")

    # --- Stage 1: FLIR<->ZED offset -----------------------------------------
    offset_doc = resolve_offset(args, session_dir, flir_frames, flir_ts, zed_frames)
    flir_zed_offset = float(offset_doc["flir_zed_offset_s"])

    # --- LiDAR poses --------------------------------------------------------
    try:
        poses = load_lidar_poses(bag, args.odom_topic, args.store)
    except ImportError:
        print("rosbags not installed (reads the LiDAR bag). "
              "pip install rosbags", file=sys.stderr)
        return 1
    print(f"LiDAR: {len(poses)} pose(s) on {args.odom_topic}")

    # --- Stage 2: triplets --------------------------------------------------
    triplets, summary = build_triplets(
        flir_frames, flir_ts, flir_src, zed_frames, poses,
        flir_zed_offset, args.lidar_zed_offset, args.max_delta,
    )

    manifest = {
        "schema": "sync_manifest/v1",
        "generated_by": "sync_manifest.py",
        "generated_utc": utc_now_iso(),
        "inputs": {
            "session_dir": str(session_dir),
            "flir_dir": str(flir_dir),
            "bag": str(bag),
            "odom_topic": args.odom_topic,
        },
        "sync": {
            "flir_zed_offset_s": flir_zed_offset,
            "flir_zed_offset_source": "flir_zed_offset.json",
            "flir_event_frame": offset_doc["flir_event_frame"],
            "zed_event_frame": offset_doc["zed_event_frame"],
            "lidar_zed_offset_s": args.lidar_zed_offset,
            "lidar_zed_offset_status": (
                "UNVERIFIED shared-clock assumption (offset applied as given; "
                "default 0) -- verify on the rig"
            ),
        },
        "matching": {
            "reference": "flir",
            "max_delta_s": args.max_delta,
            **summary,
        },
        "triplets": triplets,
    }

    out_path = session_dir / args.output
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Wrote {out_path}\n"
        f"  {summary['n_triplets']} triplet(s): {summary['n_matched']} matched, "
        f"{summary['n_low_confidence']} low-confidence (> {args.max_delta}s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
