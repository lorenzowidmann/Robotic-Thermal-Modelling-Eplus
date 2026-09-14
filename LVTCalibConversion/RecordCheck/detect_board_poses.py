"""
Automatically detect the board poses in a capture session by counting the
"stable" segments between one board move and the next.

Idea: during a pose the board is still -> the difference between consecutive
frames is low. When the board is moved -> a peak of difference appears. The
peaks separate the poses; the stable segments between two peaks are the real
poses.

Three input types are supported:

  * FLIR thermal RJPG:  YYYYMMDD_HHMMSS_R.jpg
      The timestamp lives in the filename (which is also temporal order).

  * ZED right-eye PNG:  right_NNNNNN.png (or any <prefix>_NNNNNN.png)
      No timestamp in the name; frames are ordered by their numeric index.
      Per-frame times are read from a sibling metadata.json
      (schema "zed_record/v1": recording.frame_interval_s, frames[].t_offset_s,
      session.started_utc). If metadata.json is missing, times fall back to
      index / --fps.

  * LiDAR rosbag2 (--db3 or --bag):  a .db3 file or a rosbag2 folder.
      Reads livox_ros_driver2/msg/CustomMsg (or sensor_msgs/msg/PointCloud2)
      scans. "Difference" between consecutive scans = voxel-occupancy Jaccard
      distance: low while the scene is static (a held board), a peak when the
      board is moved.
      --db3: pure stdlib (sqlite3 + struct), no extra package, but loads the
        whole message index in RAM -- fine for normal bags, a single .db3 file.
      --bag: streams via the `rosbags` package, one message at a time, never
        the whole bag in RAM -- for tens-of-GB bags; also accepts a folder
        with metadata.yaml (falls back to a lone .db3 inside if missing).

For images, accepts multiple folders: a long session may be split into separate
folders, which are treated as a single stream.

Extra flags folded in from the other RecordCheck pose-detection scripts (kept
here so there is a single tool; see each flag's --help for details):
  --eye right|left        ZED dual-eye session (zed_record_dual.py): pick one
                           eye out of a frames/ folder that mixes both.
  --gap-threshold N        FLIR alternate segmentation: split poses by time
                           gap between filenames instead of frame-diff (for
                           stop-start recordings, or when NUC/AGC noise makes
                           frame-diff unreliable).
  --check-inside           FLIR only: per pose, detect the four-hole board and
                           verdict whether it stays fully inside the frame
                           (not clipped by the image borders).
  --csv-out FILE           Save the detected-poses table (or the --check-inside
                           table) as CSV.

Usage:
    py detect_board_poses.py <folder1> [<folder2> ...]
    py detect_board_poses.py <folder1> --threshold 8.0 --min-pose-frames 20
    py detect_board_poses.py <zed_session>/frames --eye right
    py detect_board_poses.py <flir_dir> --gap-threshold 5 --check-inside --csv-out flir_check.csv
    py detect_board_poses.py --db3 <rosbag2_dir_or_file>
    py detect_board_poses.py --db3 <bag> --voxel 0.10 --db3-msg-stride 10
    py detect_board_poses.py --bag <rosbag2_dir_or_file> --db3-msg-stride 5 --mad-k 2.5

Output: list of the detected poses with index, start/end time, duration in
seconds and number of frames (LiDAR: sampled scans). Also useful to build the
-s/-u windows for `rosbag play`.

NOTE on thermal cameras: the RJPG preview uses an auto-scaled palette (AGC) and
performs a periodic flat-field correction (NUC). Both cause a full-frame
brightness change even when the board is perfectly still, producing spurious
"movement" spikes. To avoid over-counting the poses, only a RUN of at least
--min-move-frames consecutive above-threshold frames is treated as a real board
move; isolated single-frame spikes are ignored. If the FLIR recorded without
any stop, there are no such spikes to detect at all: use --gap-threshold instead.
"""

import argparse
import csv
import json
import re
import sqlite3
import struct
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python required:  py -m pip install opencv-python")


IMG_EXTS = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


def parse_flir_timestamp(name: str):
    """Extract the datetime from a FLIR filename: YYYYMMDD_HHMMSS_R.jpg"""
    stem = name.split("_R")[0]
    try:
        return datetime.strptime(stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def extract_index(name: str):
    """Extract the trailing frame index from names like right_000123.png -> 123."""
    m = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(m.group(1)) if m else None


def collect_images(dirs):
    files = []
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            sys.exit(f"Folder not found: {p}")
        found = [f for f in p.iterdir() if f.suffix in IMG_EXTS]
        if not found:
            print(f"WARNING: no image found in {p}")
        files.extend(found)

    # Sort: prefer FLIR timestamp, then numeric frame index, then plain name.
    # The category tag keeps the tuple comparisons type-safe.
    def sort_key(f):
        ts = parse_flir_timestamp(f.name)
        if ts:
            return (0, ts.timestamp(), f.name)
        idx = extract_index(f.name)
        if idx is not None:
            return (1, idx, f.name)
        return (2, 0, f.name)

    files.sort(key=sort_key)
    return files


def load_metadata_times(dirs):
    """Look for a zed_record metadata.json in each dir or its parent.

    Returns (offset_by_name, abs_start, path) or None.
      offset_by_name: {filename -> t_offset_s (float)}
      abs_start: datetime of session start (tz-aware) or None
    """
    seen = set()
    for d in dirs:
        for cand in (Path(d) / "metadata.json", Path(d).parent / "metadata.json"):
            if cand in seen or not cand.is_file():
                continue
            seen.add(cand)
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            frames = data.get("frames")
            if not frames:
                continue
            offset_by_name = {
                fr["file"]: float(fr["t_offset_s"])
                for fr in frames
                if "file" in fr and "t_offset_s" in fr
            }
            if not offset_by_name:
                continue
            abs_start = None
            started = (data.get("session") or {}).get("started_utc")
            if started:
                try:
                    abs_start = datetime.fromisoformat(started.replace("Z", "+00:00"))
                except ValueError:
                    abs_start = None
            return offset_by_name, abs_start, cand
    return None


def build_time_index(files, dirs, fps):
    """Build {filename -> offset_seconds} and the absolute session start.

    Priority: FLIR filename timestamp -> metadata.json -> index / fps.
    Returns (offset_by_name, abs_start, source_str).
    """
    # FLIR: timestamps embedded in the filename take priority. metadata.json is
    # a ZED-only sidecar and must never be applied to a FLIR session.
    if parse_flir_timestamp(files[0].name) is not None:
        t0 = parse_flir_timestamp(files[0].name)
        offsets = {}
        for f in files:
            ts = parse_flir_timestamp(f.name)
            offsets[f.name] = (ts - t0).total_seconds() if ts else 0.0
        return offsets, t0, "FLIR filename timestamps"

    meta = load_metadata_times(dirs)
    if meta is not None:
        offset_by_name, abs_start, path = meta
        # Only keep entries for files we actually loaded; warn on gaps.
        missing = [f.name for f in files if f.name not in offset_by_name]
        if missing:
            print(f"WARNING: {len(missing)} frame(s) not listed in metadata.json "
                  f"(e.g. {missing[0]}); those fall back to index/fps.")
        base = min(offset_by_name.values()) if offset_by_name else 0.0
        offsets = {}
        for i, f in enumerate(files):
            if f.name in offset_by_name:
                offsets[f.name] = offset_by_name[f.name] - base
            else:
                offsets[f.name] = i / fps
        return offsets, abs_start, f"metadata.json ({path})"

    # Fallback: assume constant frame rate
    offsets = {f.name: i / fps for i, f in enumerate(files)}
    return offsets, None, f"constant {fps} fps (no metadata)"


_ZED_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def _zed_frame_index(path):
    m = _ZED_INDEX_RE.search(path.name)
    return int(m.group(1)) if m else -1


def load_session(session, eye, fps):
    """ZED-only loader (from zed_pose_detect.py): one eye of a zed_record.py /
    zed_record_dual.py session as an ordered list of (t_offset_s, Path).

    Needed because a dual-eye session mixes right_*/left_* in the same
    frames/ folder, which collect_images()/build_time_index() do not
    disambiguate. Returns (entries, started_utc, time_source_str)."""
    session = Path(session)
    frames_dir = session / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"Cartella frames/ non trovata in {session}")
    on_disk = sorted(frames_dir.glob(f"{eye}_*.png"), key=_zed_frame_index)

    meta_path = session / "metadata.json"
    started, entries = None, []
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"ATTENZIONE: metadata.json illeggibile ({exc})")
            data = {}
        sess = data.get("session") or {}
        started = sess.get("started_utc")
        if data and sess.get("stopped_utc") is None:
            print("ATTENZIONE: metadata.json senza stopped_utc: la registrazione "
                  "non si e' chiusa correttamente.")
        for fr in data.get("frames") or []:
            name = fr.get(eye) or fr.get("file")
            if name and name.startswith(f"{eye}_") and "t_offset_s" in fr:
                entries.append((float(fr["t_offset_s"]), frames_dir / name))
    else:
        print("ATTENZIONE: metadata.json assente.")

    if entries:
        source = "metadata.json (t_offset_s)"
        if len(on_disk) != len(entries):
            print(f"ATTENZIONE: {len(entries)} frame '{eye}' nel manifest, "
                  f"{len(on_disk)} su disco.")
    else:
        entries = [(i / fps, f) for i, f in enumerate(on_disk)]
        source = f"indice / {fps:g} fps (nessun manifest frame in metadata.json)"
    entries.sort(key=lambda e: e[0])
    return entries, started, source


def diff_profile(entries, stride, downscale):
    """ZED-only frame-diff profile over a load_session() eye (from
    zed_pose_detect.py). Returns (values, idx): values[k] compares the
    previous sampled frame with entries[idx[k]]."""
    values, idx = [], []
    prev = None
    skipped = 0
    sampled = range(0, len(entries), stride)
    for n, i in enumerate(sampled, start=1):
        img = cv2.imread(str(entries[i][1]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        if downscale > 1:
            img = cv2.resize(img, (img.shape[1] // downscale,
                                   img.shape[0] // downscale))
        if prev is not None:
            values.append(float(np.mean(cv2.absdiff(prev, img))))
            idx.append(i)
        prev = img
        if n % 200 == 0:
            print(f"\r  {n}/{len(sampled)} frame letti", end="", flush=True)
    print()
    if skipped:
        print(f"ATTENZIONE: {skipped}/{len(sampled)} frame illeggibili o mancanti, saltati.")
    return np.array(values), idx


def frame_difference(prev_gray, cur_gray):
    """Mean absolute difference between two grayscale frames."""
    diff = cv2.absdiff(prev_gray, cur_gray)
    return float(np.mean(diff))


def fmt_clock(abs_start, offset):
    """HH:MM:SS at abs_start+offset, or the raw offset if no absolute start."""
    if abs_start is not None:
        return (abs_start + timedelta(seconds=offset)).strftime("%H:%M:%S")
    return f"{offset:.1f}s"


# ----------------------------------------------------------------------------
# LiDAR rosbag2 (.db3) support
# ----------------------------------------------------------------------------
_CUSTOM_MSG_TYPE = "livox_ros_driver2/msg/CustomMsg"
_POINT_CLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
_CUSTOM_POINT_STRIDE = 20  # CustomPoint: u32 offset_time, f32 x/y/z, 3x u8 -> 20


def _resolve_db3(path):
    """Accept a .db3 file or a rosbag2 folder; return the .db3 Path."""
    p = Path(path)
    if p.is_dir():
        cands = sorted(p.glob("*.db3"))
        if not cands:
            sys.exit(f"No .db3 file found in {p}")
        return cands[0]
    if not p.is_file():
        sys.exit(f"Rosbag not found: {p}")
    return p


def _custom_msg_xyz(buf, point_stride):
    """Parse livox CustomMsg CDR bytes -> (N,3) float32 xyz."""
    pos = 4                                   # skip encapsulation header
    pos += 4 + 4                              # header.stamp sec + nanosec
    slen = struct.unpack_from("<I", buf, pos)[0]; pos += 4 + slen  # frame_id
    rem = (pos - 4) % 8                       # align u64 (timebase)
    if rem:
        pos += 8 - rem
    pos += 8                                  # timebase
    point_num = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    pos += 1 + 3                              # lidar_id + rsvd[3]
    seq_len = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    n = max(point_num, seq_len)
    data = buf[pos:]
    need = n * _CUSTOM_POINT_STRIDE
    if len(data) < need:                      # CDR pads no trailing element
        data = data + b"\x00" * (need - len(data))
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"],
                   "offsets": [4, 8, 12], "itemsize": _CUSTOM_POINT_STRIDE})
    arr = np.frombuffer(data, dtype=dt, count=n)[::point_stride]
    p = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return p[np.isfinite(p).all(axis=1)]


def _pointcloud2_xyz(buf, point_stride):
    """Parse sensor_msgs/PointCloud2 CDR bytes -> (N,3) float32 xyz (f32 x/y/z)."""
    pos = 4
    pos += 4 + 4
    slen = struct.unpack_from("<I", buf, pos)[0]; pos += 4 + slen  # frame_id
    height = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    width = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    n_fields = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    fields = {}
    for _ in range(n_fields):
        fl = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        name = buf[pos:pos + fl][:-1].decode(); pos += fl
        offset = struct.unpack_from("<I", buf, pos)[0]; pos += 4
        datatype = buf[pos]; pos += 1
        pos += 4                              # count
        fields[name] = (offset, datatype)
    pos += 1                                  # is_bigendian
    point_step = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    pos += 4                                  # row_step
    dlen = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    data = buf[pos:pos + dlen]
    n = width * height
    ox, oy, oz = fields["x"][0], fields["y"][0], fields["z"][0]
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"],
                   "offsets": [ox, oy, oz], "itemsize": point_step})
    arr = np.frombuffer(data, dtype=dt, count=n)[::point_stride]
    p = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return p[np.isfinite(p).all(axis=1)]


def _voxel_keys(points, voxel):
    """Set of occupied-voxel keys (int64) for a cloud."""
    q = np.floor(points / voxel).astype(np.int64) + (1 << 19)  # keep positive
    keys = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    return set(np.unique(keys).tolist())


def lidar_profile(db3, topic, msg_stride, point_stride, voxel):
    """Build the scan-to-scan difference profile of a rosbag2 LiDAR recording.

    Returns (ts, values, abs_start, source, n_total, eff_rate).
      ts     : offset seconds of each sampled scan (len = n_samples)
      values : voxel-occupancy Jaccard distance vs the previous sample
               (len = n_samples - 1)
    """
    conn = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT id, name, type FROM topics WHERE type IN (?, ?)",
            (_CUSTOM_MSG_TYPE, _POINT_CLOUD2_TYPE),
        ).fetchall()
        if topic is not None:
            rows = [r for r in rows if r[1] == topic]
        if not rows:
            sys.exit(f"No supported LiDAR topic ({topic or 'CustomMsg/PointCloud2'}) "
                     f"in {db3}")
        if len(rows) > 1:
            names = ", ".join(f"{r[1]} [{r[2]}]" for r in rows)
            sys.exit(f"Multiple LiDAR topics: {names}. Pass --db3-topic to pick one.")
        topic_id, topic_name, msg_type = rows[0]

        idx = cur.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id=? ORDER BY id",
            (topic_id,),
        ).fetchall()
        if len(idx) < 2:
            sys.exit("Topic has fewer than 2 scans.")
        sel = idx[::msg_stride]
        t0_ns = idx[0][1]
        abs_start = datetime.fromtimestamp(t0_ns / 1e9, tz=timezone.utc)
        parse = _custom_msg_xyz if msg_type == _CUSTOM_MSG_TYPE else _pointcloud2_xyz

        print(f"LiDAR topic: {topic_name} [{msg_type}]")
        print(f"Scans: {len(idx)}  sampled: {len(sel)} (every {msg_stride})")

        ts, values = [], []
        prev = None
        for k, (mid, ns) in enumerate(sel):
            blob = cur.execute("SELECT data FROM messages WHERE id=?", (mid,)).fetchone()[0]
            keys = _voxel_keys(parse(bytes(blob), point_stride), voxel)
            ts.append((ns - t0_ns) / 1e9)
            if prev is not None:
                uni = len(prev | keys)
                values.append(1.0 - len(prev & keys) / uni if uni else 0.0)
            prev = keys
            if (k + 1) % 100 == 0:
                print(f"  ...{k + 1}/{len(sel)}")
    finally:
        conn.close()

    ts = np.array(ts)
    dur = ts[-1] - ts[0]
    eff_rate = (len(ts) - 1) / dur if dur > 0 else 0.0
    src = f"rosbag2 {Path(db3).name} ({topic_name})"
    return ts, np.array(values), abs_start, src, len(idx), eff_rate


def _require_reader():
    """Lazy import of rosbags.rosbag2.Reader (from lidar_pose_detect.py).

    Kept lazy so plain FLIR/ZED/--db3 usage of this module never needs the
    `rosbags` package -- only --bag and lidar_hole_check.py do."""
    try:
        from rosbags.rosbag2 import Reader
    except ImportError:
        sys.exit("Serve il pacchetto rosbags:  py -m pip install rosbags")
    return Reader


def resolve_bag(path):
    """Accept a rosbag2 folder (with metadata.yaml) or a .db3 file; if
    metadata.yaml is missing (recorder closed uncleanly) and the folder has
    exactly one .db3, use that. For the `rosbags`-streaming path (--bag),
    as opposed to _resolve_db3() used by the sqlite3 --db3 path."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"Bag non trovato: {p}")
    if p.is_dir() and not (p / "metadata.yaml").exists():
        db3 = sorted(p.glob("*.db3"))
        if len(db3) != 1:
            sys.exit(f"metadata.yaml assente in {p} e {len(db3)} file .db3: "
                     "passare direttamente il file .db3 con --bag.")
        print(f"ATTENZIONE: metadata.yaml assente in {p}, leggo direttamente {db3[0].name}")
        return db3[0]
    return p


def lidar_profile_stream(bag, topic, msg_stride, point_stride, voxel, time_origin):
    """Streaming counterpart of lidar_profile(), via `rosbags` (from
    lidar_pose_detect.py): one message at a time, never the whole bag in RAM
    -- fine for tens-of-GB bags, and works on a rosbag2 folder as-is (no
    up-front .db3 pick).

    Returns (value_times, values, all_offsets, info): value_times[k] is the
    offset (s) of the scan compared by values[k]; all_offsets are the
    offsets of ALL messages on the topic."""
    Reader = _require_reader()
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            available = ", ".join(sorted({f"{c.topic} [{c.msgtype}]"
                                          for c in reader.connections}))
            sys.exit(f"Topic {topic} non presente nel bag. Disponibili: {available}")
        msgtype = conns[0].msgtype
        if msgtype == _CUSTOM_MSG_TYPE:
            parse = _custom_msg_xyz
        elif msgtype == _POINT_CLOUD2_TYPE:
            parse = _pointcloud2_xyz
        else:
            sys.exit(f"Tipo {msgtype} non supportato (serve CustomMsg o PointCloud2).")
        declared = sum(c.msgcount for c in conns)
        bag_start = reader.start_time
        print(f"Topic: {topic} [{msgtype}]   messaggi dichiarati: {declared}   "
              f"analizzati: 1 ogni {msg_stride}")

        all_ns, sample_ns, values = [], [], []
        prev = None
        bad = 0
        for i, (_, t_ns, raw) in enumerate(reader.messages(connections=conns)):
            all_ns.append(t_ns)
            if i % msg_stride:
                continue
            try:
                keys = _voxel_keys(parse(bytes(raw), point_stride), voxel)
            except (struct.error, ValueError, KeyError):
                bad += 1
                continue
            if prev is not None:
                uni = len(prev | keys)
                values.append(1.0 - len(prev & keys) / uni if uni else 0.0)
            sample_ns.append(t_ns)
            prev = keys
            if len(sample_ns) % 100 == 0:
                print(f"\r  {i + 1}/{declared} messaggi letti", end="", flush=True)
        print()

    all_ns = np.sort(np.asarray(all_ns, dtype=np.int64))
    if all_ns.size == 0:
        sys.exit(f"Nessun messaggio su {topic}.")
    t0 = bag_start if time_origin == "bag" else int(all_ns[0])
    info = {
        "declared": declared,
        "read": int(all_ns.size),
        "bad": bad,
        "sampled": len(sample_ns),
        "topic_start_utc": datetime.fromtimestamp(all_ns[0] / 1e9, tz=timezone.utc),
        "topic_start_offset": (int(all_ns[0]) - bag_start) / 1e9,
    }
    value_times = (np.asarray(sample_ns[1:], dtype=np.int64) - t0) / 1e9
    return value_times, np.asarray(values), (all_ns - t0) / 1e9, info


def segment(values, threshold, min_move_frames, min_pose_frames):
    """Debounced segmentation shared by the image and LiDAR paths.

    Returns (poses, transitions): index ranges into `values`.
    """
    moving = values > threshold
    n = len(moving)
    transitions = []
    i = 0
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if j - i >= min_move_frames:
                transitions.append((i, j))
            i = j
        else:
            i += 1
    poses = []
    prev_end = 0
    for a, b in transitions:
        if a - prev_end >= min_pose_frames:
            poses.append((prev_end, a))
        prev_end = b
    if n - prev_end >= min_pose_frames:
        poses.append((prev_end, n))
    return poses, transitions


def _pose_intervals(value_times, values, args, min_move, min_pose, mad_k):
    """Segment + duration-filter -> list of (start_s, end_s) pose intervals."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    thr = (args.threshold if args.threshold is not None
           else median + mad_k * (mad if mad > 1e-6 else 1.0))
    poses, _ = segment(values, thr, min_move, min_pose)

    def span(a, b):
        return value_times[a], value_times[min(b, len(value_times)) - 1]

    intervals = [span(a, b) for a, b in poses]
    if args.min_duration is not None:
        intervals = [(s, e) for (s, e) in intervals if e - s >= args.min_duration]
    return intervals


def sensor_pose_intervals(kind, args):
    """Detect poses for one sensor. Returns (name, intervals) where intervals is
    a list of (start_s, end_s) relative to that sensor's own recording start."""
    if kind == "lidar":
        db3 = _resolve_db3(args.db3)
        print(f"[LiDAR] {db3.name}")
        vt_all, values, _abs, _src, _n, eff = lidar_profile(
            db3, args.db3_topic, args.db3_msg_stride, args.db3_point_stride, args.voxel)
        vt = vt_all[1:]
        min_move = args.min_move_frames if args.min_move_frames is not None else 1
        min_pose = (args.min_pose_frames if args.min_pose_frames is not None
                    else max(4, round(10.0 * eff)))
        mad_k = args.mad_k if args.mad_k is not None else 4.0
        name = "LiDAR"
    else:
        dirs = args.flir if kind == "flir" else args.zed
        name = "FLIR" if kind == "flir" else "ZED"
        print(f"[{name}] {', '.join(dirs)}")
        files = collect_images(dirs)
        if len(files) < 2:
            sys.exit(f"{name}: at least 2 images required.")
        vt, values, _abs, _src, _labels = image_profile(files, dirs, args)
        min_move = args.min_move_frames if args.min_move_frames is not None else 3
        min_pose = args.min_pose_frames if args.min_pose_frames is not None else 15
        mad_k = args.mad_k if args.mad_k is not None else 6.0

    intervals = _pose_intervals(vt, values, args, min_move, min_pose, mad_k)
    return name, intervals


def compare_sensors(args):
    """Detect poses on each provided sensor and compare inter-pose gaps."""
    sensors = []
    if args.flir:
        sensors.append(sensor_pose_intervals("flir", args))
    if args.zed:
        sensors.append(sensor_pose_intervals("zed", args))
    if args.db3:
        sensors.append(sensor_pose_intervals("lidar", args))
    if len(sensors) < 2:
        sys.exit("--compare needs at least two of --flir / --zed / --db3.")

    print()
    print("=" * 78)
    print("POSES DETECTED PER SENSOR")
    for name, iv in sensors:
        print(f"  {name:<6} {len(iv)} poses")
    counts = {len(iv) for _, iv in sensors}
    n = min(len(iv) for _, iv in sensors)
    if len(counts) > 1:
        print(f"  WARNING: pose counts differ -> aligning the first {n} by order; "
              "a missed pose in one sensor will misalign the rest.")
    print("=" * 78)

    if n < 2:
        sys.exit("Need at least 2 poses per sensor to have an inter-pose gap.")

    names = [name for name, _ in sensors]
    print("Pose duration = seconds each pose lasts (END - START)")
    print("(relative within each sensor, so different clocks do not matter).")
    print()
    header = f"{'pose':>4}  " + "  ".join(f"{nm:>8}" for nm in names) + "   | maxdiff"
    print(header)
    print("-" * len(header))
    for i in range(n):
        durs = [iv[i][1] - iv[i][0] for _, iv in sensors]
        maxdiff = max(durs) - min(durs)
        cells = "  ".join(f"{d:7.1f}s" for d in durs)
        print(f"{i + 1:>4}  {cells}   | {maxdiff:6.1f}s")
    print()
    print("NOTE: large maxdiff on a row = the sensors disagree on that pose's "
          "length, usually a missed/extra/merged pose in one stream. Tune "
          "per-sensor detection with --threshold / --mad-k / --min-duration.")


def split_on_gaps(files, gap_threshold):
    """FLIR-only alternate segmentation (from flir_gap_report.py): split
    `files` (already sorted, FLIR filename timestamps) into poses by
    time gap between consecutive files, instead of frame-diff.

    Use when the FLIR recorded continuously without stops between poses
    (frame-diff still segments those, this is for the opposite case: a
    stop-start recording where frame-diff has nothing to detect between
    files, or where NUC/AGC noise makes frame-diff unreliable).

    Returns a list of (a, b) index ranges into `files` (files without a
    parseable timestamp are dropped first, with a warning)."""
    ts = [(i, parse_flir_timestamp(f.name)) for i, f in enumerate(files)]
    good = [(i, t) for i, t in ts if t is not None]
    if len(good) < len(ts):
        print(f"ATTENZIONE: {len(ts) - len(good)} file senza timestamp nel nome, "
              "ignorati per la segmentazione a gap.")
    if not good:
        return []
    poses = [[good[0][0]]]
    for (i, t), (i_prev, t_prev) in zip(good[1:], good):
        if (t - t_prev).total_seconds() > gap_threshold:
            poses.append([])
        poses[-1].append(i)
    return [(grp[0], grp[-1] + 1) for grp in poses]


def image_profile(files, dirs, args):
    """Frame-difference profile of an image session.

    Returns (value_times, values, abs_start, source, labels) where value_times
    and values are aligned per consecutive-frame pair.
    """
    offsets, abs_start, source = build_time_index(files, dirs, args.fps)
    print(f"Images found: {len(files)}   Time source: {source}")
    print(f"First: {files[0].name}   Last: {files[-1].name}")

    print("Computing differences between consecutive frames...")
    vt, values, labels = [], [], []
    prev = None
    skipped = 0
    for i, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1               # 0-byte / corrupt frame
            continue
        if args.downscale > 1:
            img = cv2.resize(img, (img.shape[1] // args.downscale,
                                   img.shape[0] // args.downscale))
        if prev is not None:
            values.append(frame_difference(prev, img))
            vt.append(offsets[f.name])
            labels.append(f.name)
        prev = img
        if (i + 1) % 200 == 0:
            print(f"  ...{i + 1}/{len(files)}")
    if skipped:
        print(f"  Skipped {skipped}/{len(files)} unreadable (empty/corrupt) frames.")
    if not values:
        sys.exit("No difference could be computed.")
    return np.array(vt), np.array(values), abs_start, source, labels


def _rect_score(p):
    """Lower = the 4 points look more like corners of a centred rectangle
    (from flir_pose_check.py).

    Corners of a rectangle are equidistant from the centroid (equal half-
    diagonals) and come in opposite pairs (v and -v). A far spurious blob
    breaks both, so it scores high and is rejected.
    """
    m = p.mean(axis=0)
    v = p - m
    r = np.linalg.norm(v, axis=1)
    rmean = float(r.mean())
    if rmean < 1e-3:
        return 1e9
    rad_err = float(r.std()) / rmean           # radii should be equal
    sym_err = 0.0
    for i in range(4):
        d = np.linalg.norm(v + v[i], axis=1)   # distance from -v[i] to each v[j]
        d[i] = np.inf
        sym_err += float(d.min())
    sym_err /= 4.0 * rmean                      # each v should have an opposite
    return rad_err + sym_err


def select_holes(cand, rect_tol):
    """Pick the board holes and return (pts(4,2), centre(2,)) or None
    (from flir_pose_check.py).

    The holes are the roundest blobs (not the largest), so candidates are
    ranked by circularity. First try to find 4 that form a centred
    rectangle; if a hole is missing (weak thermal contrast in some poses),
    reconstruct the 4th from 3 that form a right angle -- the four holes
    are the corners of a rectangle.
    """
    import itertools
    if len(cand) < 3:
        return None
    cand = sorted(cand, key=lambda t: -t[3])[:8]  # by circularity, keep top 8
    areas = np.array([a for a, _, _, _ in cand], dtype=float)
    pts_all = np.array([[x, y] for _, x, y, _ in cand], dtype=np.float32)
    n = len(cand)

    def area_ok(ii):
        aa = areas[ii]
        return aa.max() / max(aa.min(), 1e-6) <= 3.5

    # --- 4 real holes forming a centred rectangle ---
    best = None
    for idx in itertools.combinations(range(n), 4):
        ii = list(idx)
        if not area_ok(ii):
            continue
        p = pts_all[ii]
        if p.std(axis=0).max() < 4.0:              # too tightly clustered = noise
            continue
        s = _rect_score(p)
        if best is None or s < best[0]:
            best = (s, p)
    if best is not None and best[0] <= rect_tol:
        return best[1], best[1].mean(axis=0)

    # --- fallback: 3 holes -> reconstruct the 4th at the right-angle corner ---
    best3 = None
    for idx in itertools.combinations(range(n), 3):
        ii = list(idx)
        if not area_ok(ii):
            continue
        p3 = pts_all[ii]
        for vtx in range(3):
            o = [j for j in range(3) if j != vtx]
            va = p3[o[0]] - p3[vtx]
            vb = p3[o[1]] - p3[vtx]
            la, lb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if la < 6 or lb < 6:
                continue
            if abs(float(va @ vb)) / (la * lb) > 0.25:   # sides not ~perpendicular
                continue
            p4 = p3[o[0]] + p3[o[1]] - p3[vtx]           # reflect vertex
            quad = np.array([p3[0], p3[1], p3[2], p4], dtype=np.float32)
            s = _rect_score(quad)
            if best3 is None or s < best3[0]:
                best3 = (s, quad)
    if best3 is not None and best3[0] <= rect_tol:
        return best3[1], best3[1].mean(axis=0)
    return None


def detect_holes(gray, hole_kernel, min_area, max_area, min_circ, rect_tol):
    """Detect the four board holes (from flir_pose_check.py). Returns
    (pts(4,2), centre(2,)) or None."""
    gb = cv2.GaussianBlur(gray, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_kernel, hole_kernel))
    bh = cv2.morphologyEx(gb, cv2.MORPH_BLACKHAT, k)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        circ = area / (np.pi * r * r)
        if circ < min_circ:
            continue
        cand.append((area, x, y, circ))
    return select_holes(cand, rect_tol)


def _scan_edge(gray, cx, cy, dx, dy, i0, drop, run=3):
    """Walk from (cx,cy) along (dx,dy) (from flir_pose_check.py). Return
    distance to the board edge (intensity drop below i0-drop for `run` px)
    or None if the ray reaches the image border while still on the panel."""
    h_img, w_img = gray.shape[:2]
    cnt = 0
    t = 1
    while True:
        x = int(round(cx + dx * t))
        y = int(round(cy + dy * t))
        if x < 0 or x >= w_img or y < 0 or y >= h_img:
            return None  # ran off the frame still on the panel
        if gray[y, x] < i0 - drop:
            cnt += 1
            if cnt >= run:
                return t - run + 1
        else:
            cnt = 0
        t += 1


def board_extent(gray, ctr, drop):
    """Measure board half-extents (hx, hy) about the hole centre using the
    symmetric edge scan (from flir_pose_check.py). Returns dict with hx, hy
    (or None), i0."""
    cx, cy = float(ctr[0]), float(ctr[1])
    x0, y0 = int(round(cx)), int(round(cy))
    patch = gray[max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x0 + 3]
    i0 = float(np.median(patch)) if patch.size else float(gray[y0, x0])

    eL = _scan_edge(gray, cx, cy, -1, 0, i0, drop)
    eR = _scan_edge(gray, cx, cy, 1, 0, i0, drop)
    eU = _scan_edge(gray, cx, cy, 0, -1, i0, drop)
    eD = _scan_edge(gray, cx, cy, 0, 1, i0, drop)

    def half(a, b):
        vals = [e for e in (a, b) if e is not None]
        return sum(vals) / len(vals) if vals else None

    return {"i0": i0, "hx": half(eL, eR), "hy": half(eU, eD),
            "edges": (eL, eR, eU, eD)}


def board_inside(gray, ctr, ext, margin):
    """Return (inside_bool_or_None, clipped_sides list) (from
    flir_pose_check.py). None = inconclusive (an axis had no measurable
    edge on either side)."""
    h_img, w_img = gray.shape[:2]
    cx, cy = float(ctr[0]), float(ctr[1])
    hx, hy = ext["hx"], ext["hy"]
    if hx is None or hy is None:
        return None, []
    sides = []
    if cx - hx < margin:
        sides.append("L")
    if cx + hx > w_img - 1 - margin:
        sides.append("R")
    if cy - hy < margin:
        sides.append("U")
    if cy + hy > h_img - 1 - margin:
        sides.append("D")
    return (len(sides) == 0), sides


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"CSV saved: {path}")


def check_poses_inside(files, poses, args):
    """FLIR-only per-pose "is the board fully inside the frame" check (from
    flir_pose_check.py, --check-inside). `poses` are (a, b) index ranges
    into `files`. Returns csv_rows (posa,primo_file,ultimo_file,...,verdetto)
    and prints a report."""
    long_poses = [(a, b) for (a, b) in poses if (b - a) >= args.check_min_frames]
    print(f"Poses with >= {args.check_min_frames} frames: {len(long_poses)}")
    print()

    debug_dir = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(f"{'#':>3}  {'first':<22} {'last':<22} {'frames':>6}  "
          f"{'det':>4} {'inside':>6}  verdict")
    print("-" * 90)

    csv_rows = []
    for k, (a, b) in enumerate(long_poses, start=1):
        pose_files = files[a:b]
        sample = pose_files[::args.step] or pose_files

        n_analyzed = len(sample)
        n_detected = 0
        n_inside = 0
        clip_tally = {}
        first_annot = None

        for f in sample:
            gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            hole = detect_holes(gray, args.hole_kernel, args.hole_min_area,
                                args.hole_max_area, args.hole_min_circ,
                                args.rect_tol)
            if hole is None:
                continue
            n_detected += 1
            pts, ctr = hole
            ext = board_extent(gray, ctr, args.edge_drop)
            inside, sides = board_inside(gray, ctr, ext, args.margin_px)
            if inside:
                n_inside += 1
            for s in sides:
                clip_tally[s] = clip_tally.get(s, 0) + 1
            if debug_dir is not None and first_annot is None:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for (px, py) in pts.astype(int):
                    cv2.circle(vis, (px, py), 4, (0, 165, 255), -1)
                col = (0, 200, 0) if inside else (0, 0, 255)
                if ext["hx"] is not None and ext["hy"] is not None:
                    cx, cy = int(ctr[0]), int(ctr[1])
                    hx, hy = int(ext["hx"]), int(ext["hy"])
                    cv2.rectangle(vis, (cx - hx, cy - hy), (cx + hx, cy + hy), col, 2)
                first_annot = (f.stem, vis)

        if n_analyzed == 0 or n_detected < args.detect_frac * n_analyzed:
            verdict = "INCONCLUSIVE (holes rarely detected)"
        else:
            frac_inside = n_inside / n_detected
            if frac_inside >= args.inside_frac:
                verdict = "INSIDE"
            else:
                worst = "".join(sorted(clip_tally, key=lambda s: -clip_tally[s]))
                verdict = f"CLIPPED [{worst}]"

        print(f"{k:>3}  {pose_files[0].name:<22} {pose_files[-1].name:<22} "
              f"{b - a:>6}  {n_detected:>2}/{n_analyzed:<2} "
              f"{n_inside:>3}/{max(n_detected,1):<2}  {verdict}")

        if debug_dir is not None and first_annot is not None:
            out = debug_dir / f"pose{k:02d}_{first_annot[0]}.png"
            cv2.imwrite(str(out), first_annot[1])

        csv_rows.append([k, pose_files[0].name, pose_files[-1].name,
                         b - a, n_analyzed, n_detected, n_inside, verdict])

    print("=" * 90)
    inside_ok = [r[0] for r in csv_rows if r[-1] == "INSIDE"]
    print(f"Poses fully inside: {len(inside_ok)} / {len(long_poses)}  -> {inside_ok}")
    print()
    print("NOTES:")
    print("  - 'det'    = frames where the 4 holes were detected / frames analyzed.")
    print("  - 'inside' = detected frames whose board rectangle is fully in-frame.")
    print("  - Tune with --edge-drop (panel<->background contrast), --hole-*, "
          "and verify with --debug-dir.")
    return csv_rows


def main():
    ap = argparse.ArgumentParser(
        description="Count the board poses in a capture session (images or LiDAR)."
    )
    ap.add_argument("dirs", nargs="*", help="One or more image folders (in order)")
    ap.add_argument("--db3", default=None,
                    help="LiDAR mode: a rosbag2 .db3 file or folder. Detects poses "
                         "from scan-to-scan voxel-occupancy change instead of images. "
                         "Pure stdlib (sqlite3), loads the whole message index in RAM: "
                         "fine for normal bags. For very large bags use --bag instead.")
    ap.add_argument("--bag", default=None,
                    help="LiDAR mode, streaming variant of --db3 (needs the `rosbags` "
                         "package): reads a rosbag2 folder (with metadata.yaml) or a "
                         ".db3 file one message at a time, never the whole bag in RAM. "
                         "Use for tens-of-GB bags. --db3 and --bag are mutually exclusive.")
    ap.add_argument("--bag-topic", default="/livox/lidar",
                    help="LiDAR topic for --bag (default /livox/lidar).")
    ap.add_argument("--time-origin", choices=("bag", "topic"), default="bag",
                    help="--bag only: zero of the offsets, bag start or first message "
                         "on the topic (default bag).")
    ap.add_argument("--eye", choices=("right", "left"), default=None,
                    help="ZED only: select one eye when frames/ mixes right_*/left_* "
                         "(zed_record_dual.py sessions). Filters `dirs` by filename "
                         "prefix before processing.")
    ap.add_argument("--gap-threshold", type=float, default=None,
                    help="FLIR only: segment poses by time gap between consecutive "
                         "filenames (>N seconds = new pose) instead of frame-diff. Use "
                         "for stop-start recordings where frame-diff has nothing to "
                         "detect between files, or where NUC/AGC noise makes frame-diff "
                         "unreliable. If the FLIR recorded continuously (no real gaps), "
                         "use frame-diff (default, omit this flag) instead.")
    ap.add_argument("--check-inside", action="store_true",
                    help="FLIR only: for each pose with >= --check-min-frames frames, "
                         "detect the four-hole board and verdict whether it stays fully "
                         "inside the frame (not clipped by the image borders). Adds "
                         "hole-detection columns to --csv-out.")
    ap.add_argument("--check-min-frames", type=int, default=60,
                    help="--check-inside only: only check poses with at least this many "
                         "frames (default 60).")
    ap.add_argument("--step", type=int, default=3,
                    help="--check-inside only: analyze every Nth frame within a pose "
                         "(default 3).")
    ap.add_argument("--margin-px", type=int, default=3,
                    help="--check-inside only: required clear border, in pixels, on "
                         "every image edge (default 3).")
    ap.add_argument("--inside-frac", type=float, default=0.9,
                    help="--check-inside only: pose passes if >= this fraction of "
                         "DETECTED frames are fully inside (default 0.90).")
    ap.add_argument("--detect-frac", type=float, default=0.5,
                    help="--check-inside only: pose is inconclusive if the board is "
                         "detected in fewer than this fraction of analyzed frames "
                         "(default 0.50).")
    ap.add_argument("--hole-kernel", type=int, default=41,
                    help="--check-inside only: black-hat kernel (px), must exceed the "
                         "hole diameter (default 41).")
    ap.add_argument("--hole-min-area", type=float, default=100.0,
                    help="--check-inside only: min hole blob area in px (default 100).")
    ap.add_argument("--hole-max-area", type=float, default=3000.0,
                    help="--check-inside only: max hole blob area in px (default 3000).")
    ap.add_argument("--hole-min-circ", type=float, default=0.45,
                    help="--check-inside only: min hole circularity gate (default 0.45).")
    ap.add_argument("--rect-tol", type=float, default=0.35,
                    help="--check-inside only: max 'rectangle score' for the 4 holes "
                         "(default 0.35, lower rejects spurious sets more aggressively).")
    ap.add_argument("--edge-drop", type=float, default=20.0,
                    help="--check-inside only: intensity drop (0-255) from panel to "
                         "background marking a board edge in the outward scan (default 20).")
    ap.add_argument("--debug-dir", default=None,
                    help="--check-inside only: save one annotated frame per checked pose here.")
    ap.add_argument("--csv-out", default=None, metavar="CSV",
                    help="Also save the detected-poses table as CSV (the --check-inside "
                         "table, if that flag is set).")
    ap.add_argument("--compare", action="store_true",
                    help="Cross-check 2-3 sensors: detect poses on each of --flir / "
                         "--zed / --db3 and report the gap between consecutive poses "
                         "per sensor (clock-independent, so FLIR's different clock is "
                         "fine).")
    ap.add_argument("--flir", nargs="+", default=None,
                    help="FLIR image folder(s) for --compare.")
    ap.add_argument("--zed", nargs="+", default=None,
                    help="ZED image folder(s) for --compare.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Movement threshold. If omitted, estimated as "
                         "median + K*MAD (see --mad-k).")
    ap.add_argument("--mad-k", type=float, default=None,
                    help="Auto-threshold robustness factor (default 6 for images, "
                         "4 for LiDAR).")
    ap.add_argument("--min-pose-frames", type=int, default=None,
                    help="Min stable frames/scans for a pose (default 15 images; "
                         "LiDAR ~10 s worth).")
    ap.add_argument("--min-move-frames", type=int, default=None,
                    help="Min consecutive above-threshold frames/scans for a real "
                         "move (default 3 images, 1 LiDAR).")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="Keep only poses lasting at least this many SECONDS "
                         "(applies to images and LiDAR alike).")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="Fallback frame rate for images without timestamps.")
    ap.add_argument("--downscale", type=int, default=2,
                    help="Image downscale factor (default 2).")
    # LiDAR-only
    ap.add_argument("--db3-topic", default=None,
                    help="LiDAR topic name if the bag has more than one.")
    ap.add_argument("--db3-msg-stride", type=int, default=10,
                    help="Use every Nth scan (default 10). Lower = finer, slower.")
    ap.add_argument("--db3-point-stride", type=int, default=4,
                    help="Subsample every Nth point per scan (default 4).")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="Voxel size (m) for the occupancy difference (default 0.10).")
    ap.add_argument("--show-profile", action="store_true",
                    help="Print the full sample-by-sample difference profile.")
    args = ap.parse_args()

    if args.compare:
        compare_sensors(args)
        return

    if args.db3 is not None and args.bag is not None:
        sys.exit("Pass either --db3 or --bag, not both.")
    is_lidar = args.db3 is not None or args.bag is not None
    if is_lidar and args.dirs:
        sys.exit("Pass either image folders or --db3/--bag, not both.")
    if not is_lidar and not args.dirs:
        sys.exit("Nothing to do: pass image folder(s), --db3/--bag <rosbag>, or --compare.")
    gap_mode = not is_lidar and args.gap_threshold is not None

    files = None  # set in the image branch; used by --eye/--gap-threshold/--check-inside
    if args.bag is not None:
        bag = resolve_bag(args.bag)
        print(f"Bag: {bag}")
        vt_all, values, all_off, info = lidar_profile_stream(
            bag, args.bag_topic, args.db3_msg_stride, args.db3_point_stride,
            args.voxel, args.time_origin)
        value_times = vt_all
        abs_start = None
        source = f"rosbag2 stream ({Path(bag).name})"
        count_unit = "scans"
        dur = value_times[-1] - value_times[0] if value_times.size else 0.0
        eff_rate = (value_times.size - 1) / dur if dur > 0 else 0.0
        min_move = args.min_move_frames if args.min_move_frames is not None else 1
        min_pose = (args.min_pose_frames if args.min_pose_frames is not None
                    else max(4, round(10.0 * eff_rate)))
        mad_k = args.mad_k if args.mad_k is not None else 2.5
        print(f"First message on topic: {info['topic_start_utc']:%Y-%m-%d %H:%M:%S.%f} UTC "
              f"({info['topic_start_offset']:+.2f} s from bag start)")
        print(f"Effective sample rate: {eff_rate:.2f} Hz "
              f"(min-pose-frames={min_pose} ~ {min_pose/max(eff_rate,1e-6):.0f} s)")
    elif args.db3 is not None:
        db3 = _resolve_db3(args.db3)
        vt_all, values, abs_start, source, n_total, eff_rate = lidar_profile(
            db3, args.db3_topic, args.db3_msg_stride, args.db3_point_stride, args.voxel)
        value_times = vt_all[1:]                       # value i compares scan i-1,i
        count_unit = "scans"
        # a board move spans ~1 sampled scan at coarse msg-stride, so no debounce
        min_move = args.min_move_frames if args.min_move_frames is not None else 1
        min_pose = (args.min_pose_frames if args.min_pose_frames is not None
                    else max(4, round(10.0 * eff_rate)))
        mad_k = args.mad_k if args.mad_k is not None else 4.0
        print(f"Effective sample rate: {eff_rate:.2f} Hz "
              f"(min-pose-frames={min_pose} ~ {min_pose/max(eff_rate,1e-6):.0f} s)")
    else:
        files = collect_images(args.dirs)
        if args.eye:
            files = [f for f in files if f.name.startswith(f"{args.eye}_")]
        if len(files) < 2:
            sys.exit("At least 2 images are required.")
        offsets, abs_start, source = build_time_index(files, args.dirs, args.fps)
        count_unit = "frames"
        min_move = args.min_move_frames if args.min_move_frames is not None else 3
        min_pose = args.min_pose_frames if args.min_pose_frames is not None else 15
        mad_k = args.mad_k if args.mad_k is not None else 6.0
        if not gap_mode:
            value_times, values, abs_start, source, labels = image_profile(
                files, args.dirs, args)

    if gap_mode:
        print(f"Images found: {len(files)}   Time source: {source}")
        print(f"First: {files[0].name}   Last: {files[-1].name}")
        file_poses = split_on_gaps(files, args.gap_threshold)
        dropped = 0
        if args.min_duration is not None:
            def _fp_dur(a, b):
                return offsets[files[b - 1].name] - offsets[files[a].name]
            kept = [(a, b) for (a, b) in file_poses if _fp_dur(a, b) >= args.min_duration]
            dropped = len(file_poses) - len(kept)
            file_poses = kept

        print()
        print("=" * 78)
        print(f"DETECTED POSES: {len(file_poses)}   (gap > {args.gap_threshold:g} s"
              f"{f'; {dropped} shorter than {args.min_duration:g}s dropped' if dropped else ''})")
        print("=" * 78)
        print(f"{'#':>3}  {'start':<10} {'end':<10} {'duration':>8}  {count_unit:>6}")
        print("-" * 78)
        rows = []
        for k, (a, b) in enumerate(file_poses, start=1):
            off_a, off_b = offsets[files[a].name], offsets[files[b - 1].name]
            print(f"{k:>3}  {fmt_clock(abs_start, off_a):<10} "
                  f"{fmt_clock(abs_start, off_b):<10} "
                  f"{off_b - off_a:>7.0f}s  {b - a:>6}     "
                  f"[session offset: {off_a:.0f}s - {off_b:.0f}s]")
            rows.append([k, files[a].name, files[b - 1].name,
                        f"{off_a:.1f}", f"{off_b:.1f}", f"{off_b - off_a:.1f}", b - a])
        print()
        print("NOTE: segmentation by filename-timestamp gap (--gap-threshold), not "
              "frame-diff. If this finds only 1 pose, the FLIR likely recorded "
              "continuously: drop --gap-threshold to use frame-diff instead.")

        if args.check_inside:
            print()
            csv_rows = check_poses_inside(files, file_poses, args)
            if args.csv_out:
                print()
                write_csv(args.csv_out, ["posa", "primo_file", "ultimo_file", "n_frame",
                                         "n_analizzati", "n_rilevati", "n_inside", "verdetto"],
                          csv_rows)
        elif args.csv_out:
            print()
            write_csv(args.csv_out, ["pose", "first_file", "last_file", "start_s",
                                     "end_s", "duration_s", "n_frame"], rows)
        return

    dur_total = value_times[-1] - value_times[0]
    print(f"Total session duration: {dur_total:.0f} s ({dur_total/60:.1f} min)")
    if abs_start is not None:
        print(f"Session start: {abs_start.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if args.threshold is None:
        threshold = median + mad_k * (mad if mad > 1e-6 else 1.0)
    else:
        threshold = args.threshold

    print(f"Difference statistics: median={median:.3f}  MAD={mad:.3f}")
    print(f"Movement threshold used: {threshold:.3f}"
          f"{f'  (auto: median+{mad_k:g}*MAD)' if args.threshold is None else '  (provided)'}")
    print()

    if args.show_profile:
        print("Difference profile:")
        for i, v in enumerate(values):
            marker = "  <-- MOVEMENT" if v > threshold else ""
            print(f"  {fmt_clock(abs_start, value_times[i])}  {v:8.3f}{marker}")
        print()

    poses, transitions = segment(values, threshold, min_move, min_pose)

    def pose_duration(a, b):
        return value_times[min(b, len(value_times)) - 1] - value_times[a]

    dropped = 0
    if args.min_duration is not None:
        kept = [(a, b) for (a, b) in poses if pose_duration(a, b) >= args.min_duration]
        dropped = len(poses) - len(kept)
        poses = kept

    print("=" * 78)
    print(f"DETECTED POSES: {len(poses)}   (real transitions: {len(transitions)}"
          f"{f'; {dropped} shorter than {args.min_duration:g}s dropped' if dropped else ''})")
    print("=" * 78)
    print(f"{'#':>3}  {'start':<10} {'end':<10} {'duration':>8}  {count_unit:>6}")
    print("-" * 78)

    for k, (a, b) in enumerate(poses, start=1):
        off_a = value_times[a]
        off_b = value_times[min(b, len(value_times)) - 1]
        dur = off_b - off_a
        print(f"{k:>3}  {fmt_clock(abs_start, off_a):<10} "
              f"{fmt_clock(abs_start, off_b):<10} "
              f"{dur:>7.0f}s  {b - a:>6}     "
              f"[session offset: {off_a:.0f}s - {off_b:.0f}s]")

    print()
    print("NOTES:")
    print(f"  - Stable segments shorter than {min_pose} {count_unit} were discarded")
    print("    (likely transitions, not real poses). Tune with --min-pose-frames.")
    print(f"  - Above-threshold bursts shorter than {min_move} {count_unit} were")
    print("    ignored as noise. Tune with --min-move-frames.")
    if args.min_duration is not None:
        print(f"  - Only poses lasting >= {args.min_duration:g}s are shown "
              "(--min-duration).")
    print("  - If the pose count is off, try --threshold or --mad-k "
          "(higher = fewer moves).")
    if is_lidar:
        print("  - LiDAR 'difference' = voxel-occupancy Jaccard distance per scan; "
              "tune --voxel / --db3-msg-stride.")
    print("  - Use --show-profile to inspect the sample-by-sample detail.")
    print("  - Offsets are relative to the start of THIS session. The camera and")
    print("    LiDAR recordings start at different moments and must be aligned by")
    print("    comparing their absolute start times.")

    if files is not None:
        # image mode: value index k <-> file index k+1 (image_profile skips the
        # first frame as "prev", emitting no diff for it)
        file_poses = [(a + 1, b + 1) for a, b in poses]
        if args.check_inside:
            print()
            csv_rows = check_poses_inside(files, file_poses, args)
            if args.csv_out:
                print()
                write_csv(args.csv_out, ["posa", "primo_file", "ultimo_file", "n_frame",
                                         "n_analizzati", "n_rilevati", "n_inside", "verdetto"],
                          csv_rows)
        elif args.csv_out:
            rows = [[k, files[a].name, files[b - 1].name, f"{offsets[files[a].name]:.1f}",
                    f"{offsets[files[b - 1].name]:.1f}",
                    f"{offsets[files[b - 1].name] - offsets[files[a].name]:.1f}", b - a]
                    for k, (a, b) in enumerate(file_poses, start=1)]
            print()
            write_csv(args.csv_out, ["pose", "first_file", "last_file", "start_s",
                                     "end_s", "duration_s", "n_frame"], rows)
    elif args.csv_out:
        rows = [[k, f"{value_times[a]:.1f}", f"{value_times[min(b, len(value_times)) - 1]:.1f}",
                f"{value_times[min(b, len(value_times)) - 1] - value_times[a]:.1f}", b - a]
                for k, (a, b) in enumerate(poses, start=1)]
        print()
        write_csv(args.csv_out, ["pose", "start_s", "end_s", "duration_s", "n_scans"], rows)


if __name__ == "__main__":
    main()
