"""Full-FOV world clouds from a rosbag's RAW Livox topic plus its odometry.

Why this exists
---------------
FAST-LIO's `/cloud_registered` is not what the sensor recorded. Measured on
session 9 (`rosbag2_2026_07_30-18_12_20`), first triplet:

    topic                  points/msg   azimuth          elevation      min range
    /livox/lidar               83 096   -59.5 .. +60.8   -13.4 .. +13.2     1.02 m
    /cloud_registered           6 465   -17.2 .. +17.4    -4.3 ..  +8.7     4.05 m

Projected into the ZED frame that leaves a patch of u in [666, 1309],
v in [369, 628] -- about **8% of the frame area** -- and every one of those
6465 points already lands inside the image, so the camera FOV clips nothing.
The cloud arrives pre-cropped. The 4.05 m floor looks like FAST-LIO's
`preprocess/blind`, the +-17 deg cone like `mapping/fov_degree`.

Consequence downstream: WindowsDoorsDetection's stage 2 pools a vote per LiDAR
point, so a region no point reaches can never win a voxel. Both window bays in
session 9 sit outside the cone and get **zero returns even inside their
bounding boxes** -- which is also why that module's long-standing claim that
"glazing returns no LiDAR" cannot be trusted as measured: it was measured on
this cloud, and it cannot tell glass apart from out-of-footprint.

Re-running FAST-LIO with those parameters relaxed is the proper fix and also
buys better odometry, since the trajectory was estimated from the cropped scans
too. This module is the cheap one: the raw points are already in the bag, and
`/Odometry`'s pose maps lidar-local straight to world with no extra extrinsic.

Verified before it was written: transforming one raw scan by the manifest's
pose puts the matching `/cloud_registered` points 4.9 cm away at the median,
93.6% within 10 cm -- i.e. the same cloud, plus the 92% FAST-LIO threw away.
Read that residual as agreement, not as an error bar: it is a nearest-neighbour
distance into a cloud 13x denser, so it mostly measures surface sampling. It
does NOT shrink with `deskew=True` on these frames, because there is no motion
here to remove.

De-skewing: cheap, on by default, and small on THIS data
--------------------------------------------------------
Each CustomMsg covers **200 ms** of scanning (measured: `offset_time` spans
200.1 and 200.4 ms on the first two messages), not the 100 ms one might assume,
so a moving sensor smears every surface across a fifth of a second of travel.
Each point is therefore transformed by the pose interpolated at its own
timestamp.

How much that is worth depends entirely on speed, and on session 9's first
frames it is worth almost nothing: the rover is nearly stationary there (0.6 cm
between consecutive odometry poses), and deskew-on against deskew-off moves the
same points by p50 0.2 cm, max 0.6 cm. Do not quote "de-skew fixed the
registration" from this bag -- it did not, because there was no motion to fix.
It stays on because it costs one interpolation per point and the same session
does contain walking segments.

Drop-in contract
----------------
`nearest_clouds_for_targets` mirrors
EmissivityCalculation/project_to_flir.py's function of the same name --
`(bag, target_epochs, topic, store) -> [(t, points_world) | None]` -- so it
substitutes wherever that one is imported. Note the third positional is the
CLOUD topic, which for us is `/livox/lidar`; the odometry topic is a separate
keyword. Callers whose CLI names that argument `--odom-topic` (both
WindowsDoorsDetection stages do, confusingly) must pass `/livox/lidar` there.

Usage:
    py livox_odometry_loader.py --bag ...\\rosbag2_2026_07_30-18_12_20 --inspect
    py livox_odometry_loader.py --bag ... --session-dir ...\\fullrate --limit 5 --compare
    py livox_odometry_loader.py --bag ... --session-dir ...\\fullrate --limit 1 --export-ply out.ply
"""

import argparse
import json
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import get_types_from_msg

LIDAR_TOPIC = "/livox/lidar"
ODOM_TOPIC = "/Odometry"

# livox_ros_driver2's message definitions. Registered by hand because the
# driver's .msg files are not installed on this machine -- rosbags only needs
# the field layout to deserialize, and a bag recorded by the driver carries no
# type definitions of its own (AnyReader raises "Bag contains no type
# definitions" without a default_typestore).
CUSTOM_POINT_MSG = """uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line"""

CUSTOM_MSG = """std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
livox_ros_driver2/CustomPoint[] points"""

# CustomPoint is 19 bytes of fields padded to a 4-byte boundary. Verified
# against the bag rather than assumed: (len(rawdata) - header) / point_num came
# out at exactly 20.00 on both messages checked. `_points_from_raw` re-checks
# this per message and falls back to full deserialisation if it ever fails.
POINT_STRIDE = 20
_POINT_DTYPE = np.dtype([
    ("offset_time", "<u4"),
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("reflectivity", "u1"), ("tag", "u1"), ("line", "u1"), ("_pad", "u1"),
])


def make_typestore(store: str = "ROS2_HUMBLE"):
    ts = get_typestore(Stores[store])
    ts.register(get_types_from_msg(CUSTOM_POINT_MSG, "livox_ros_driver2/msg/CustomPoint"))
    ts.register(get_types_from_msg(CUSTOM_MSG, "livox_ros_driver2/msg/CustomMsg"))
    return ts


# --- reading the bag without paying for deserialisation ---------------------

def _stamp_from_raw(rawdata: bytes) -> float:
    """header.stamp, straight out of the CDR bytes.

    A CDR payload opens with a 4-byte encapsulation header, and `std_msgs/Header`
    puts `builtin_interfaces/Time` first, so sec/nsec sit at offset 4 with no
    alignment padding in between. Verified against rosbags' own deserialisation
    on this bag: exact match to the nanosecond.

    Worth the 30 lines: deserialising a CustomMsg costs ~128 ms because rosbags
    materialises 90 000 Python objects for the point array, and the first pass
    only needs to know which message is closest to each target.
    """
    sec, nsec = struct.unpack_from("<iI", rawdata, 4)
    return float(sec) + float(nsec) * 1e-9


def _points_from_raw(rawdata: bytes, reader=None, connection=None):
    """(offset_time_ns, xyz) for one CustomMsg, by viewing the CDR bytes.

    The points sequence is the last member of the message, so it occupies the
    tail of the payload: `point_num * 20` bytes preceded by the sequence-length
    uint32. That length is read back and checked against the tail size, and any
    mismatch falls through to rosbags -- so a driver that ever changes the
    struct layout gets slow, not wrong.
    """
    n_tail = (len(rawdata) - 4) // POINT_STRIDE
    for n in (n_tail, n_tail - 1):          # allow up to 3 bytes of end padding
        if n <= 0:
            break
        start = len(rawdata) - n * POINT_STRIDE
        if start >= 8 and struct.unpack_from("<I", rawdata, start - 4)[0] == n:
            arr = np.frombuffer(rawdata, dtype=_POINT_DTYPE, count=n, offset=start)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
            return arr["offset_time"].astype(np.int64), xyz

    if reader is None:
        raise ValueError("CustomMsg point array not where expected and no reader "
                         "given to fall back on")
    msg = reader.deserialize(rawdata, connection.msgtype)
    off = np.fromiter((p.offset_time for p in msg.points), dtype=np.int64, count=len(msg.points))
    xyz = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float64)
    return off, xyz


# --- the trajectory ---------------------------------------------------------

@dataclass
class Trajectory:
    """Time-ordered odometry poses, queryable at arbitrary instants.

    Poses map LIDAR-LOCAL coordinates to world directly -- there is no
    lidar->body extrinsic to compose. That is not an assumption: transforming a
    raw scan by the pose lands `/cloud_registered`'s own points 4.9 cm away at
    the median, which is the motion-smear residual and nothing else.
    """
    times: np.ndarray       # (N,) epoch seconds, increasing
    positions: np.ndarray   # (N, 3)
    quats: np.ndarray       # (N, 4) xyzw

    def __len__(self):
        return len(self.times)

    def pose_at(self, t: np.ndarray):
        """Interpolated (positions (M,3), quats (M,4)) at each time in `t`.

        Clamped at both ends rather than extrapolated: a point 5 ms past the
        last odometry sample should get the last pose, not a linear guess.
        """
        t = np.asarray(t, dtype=float)
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 2)
        t0, t1 = self.times[idx], self.times[idx + 1]
        u = np.clip((t - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)[:, None]
        pos = self.positions[idx] * (1 - u) + self.positions[idx + 1] * u
        return pos, _slerp(self.quats[idx], self.quats[idx + 1], u[:, 0])


def _slerp(q0: np.ndarray, q1: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion slerp, row-wise. xyzw in, xyzw out.

    Written out rather than pulled from scipy: the consensus stage's venv has
    no scipy, and this module is meant to be importable from both.
    """
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0, -q1, q1)
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6                 # parallel: slerp degenerates to lerp
    safe = np.where(near, 1.0, sin_theta)
    s0 = np.where(near, 1.0 - u, np.sin((1.0 - u) * theta) / safe)
    s1 = np.where(near, u, np.sin(u * theta) / safe)
    q = s0[:, None] * q0 + s1[:, None] * q1
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v (M,3) by q (M,4, xyzw), without building M rotation matrices."""
    u = q[:, :3]
    w = q[:, 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def read_trajectory(bag: Path, odom_topic: str = ODOM_TOPIC,
                    store: str = "ROS2_HUMBLE", typestore=None) -> Trajectory:
    """All `nav_msgs/Odometry` poses in the bag, sorted by header stamp."""
    typestore = typestore or make_typestore(store)
    times, pos, quats = [], [], []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == odom_topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Odometry topic {odom_topic!r} not in bag. Available: {topics}")
        for connection, _bagts, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            times.append(_stamp_from_raw(rawdata))
            pos.append((p.x, p.y, p.z))
            quats.append((o.x, o.y, o.z, o.w))
    if len(times) < 2:
        raise SystemExit(f"{odom_topic} has {len(times)} pose(s); need at least 2 to interpolate.")
    order = np.argsort(times)
    return Trajectory(np.asarray(times)[order], np.asarray(pos)[order], np.asarray(quats)[order])


# --- the scans --------------------------------------------------------------

def scan_stamps(bag: Path, topic: str = LIDAR_TOPIC, store: str = "ROS2_HUMBLE",
                typestore=None) -> list[float]:
    """header.stamp of every message on `topic`, in bag order, cheaply."""
    typestore = typestore or make_typestore(store)
    out = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not in bag. Available: {topics}")
        for _connection, _bagts, rawdata in reader.messages(connections=conns):
            out.append(_stamp_from_raw(rawdata))
    return out


def world_cloud(offsets_ns: np.ndarray, xyz: np.ndarray, base_time: float,
                traj: Trajectory, deskew: bool = True,
                min_range: float = 0.0, max_range: float = 0.0,
                point_filter_num: int = 1) -> np.ndarray:
    """One raw scan -> (M, 3) world points.

    Filtering happens BEFORE the pose lookup, so a decimated run does not pay
    to interpolate poses it is about to throw away. Zero returns (the driver
    emits x=y=z=0 for a non-detection) are always dropped: ~8% of a HAP message
    on this bag, and every one of them would land on the sensor origin and vote
    for whatever segment happens to sit there.
    """
    r = np.linalg.norm(xyz, axis=1)
    keep = r > 1e-6
    if min_range > 0:
        keep &= r >= min_range
    if max_range > 0:
        keep &= r <= max_range
    idx = np.nonzero(keep)[0]
    if point_filter_num > 1:
        idx = idx[::point_filter_num]
    if idx.size == 0:
        return np.zeros((0, 3))

    pts = xyz[idx]
    if deskew:
        # Each point carries its own offset from the message stamp, and a
        # CustomMsg spans 200 ms on this bag -- see the module docstring.
        t = base_time + offsets_ns[idx] * 1e-9
    else:
        t = np.full(idx.size, base_time)
    pos, quat = traj.pose_at(t)
    return _rotate(quat, pts) + pos


def nearest_clouds_for_targets(bag: Path, target_epochs: list[float],
                               topic: str = LIDAR_TOPIC, store: str = "ROS2_HUMBLE",
                               odom_topic: str = ODOM_TOPIC, deskew: bool = True,
                               min_range: float = 0.0, max_range: float = 0.0,
                               point_filter_num: int = 1, traj: Trajectory | None = None,
                               ) -> list[tuple[float, np.ndarray] | None]:
    """For each target epoch, the nearest raw scan, transformed to world.

    Same shape as project_to_flir.nearest_clouds_for_targets, so it drops in
    wherever that is imported -- but reading the RAW topic, which on this bag
    means ~13x the points over 3.5x the azimuth from 1 m instead of 4 m.

    Two passes over the bag on purpose. The first reads only header stamps out
    of the CDR bytes to decide which messages are wanted; the second
    deserialises just those. Deserialising every message to find the nearest
    one would cost ~128 ms x 622 messages for data almost all of which is
    discarded.
    """
    typestore = make_typestore(store)
    traj = traj if traj is not None else read_trajectory(bag, odom_topic, store, typestore)

    stamps = scan_stamps(bag, topic, store, typestore)
    if not stamps:
        return [None] * len(target_epochs)
    arr = np.asarray(stamps)
    wanted: dict[int, list[int]] = {}
    for i, target in enumerate(target_epochs):
        k = int(np.abs(arr - float(target)).argmin())
        wanted.setdefault(k, []).append(i)

    out: list[tuple[float, np.ndarray] | None] = [None] * len(target_epochs)
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for k, (connection, _bagts, rawdata) in enumerate(reader.messages(connections=conns)):
            if k not in wanted:
                continue
            base = _stamp_from_raw(rawdata)
            offsets, xyz = _points_from_raw(rawdata, reader, connection)
            pts = world_cloud(offsets, xyz, base, traj, deskew=deskew,
                              min_range=min_range, max_range=max_range,
                              point_filter_num=point_filter_num)
            for i in wanted[k]:
                out[i] = (base, pts)
    return out


# --- CLI --------------------------------------------------------------------

def read_pointcloud2_xyz(msg) -> np.ndarray:
    """x/y/z out of a sensor_msgs/PointCloud2, for --compare only."""
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}
    return np.stack([raw[:, off[a]:off[a] + 4].copy().view(np.float32).ravel()
                     for a in ("x", "y", "z")], axis=1).astype(np.float64)


def targets_from_session(session_dir: Path, limit: int | None, every_n: int):
    """The lidar epochs WindowsDoorsDetection would ask for, from its manifest."""
    man = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    trips = man["triplets"][::every_n]
    if limit:
        trips = trips[:limit]
    return [t["lidar"]["timestamp_zedclock"] for t in trips], trips


def write_ply(path: Path, pts: np.ndarray):
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(pts)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "end_header\n").encode("ascii")
    path.write_bytes(header + pts.astype("<f4").tobytes())


def fix_rosbag2_digests(bag: Path) -> int:
    """Fill in the empty `type_description_hash` MATLAB's ros2bagwriter leaves.

    A bag written by MATLAB (`3DModelPointCloudExtraction/BagFilter*.m`, the
    corrected-odometry export) stores the message definition but leaves its
    RIHS01 digest as "". rosbags asserts that digest against one it recomputes
    from the stored definition, so such a bag fails to open with
    `AssertionError: Failed to parse nav_msgs/msg/Odometry` -- before a single
    message is read. The definition itself is correct; only the hash is absent.

    This recomputes the digest exactly as rosbags would and writes it back, to
    the sqlite tables and to metadata.yaml. It only ever fills an EMPTY field,
    so it is idempotent and can never paper over a genuine type mismatch.

    Returns the number of definitions patched.
    """
    import sqlite3
    from rosbags.typesys import get_types_from_msg
    from rosbags.typesys.store import Typestore

    dbs = sorted(bag.glob("*.db3"))
    if not dbs:
        raise SystemExit(f"No .db3 in {bag} -- not a sqlite3 rosbag2.")

    n_fixed, last_hash = 0, None
    for db in dbs:
        con = sqlite3.connect(db)
        try:
            rows = list(con.execute(
                "SELECT id, topic_type, encoded_message_definition, type_description_hash"
                " FROM message_definitions"))
        except sqlite3.OperationalError:
            print(f"  {db.name}: no message_definitions table, nothing to fix")
            con.close()
            continue
        for mid, name, msgdef, digest in rows:
            if digest:
                print(f"  {db.name}: {name} already has a digest, left alone")
                continue
            store = Typestore()
            store.register(get_types_from_msg(msgdef, name))
            last_hash = store.hash_rihs01(name)
            con.execute("UPDATE message_definitions SET type_description_hash = ?"
                        " WHERE id = ?", (last_hash, mid))
            con.execute("UPDATE topics SET type_description_hash = ? WHERE type = ?",
                        (last_hash, name))
            print(f"  {db.name}: {name} -> {last_hash}")
            n_fixed += 1
        con.commit()
        con.close()

    meta = bag / "metadata.yaml"
    if last_hash and meta.is_file():
        text = meta.read_text(encoding="utf-8")
        if 'type_description_hash: ""' in text:
            # The reader takes the digest from sqlite; this is only so the
            # manifest agrees with it for anything else that reads the bag.
            meta.write_text(text.replace('type_description_hash: ""',
                                         f'type_description_hash: "{last_hash}"'),
                            encoding="utf-8")
    return n_fixed


def parse_args():
    p = argparse.ArgumentParser(
        description="World clouds from the raw Livox topic + odometry, bypassing "
                    "FAST-LIO's cropped /cloud_registered")
    p.add_argument("--bag", required=True, metavar="DIR")
    p.add_argument("--lidar-topic", default=LIDAR_TOPIC)
    p.add_argument("--odom-topic", default=ODOM_TOPIC)
    p.add_argument("--store", default="ROS2_HUMBLE")
    p.add_argument("--session-dir", default=None, metavar="DIR",
                   help="Take target epochs from this session's sync_manifest.json.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--every-n", type=int, default=1)
    p.add_argument("--min-range", type=float, default=0.0, metavar="M")
    p.add_argument("--max-range", type=float, default=0.0, metavar="M",
                   help="0 disables (default).")
    p.add_argument("--point-filter-num", type=int, default=1, metavar="N",
                   help="Keep every Nth surviving point (default 1, keep all).")
    p.add_argument("--no-deskew", action="store_true",
                   help="One pose per scan instead of one per point. A CustomMsg spans "
                        "200 ms here, so this smears by however far the rover moved in "
                        "that time -- 0.6 cm on session 9's opening frames, where it is "
                        "nearly stationary. Diagnostic only.")
    p.add_argument("--inspect", action="store_true",
                   help="Report what each topic in the bag actually contains.")
    p.add_argument("--fix-digests", action="store_true",
                   help="Repair a bag written by MATLAB's ros2bagwriter, which leaves "
                        "type_description_hash empty and so cannot be opened by rosbags "
                        "at all. Fills only empty digests; safe to re-run. Needed once on "
                        "each corrected-odometry bag out of BagFilter*.m.")
    p.add_argument("--compare", action="store_true",
                   help="Per target, compare the rebuilt cloud against /cloud_registered.")
    p.add_argument("--compare-topic", default="/cloud_registered")
    p.add_argument("--export-ply", default=None, metavar="PLY",
                   help="Write the first target's rebuilt cloud to a PLY.")
    return p.parse_args()


def angular_summary(pts_local: np.ndarray) -> str:
    r = np.linalg.norm(pts_local, axis=1)
    ok = r > 1e-6
    r = r[ok]
    p = pts_local[ok]
    az = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
    el = np.degrees(np.arcsin(np.clip(p[:, 2] / r, -1, 1)))
    return (f"{ok.sum():7d} pts   az {az.min():7.1f}..{az.max():6.1f}   "
            f"el {el.min():6.1f}..{el.max():5.1f}   range {r.min():5.2f}..{r.max():6.2f} m")


def main():
    args = parse_args()
    bag = Path(args.bag)
    typestore = make_typestore(args.store)

    if args.fix_digests:
        # Before anything opens the bag: an unpatched MATLAB bag cannot be read
        # at all, so this has to run ahead of make_typestore's first use of it.
        n = fix_rosbag2_digests(bag)
        print(f"{bag.name}: {n} definition(s) patched"
              if n else f"{bag.name}: nothing to patch")
        return 0

    if args.inspect:
        with AnyReader([bag], default_typestore=typestore) as reader:
            print(f"{bag.name}")
            for c in sorted(reader.connections, key=lambda c: -c.msgcount):
                print(f"  {c.topic:<24} {c.msgtype:<44} msgs={c.msgcount}")
        raw_stamps = scan_stamps(bag, args.lidar_topic, args.store, typestore)
        print(f"\n{args.lidar_topic}: {len(raw_stamps)} messages, "
              f"{raw_stamps[-1] - raw_stamps[0]:.1f} s span")
        with AnyReader([bag], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic == args.lidar_topic]
            for connection, _b, rawdata in reader.messages(connections=conns):
                offsets, xyz = _points_from_raw(rawdata, reader, connection)
                print(f"  first message, sensor frame: {angular_summary(xyz)}")
                print(f"  offset_time spans {(offsets.max() - offsets.min()) / 1e6:.1f} ms "
                      f"-- that is how much motion one 'scan' contains")
                break
        return 0

    if not args.session_dir:
        print("Nothing to do: pass --session-dir (targets) or --inspect.", file=sys.stderr)
        return 1

    session_dir = Path(args.session_dir)
    targets, trips = targets_from_session(session_dir, args.limit, args.every_n)
    t0 = time.time()
    traj = read_trajectory(bag, args.odom_topic, args.store, typestore)
    print(f"{args.odom_topic}: {len(traj)} poses over "
          f"{traj.times[-1] - traj.times[0]:.1f} s ({time.time() - t0:.1f}s to read)")

    t0 = time.time()
    clouds = nearest_clouds_for_targets(
        bag, targets, args.lidar_topic, args.store, odom_topic=args.odom_topic,
        deskew=not args.no_deskew, min_range=args.min_range, max_range=args.max_range,
        point_filter_num=args.point_filter_num, traj=traj)
    print(f"{len(targets)} target(s) -> {sum(c is not None for c in clouds)} cloud(s) "
          f"in {time.time() - t0:.1f}s "
          f"(deskew {'off' if args.no_deskew else 'on'})")

    ref = {}
    if args.compare:
        with AnyReader([bag], default_typestore=typestore) as reader:
            conns = [c for c in reader.connections if c.topic == args.compare_topic]
            for connection, _b, rawdata in reader.messages(connections=conns):
                t = _stamp_from_raw(rawdata)
                for i, target in enumerate(targets):
                    if abs(t - target) < 0.06:
                        ref[i] = read_pointcloud2_xyz(reader.deserialize(rawdata,
                                                                        connection.msgtype))

    for i, (target, cloud) in enumerate(zip(targets, clouds)):
        stem = Path(trips[i]["flir"]["file"]).stem
        if cloud is None:
            print(f"  [{i}] {stem}: no scan")
            continue
        t, pts = cloud
        print(f"  [{i}] {stem}: {len(pts):7d} world points, scan stamp {t - target:+.3f}s "
              f"from target")
        if i in ref and len(ref[i]):
            tree_pts = pts
            d = np.array([np.min(np.sum((tree_pts - q) ** 2, axis=1)) for q in ref[i][::20]])
            d = np.sqrt(d)
            print(f"        vs {args.compare_topic} ({len(ref[i])} pts, every 20th checked): "
                  f"p50={np.median(d) * 100:.1f} cm  p90={np.percentile(d, 90) * 100:.1f} cm  "
                  f"gain={len(pts) / len(ref[i]):.1f}x points")

    if args.export_ply and clouds and clouds[0] is not None:
        out = Path(args.export_ply)
        write_ply(out, clouds[0][1])
        print(f"Wrote {len(clouds[0][1])} points to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
