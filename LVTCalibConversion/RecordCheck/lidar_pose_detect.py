"""
Rileva le pose statiche della board in un bag ROS2 del LiDAR Livox (.db3,
topic /livox/lidar, livox_ros_driver2/msg/CustomMsg), per differenza tra scan.

Idea (la stessa di Thesis/Calibration/detect_board_poses.py --db3, gia' usata
su Exttr_tryN): ogni scan campionato viene voxelizzato (--voxel) e confrontato
con il precedente tramite distanza di Jaccard sull'occupazione dei voxel
    d = 1 - |A intersezione B| / |A unione B|
bassa finche' la scena e' ferma, un picco quando la board viene spostata.
Gli scan anomali (spostamento) sono gli outlier sopra mediana + K*MAD
(--mad-k); i tratti stabili tra due spostamenti sono le pose. Su Exttr_tryN
--db3-msg-stride 5 --voxel 0.10 --mad-k 2.5 ha dato 19 segmenti, 12 pose
pulite dopo aver scartato quelli corti (--min-duration).

Il bag viene letto in streaming con `rosbags` (un messaggio alla volta, mai
tutto in RAM): va bene anche per bag da decine di GB. Il payload CDR del
CustomMsg viene decodificato direttamente (solo x/y/z), senza typestore.
--db3-msg-stride N analizza uno scan ogni N (piu' veloce, meno risoluzione);
i messaggi saltati passano comunque dal disco.

Offset in secondi dall'inizio del bag (--time-origin bag, default) o dal primo
messaggio del topic (--time-origin topic). n_messaggi conta tutti i messaggi
del topic nella finestra della posa, non solo quelli campionati.

Lo script legge solo: il bag non viene copiato, spostato o modificato.

Uso tipico:
    py lidar_pose_detect.py --bag <cartella_bag>
    py lidar_pose_detect.py --bag <file.db3> --db3-msg-stride 5 --voxel 0.10 --mad-k 2.5
    py lidar_pose_detect.py --bag <cartella_bag> --min-duration 45 --csv-out lidar_pose.csv
"""

import argparse
import csv
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import Reader
except ImportError:
    sys.exit("Serve il pacchetto rosbags:  py -m pip install rosbags")

_CUSTOM_MSG_TYPE = "livox_ros_driver2/msg/CustomMsg"
_POINT_CLOUD2_TYPE = "sensor_msgs/msg/PointCloud2"
_CUSTOM_POINT_STRIDE = 20  # CustomPoint: u32 offset_time, f32 x/y/z, 3x u8 -> 20


def resolve_bag(path):
    """Cartella rosbag2 (con metadata.yaml) o file .db3. Se metadata.yaml manca
    (recorder chiuso male) e la cartella ha un solo .db3, usa quello."""
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


def _custom_msg_xyz(buf, point_stride):
    """Byte CDR di un livox CustomMsg -> (N,3) float32 xyz."""
    pos = 4                                   # header di encapsulation
    pos += 4 + 4                              # header.stamp sec + nanosec
    slen = struct.unpack_from("<I", buf, pos)[0]; pos += 4 + slen  # frame_id
    rem = (pos - 4) % 8                       # allineamento u64 (timebase)
    if rem:
        pos += 8 - rem
    pos += 8                                  # timebase
    point_num = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    pos += 1 + 3                              # lidar_id + rsvd[3]
    seq_len = struct.unpack_from("<I", buf, pos)[0]; pos += 4
    n = max(point_num, seq_len)
    data = buf[pos:]
    need = n * _CUSTOM_POINT_STRIDE
    if len(data) < need:                      # il CDR non padda l'ultimo elemento
        data = data + b"\x00" * (need - len(data))
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4", "<f4", "<f4"],
                   "offsets": [4, 8, 12], "itemsize": _CUSTOM_POINT_STRIDE})
    arr = np.frombuffer(data, dtype=dt, count=n)[::point_stride]
    p = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    return p[np.isfinite(p).all(axis=1)]


def _pointcloud2_xyz(buf, point_stride):
    """Byte CDR di un sensor_msgs/PointCloud2 (x/y/z f32) -> (N,3) float32."""
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
    """Insieme delle chiavi (int64) dei voxel occupati."""
    q = np.floor(points / voxel).astype(np.int64) + (1 << 19)
    keys = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    return set(np.unique(keys).tolist())


def lidar_profile(bag, topic, msg_stride, point_stride, voxel, time_origin):
    """Profilo di Jaccard tra scan campionati consecutivi, letto in streaming.

    Restituisce (value_times, values, all_offsets, info): value_times[k] e'
    l'offset (s) dello scan confrontato da values[k]; all_offsets sono gli
    offset di TUTTI i messaggi del topic."""
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
    """Segmentazione con debounce (da detect_board_poses.py).

    Restituisce (poses, transitions): intervalli di indici in `values`."""
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


def print_table(header, rows):
    cells = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in header]
    for r in cells:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"CSV salvato: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Rileva le pose statiche della board in un bag ROS2 Livox."
    )
    ap.add_argument("--bag", required=True,
                    help="Cartella del bag ROS2 (con metadata.yaml) o file .db3")
    ap.add_argument("--topic", default="/livox/lidar",
                    help="Topic LiDAR (default /livox/lidar)")
    ap.add_argument("--voxel", type=float, default=0.10,
                    help="Lato del voxel in metri (default 0.10)")
    ap.add_argument("--mad-k", type=float, default=2.5,
                    help="Fattore K della soglia outlier mediana + K*MAD (default 2.5)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Soglia di Jaccard fissa al posto di quella automatica")
    ap.add_argument("--db3-msg-stride", type=int, default=5,
                    help="Analizza uno scan ogni N (default 5)")
    ap.add_argument("--db3-point-stride", type=int, default=4,
                    help="Usa un punto ogni N per scan (default 4)")
    ap.add_argument("--min-move-frames", type=int, default=1,
                    help="Scan campionati consecutivi sopra soglia per contare come "
                         "spostamento (default 1: a stride 5 uno spostamento dura "
                         "~1 campione)")
    ap.add_argument("--min-pose-frames", type=int, default=None,
                    help="Scan campionati stabili minimi per una posa (default ~10 s)")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="Tiene solo le pose lunghe almeno N secondi")
    ap.add_argument("--time-origin", choices=("bag", "topic"), default="bag",
                    help="Zero degli offset: inizio del bag o primo messaggio del "
                         "topic (default bag)")
    ap.add_argument("--csv-out", default=None, metavar="CSV",
                    help="Salva anche la tabella delle pose in CSV")
    args = ap.parse_args()
    if args.db3_msg_stride < 1 or args.db3_point_stride < 1:
        ap.error("--db3-msg-stride e --db3-point-stride devono essere >= 1")
    if args.voxel <= 0:
        ap.error("--voxel deve essere > 0")

    bag = resolve_bag(args.bag)
    print(f"Bag: {bag}")
    value_times, values, all_off, info = lidar_profile(
        bag, args.topic, args.db3_msg_stride, args.db3_point_stride, args.voxel,
        args.time_origin)

    if info["read"] != info["declared"]:
        print(f"ATTENZIONE: letti {info['read']} messaggi, i metadati del bag "
              f"ne dichiarano {info['declared']}.")
    if info["bad"]:
        print(f"ATTENZIONE: {info['bad']} scan non decodificabili, saltati.")
    if values.size < 2:
        sys.exit("Troppo pochi scan campionati per rilevare le pose.")

    dur = value_times[-1] - value_times[0]
    eff_rate = (value_times.size - 1) / dur if dur > 0 else 0.0
    min_pose = (args.min_pose_frames if args.min_pose_frames is not None
                else max(4, round(10.0 * eff_rate)))
    print(f"Primo messaggio del topic: {info['topic_start_utc']:%Y-%m-%d %H:%M:%S.%f} UTC "
          f"({info['topic_start_offset']:+.2f} s dall'inizio del bag)")
    print(f"Durata: {dur:.0f} s ({dur / 60:.1f} min)   campionamento effettivo "
          f"{eff_rate:.2f} Hz   min-pose-frames {min_pose} (~{min_pose / max(eff_rate, 1e-6):.0f} s)")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = (args.threshold if args.threshold is not None
                 else median + args.mad_k * (mad if mad > 1e-6 else 1.0))
    print(f"Jaccard: mediana={median:.3f}  MAD={mad:.3f}  soglia={threshold:.3f}"
          + ("  (fissa)" if args.threshold is not None else f"  (mediana+{args.mad_k:g}*MAD)"))
    print()

    poses, transitions = segment(values, threshold, args.min_move_frames, min_pose)
    spans = [(value_times[a], value_times[min(b, value_times.size) - 1]) for a, b in poses]
    dropped = 0
    if args.min_duration is not None:
        kept = [(s, e) for s, e in spans if e - s >= args.min_duration]
        dropped = len(spans) - len(kept)
        spans = kept

    header = ["posa", "offset_inizio_s", "offset_fine_s", "durata_s", "n_messaggi"]
    rows = []
    for k, (s, e) in enumerate(spans, start=1):
        n_msg = int(np.searchsorted(all_off, e, side="right")
                    - np.searchsorted(all_off, s, side="left"))
        rows.append([k, f"{s:.1f}", f"{e:.1f}", f"{e - s:.1f}", n_msg])

    print(f"POSE: {len(rows)}   (spostamenti: {len(transitions)}"
          + (f"; {dropped} pose piu' corte di {args.min_duration:g} s scartate" if dropped else "")
          + f")   offset dall'inizio del {'bag' if args.time_origin == 'bag' else 'topic'}")
    print_table(header, rows)
    print()
    print("Se il numero di pose non torna: --min-duration scarta i segmenti spuri, "
          "--mad-k piu' alto = meno spostamenti, --db3-msg-stride piu' basso = "
          "risoluzione piu' fine.")

    if args.csv_out:
        print()
        write_csv(args.csv_out, header, rows)


if __name__ == "__main__":
    main()
