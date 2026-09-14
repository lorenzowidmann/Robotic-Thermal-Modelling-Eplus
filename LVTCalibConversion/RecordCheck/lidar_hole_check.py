"""
Check al volo: il LiDAR Livox (HAP) risolve i 4 fori del target di
calibrazione? Legge direttamente il bag ROS2 grezzo (.db3, topic /livox/lidar,
livox_ros_driver2/msg/CustomMsg), senza conversione ROS1, Docker o lvt2calib.

Passi:
1. Legge SOLO i messaggi del topic nella finestra [--start, --start + --seconds]
   secondi dall'inizio del bag (default: primi 60 s). La finestra va a
   Reader.messages(start, stop), che
   sul .db3 diventa una query SQL filtrata per timestamp, quindi il resto del
   file non viene letto. Il payload CDR e' decodificato a struct (solo x/y/z,
   stesso parser di detect_board_poses.py), senza typestore.
2. Accumula tutti gli scan in un'unica nuvola (assunzione: target statico
   nella finestra). I punti a ~0 m (nessun ritorno) vengono scartati.
3. Ritaglio opzionale --roi xmin xmax ymin ymax zmin zmax (frame del sensore),
   poi RANSAC del piano dominante (open3d segment_plane) su una copia
   sotto-campionata; gli inlier sono poi ricalcolati su tutta la nuvola.
4. Proiezione degli inlier sulla base 2D del piano, come fit_oriented_rect in
   OcTreeVoxel/fit_closed_planes.py: normale quasi verticale -> u,v = X/Y del
   sensore proiettati; altrimenti u = cross(normale, up), v = up proiettato
   sul piano (qui ortogonalizzato, cosi' i diametri non si accorciano se la
   board e' inclinata).
5. Occupancy grid 2D (--grid-res). I vuoti connessi al bordo della griglia
   sono esterno alla board; i vuoti chiusi sono candidati foro.
6. Candidato foro = vuoto chiuso con diametro equivalente
   2*sqrt(area/pi) in [--min-hole-diam, --max-hole-diam].
7. Verdetto: >= 4 fori plausibili -> i fori si vedono.

Frame del sensore: si assume Z in alto (LiDAR montato dritto).
Lo script legge solo: il bag non viene copiato, spostato o modificato.

Uso tipico (venv C:\\venvs\\planefit, ha open3d e rosbags):
    C:\\venvs\\planefit\\Scripts\\python.exe lidar_hole_check.py --bag <cartella_bag>
    ... --bag <cartella_bag> --seconds 30 --roi 1 4 -1.5 1.5 -1 1.5
    ... --bag <file.db3> --grid-res 0.03 --min-cell-points 1
    ... --bag <cartella_bag> --start 111.9 --seconds 60 --roi ...   (posa 2)
"""

import argparse
import struct
import sys
from collections import deque

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("Serve il pacchetto open3d (venv C:\\venvs\\planefit).")

# era lidar_pose_detect.py, consolidato in detect_board_poses.py
from detect_board_poses import (_CUSTOM_MSG_TYPE, _POINT_CLOUD2_TYPE,
                                _custom_msg_xyz, _pointcloud2_xyz, resolve_bag,
                                _require_reader)

Reader = _require_reader()

_RANSAC_MAX_POINTS = 2_000_000   # il RANSAC gira su un sotto-campione
_MAX_GRID_CELLS = 20_000_000


def read_window(bag, topic, start, seconds, point_stride, min_range):
    """Nuvola (N,3) float32 dei messaggi del topic in [start, start + seconds]
    secondi dall'inizio del bag."""
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
        bag_start = reader.start_time
        start_ns = bag_start + int(start * 1e9)
        stop_ns = start_ns + int(seconds * 1e9)
        print(f"Topic: {topic} [{msgtype}]   finestra: da {start:g} s a "
              f"{start + seconds:g} s dall'inizio del bag")

        frames, times = [], []
        bad = n_raw = 0
        for _, t_ns, raw in reader.messages(connections=conns, start=start_ns, stop=stop_ns):
            if t_ns >= stop_ns:                   # sicurezza: fuori finestra, stop
                break
            times.append(t_ns)
            try:
                p = parse(bytes(raw), point_stride)
            except (struct.error, ValueError, KeyError):
                bad += 1
                continue
            n_raw += len(p)
            frames.append(p[np.einsum("ij,ij->i", p, p) > min_range ** 2])
            if len(times) % 20 == 0:
                print(f"\r  {len(times)} messaggi letti  "
                      f"(t = {(t_ns - bag_start) / 1e9:.1f} s)", end="", flush=True)
        print()

    if not frames:
        sys.exit(f"Nessun messaggio {topic} decodificabile tra {start:g} s e "
                 f"{start + seconds:g} s.")
    xyz = np.concatenate(frames)
    info = {
        "msgs": len(times),
        "bad": bad,
        "raw": n_raw,
        "t_first": (times[0] - bag_start) / 1e9,
        "t_last": (times[-1] - bag_start) / 1e9,
    }
    return xyz, info


def crop_roi(xyz, roi):
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    m = ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) & (xyz[:, 1] >= ymin)
         & (xyz[:, 1] <= ymax) & (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax))
    return xyz[m]


def fit_dominant_plane(xyz, distance_threshold, num_iterations, voxel):
    """RANSAC su un sotto-campione, poi inlier = tutti i punti entro soglia.
    Restituisce (normale unitaria, d, maschera inlier)."""
    step = max(1, len(xyz) // _RANSAC_MAX_POINTS)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[::step].astype(np.float64))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    model, _ = pcd.segment_plane(distance_threshold=distance_threshold,
                                 ransac_n=3, num_iterations=num_iterations)
    n = np.asarray(model[:3], dtype=np.float64)
    norm = np.linalg.norm(n)
    n, d = n / norm, model[3] / norm
    mask = np.abs(xyz @ n + d) < distance_threshold
    return n, d, mask


def plane_basis(n):
    """Base 2D (u, v) sul piano, come fit_oriented_rect (fit_closed_planes.py)."""
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(n[2]) > 0.5:                           # pavimento/soffitto: X/Y proiettati
        u = np.array([1.0, 0.0, 0.0]) - n[0] * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
    else:                                         # parete/board: u orizzontale, v ~ up
        u = np.cross(n, world_up)
        u /= np.linalg.norm(u)
        v = np.cross(u, n)                        # = up proiettato sul piano
    return u, v


def label_empty(empty):
    """Componenti 4-connesse delle celle vuote. Restituisce (labels piatte,
    lista di (celle, tocca_bordo))."""
    h, w = empty.shape
    flat = empty.ravel()
    labels = np.full(h * w, -1, dtype=np.int32)
    comps = []
    for seed in np.flatnonzero(flat):
        if labels[seed] >= 0:
            continue
        lab = len(comps)
        labels[seed] = lab
        cells, border = [], False
        q = deque([int(seed)])
        while q:
            c = q.popleft()
            cells.append(c)
            r, k = divmod(c, w)
            if r == 0 or k == 0 or r == h - 1 or k == w - 1:
                border = True
            for nb, ok in ((c - w, r > 0), (c + w, r < h - 1),
                           (c - 1, k > 0), (c + 1, k < w - 1)):
                if ok and flat[nb] and labels[nb] < 0:
                    labels[nb] = lab
                    q.append(nb)
        comps.append((np.asarray(cells), border))
    return labels, comps


def find_holes(uv, res, min_cell_points):
    """Occupancy grid + vuoti chiusi. Restituisce (lista vuoti chiusi, shape)."""
    lo = uv.min(axis=0)
    ij = np.floor((uv - lo) / res).astype(np.int64) + 1    # 1 cella di bordo vuota
    shape = tuple(int(s) for s in ij.max(axis=0) + 2)
    if shape[0] * shape[1] > _MAX_GRID_CELLS:
        sys.exit(f"Griglia {shape[0]}x{shape[1]} troppo grande: il piano fittato e' "
                 "enorme (muro/pavimento?). Usare --roi attorno alla board o "
                 "--grid-res piu' grande.")
    counts = np.bincount(ij[:, 0] * shape[1] + ij[:, 1],
                         minlength=shape[0] * shape[1]).reshape(shape)
    occupied = counts >= min_cell_points
    _, comps = label_empty(~occupied)

    voids = []
    for cells, border in comps:
        if border:
            continue                               # esterno alla board
        r, k = np.divmod(cells, shape[1])
        area = cells.size * res * res
        voids.append({
            "cells": int(cells.size),
            "area": area,
            "diam": 2.0 * np.sqrt(area / np.pi),
            "u": float(lo[0] + (r.mean() - 1 + 0.5) * res),
            "v": float(lo[1] + (k.mean() - 1 + 0.5) * res),
            "w": float((r.max() - r.min() + 1) * res),
            "h": float((k.max() - k.min() + 1) * res),
        })
    return voids, shape, int(occupied.sum())


def main():
    ap = argparse.ArgumentParser(
        description="Check al volo: il LiDAR vede i 4 fori del target? (bag ROS2 grezzo)"
    )
    ap.add_argument("--bag", required=True,
                    help="Cartella del bag ROS2 (con metadata.yaml) o file .db3")
    ap.add_argument("--topic", default="/livox/lidar",
                    help="Topic LiDAR (default /livox/lidar)")
    ap.add_argument("--start", type=float, default=0.0,
                    help="Inizio della finestra in secondi dall'inizio del bag (default 0): "
                         "per una posa successiva usare offset_inizio_s di detect_board_poses.py --bag")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="Durata della finestra letta, da --start (default 60)")
    ap.add_argument("--point-stride", type=int, default=1,
                    help="Usa un punto ogni N per scan (default 1 = tutti)")
    ap.add_argument("--min-range", type=float, default=0.1,
                    help="Scarta i punti piu' vicini di N m (nessun ritorno, default 0.1)")
    ap.add_argument("--roi", type=float, nargs=6, default=None,
                    metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                    help="Ritaglio in metri nel frame del sensore prima del RANSAC")
    ap.add_argument("--distance-threshold", type=float, default=0.02,
                    help="Distanza inlier del RANSAC in metri (default 0.02)")
    ap.add_argument("--num-iterations", type=int, default=1000,
                    help="Iterazioni RANSAC (default 1000)")
    ap.add_argument("--ransac-voxel", type=float, default=0.01,
                    help="Voxel del sotto-campione usato solo per il RANSAC (default 0.01, 0 = off)")
    ap.add_argument("--grid-res", type=float, default=0.02,
                    help="Lato cella dell'occupancy grid in metri (default 0.02)")
    ap.add_argument("--min-cell-points", type=int, default=3,
                    help="Punti minimi perche' una cella conti come piena: filtra i "
                         "flying pixel isolati dentro i fori (default 3)")
    ap.add_argument("--min-hole-diam", type=float, default=0.08,
                    help="Diametro equivalente minimo di un foro in metri (default 0.08)")
    ap.add_argument("--max-hole-diam", type=float, default=0.20,
                    help="Diametro equivalente massimo di un foro in metri (default 0.20)")
    ap.add_argument("--expected-holes", type=int, default=4,
                    help="Fori attesi sul target (default 4)")
    args = ap.parse_args()
    if args.seconds <= 0 or args.start < 0:
        ap.error("--seconds deve essere > 0 e --start >= 0")
    if args.point_stride < 1 or args.min_cell_points < 1:
        ap.error("--point-stride e --min-cell-points devono essere >= 1")
    if args.grid_res <= 0 or args.distance_threshold <= 0:
        ap.error("--grid-res e --distance-threshold devono essere > 0")
    if args.min_hole_diam >= args.max_hole_diam:
        ap.error("--min-hole-diam deve essere < --max-hole-diam")

    bag = resolve_bag(args.bag)
    print(f"Bag: {bag}")
    xyz, info = read_window(bag, args.topic, args.start, args.seconds, args.point_stride,
                            args.min_range)
    print(f"Messaggi letti: {info['msgs']}  (da {info['t_first']:.2f} s a {info['t_last']:.2f} s)"
          + (f"   non decodificabili: {info['bad']}" if info["bad"] else ""))
    print(f"Punti letti: {info['raw']}   validi (> {args.min_range:g} m): {len(xyz)}")

    if args.roi is not None:
        xyz = crop_roi(xyz, args.roi)
        print(f"Dopo --roi {' '.join(f'{r:g}' for r in args.roi)}: {len(xyz)} punti")
    if len(xyz) < 100:
        sys.exit("Troppi pochi punti per il RANSAC: controllare --roi / --seconds.")

    n, d, mask = fit_dominant_plane(xyz, args.distance_threshold, args.num_iterations,
                                    args.ransac_voxel)
    inl = xyz[mask].astype(np.float64)
    centroid = inl.mean(axis=0)
    u, v = plane_basis(n)
    uv = np.column_stack([(inl - centroid) @ u, (inl - centroid) @ v])
    ext = uv.max(axis=0) - uv.min(axis=0)
    print()
    print(f"Piano dominante: normale {np.round(n, 3)}  d = {d:.3f} m  "
          f"distanza dal sensore {abs(d):.2f} m")
    print(f"Inlier del piano: {len(inl)} ({100.0 * len(inl) / len(xyz):.1f}%)   "
          f"estensione u x v: {ext[0]:.2f} x {ext[1]:.2f} m   "
          f"centroide {np.round(centroid, 3)}")

    voids, shape, n_occ = find_holes(uv, args.grid_res, args.min_cell_points)
    holes = [h for h in voids if args.min_hole_diam <= h["diam"] <= args.max_hole_diam]
    small = sum(h["diam"] < args.min_hole_diam for h in voids)
    big = [h for h in voids if h["diam"] > args.max_hole_diam]
    print(f"Griglia {shape[0]}x{shape[1]} celle da {args.grid_res:g} m, piene: {n_occ}   "
          f"vuoti chiusi: {len(voids)} ({small} troppo piccoli, {len(big)} troppo grandi)")
    print()

    holes.sort(key=lambda h: (-h["v"], h["u"]))    # dall'alto a sinistra
    print(f"CANDIDATI FORO: {len(holes)}   (diametro equivalente in "
          f"[{args.min_hole_diam:g}, {args.max_hole_diam:g}] m, u,v rispetto al centroide)")
    if holes:
        print(f"  {'#':>2}  {'diam_m':>7}  {'u_m':>7}  {'v_m':>7}  {'bbox_m':>11}  {'celle':>5}")
        for k, h in enumerate(holes, start=1):
            print(f"  {k:>2}  {h['diam']:7.3f}  {h['u']:+7.3f}  {h['v']:+7.3f}  "
                  f"{h['w']:.2f} x {h['h']:.2f}  {h['cells']:>5}")
    for h in sorted(big, key=lambda h: -h["diam"])[:3]:
        print(f"  (scartato, troppo grande: diam {h['diam']:.3f} m a u={h['u']:+.3f} v={h['v']:+.3f})")
    print()

    exp = args.expected_holes
    if len(holes) >= exp:
        print(f"VERDETTO: {len(holes)} fori plausibili trovati (>= {exp}) -> i fori si vedono.")
        if len(holes) > exp:
            print(f"  Nota: piu' di {exp} candidati, alcuni possono essere vuoti spuri "
                  "(ombre/occlusioni): controllare diametri e posizioni.")
    else:
        print(f"VERDETTO: solo {len(holes)} fori plausibili su {exp} -> "
              f"ne mancano {exp - len(holes)}.")
        if ext.max() > 1.5:
            print(f"  Il piano fittato misura {ext[0]:.2f} x {ext[1]:.2f} m, molto piu' "
                  "della board: probabilmente e' un muro/pavimento. Ripetere con "
                  "--roi xmin xmax ymin ymax zmin zmax attorno alla board.")
        else:
            print("  Se il piano e' la board: provare --grid-res piu' grande (nuvola rada), "
                  "--min-cell-points piu' basso, o --seconds maggiore.")


if __name__ == "__main__":
    main()
