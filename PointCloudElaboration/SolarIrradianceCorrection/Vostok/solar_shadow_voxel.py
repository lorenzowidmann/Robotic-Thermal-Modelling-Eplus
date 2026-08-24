"""Shadow-aware sunlit/occluded mask per surface voxel, via VOSTOK.

Integrates VOSTOK (github.com/3dgeo-heidelberg/vostok, GPL-3.0 -- see
../../../../vostok, built from source, sibling to Easy3D) to get a proper
raycast/octree shadow mask for OcTreeVoxel's surface voxels at Session 9's
actual capture time, instead of trusting VOSTOK's own built-in clear-sky
irradiance model (a cruder Linke-turbidity-fixed-to-3 approximation than the
already-validated pvlib/Erbs/Perez pipeline in
OpenStudioModel/sun_incidence.py + parse_arpav.py).

This script stops at producing the shadow mask -- it does NOT multiply by
irradiance magnitude to get a sol-air correction. That combination is a
separate follow-up once this output exists and has been sanity-checked
(e.g. does the mask show the corridor's own south-facing windows plausibly
self-shadowing at 18:12 in late July?).

Pipeline:
  1. Load OcTreeVoxel/transform.json's rotation/translation and
     planes_aligned.json's 6 closed-box face normals (both already in the
     *aligned* building frame -- see OcTreeVoxel/README.md). Derive an
     additional rotation about Z from --north-bearing-deg
     (north_alignment_rotation) so the horizontal axes become true
     North/East-aligned: VOSTOK's IrradianceCalc.cpp assumes +Y = North,
     +X = East, Z = up (confirmed directly by reading getIncidenceAngle's
     sun-vector construction: snX=sin(azimuth)*sin(zenith),
     snY=cos(azimuth)*sin(zenith), with azimuth measured from North -- azimuth
     0 (North) gives a unit vector in +Y, azimuth 90 (East) gives +X). This
     is not optional -- VOSTOK has no other way to know which way is North.
  2. Reload the raw point cloud from the same filtered bag OcTreeVoxel
     used, apply transform.json's rotation+translation, then the new
     north-alignment rotation. Write as VOSTOK's shadow-point file
     ("x y z" per line, no normals needed -- shadow points only cast
     shadows, they aren't evaluated for irradiance).
  3. From voxels.npz, keep only occupied voxels within --face-tolerance of
     one of the 6 closed-box faces (in the *aligned* frame, matching
     planes_aligned.json's own frame directly -- the north rotation is
     applied only afterward, to the kept voxels' centers/normals, for the
     query file). Each kept voxel gets the (north-aligned, outward-pointing)
     normal of whichever face it's nearest to; voxels that don't clear any
     face within tolerance have no single meaningful surface orientation
     and are dropped. Write as VOSTOK's query-point file
     ("x y z nx ny nz" per line -- normals are a hard VOSTOK requirement
     for query points).
  4. Generate a .sol config file (line-by-line format straight from
     VOSTOK's own README/ProjectConfig.cpp::loadFromFile, verified by
     reading the source, not just trusting the README prose): shadow file
     from step 2, query file from step 3, voxel_size from voxels.npz
     (VOSTOK's own internal shadow octree resolution), Padova lat/lon,
     timezone 2 (CEST -- SOLPOS's convention is "east positive", confirmed
     in solpos00.h; July is under daylight saving, NOT the standard-time
     UTC+1), year 2026, day-start = day-end = day-of-year for 30 July 2026
     (computed via Python's datetime, not hand-computed), day-step 1,
     minute-step 5, shadow mode 2 (output shadow clouds -- the actual
     deliverable; VOSTOK's cumulative "total irradiation" column is
     ignored/discarded here, see module docstring above).
  5. Run vostok.exe on the generated .sol file.
  6. VOSTOK's day loop only evaluates minutes from that day's sunrise to
     sunset (not midnight-aligned), so a 5-minute step doesn't necessarily
     land exactly on 18:12 -- parse whichever
     shadow_clouds/<year>_<day>_<hour>-<minute>_shadow.txt is closest to
     the requested capture time. Row order is guaranteed to match the query
     file's order (VOSTOK forces single-threaded, sequential-cursor writes
     whenever shadow-cloud output is requested -- confirmed by reading
     main.cpp, not assumed), so rows are matched back to their source voxel
     by position, with the echoed-back x/y/z used only as a sanity
     cross-check. Writes voxel center (aligned frame, NOT north-rotated --
     consistent with the rest of the OcTreeVoxel
     pipeline), face id, outward normal (aligned frame), and illuminated
     (bool) per surface voxel.

Usage:
    python solar_shadow_voxel.py --north-bearing-deg <measured value>
        [--voxels ../../OcTreeVoxel/OcTreeVoxel_out/voxels.npz]
        [--transform ../../OcTreeVoxel/OcTreeVoxel_out/transform.json]
        [--planes-aligned ../../OcTreeVoxel/OcTreeVoxel_out/planes_aligned.json]
        [--bag <rosbag2_folder>] [--capture-datetime 2026-07-30T18:12:00]
        [--vostok-exe ...\\vostok\\build\\vostok.exe]
        [--face-tolerance-m <default: 0.5 * voxel_size>]
        [--minute-step 5]
        [--workdir solar_shadow_voxel_out]
        [--out solar_shadow_voxel_out/shadow_mask.json]

Every generated file (VOSTOK's inputs, VOSTOK's own shadow_clouds/ output,
and the final shadow_mask.json) lands in solar_shadow_voxel_out/ by default.

Prerequisite: ../../OcTreeVoxel must have been run first -- this script consumes
the voxels.npz / transform.json / planes_aligned.json it writes to its own
OcTreeVoxel_out/ folder, none of which are committed (see README.md
'Prerequisites').

--north-bearing-deg is REQUIRED, no default, fails loudly if omitted -- the
project's own working notes flag this as unresolved ("resolve south wall /
spatial mask"). It is the true compass bearing (degrees from true North,
clockwise) of the ALIGNED building frame's +X axis (aligned_octree.py's
"forward" -- the dominant wall pair's direction, not +Y/"right"). Note this
is NOT the same number as EmissivityCalculation/voxel_solar_ns.py's
--north-offset-deg default of 193 -- that is the bearing of the RAW,
pre-alignment SLAM +Y axis, and OcTreeVoxel/aligned_octree.py's
compute_building_frame applies its own extra rotation (a measured ~2.6 deg
yaw correction plus whatever small tilt correction) on top of the raw SLAM
frame, so the aligned frame's own axis bearing is close to but not exactly
193 (or 193-90=103, translating axes) -- reusing that number directly here
without re-deriving it would silently be wrong by roughly that residual
rotation.

Venv: needs numpy + rosbags (rosbags for load_merged_cloud, same as
OcTreeVoxel) -- C:\\venvs\\planefit already has both.
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

HERE = Path(__file__).resolve().parent
OCTREE_VOXEL_DIR = HERE.parent.parent / "OcTreeVoxel" / "OcTreeVoxel_out"
OUT_DIR = HERE / "solar_shadow_voxel_out"

DEFAULT_VOXELS = OCTREE_VOXEL_DIR / "voxels.npz"
DEFAULT_TRANSFORM = OCTREE_VOXEL_DIR / "transform.json"
DEFAULT_PLANES_ALIGNED = OCTREE_VOXEL_DIR / "planes_aligned.json"
DEFAULT_VOSTOK_EXE = Path(r"C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\vostok\build\vostok.exe")

# DII, Via Gradenigo 6/a, Padova.
DEFAULT_LAT = 45.411150270162175
DEFAULT_LON = 11.891977961725079
DEFAULT_CAPTURE_DATETIME = "2026-07-30T18:12:00"  # local time, CEST
DEFAULT_TIMEZONE = 2  # CEST = UTC+2; SOLPOS convention is east-positive (solpos00.h)
DEFAULT_MINUTE_STEP = 5


# ---------------------------------------------------------------------------
# Point cloud loading -- same convention as OcTreeVoxel/fit_closed_planes.py
# and TemperatureToVoxel/temperature_to_voxel.py (self-contained, no
# cross-import): raw, unaligned bag points in the SLAM/map frame.
# ---------------------------------------------------------------------------

def read_pointcloud2(msg):
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_merged_cloud(bag, topic, store):
    typestore = get_typestore(Stores[store])
    frames = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise SystemExit(f"Topic {topic!r} not found. Available: {topics}")
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            frames.append(read_pointcloud2(msg))
    xyz = np.vstack(frames)
    print(f"{len(frames)} frame(s), {len(xyz)} points")
    return xyz.astype(np.float64)


# ---------------------------------------------------------------------------
# North alignment.
# ---------------------------------------------------------------------------

def north_alignment_rotation(north_bearing_deg):
    """Rotation about Z that takes the aligned frame's +X axis (whose true
    compass bearing is `north_bearing_deg`) to VOSTOK's convention:
    +Y = North (bearing 0), +X = East (bearing 90).

    Derivation (row-vector convention throughout this project, v_new =
    v_old @ R.T): a horizontal direction at bearing beta has target-frame
    components (sin(beta), cos(beta), 0) -- matches VOSTOK's own
    IrradianceCalc.cpp::getIncidenceAngle sun-vector construction exactly
    (snX=sin(azimuth)*sin(zenith), snY=cos(azimuth)*sin(zenith)), confirming
    the convention independently. A standard Z-rotation by theta maps
    (1,0,0) -> (cos(theta), sin(theta), 0). Setting
    (cos(theta), sin(theta)) = (sin(beta), cos(beta)) gives
    theta = 90 - beta.
    """
    theta = np.radians(90.0 - north_bearing_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def compose_transform(R, t, R_extra):
    """Chain (aligned = p @ R.T + t) then (north = aligned @ R_extra.T):
    north = p @ (R_extra @ R).T + (t @ R_extra.T). Returns (total_R, total_t)
    in the same row-vector convention (p @ total_R.T + total_t)."""
    total_R = R_extra @ R
    total_t = t @ R_extra.T
    return total_R, total_t


# ---------------------------------------------------------------------------
# Surface-voxel <-> closed-box-face matching (aligned frame, pre-north-rotation).
# ---------------------------------------------------------------------------

def face_bounds(plane):
    """Which axis this (axis-aligned, aligned-frame) face is fixed on, its
    fixed coordinate, and the [lo, hi] bounds on the other two axes (from
    corners_3d) -- used instead of the plane's basis_u/basis_v projection
    since every face is exactly axis-aligned here, so a plain per-axis
    bounding range is simpler and equally correct."""
    corners = np.asarray(plane["corners_3d"], dtype=float)
    normal = np.asarray(plane["normal"], dtype=float)
    axis = int(np.argmax(np.abs(normal)))
    fixed = float(corners[:, axis].mean())
    other_axes = [a for a in (0, 1, 2) if a != axis]
    lo = corners[:, other_axes].min(axis=0)
    hi = corners[:, other_axes].max(axis=0)
    return axis, fixed, other_axes, lo, hi


def outward_normal(normal, face_center, box_center):
    """Force the normal to point away from the box's own centroid.

    fit_planes.py/close_geometry don't guarantee consistent normal
    orientation (a plane can point into or out of the room) -- same issue
    EmissivityCalculation/voxel_solar_ns.py already ran into and fixed the
    same way ("every plane is re-oriented to point away from the scene
    interior"). Solar/sol-air analysis needs the *exterior*-facing normal.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    if np.dot(normal, np.asarray(face_center, dtype=float) - np.asarray(box_center, dtype=float)) < 0:
        normal = -normal
    return normal


def assign_surface_voxels(centers, planes, tolerance_m):
    """Which occupied voxels sit on the closed box's surface, and which
    face (by nearest perpendicular distance, among faces whose in-plane
    bounds the voxel also falls within) each belongs to.

    Returns (voxel_row_idx, face_id, outward_normal_per_voxel) -- all in
    the ALIGNED frame (not north-rotated). voxel_row_idx indexes into the
    original `centers` (and therefore voxels.npz's counts/etc) array.
    """
    box_center = np.mean([p["center_3d"] for p in planes], axis=0)
    n = len(centers)
    best_dist = np.full(n, np.inf)
    best_face = np.full(n, -1, dtype=np.int64)
    face_outward_normals = []

    for face_id, p in enumerate(planes):
        axis, fixed, other_axes, lo, hi = face_bounds(p)
        face_outward_normals.append(outward_normal(p["normal"], p["center_3d"], box_center))

        dist = np.abs(centers[:, axis] - fixed)
        a0, a1 = other_axes
        in_bounds = (
            (centers[:, a0] >= lo[0] - tolerance_m) & (centers[:, a0] <= hi[0] + tolerance_m)
            & (centers[:, a1] >= lo[1] - tolerance_m) & (centers[:, a1] <= hi[1] + tolerance_m)
        )
        candidate = (dist <= tolerance_m) & in_bounds & (dist < best_dist)
        best_dist[candidate] = dist[candidate]
        best_face[candidate] = face_id

    keep = best_face >= 0
    voxel_row_idx = np.where(keep)[0]
    face_id_kept = best_face[keep]
    normals_kept = np.asarray(face_outward_normals)[face_id_kept]
    return voxel_row_idx, face_id_kept, normals_kept


# ---------------------------------------------------------------------------
# VOSTOK .sol file + run + parse.
# ---------------------------------------------------------------------------

def write_sol_file(sol_path, shadow_file, query_file, voxel_size, lat, lon, timezone,
                    year, day_of_year, day_step, minute_step, output_file, min_sun_angle=-99.9):
    """Line-by-line format per VOSTOK's ProjectConfig.cpp::loadFromFile
    (verified by reading the source, matches the README's documented
    format exactly)."""
    lines = [
        str(shadow_file), "x y z",
        str(query_file), "x y z nx ny nz",
        f"{voxel_size}",
        f"{lat}",
        f"{lon}",
        f"{timezone}",
        f"{year}",
        f"{day_of_year}",
        f"{day_of_year}",
        f"{day_step}",
        f"{minute_step}",
        "2",  # shadow mode: output shadow clouds
        str(output_file),
        "1",  # multi-threading (irrelevant for shadow-cloud output -- VOSTOK
              # forces single-threaded whenever computeShadows>1, see main.cpp)
        f"{min_sun_angle}",
    ]
    sol_path.write_text("\n".join(lines) + "\n")


def run_vostok(vostok_exe, sol_path, workdir):
    print(f"running {vostok_exe} {sol_path.name} (cwd={workdir})")
    result = subprocess.run(
        [str(vostok_exe), str(sol_path.name)], cwd=str(workdir),
        capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(f"vostok.exe exited with code {result.returncode}")


def find_closest_shadow_file(shadow_clouds_dir, year, day_of_year, target_hour, target_minute):
    target = target_hour * 60 + target_minute
    best_path, best_diff = None, None
    prefix = f"{year}_{day_of_year:03d}_"
    for path in sorted(shadow_clouds_dir.glob(f"{prefix}*_shadow.txt")):
        # filename: <year>_<day>_<hour>-<minute>_shadow.txt
        hm = path.stem[len(prefix):-len("_shadow")]
        hour_str, minute_str = hm.split("-")
        minutes = int(hour_str) * 60 + int(minute_str)
        diff = abs(minutes - target)
        if best_diff is None or diff < best_diff:
            best_diff, best_path = diff, path
    if best_path is None:
        raise SystemExit(f"no shadow_clouds/{prefix}*_shadow.txt files found in {shadow_clouds_dir}")
    print(f"closest shadow file to {target_hour:02d}:{target_minute:02d}: "
          f"{best_path.name} ({best_diff} min away)")
    return best_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--north-bearing-deg", type=float, required=True,
                    help="REQUIRED, no default. True compass bearing (deg from "
                         "North, clockwise) of the ALIGNED building frame's +X "
                         "axis. See module docstring for why this can't be "
                         "guessed or defaulted, and why voxel_solar_ns.py's "
                         "--north-offset-deg=193 (a different, RAW-SLAM-frame "
                         "number) isn't directly reusable here.")
    ap.add_argument("--voxels", type=Path, default=DEFAULT_VOXELS)
    ap.add_argument("--transform", type=Path, default=DEFAULT_TRANSFORM)
    ap.add_argument("--planes-aligned", type=Path, default=DEFAULT_PLANES_ALIGNED)
    ap.add_argument("--bag", type=Path, default=None,
                    help="rosbag2 folder (default: the 'bag' field recorded in --transform)")
    ap.add_argument("--topic", default=None,
                    help="default: the 'topic' field recorded in --transform")
    ap.add_argument("--store", default=None,
                    help="default: the 'store' field recorded in --transform")
    ap.add_argument("--capture-datetime", default=DEFAULT_CAPTURE_DATETIME,
                    help="local capture time, ISO format (default "
                         f"{DEFAULT_CAPTURE_DATETIME}). Day-of-year is computed "
                         "from this via Python's datetime.")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT)
    ap.add_argument("--lon", type=float, default=DEFAULT_LON)
    ap.add_argument("--timezone", type=int, default=DEFAULT_TIMEZONE,
                    help="SOLPOS convention, east-positive (default 2 = CEST)")
    ap.add_argument("--minute-step", type=int, default=DEFAULT_MINUTE_STEP)
    ap.add_argument("--face-tolerance-m", type=float, default=None,
                    help="max distance (m) from a voxel to a closed-box face to "
                         "count as a surface voxel (default: half the voxel edge)")
    ap.add_argument("--vostok-exe", type=Path, default=DEFAULT_VOSTOK_EXE)
    ap.add_argument("--workdir", type=Path, default=OUT_DIR,
                    help="where shadow_points.xyz/query_points.xyz/run.sol/"
                         "shadow_clouds/ get written (default: "
                         "solar_shadow_voxel_out/ next to this script)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "shadow_mask.json",
                    help="per-voxel shadow mask JSON (default: "
                         "solar_shadow_voxel_out/shadow_mask.json)")
    args = ap.parse_args()

    if not args.vostok_exe.exists():
        raise SystemExit(f"vostok.exe not found at {args.vostok_exe} -- build it first "
                          f"(see ../../../../vostok/README.md, or this folder's README.md "
                          f"'Building VOSTOK' section)")

    capture_dt = datetime.strptime(args.capture_datetime, "%Y-%m-%dT%H:%M:%S")
    day_of_year = capture_dt.timetuple().tm_yday
    print(f"capture time: {capture_dt.isoformat()} local -> day-of-year {day_of_year}, "
          f"target {capture_dt.hour:02d}:{capture_dt.minute:02d}")

    # --- step 1: transforms ---
    transform = json.loads(args.transform.read_text())
    R = np.array(transform["rotation"])
    t = np.array(transform["translation"])
    bag = args.bag if args.bag is not None else Path(transform["bag"])
    topic = args.topic if args.topic is not None else transform["topic"]
    store = args.store if args.store is not None else transform.get("store", "ROS2_HUMBLE")

    R_extra = north_alignment_rotation(args.north_bearing_deg)
    total_R, total_t = compose_transform(R, t, R_extra)
    print(f"north-bearing-deg={args.north_bearing_deg} -> extra Z-rotation "
          f"{90.0 - args.north_bearing_deg:.3f} deg")

    # --- step 3: surface voxels + face normals (aligned frame, pre-north-rotation) ---
    voxels = np.load(args.voxels)
    centers = voxels["centers"]
    voxel_size = float(voxels["voxel_size"])
    tol = args.face_tolerance_m if args.face_tolerance_m is not None else voxel_size * 0.5

    planes = json.loads(args.planes_aligned.read_text())["planes"]
    voxel_row_idx, face_id, normals_aligned = assign_surface_voxels(centers, planes, tol)
    print(f"surface voxels: {len(voxel_row_idx)} / {len(centers)} "
          f"(tolerance={tol:.4f} m, {len(planes)} faces)")

    surface_centers_aligned = centers[voxel_row_idx]
    surface_centers_north = surface_centers_aligned @ R_extra.T
    normals_north = normals_aligned @ R_extra.T

    # --- step 2: raw cloud -> shadow points ---
    xyz = load_merged_cloud(bag, topic, store)
    print(f"aligning + north-rotating {len(xyz)} points for the shadow point cloud")
    shadow_points = xyz @ total_R.T + total_t

    args.workdir.mkdir(parents=True, exist_ok=True)
    shadow_path = args.workdir / "shadow_points.xyz"
    query_path = args.workdir / "query_points.xyz"
    sol_path = args.workdir / "run.sol"
    output_path = args.workdir / "run_irradiation_ignored.xyz"

    for stale in (shadow_path, query_path):
        meta = stale.with_suffix(stale.suffix + ".vostokmeta")
        if meta.exists():
            meta.unlink()

    np.savetxt(shadow_path, shadow_points, fmt="%.4f")
    print(f"wrote {len(shadow_points)} shadow points to {shadow_path}")

    query_rows = np.hstack([surface_centers_north, normals_north])
    np.savetxt(query_path, query_rows, fmt="%.4f")
    print(f"wrote {len(query_rows)} query points to {query_path}")

    # --- step 4/5: run VOSTOK ---
    write_sol_file(
        sol_path, shadow_path.name, query_path.name, voxel_size,
        args.lat, args.lon, args.timezone, capture_dt.year, day_of_year,
        day_step=1, minute_step=args.minute_step, output_file=output_path.name)
    run_vostok(args.vostok_exe, sol_path, args.workdir)

    # --- step 6: parse + match back to voxels ---
    shadow_clouds_dir = args.workdir / "shadow_clouds"
    shadow_file = find_closest_shadow_file(
        shadow_clouds_dir, capture_dt.year, day_of_year, capture_dt.hour, capture_dt.minute)

    rows = np.loadtxt(shadow_file)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if len(rows) != len(query_rows):
        print(f"WARNING: shadow file has {len(rows)} rows, expected {len(query_rows)} "
              f"(one per query point) -- results below may be misaligned")

    xyz_echo = rows[:, :3]
    illuminated = rows[:, 3].astype(bool)
    mismatch = np.abs(xyz_echo - surface_centers_north[:len(rows)]).max() if len(rows) else 0.0
    print(f"sanity check: max |echoed xyz - query xyz| = {mismatch:.4f} m "
          f"(should be ~0, VOSTOK echoes back the query point verbatim)")

    result = {
        "shadow_file": str(shadow_file),
        "capture_datetime": args.capture_datetime,
        "north_bearing_deg": args.north_bearing_deg,
        "voxel_index": voxel_row_idx.tolist(),
        "center_aligned": surface_centers_aligned.tolist(),
        "face_id": face_id.tolist(),
        "normal_aligned": normals_aligned.tolist(),
        "illuminated": illuminated.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    n_lit = int(illuminated.sum())
    print(f"illuminated: {n_lit} / {len(illuminated)} ({100 * n_lit / max(len(illuminated), 1):.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
