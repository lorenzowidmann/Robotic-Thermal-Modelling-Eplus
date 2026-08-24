"""RANSAC-fit a closed wall/floor/ceiling box from the raw LiDAR point cloud.

This is the geometry step behind the leveling idea in De Pazzi, Chiodini,
Pertile (Sensors 2022), "3D Radiometric Mapping by Means of LiDAR SLAM and
Thermal Camera Data Fusion", generalized from their single ground-plane
gravity-leveling (Sec. 4.2, Eq. 14) to a full closed set of walls/floor/
ceiling -- needed because this corridor sits at a real yaw in the SLAM frame,
not just a tilt, so one plane isn't enough to define an orientation.

The RANSAC plane-fitting logic below (load_merged_cloud, segment_planes,
dedupe_planes, close_geometry, tilt_from_structure_deg, and their helpers) is
the same logic originally written for Thesis/OpenStudioModel/fit_planes.py,
folded directly into this file rather than kept as a separate imported
module -- this pipeline only ever uses it through fit_closed_planes.py, so a
second file existing solely to be imported once added nothing.

Unlike fit_planes.py's own CLI, there is intentionally no --snap-axis here:
snapping every normal to the nearest *world* axis would erase the corridor's
real yaw, which is exactly what aligned_octree.py (script 2) needs from the
wall planes to build its rotation. --close-geometry's own per-axis grouping
(largest plane per side per axis of the *original*, unsnapped normals) still
works to pair opposing walls as long as the yaw is well under 45 deg, which
holds for this corridor.

close_geometry always runs with cap_open_faces=True here: the whole point of
this script is a single watertight box (floor + ceiling + all 4 walls) for
script 2 to level against, not a partial plane set.

Also dropped versus the original fit_planes.py: crop_roi, statistical_outlier_
removal, and declutter (the manual-ROI / SOR / "drop points outside the main
body" options) -- fit_closed_planes.py's CLI never exposed them, so their
code was dead weight here.

Usage:
    python fit_closed_planes.py [--bag <rosbag2_folder>] [--topic /cloud_registered]
        [--distance-threshold 0.02] [--ransac-n 3] [--num-iterations 1000]
        [--min-inliers 500] [--max-planes 20] [--max-attempts 0]
        [--max-tilt-deg 15] [--rect-cluster-gap 0.3]
        [--no-dedupe] [--dedupe-normal-deg 10] [--dedupe-offset-m 0.15]
        [--close-retry-margin 0.5] [--close-retry-min-inliers 150]
        [--close-retry-min-density-ratio 0.4]
        [--out planes.json]

Venv: C:\\venvs\\planefit (Python 3.12 -- open3d has no cp313 wheel yet, see
requirements.txt).
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

DEFAULT_BAG = Path(
    r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20"
    r"\rosbag2_2026_07_30-18_12_20_filtered"
)


# Every generated file goes here, never next to the sources -- keeps the module
# folder clean and lets one .gitignore rule cover the whole pipeline's output.
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "OcTreeVoxel_out"


def read_pointcloud2(msg):
    # x,y,z as float32 at offsets 0,4,8; slice by point_step to skip extra fields
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
    return xyz


def cluster_labels(points, gap):
    """Label points by connectivity (works for 2D or 3D points): two points in
    touching grid cells of edge `gap` (full neighbourhood) belong to the same
    cluster."""
    keys = np.floor(points / gap).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    index = {tuple(v): i for i, v in enumerate(uniq)}
    parent = np.arange(len(uniq))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    dim = points.shape[1]
    neigh = [d for d in itertools.product((-1, 0, 1), repeat=dim) if any(d)]
    for i, v in enumerate(uniq):
        for d in neigh:
            j = index.get(tuple(v + d))
            if j is not None and j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(len(uniq))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[inv]


def largest_cluster_mask(points, gap):
    labels = cluster_labels(points, gap)
    counts = np.bincount(labels)
    main = int(counts.argmax())
    return labels == main


def fit_oriented_rect(points, normal, centroid, rect_cluster_gap=0.0, align_to_structure=True):
    """Project inlier points into the plane's own 2D basis and fit a rectangle.

    align_to_structure=True (default): the 2D basis is locked to the
    building's real orientation instead of picked arbitrarily -- for a
    floor/ceiling (normal near vertical) the axes are world X/Y; for a wall
    (normal near horizontal) one axis is world-up (Z) and the other is
    horizontal along the wall. The rectangle is then the axis-aligned
    bounding box in that basis, so it can never come out diamond-rotated the
    way an unconstrained min-area rectangle can on a sparse/irregular patch.
    align_to_structure=False falls back to cv2.minAreaRect's true
    minimum-area (possibly rotated) box on an arbitrary basis.

    If rect_cluster_gap > 0, first trim to the largest connected 2D patch so
    a few stray inliers (e.g. seen through a doorway) can't stretch the
    rectangle far beyond the plane's real extent."""
    n = normal / np.linalg.norm(normal)

    if align_to_structure:
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(n[2]) > 0.5:  # floor/ceiling: axes = world X, Y (projected onto the plane)
            u = np.array([1.0, 0.0, 0.0]) - n[0] * n
            u /= np.linalg.norm(u)
            v = np.cross(n, u)
        else:  # wall: one axis is world-up, the other horizontal along the wall
            u = np.cross(n, world_up)
            norm_u = np.linalg.norm(u)
            if norm_u < 1e-6:  # degenerate: normal ~parallel to world_up, shouldn't happen for a wall
                u = np.array([1.0, 0.0, 0.0]) - n[0] * n
                norm_u = np.linalg.norm(u)
            u /= norm_u
            v = world_up.copy()
    else:
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, arbitrary)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

    rel = points - centroid
    pts2d = np.stack([rel @ u, rel @ v], axis=1).astype(np.float32)

    if rect_cluster_gap > 0 and len(pts2d) > 10:
        mask = largest_cluster_mask(pts2d, rect_cluster_gap)
        dropped = len(pts2d) - int(mask.sum())
        if dropped:
            print(f"  rect trim: dropped {dropped} stray inlier(s) outside main patch")
        pts2d = pts2d[mask]

    if align_to_structure:
        xmin, ymin = pts2d.min(axis=0)
        xmax, ymax = pts2d.max(axis=0)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        w, h = float(xmax - xmin), float(ymax - ymin)
        angle = 0.0
        box2d = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32)
    else:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts2d)
        box2d = cv2.boxPoints(((cx, cy), (w, h), angle))  # 4x2, CCW in local basis

    box3d = centroid + box2d[:, 0:1] * u + box2d[:, 1:2] * v
    center3d = centroid + cx * u + cy * v

    return {
        "width_m": float(w),
        "height_m": float(h),
        "area_m2": float(w * h),
        "angle_deg": float(angle),
        "center_3d": center3d.tolist(),
        "corners_3d": box3d.tolist(),
        "basis_u": u.tolist(),
        "basis_v": v.tolist(),
    }


def tilt_from_structure_deg(normal):
    """Angle (deg) of `normal` from the nearest Manhattan-aligned orientation:
    0 deg = exactly vertical normal (floor/ceiling), 90 deg = exactly
    horizontal normal (wall). Assumes Z is up (world/map frame convention
    used throughout this script)."""
    n = normal / np.linalg.norm(normal)
    angle_to_vertical = np.degrees(np.arccos(np.clip(abs(n[2]), 0.0, 1.0)))
    return min(angle_to_vertical, abs(90.0 - angle_to_vertical))


_CANONICAL_AXES = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
])


def snap_normal_to_axis(normal):
    """Snap `normal` to the nearest of the 6 world axes (+-X, +-Y, +-Z), so
    the plane's tilt becomes exactly 0 deg. Assumes the structure really is
    Manhattan-aligned with the world frame."""
    n = normal / np.linalg.norm(normal)
    return _CANONICAL_AXES[np.argmax(_CANONICAL_AXES @ n)]


def segment_planes(xyz, distance_threshold, ransac_n, num_iterations, min_inliers, max_planes,
                    rect_cluster_gap=0.0, max_tilt_deg=0.0, max_attempts=0, align_to_structure=True,
                    snap_axis=False):
    """Iteratively RANSAC-fit planes. If max_tilt_deg > 0, a candidate plane
    whose normal isn't close to vertical (floor/ceiling) or horizontal (wall)
    -- i.e. doesn't match the building's real structure orientation -- is
    rejected: its inliers are discarded as noise (not returned as a plane)
    and fitting continues on what's left, up to max_attempts total tries.
    If snap_axis, every accepted plane's normal is snapped to the nearest
    world axis (tilt_deg forced to exactly 0) before the rectangle is fit."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if max_attempts <= 0:
        max_attempts = max_planes * 5

    planes = []
    remaining = pcd
    attempt = 0
    while len(planes) < max_planes and attempt < max_attempts:
        if len(remaining.points) < max(ransac_n, min_inliers):
            print(f"only {len(remaining.points)} points left, stopping")
            break

        model, inlier_idx = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        attempt += 1
        if len(inlier_idx) < min_inliers:
            print(f"best fit {len(inlier_idx)} inliers < --min-inliers {min_inliers}, stopping")
            break

        a, b, c, d = model
        normal = np.array([a, b, c])
        inlier_pts = np.asarray(remaining.points)[inlier_idx]

        tilt = tilt_from_structure_deg(normal)
        n_unit = normal / np.linalg.norm(normal)
        orientation = "floor_ceiling" if abs(n_unit[2]) > 0.5 else "wall"

        if max_tilt_deg > 0 and tilt > max_tilt_deg:
            print(f"reject: {len(inlier_idx)}-point plane tilted {tilt:.1f} deg off "
                  f"vertical/horizontal (> --max-tilt-deg {max_tilt_deg}), discarding as noise")
            remaining = remaining.select_by_index(inlier_idx, invert=True)
            continue

        centroid = inlier_pts.mean(axis=0)

        if snap_axis:
            normal_out = snap_normal_to_axis(normal)
            d_out = -float(np.dot(normal_out, centroid))  # plane still passes through the centroid
            tilt_out = 0.0
        else:
            normal_out = normal
            d_out = float(d)
            tilt_out = tilt

        rect = fit_oriented_rect(inlier_pts, normal_out, centroid, rect_cluster_gap=rect_cluster_gap,
                                  align_to_structure=align_to_structure)

        plane = {
            "id": len(planes),
            "normal": normal_out.tolist(),
            "d": d_out,
            "inlier_count": len(inlier_idx),
            "centroid_3d": centroid.tolist(),
            "orientation": orientation,
            "tilt_deg": tilt_out,
            **rect,
        }
        planes.append(plane)
        print(f"plane {plane['id']}: {len(inlier_idx)} inliers, normal {normal_out.round(3)}, "
              f"orientation={orientation} tilt={tilt_out:.1f}deg (raw fit was {tilt:.1f}deg), "
              f"rect {rect['width_m']:.2f}x{rect['height_m']:.2f} m")

        remaining = remaining.select_by_index(inlier_idx, invert=True)

    print(f"{len(planes)} plane(s) extracted, {len(remaining.points)} points left unassigned")
    return planes


def canonical_normal_offset(normal, d):
    """Sign-canonicalize (normal, d) so two fits of the *same* physical plane
    compare equal regardless of which side segment_plane's normal happened to
    point to: flip both if the largest-magnitude normal component is negative."""
    n = np.array(normal, dtype=float)
    n = n / np.linalg.norm(n)
    if n[np.argmax(np.abs(n))] < 0:
        n, d = -n, -d
    return n, d


def dedupe_planes(planes, normal_deg=10.0, offset_m=0.15):
    """A real corridor floor/wall/ceiling often gets RANSAC-fit as several
    disjoint fragments (clutter, occlusion, gaps) instead of one plane.
    Group planes that are effectively the same physical surface -- same
    orientation, near-parallel normals, near-identical offset -- and keep
    only the largest (by inlier_count) of each group, dropping the rest."""
    order = sorted(range(len(planes)), key=lambda i: -planes[i]["inlier_count"])
    keyed = [canonical_normal_offset(planes[i]["normal"], planes[i]["d"]) for i in range(len(planes))]

    used = [False] * len(planes)
    kept = []
    for i in order:
        if used[i]:
            continue
        used[i] = True
        kept.append(i)
        ni, di = keyed[i]
        dropped = []
        for j in order:
            if used[j]:
                continue
            nj, dj = keyed[j]
            angle = np.degrees(np.arccos(np.clip(np.dot(ni, nj), -1.0, 1.0)))
            if angle < normal_deg and abs(di - dj) < offset_m:
                used[j] = True
                dropped.append(j)
        if dropped:
            print(f"dedupe: plane {planes[i]['id']} ({planes[i]['inlier_count']} inliers) absorbs "
                  f"{[planes[j]['id'] for j in dropped]} as the same surface")

    kept.sort()
    result = [planes[i] for i in kept]
    for new_id, p in enumerate(result):
        p["id"] = new_id
    print(f"dedupe: {len(planes)} -> {len(result)} plane(s)")
    return result


def _retry_fit_closing_plane(xyz, axis, missing_side, distance_threshold, ransac_n, num_iterations,
                              retry_min_inliers, retry_margin, retry_min_density_ratio, density_ref,
                              rect_cluster_gap, align_to_structure, snap_axis, plane_id):
    """A big open face (e.g. a whole missing wall) usually means real data
    is there but it lost out to --min-inliers or just wasn't the largest fit
    on the first pass. Search a thin slab of the working cloud right at the
    boundary that would close this axis/side for a plane matching that
    orientation, using a relaxed inlier threshold. Returns a plane dict, or
    None if nothing convincing enough was found (in which case the caller
    leaves the face open, same as before)."""
    if xyz is None or len(xyz) == 0:
        return None
    col = xyz[:, axis]
    if missing_side == "-":
        edge = float(col.min())
        mask = col <= edge + retry_margin
    else:
        edge = float(col.max())
        mask = col >= edge - retry_margin
    slab = xyz[mask]
    if len(slab) < max(ransac_n, retry_min_inliers):
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(slab)
    model, inlier_idx = pcd.segment_plane(distance_threshold=distance_threshold,
                                           ransac_n=ransac_n, num_iterations=num_iterations)
    if len(inlier_idx) < retry_min_inliers:
        return None

    a, b, c, d = model
    normal = np.array([a, b, c])
    if int(np.argmax(np.abs(normal))) != axis:
        return None  # the slab's best plane isn't even oriented the way we need

    inlier_pts = slab[inlier_idx]
    centroid = inlier_pts.mean(axis=0)
    tilt = tilt_from_structure_deg(normal)
    n_unit = normal / np.linalg.norm(normal)
    orientation = "floor_ceiling" if abs(n_unit[2]) > 0.5 else "wall"

    if snap_axis:
        normal = snap_normal_to_axis(normal)
        d = -float(np.dot(normal, centroid))
        tilt = 0.0

    rect = fit_oriented_rect(inlier_pts, normal, centroid, rect_cluster_gap=rect_cluster_gap,
                              align_to_structure=align_to_structure)

    density = len(inlier_idx) / rect["area_m2"] if rect["area_m2"] > 0 else 0.0
    if density_ref is not None and density < retry_min_density_ratio * density_ref:
        print(f"  retry candidate point density {density:.0f}/m2 is only "
              f"{density / density_ref:.0%} of confirmed walls' {density_ref:.0f}/m2 "
              f"(< --close-retry-min-density-ratio {retry_min_density_ratio:.0%}) -- "
              f"thin/scattered like a door or opening, not a solid wall, rejecting")
        return None

    return {
        "id": plane_id, "normal": normal.tolist(), "d": float(d),
        "inlier_count": len(inlier_idx), "centroid_3d": centroid.tolist(),
        "orientation": orientation, "tilt_deg": float(tilt), **rect,
    }


def close_geometry(planes, xyz=None, distance_threshold=0.02, ransac_n=3, num_iterations=1000,
                    retry_min_inliers=150, retry_margin=0.5, retry_min_density_ratio=0.4,
                    rect_cluster_gap=0.0, align_to_structure=True, snap_axis=False, roi=None,
                    cap_open_faces=False):
    """Reduce the plane set to a single watertight box, importable to
    OpenStudio as a shoebox Space: for each axis (X/Y/Z) and each side of the
    room's centroid along that axis, keep only the single largest (by
    inlier_count) plane -- dropping the other, redundant fragments on that
    side -- then stretch every kept plane's rectangle to the shared bounding
    box on its other two axes, so every face's edges meet its neighbors
    exactly instead of stopping wherever its own inliers happened to end.

    If a whole side of an axis has no plane at all (a big open face, e.g. a
    missing wall) and `xyz` (the working point cloud) is given, retries a
    targeted RANSAC fit in a thin slab at that boundary before giving up --
    the open face is only kept if that retry also fails to find anything.
    A point-density check rejects thin/scattered fits (a corridor
    continuation or doorway can span as much area as a real wall without
    being solidly covered), but that alone isn't reliable when the open
    side sits exactly on a manually-cropped --roi boundary -- that boundary
    is where *we* cut the data, not a real edge, so if `roi` is given
    (xmin,xmax,ymin,ymax,zmin,zmax) any open side that coincides with it is
    left open without even attempting a retry."""
    if not planes:
        return planes

    total_w = sum(p["inlier_count"] for p in planes) or 1
    center = np.zeros(3)
    for p in planes:
        center += np.array(p["centroid_3d"]) * p["inlier_count"]
    center /= total_w

    def axis_of(p):
        return int(np.argmax(np.abs(np.array(p["normal"]))))

    axis_groups = {0: [], 1: [], 2: []}
    for p in planes:
        axis_groups[axis_of(p)].append(p)

    # pass 1: keep the largest plane per non-empty side, note which (axis, side) are missing
    kept = []
    missing = []
    for axis in (0, 1, 2):
        group = axis_groups[axis]
        neg = [p for p in group if p["centroid_3d"][axis] < center[axis]]
        pos = [p for p in group if p["centroid_3d"][axis] >= center[axis]]
        for side_name, side in (("-", neg), ("+", pos)):
            if side:
                best = max(side, key=lambda p: p["inlier_count"])
                kept.append(best)
                dropped = [p["id"] for p in side if p is not best]
                if dropped:
                    print(f"close-geometry: axis {'XYZ'[axis]}{side_name} kept plane {best['id']} "
                          f"({best['inlier_count']} inliers), dropped {dropped} as redundant")
            elif group:  # some plane exists on this axis, just not this side -- worth a retry
                missing.append((axis, side_name))

    densities = [p["inlier_count"] / p["area_m2"] for p in kept if p.get("area_m2", 0) > 0]
    density_ref = float(np.median(densities)) if densities else None

    next_id = max((p["id"] for p in planes), default=-1) + 1
    for axis, side_name in missing:
        if roi is not None:
            roi_bound = roi[2 * axis] if side_name == "-" else roi[2 * axis + 1]
            edge = float(xyz[:, axis].min() if side_name == "-" else xyz[:, axis].max()) if xyz is not None else None
            if edge is not None and abs(edge - roi_bound) < 0.05:
                print(f"close-geometry: axis {'XYZ'[axis]}{side_name} sits right on the --roi crop "
                      f"boundary ({roi_bound}) -- that's where we cut the data, not a real edge, "
                      f"leaving it open without retrying")
                continue
        print(f"close-geometry: axis {'XYZ'[axis]}{side_name} is a big open face, "
              f"retrying a targeted fit to close it...")
        found = _retry_fit_closing_plane(
            xyz, axis, side_name, distance_threshold, ransac_n, num_iterations,
            retry_min_inliers, retry_margin, retry_min_density_ratio, density_ref,
            rect_cluster_gap, align_to_structure, snap_axis, next_id)
        if found is not None:
            print(f"close-geometry: retry fit plane {next_id} ({found['inlier_count']} inliers) "
                  f"closes axis {'XYZ'[axis]}{side_name}")
            kept.append(found)
            next_id += 1
        else:
            print(f"close-geometry: retry found nothing convincing for axis {'XYZ'[axis]}{side_name}, "
                  f"leaving it open")

    if len(kept) < 3:
        print(f"close-geometry: only {len(kept)} independent face(s) found -- "
              f"not enough to close a box, keeping planes as-is")
        return planes

    def data_extent(axis):
        if xyz is not None and len(xyz):
            return float(xyz[:, axis].min()), float(xyz[:, axis].max())
        allc = np.array([c for p in kept for c in p["corners_3d"]])
        return float(allc[:, axis].min()), float(allc[:, axis].max())

    bounds = {}
    open_sides = []
    for axis in (0, 1, 2):
        vals = [p["centroid_3d"][axis] for p in kept if axis_of(p) == axis]
        lo_data, hi_data = data_extent(axis)
        if len(vals) >= 2 and max(vals) - min(vals) > 1e-9:
            bounds[axis] = (min(vals), max(vals))
        elif vals:
            known = vals[0]
            if abs(known - hi_data) < abs(known - lo_data):
                bounds[axis] = (lo_data, known)
                open_sides.append((axis, "-"))
            else:
                bounds[axis] = (known, hi_data)
                open_sides.append((axis, "+"))
            print(f"close-geometry: only one {'XYZ'[axis]}-side plane found -- that end is "
                  f"open (e.g. a corridor cut, not a real wall); using the data extent "
                  f"{bounds[axis][0]:.2f}..{bounds[axis][1]:.2f} so the other faces still line up")
        else:
            bounds[axis] = (lo_data, hi_data)
            open_sides.extend([(axis, "-"), (axis, "+")])

    if cap_open_faces:
        for axis, side_name in open_sides:
            fixed = bounds[axis][0] if side_name == "-" else bounds[axis][1]
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0
            centroid = [(bounds[a][0] + bounds[a][1]) / 2 for a in (0, 1, 2)]
            centroid[axis] = fixed
            kept.append({
                "id": next_id, "normal": normal, "d": -fixed,
                "inlier_count": 0, "centroid_3d": centroid,
                "orientation": "floor_ceiling" if axis == 2 else "wall",
                "tilt_deg": 0.0, "synthetic": True, "boundary_condition": "adiabatic",
                "width_m": 0.0, "height_m": 0.0, "area_m2": 0.0, "angle_deg": 0.0,
                "center_3d": centroid, "corners_3d": [centroid] * 4,
                "basis_u": [0.0, 0.0, 0.0], "basis_v": [0.0, 0.0, 0.0],
            })
            print(f"close-geometry: capping open face {'XYZ'[axis]}{side_name} at {fixed:.2f} with a "
                  f"synthetic adiabatic surface (plane {next_id}) -- no LiDAR evidence of a wall "
                  f"there, it exists only to make the solid watertight for OpenStudio")
            next_id += 1

    for p in kept:
        axis = axis_of(p)
        other = [a for a in (0, 1, 2) if a != axis]
        fixed = p["centroid_3d"][axis]
        ranges = [bounds[a] for a in other]
        (lo0, hi0), (lo1, hi1) = ranges
        a0, a1 = other

        corners3d = []
        for v0, v1 in [(lo0, lo1), (hi0, lo1), (hi0, hi1), (lo0, hi1)]:
            c = [0.0, 0.0, 0.0]
            c[axis], c[a0], c[a1] = fixed, v0, v1
            corners3d.append(c)
        p["corners_3d"] = corners3d

        center3d = [0.0, 0.0, 0.0]
        center3d[axis], center3d[a0], center3d[a1] = fixed, (lo0 + hi0) / 2, (lo1 + hi1) / 2
        p["center_3d"] = center3d

        p["width_m"] = float(hi0 - lo0)
        p["height_m"] = float(hi1 - lo1)
        p["area_m2"] = float(p["width_m"] * p["height_m"])

    kept.sort(key=lambda p: p["id"])
    for new_id, p in enumerate(kept):
        p["id"] = new_id
    print(f"close-geometry: {len(planes)} -> {len(kept)} plane(s), closed box "
          f"x={bounds.get(0)} y={bounds.get(1)} z={bounds.get(2)}")
    return kept


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, default=DEFAULT_BAG, help="rosbag2 folder")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--distance-threshold", type=float, default=0.02,
                    help="RANSAC inlier distance (m)")
    ap.add_argument("--ransac-n", type=int, default=3)
    ap.add_argument("--num-iterations", type=int, default=1000)
    ap.add_argument("--min-inliers", type=int, default=500,
                    help="stop once the next best plane has fewer inliers than this")
    ap.add_argument("--max-planes", type=int, default=20, help="safety cap")
    ap.add_argument("--max-attempts", type=int, default=0,
                    help="total RANSAC tries (accepted + rejected); 0 = --max-planes * 5")
    ap.add_argument("--max-tilt-deg", type=float, default=15.0,
                    help="reject a candidate plane if its normal is more than this many "
                         "degrees off vertical (floor/ceiling) or horizontal (wall) -- "
                         "discarded as noise, not returned as a plane (0 = accept any orientation)")
    ap.add_argument("--rect-cluster-gap", type=float, default=0.3,
                    help="per-plane rectangle trim: max gap (m) for a plane's inlier "
                         "points to count as the same 2D patch (0 = off)")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep every RANSAC fragment as its own plane instead of merging "
                         "near-duplicate fragments of the same physical surface")
    ap.add_argument("--dedupe-normal-deg", type=float, default=10.0,
                    help="max angle (deg) between normals to count as the same surface")
    ap.add_argument("--dedupe-offset-m", type=float, default=0.15,
                    help="max offset difference (m) to count as the same surface")
    ap.add_argument("--close-retry-margin", type=float, default=0.5,
                    help="thickness (m) of the slab searched at an open boundary "
                         "when retrying a fit to close it")
    ap.add_argument("--close-retry-min-inliers", type=int, default=150,
                    help="relaxed --min-inliers used only for the retry fit")
    ap.add_argument("--close-retry-min-density-ratio", type=float, default=0.4,
                    help="reject a retry fit whose point density (inliers/area) is below "
                         "this fraction of the confirmed real walls' density")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "planes.json",
                    help="default: OcTreeVoxel_out/planes.json")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    xyz = load_merged_cloud(args.bag, args.topic, args.store)

    print(f"RANSAC input: {len(xyz)} points")
    planes = segment_planes(
        xyz, args.distance_threshold, args.ransac_n, args.num_iterations,
        args.min_inliers, args.max_planes, rect_cluster_gap=args.rect_cluster_gap,
        max_tilt_deg=args.max_tilt_deg, max_attempts=args.max_attempts,
        align_to_structure=True, snap_axis=False)

    if not args.no_dedupe:
        planes = dedupe_planes(planes, normal_deg=args.dedupe_normal_deg,
                                offset_m=args.dedupe_offset_m)

    planes = close_geometry(
        planes, xyz=xyz, distance_threshold=args.distance_threshold,
        ransac_n=args.ransac_n, num_iterations=args.num_iterations,
        retry_min_inliers=args.close_retry_min_inliers,
        retry_margin=args.close_retry_margin,
        retry_min_density_ratio=args.close_retry_min_density_ratio,
        rect_cluster_gap=args.rect_cluster_gap, align_to_structure=True,
        snap_axis=False, roi=None, cap_open_faces=True)

    args.out.write_text(json.dumps(
        {"bag": str(args.bag), "topic": args.topic, "store": args.store, "planes": planes},
        indent=2))
    print(f"wrote {len(planes)} plane(s) to {args.out}")


if __name__ == "__main__":
    main()
