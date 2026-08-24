"""Rectilinear box (room/hall volume) fitting on a point cloud -- not walls.

Unlike PlaneFittingAttempt/fit_planes.py (which RANSAC-fits individual wall/
floor/ceiling *planes*), this script partitions the footprint of EVERY hall
in the cloud into a small set of axis-aligned rectangular *boxes* (extruded
to each hall's height) that together exactly tile the occupied floor area:

  - every box is axis-aligned to the same X/Y grid, so any two box edges
    differ by an exact multiple of 90 deg -- there is no other orientation
    a box could have;
  - boxes are produced by repeatedly carving the single largest all-occupied
    rectangle out of the footprint and marking it used, so the boxes that
    result form an exact, gap-free tiling of the footprint: any two that are
    adjacent share a full grid edge (they touch), and together they cover
    the whole footprint (down to --min-box-area, see below). This also
    means two SEPARATE halls -- disconnected in the point cloud, so there is
    no occupied cell bridging them -- can never accidentally end up sharing
    one box: a rectangle can only ever grow through occupied cells;
  - every box gets the SAME z_min/z_max within its own hall (that hall's
    floor/ceiling, estimated from the point cloud), so every box in a given
    hall has the same height. Different halls at different heights (e.g. a
    mezzanine glimpsed through a doorway) still each come out uniform;
  - the footprint itself -- where the boxes go, and how many are needed --
    comes directly from the point cloud's own occupied-cell grid, not a
    guessed shape (a corridor, an L-shaped hall, a T-junction all decompose
    differently, automatically).

Default behaviour fits boxes to EVERY hall found in the cloud (see
--single-hall to isolate only the largest one instead, e.g. combined with
--roi to manually pick which).

HEIGHT IS PER ROOM, NOT PER STRUCTURE. Note that "a connected cluster of
points" is NOT the same thing as "one room": in a real building every
corridor joins every other through the floor, so the whole structure is a
single connected cluster. Estimating one floor_z/ceiling_z per cluster
therefore forces the entire building to share a height, and a low corridor
ends up inside a box reaching a metre above its real ceiling. So instead:

  1. each tiled rectangle measures its own floor/ceiling from the points
     under its OWN footprint (robust percentiles, --floor-pct/--ceiling-pct);
  2. rectangles that touch AND agree on height within --height-group-tol
     are grouped into one room, which then takes a single consensus height
     recomputed from all its points together -- so a room still comes out
     flat-topped rather than each box wobbling by its own sampling noise;
  3. only rectangles at least --room-seed-width wide may define a room.
     The tiling leaves thin slivers along walls whose ceiling the scan
     barely reaches at a grazing angle, so their own height reads far too
     low; each sliver inherits the room of the neighbour it shares the
     longest edge with instead of inventing a spurious short room.

The net effect is what you want architecturally: every box within one hall
shares that hall's height, while a taller hall next door keeps its own --
even though the two are physically connected. Pass --floor-z/--ceiling-z
to bypass all of this and force one fixed height everywhere, or
--no-height-groups to go back to one height per connected cluster.

ASSUMES the input is already yaw-aligned (walls parallel to world X/Y) --
e.g. LoopClosure_vFinal/BagFilter.m or BagFilter_NoLoopClosure.m's output
.pcd, which corrects exactly this (see their Sezione 6c). Raw /cloud_
registered still carries FAST-LIO yaw drift, so a corridor there is NOT
grid-aligned and this script's rectangles would cut across it at an angle.
If you know the yaw is off by some fixed amount instead, pass --yaw-deg to
rotate the cloud (about Z) before gridding; the fitted boxes are rotated
back so the output stays in the original frame.

PIPELINE (steps 4-9 run once PER CONNECTED CLUSTER, independently)
  1. load the cloud (--bag or --pcd)
  2. optional --roi crop, --sor (statistical outlier removal)
  3. declutter (on by default): split into connected clusters of >=
     --min-cluster-points points, dropping smaller ones as noise
     (--single-hall to keep only the largest cluster)
  4. estimate this cluster's overall floor_z/ceiling_z from robust height
     percentiles -- used to bound the grid and as the fallback for any
     rectangle too small to measure its own, NOT as the boxes' height
  5. rasterize the XY footprint (points between those bounds) into an
     occupied/empty grid at --cell-size resolution; a cell counts as
     occupied only with >= --min-points-per-cell points, so sparse noise
     reaching into open space can't fake floor there
  6. at grid resolution: drop connected blobs below --min-blob-cells as
     noise, optionally fill small enclosed holes (furniture/columns
     splitting the floor into islands that are still obviously one room)
  7. repeatedly extract the largest all-occupied rectangle (classic
     "maximum rectangle in a binary matrix", via the largest-rectangle-in-
     histogram algorithm run per grid row) until what is left is smaller
     than --min-box-area, tiling the whole footprint
  8. measure each rectangle's own height, group rectangles into rooms and
     give each room one consensus height -- see "HEIGHT IS PER ROOM" above
  9. extrude every rectangle from its room's floor_z to ceiling_z, collect
     every cluster's boxes together, renumber 0..N-1 -> boxes.json

Usage:
    python fit_boxes.py (--bag <rosbag2_folder> | --pcd <file.pcd>)
        [--topic /cloud_registered] [--yaw-deg 0]
        [--roi xmin,xmax,ymin,ymax,zmin,zmax]
        [--sor] [--sor-k 16] [--sor-std 1.5]
        [--no-declutter] [--single-hall]
        [--cluster-gap 0.30] [--min-cluster-points 200]
        [--floor-pct 1] [--ceiling-pct 99] [--floor-z Z] [--ceiling-z Z]
        [--room-seed-width 1.0] [--height-group-tol 0.30]
        [--no-height-groups] [--min-box-points 50]
        [--cell-size 0.20] [--min-points-per-cell 3]
        [--no-fill-holes] [--min-blob-cells 10]
        [--min-box-area 0.5] [--max-boxes 200]
        [--out SavedBoxes/boxes.json]

Output goes to SavedBoxes/boxes.json next to this script by default (the
directory is created if missing), which is also where show_boxes.py and
interactive_boxes.py look by default -- so the fit -> view -> edit chain
needs no --boxes/--out arguments at all.

boxes.json carries a "halls" list -- one entry per ROOM, with its own
floor_z/ceiling_z/height_m -- and every box has a "hall" (its room id) and
a "cluster" (which connected cluster it came from) field.

--sor/--roi/--cluster-gap have the same meaning and defaults as in
fit_planes.py (SOR: k=16, std=1.5; declutter gap: 0.30 m); pass
--no-declutter only if the cloud is already clean hall(s) with no stray
noise and you want every leftover speck included in the grid.

Venv: C:\\venvs\\planefit (same as fit_planes.py/show_planes.py -- open3d is
not needed here, but rosbags/scipy are).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy import ndimage
from scipy.spatial import cKDTree

# Default home for every boxes.json, resolved against THIS FILE's location rather than the
# cwd -- so box files land in the same place whether the script is run from
# LoopClosure_vFinal/ or from anywhere else. Created on demand at save time.
SAVED_BOXES_DIR = Path(__file__).resolve().parent / "SavedBoxes"


# ---------------------------------------------------------------------------
# Point cloud loading / preprocessing -- same logic as fit_planes.py, kept as
# a self-contained copy so this script doesn't depend on importing across
# PlaneFittingAttempt/ (open3d isn't even needed here).
# ---------------------------------------------------------------------------

def read_pointcloud2(msg):
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_pcd_cloud(path):
    """Load xyz straight from a .pcd file -- e.g. BagFilter.m's
    loop_closed_map.pcd or BagFilter_NoLoopClosure.m's
    noloop_corrected_map.pcd, already gravity/yaw-aligned so a corridor
    lands Manhattan-straight on the X/Y grid instead of skewed by raw
    FAST-LIO drift the way /cloud_registered is."""
    if not Path(path).exists():
        raise SystemExit(f"--pcd {path} does not exist (check the path -- "
                          f"open3d fails silently on a bad path otherwise)")
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if len(xyz) == 0:
        raise SystemExit(f"--pcd {path} loaded 0 points -- not a valid/readable .pcd file")
    print(f"{len(xyz)} points from {path}")
    return xyz


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


def crop_roi(xyz, roi):
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    mask = (
        (xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax)
        & (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax)
        & (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    )
    print(f"ROI crop: kept {int(mask.sum())} / {len(xyz)} points")
    return xyz[mask]


def statistical_outlier_removal(xyz, k=16, std_ratio=1.5):
    if len(xyz) <= k:
        return xyz
    tree = cKDTree(xyz)
    dist, _ = tree.query(xyz, k=k + 1, workers=-1)
    mean_d = dist[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d < thresh
    print(f"SOR: removed {int((~keep).sum())} / {len(xyz)} points")
    return xyz[keep]


def cluster_labels(points, gap):
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


def split_into_halls(xyz, gap, min_points, single_hall=False):
    """Split xyz into one array PER REAL HALL -- each connected cluster of
    points, dropping small ones as noise -- instead of one merged array.
    main() estimates floor_z/ceiling_z and tiles each returned hall
    SEPARATELY, so two halls at genuinely different heights (or just two
    disconnected wings of the same building) each get their own correct
    height instead of being forced to share one.

    single_hall=True: return only the single largest connected cluster, as
    one hall -- use this to deliberately isolate/edit one room (typically
    combined with --roi).

    single_hall=False (default): return every cluster with >= min_points
    points, dropping only genuinely small clusters (stray reflections, a
    few points glimpsed through a doorway) as noise."""
    labels = cluster_labels(xyz, gap)
    counts = np.bincount(labels)
    n_clusters = len(counts)

    if single_hall:
        keep_labels = [int(counts.argmax())]
        print(f"declutter (gap={gap}m, --single-hall): {n_clusters} cluster(s), "
              f"kept only the largest as 1 hall ({int(counts[keep_labels[0]])} / "
              f"{len(xyz)} points)")
    else:
        keep_labels = [i for i in range(n_clusters) if counts[i] >= min_points]
        kept_pts = int(sum(counts[i] for i in keep_labels))
        print(f"declutter (gap={gap}m): {n_clusters} cluster(s), kept {len(keep_labels)} "
              f"hall(s) with >= {min_points} points each ({kept_pts} / {len(xyz)} points), "
              f"dropped {n_clusters - len(keep_labels)} small cluster(s) as noise")

    return [xyz[labels == lbl] for lbl in keep_labels]


def declutter_clusters(xyz, gap, min_points, single_hall=False):
    """Same noise-dropping as split_into_halls, but merged back into one
    array -- for callers (show_boxes.py, interactive_boxes.py) that only
    want a clean display cloud, not per-hall geometry."""
    halls = split_into_halls(xyz, gap, min_points, single_hall)
    return np.vstack(halls) if halls else xyz[:0]


# ---------------------------------------------------------------------------
# Floor/ceiling height
# ---------------------------------------------------------------------------

def estimate_floor_ceiling(xyz, floor_pct, ceiling_pct):
    """Robust floor_z/ceiling_z from height percentiles (not min/max, which a
    single stray point above the ceiling or below the floor would wreck)."""
    z = xyz[:, 2]
    floor_z = float(np.percentile(z, floor_pct))
    ceiling_z = float(np.percentile(z, ceiling_pct))
    if ceiling_z <= floor_z:
        raise SystemExit(f"estimated ceiling_z ({ceiling_z:.2f}) <= floor_z ({floor_z:.2f}) -- "
                          f"--floor-pct/--ceiling-pct too close together, or pass "
                          f"--floor-z/--ceiling-z directly")
    print(f"  height: floor_z={floor_z:.3f} ceiling_z={ceiling_z:.3f} "
          f"height={ceiling_z - floor_z:.3f} m (percentiles {floor_pct}/{ceiling_pct})")
    return floor_z, ceiling_z


# ---------------------------------------------------------------------------
# Occupancy grid
# ---------------------------------------------------------------------------

def build_occupancy_grid(xyz, cell_size, floor_z, ceiling_z, min_points_per_cell):
    """Rasterize the XY footprint of points within [floor_z, ceiling_z] into
    a boolean grid. A cell is occupied only with >= min_points_per_cell
    points landing in it, so a handful of stray/noise points reaching past
    the real wall can't fake occupied floor out there."""
    mask = (xyz[:, 2] >= floor_z) & (xyz[:, 2] <= ceiling_z)
    pts = xyz[mask]
    if len(pts) == 0:
        raise SystemExit("no points fall within [floor_z, ceiling_z] -- check "
                          "--floor-z/--ceiling-z or --floor-pct/--ceiling-pct")

    x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
    cols = int(np.floor((pts[:, 0].max() - x0) / cell_size)) + 1
    rows = int(np.floor((pts[:, 1].max() - y0) / cell_size)) + 1

    c_idx = np.clip(np.floor((pts[:, 0] - x0) / cell_size).astype(np.int64), 0, cols - 1)
    r_idx = np.clip(np.floor((pts[:, 1] - y0) / cell_size).astype(np.int64), 0, rows - 1)

    counts = np.zeros((rows, cols), dtype=np.int64)
    np.add.at(counts, (r_idx, c_idx), 1)
    occ = counts >= min_points_per_cell
    print(f"  occupancy grid: {rows}x{cols} cells @ {cell_size}m, "
          f"{int(occ.sum())} occupied (>= {min_points_per_cell} pts/cell)")
    return occ, x0, y0


def clean_grid(occ, fill_holes, min_blob_cells):
    """Belt-and-braces cleanup at grid resolution, on top of the 3D
    declutter (split_into_halls) already done -- occupancy thresholding/
    gridding can itself fragment or reveal blobs the 3D clustering didn't
    see (e.g. a doorway alcove that's disconnected only at grid
    resolution). Keeps every connected blob with >= min_blob_cells cells
    and drops smaller ones as noise; called once per hall by main(), so
    "every blob" here means "every sub-region of THIS hall", not every
    hall in the building.

    Small fully-enclosed holes (furniture/columns splitting a room's floor
    into islands that are still obviously one room) are filled by
    default."""
    structure = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    labels, n = ndimage.label(occ, structure=structure)
    if n == 0:
        raise SystemExit("occupancy grid is empty -- nothing to tile "
                          "(check --cell-size / --min-points-per-cell)")
    sizes = ndimage.sum(occ, labels, index=range(1, n + 1))

    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_blob_cells]
    keep_mask = np.isin(labels, keep_labels)
    dropped = int(occ.sum() - keep_mask.sum())
    if dropped or len(keep_labels) > 1:
        print(f"  grid declutter: {n} blob(s), kept {len(keep_labels)} "
              f">= {min_blob_cells} cells each ({int(keep_mask.sum())} cells total), "
              f"dropped {dropped} cell(s) in {n - len(keep_labels)} speckle(s)")

    if fill_holes:
        filled = ndimage.binary_fill_holes(keep_mask)
        added = int(filled.sum() - keep_mask.sum())
        if added:
            print(f"  fill-holes: filled {added} enclosed cell(s) (furniture/columns/clutter "
                  f"inside this hall that isn't a real gap in the floor)")
        keep_mask = filled

    return keep_mask


# ---------------------------------------------------------------------------
# Rectangle tiling: repeatedly carve the largest all-occupied axis-aligned
# rectangle out of the grid until nothing big enough is left.
# ---------------------------------------------------------------------------

def _largest_rectangle_in_histogram(heights):
    """Classic largest-rectangle-in-histogram, via a monotonic stack.
    Returns (area, col_left, col_right, height) of the best rectangle, or
    (0, 0, 0, 0) if every bar is zero."""
    stack = []  # (start_col, height)
    best = (0, 0, 0, 0)
    for i, h in enumerate(heights + [0]):  # sentinel 0 flushes the stack at the end
        start = i
        while stack and stack[-1][1] > h:
            idx, sh = stack.pop()
            area = sh * (i - idx)
            if area > best[0]:
                best = (area, idx, i - 1, sh)
            start = idx
        stack.append((start, h))
    return best


def _largest_rectangle(grid):
    """Largest all-True axis-aligned rectangle anywhere in a 2D boolean
    grid. Returns (area, row_top, row_bottom, col_left, col_right), or None
    if the grid has no True cell left. O(rows*cols): builds a per-column
    "how many True cells stacked above and including this row" histogram
    incrementally, one row at a time, and solves the histogram sub-problem
    at every row (the row where the rectangle's bottom edge sits)."""
    rows, cols = grid.shape
    heights = [0] * cols
    best = None
    for r in range(rows):
        row = grid[r]
        heights = [(heights[c] + 1) if row[c] else 0 for c in range(cols)]
        area, cl, cr, h = _largest_rectangle_in_histogram(heights)
        if area > 0 and (best is None or area > best[0]):
            best = (area, r - h + 1, r, cl, cr)
    return best


def tile_boxes(occ, min_box_cells, max_boxes):
    """Repeatedly extract the single largest all-occupied rectangle and mark
    it used, until what remains is smaller than min_box_cells or max_boxes
    is hit. The result is an exact, non-overlapping tiling of the occupied
    area (down to that cutoff): any two kept rectangles that end up
    touching share a full grid edge, by construction -- there is no gap
    between them to not touch across."""
    grid = occ.copy()
    boxes = []
    while len(boxes) < max_boxes:
        result = _largest_rectangle(grid)
        if result is None:
            break
        area, r0, r1, c0, c1 = result
        if area < min_box_cells:
            break
        boxes.append((r0, r1, c0, c1))
        grid[r0:r1 + 1, c0:c1 + 1] = False
    leftover = int(grid.sum())
    return boxes, leftover


# ---------------------------------------------------------------------------
# Per-room height: a physically connected cluster of points is NOT the same
# thing as "one room at one height". A building whose corridors all join
# through the floor is a single connected cluster, so estimating one
# floor_z/ceiling_z per cluster forces the whole structure to share a height
# -- a low corridor then gets a box reaching far above its real ceiling.
# Instead each tiled rectangle gets its height from the points under ITS OWN
# footprint, and rectangles that both touch and agree on height are then
# grouped into a room that shares one consensus height (so a room still comes
# out flat-topped, rather than each box wobbling by its own sampling noise).
# ---------------------------------------------------------------------------

def grid_indices(xyz, x0, y0, cell_size):
    """Grid (row, col) index of every point, same convention as
    build_occupancy_grid -- computed once so per-rectangle point lookups are
    a couple of cheap comparisons instead of a fresh floor/divide each."""
    c_idx = np.floor((xyz[:, 0] - x0) / cell_size).astype(np.int64)
    r_idx = np.floor((xyz[:, 1] - y0) / cell_size).astype(np.int64)
    return r_idx, c_idx


def rect_point_mask(r_idx, c_idx, rect):
    r0, r1, c0, c1 = rect
    return (r_idx >= r0) & (r_idx <= r1) & (c_idx >= c0) & (c_idx <= c1)


def estimate_rect_heights(xyz, r_idx, c_idx, rects, floor_pct, ceiling_pct,
                           min_points, fallback):
    """floor/ceiling per rectangle, from the points under that rectangle
    only. Percentiles (not min/max) again, so one stray point above the
    ceiling doesn't stretch a box. A rectangle with fewer than min_points
    points under it falls back to the whole cluster's height -- a 0.2 x
    0.2 m sliver has too few points for its own percentile to mean
    anything."""
    z = xyz[:, 2]
    out = []
    for rect in rects:
        m = rect_point_mask(r_idx, c_idx, rect)
        n = int(m.sum())
        if n < min_points:
            out.append((fallback[0], fallback[1], n))
        else:
            zz = z[m]
            out.append((float(np.percentile(zz, floor_pct)),
                        float(np.percentile(zz, ceiling_pct)), n))
    return out


def shared_edge_cells(a, b):
    """Length (in grid cells) of the edge two rectangles share, 0 if they
    only touch at a corner or not at all. Corner-only contact must not
    count -- two rooms diagonally catty-corner across a junction are not
    the same room -- and the length is what lets a sliver pick the
    neighbour it is most attached to."""
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    row_overlap = min(ar1, br1) - max(ar0, br0) + 1
    col_overlap = min(ac1, bc1) - max(ac0, bc0) + 1
    if row_overlap > 0 and (ac1 + 1 == bc0 or bc1 + 1 == ac0):
        return row_overlap
    if col_overlap > 0 and (ar1 + 1 == br0 or br1 + 1 == ar0):
        return col_overlap
    if row_overlap > 0 and col_overlap > 0:
        return row_overlap * col_overlap   # overlapping; shouldn't happen in a tiling
    return 0


def group_rects_into_rooms(rects, heights, tol, is_seed):
    """Assign every rectangle a room id.

    Only "seed" rectangles -- those wide enough to actually be a room, see
    --room-seed-width -- get to define one: seeds that touch AND agree on
    floor and ceiling within tol are union-found into the same room, so a
    long corridor tiled into several wide rectangles collapses back to one
    room, while a taller hall joined to it end-on stays separate.

    The tiling also produces thin slivers along walls (0.2 m wide strips
    left over after the big rectangles are carved out). Those must NOT
    define rooms of their own: at a grazing angle the scan barely reaches
    their ceiling, so their measured height is far too low and they would
    show up as spurious short rooms. Each sliver instead joins whichever
    already-assigned neighbour it shares the longest edge with, repeatedly
    until none are left (a sliver touching only another sliver gets
    resolved on a later pass). A sliver that touches nothing assigned
    falls back to a room of its own."""
    n = len(rects)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seeds = [i for i in range(n) if is_seed[i]]
    for si, i in enumerate(seeds):
        for j in seeds[si + 1:]:
            if shared_edge_cells(rects[i], rects[j]) <= 0:
                continue
            if (abs(heights[i][0] - heights[j][0]) <= tol
                    and abs(heights[i][1] - heights[j][1]) <= tol):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = [None] * n
    renumber = {}
    for i in seeds:
        root = find(i)
        if root not in renumber:
            renumber[root] = len(renumber)
        groups[i] = renumber[root]

    pending = [i for i in range(n) if groups[i] is None]
    while pending:
        progressed = False
        for i in list(pending):
            best_j, best_share = None, 0
            for j in range(n):
                if groups[j] is None:
                    continue
                share = shared_edge_cells(rects[i], rects[j])
                if share > best_share:
                    best_j, best_share = j, share
            if best_j is not None:
                groups[i] = groups[best_j]
                pending.remove(i)
                progressed = True
        if not progressed:
            break

    for i in pending:   # touches nothing assigned -- an island, its own room
        groups[i] = len(renumber)
        renumber[f"island{i}"] = len(renumber)
    return groups


def consensus_room_height(xyz, r_idx, c_idx, rects, floor_pct, ceiling_pct, fallback):
    """One floor/ceiling for a whole room, recomputed from the union of the
    points under all its rectangles -- not an average of the per-rectangle
    numbers, which would let a tiny sliver rectangle count as much as the
    20 m one next to it."""
    mask = np.zeros(len(r_idx), dtype=bool)
    for rect in rects:
        mask |= rect_point_mask(r_idx, c_idx, rect)
    n = int(mask.sum())
    if n == 0:
        return fallback[0], fallback[1], 0
    zz = xyz[mask, 2]
    return (float(np.percentile(zz, floor_pct)),
            float(np.percentile(zz, ceiling_pct)), n)


def boxes_to_geometry(boxes, x0, y0, cell_size, box_z, yaw_deg):
    """Convert (row0,row1,col0,col1) grid rectangles into world-frame box
    dicts. box_z is a parallel list of (floor_z, ceiling_z) per rectangle --
    every box in one room carries that room's shared pair, so a room comes
    out flat-topped, while different rooms are free to differ. If yaw_deg
    != 0 (the cloud was pre-rotated by -yaw_deg before gridding, see
    main()), corners are rotated back by +yaw_deg so the output lines up
    with the original, un-rotated point cloud."""
    cos_t, sin_t = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))

    def rot(x, y):
        return x * cos_t - y * sin_t, x * sin_t + y * cos_t

    out = []
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        floor_z, ceiling_z = box_z[i]
        x_lo, x_hi = x0 + c0 * cell_size, x0 + (c1 + 1) * cell_size
        y_lo, y_hi = y0 + r0 * cell_size, y0 + (r1 + 1) * cell_size
        width, depth, height = x_hi - x_lo, y_hi - y_lo, ceiling_z - floor_z

        footprint = [(x_lo, y_lo), (x_hi, y_lo), (x_hi, y_hi), (x_lo, y_hi)]
        footprint = [rot(x, y) for x, y in footprint]
        corners_3d = ([[x, y, floor_z] for x, y in footprint]
                       + [[x, y, ceiling_z] for x, y in footprint])

        out.append({
            "id": i,
            "x_min": min(p[0] for p in footprint), "x_max": max(p[0] for p in footprint),
            "y_min": min(p[1] for p in footprint), "y_max": max(p[1] for p in footprint),
            "z_min": floor_z, "z_max": ceiling_z,
            "width_m": float(width), "depth_m": float(depth), "height_m": float(height),
            "area_m2": float(width * depth), "volume_m3": float(width * depth * height),
            "cell_count": int((r1 - r0 + 1) * (c1 - c0 + 1)),
            "corners_3d": corners_3d,
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered)")
    ap.add_argument("--pcd", type=Path,
                    help="load xyz straight from a .pcd instead of a bag -- e.g. "
                         "BagFilter.m's loop_closed_map.pcd or "
                         "BagFilter_NoLoopClosure.m's noloop_corrected_map.pcd, both "
                         "already gravity/yaw-aligned. Mutually exclusive with --bag; "
                         "--topic/--store are ignored with --pcd")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE",
                    help="typestore for bags without embedded type defs")
    ap.add_argument("--yaw-deg", type=float, default=0.0,
                    help="rotate the cloud by -yaw_deg about Z before gridding (undoes a "
                         "known residual yaw so walls land on the X/Y grid), then rotate "
                         "the fitted boxes back by +yaw_deg so output stays in the "
                         "original frame. Leave at 0 for cloud already yaw-aligned "
                         "(e.g. BagFilter.m output)")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop (metres)")
    ap.add_argument("--sor", action="store_true", help="apply Statistical Outlier Removal")
    ap.add_argument("--sor-k", type=int, default=16)
    ap.add_argument("--sor-std", type=float, default=1.5)
    ap.add_argument("--no-declutter", action="store_true",
                    help="skip cluster-based noise removal entirely -- use every point in "
                         "the (cropped) cloud as-is")
    ap.add_argument("--single-hall", action="store_true",
                    help="isolate and fit only the single largest connected hall, dropping "
                         "every other one -- default (without this flag) fits boxes to "
                         "every hall in the cloud (each disconnected group of rooms is "
                         "gridded and tiled independently, see --min-cluster-points/"
                         "--min-blob-cells for the noise cutoff)")
    ap.add_argument("--cluster-gap", type=float, default=0.30,
                    help="max gap (m) for points to count as the same cluster (declutter)")
    ap.add_argument("--min-cluster-points", type=int, default=200,
                    help="3D declutter: drop a cluster entirely if it has fewer points than "
                         "this (noise/stray reflections, not a real hall); ignored with "
                         "--single-hall")
    ap.add_argument("--floor-pct", type=float, default=1.0,
                    help="height percentile used as floor_z (ignored if --floor-z is set)")
    ap.add_argument("--ceiling-pct", type=float, default=99.0,
                    help="height percentile used as ceiling_z (ignored if --ceiling-z is set)")
    ap.add_argument("--floor-z", type=float, default=None,
                    help="override every hall's estimated floor_z with this one fixed value "
                         "(must be given together with --ceiling-z)")
    ap.add_argument("--ceiling-z", type=float, default=None,
                    help="override every hall's estimated ceiling_z with this one fixed value "
                         "(must be given together with --floor-z)")
    ap.add_argument("--cell-size", type=float, default=0.20,
                    help="grid resolution (m) for the footprint -- also the smallest "
                         "possible gap/overhang a box boundary can resolve")
    ap.add_argument("--min-points-per-cell", type=int, default=3,
                    help="min points landing in a cell for it to count as occupied floor")
    ap.add_argument("--no-fill-holes", action="store_true",
                    help="don't fill small fully-enclosed gaps in the footprint (furniture/"
                         "columns) -- leave them as holes the tiling has to route around")
    ap.add_argument("--min-blob-cells", type=int, default=10,
                    help="grid declutter, applied within each hall separately: drop a "
                         "connected occupied blob entirely if it has fewer cells than this "
                         "(grid-resolution noise, not a real sub-region of that hall)")
    ap.add_argument("--room-seed-width", type=float, default=1.0,
                    help="m -- only boxes at least this wide (in their SHORTER footprint "
                         "dimension) may define a room and measure their own height. The "
                         "tiling leaves thin slivers along walls whose ceiling the scan "
                         "barely reaches at a grazing angle, so their own height reads far "
                         "too low; each sliver instead inherits the room of the neighbour it "
                         "shares the longest edge with. Raise it if narrow tiling residue is "
                         "still forming its own short rooms")
    ap.add_argument("--height-group-tol", type=float, default=0.30,
                    help="m -- two touching boxes are treated as the same room (and so share "
                         "one height) when both their floors and their ceilings agree within "
                         "this. Raise it to merge rooms that differ only slightly; lower it "
                         "to let a modest step in ceiling height split a room in two")
    ap.add_argument("--no-height-groups", action="store_true",
                    help="don't measure height per box/room -- go back to one floor_z/"
                         "ceiling_z per connected point cluster, i.e. the whole structure "
                         "shares one height if its rooms are joined through the floor")
    ap.add_argument("--min-box-points", type=int, default=50,
                    help="a box with fewer points than this under its footprint can't measure "
                         "its own height reliably (a 0.2 x 0.2 m sliver), so it inherits the "
                         "cluster-wide estimate and is grouped by that")
    ap.add_argument("--min-box-area", type=float, default=0.5,
                    help="m^2 -- stop tiling once the largest remaining rectangle would be "
                         "smaller than this; what's left is reported as leftover, uncovered "
                         "footprint rather than forced into a sliver box")
    ap.add_argument("--max-boxes", type=int, default=200, help="safety cap")
    ap.add_argument("--min-volume", type=float, default=0.0,
                    help="m^3 -- drop boxes smaller than this (width*depth*height) right "
                         "before writing the json, e.g. to discard slivers that survived "
                         "tiling. 0 (default) = no filtering. Unlike --min-box-area (which "
                         "steers the tiling itself), this is a post-filter -- dropped boxes "
                         "leave their footprint uncovered rather than being folded into "
                         "leftover_cells")
    ap.add_argument("--out", type=Path, default=SAVED_BOXES_DIR / "boxes.json",
                    help="where to write the fitted boxes (default: SavedBoxes/boxes.json "
                         "next to this script; the directory is created if missing)")
    args = ap.parse_args()

    if bool(args.bag) == bool(args.pcd):
        raise SystemExit("pass exactly one of --bag or --pcd")

    if args.pcd:
        xyz = load_pcd_cloud(args.pcd)
        source = args.pcd
    else:
        xyz = load_merged_cloud(args.bag, args.topic, args.store)
        source = args.bag

    roi = None
    if args.roi:
        roi = tuple(float(x) for x in args.roi.split(","))
        if len(roi) != 6:
            raise SystemExit("--roi needs 6 comma-separated values: xmin,xmax,ymin,ymax,zmin,zmax")
        xyz = crop_roi(xyz, roi)

    if args.sor:
        xyz = statistical_outlier_removal(xyz, k=args.sor_k, std_ratio=args.sor_std)

    if args.no_declutter:
        halls = [xyz]
    else:
        halls = split_into_halls(xyz, gap=args.cluster_gap, min_points=args.min_cluster_points,
                                  single_hall=args.single_hall)
    if not halls:
        raise SystemExit("no hall left after declutter -- lower --min-cluster-points, "
                          "raise --cluster-gap, or check the input")

    if args.yaw_deg != 0.0:
        cos_t, sin_t = np.cos(np.radians(-args.yaw_deg)), np.sin(np.radians(-args.yaw_deg))

        def rotate(a):
            a = a.copy()
            x, y = a[:, 0].copy(), a[:, 1].copy()
            a[:, 0] = x * cos_t - y * sin_t
            a[:, 1] = x * sin_t + y * cos_t
            return a

        halls = [rotate(h) for h in halls]
        print(f"rotated {len(halls)} hall(s) by {-args.yaw_deg:.2f} deg about Z before "
              f"gridding (boxes will be rotated back by {args.yaw_deg:.2f} deg)")

    # Every step from here down runs once PER HALL: each hall gets its own floor_z/
    # ceiling_z (so two halls at different heights come out right) and its own grid/
    # tiling (so a rectangle can never straddle the empty gap between two halls).
    min_box_cells = max(1, int(round(args.min_box_area / (args.cell_size ** 2))))
    all_boxes = []
    hall_summaries = []
    total_leftover_cells = 0
    next_box_id = 0

    next_room_id = 0

    for h_idx, hall_xyz in enumerate(halls):
        print(f"\n--- cluster {h_idx} ({len(hall_xyz)} points) ---")

        # Cluster-wide height: used to bound the occupancy grid, and as the fallback for
        # any rectangle too small to measure its own. NOT what the boxes end up using --
        # that's the per-room consensus below.
        if args.floor_z is not None and args.ceiling_z is not None:
            floor_z, ceiling_z = args.floor_z, args.ceiling_z
        else:
            est_floor, est_ceiling = estimate_floor_ceiling(hall_xyz, args.floor_pct, args.ceiling_pct)
            floor_z = args.floor_z if args.floor_z is not None else est_floor
            ceiling_z = args.ceiling_z if args.ceiling_z is not None else est_ceiling

        occ, x0, y0 = build_occupancy_grid(hall_xyz, args.cell_size, floor_z, ceiling_z,
                                            args.min_points_per_cell)
        occ = clean_grid(occ, fill_holes=not args.no_fill_holes, min_blob_cells=args.min_blob_cells)

        budget = args.max_boxes - len(all_boxes)
        if budget <= 0:
            print(f"  --max-boxes ({args.max_boxes}) already reached by earlier clusters, "
                  f"skipping this one entirely")
            continue
        raw_boxes, leftover_cells = tile_boxes(occ, min_box_cells=min_box_cells, max_boxes=budget)
        print(f"  tiling: {len(raw_boxes)} box(es), {leftover_cells} leftover occupied cell(s) "
              f"(< {args.min_box_area} m^2 threshold, "
              f"{leftover_cells * args.cell_size ** 2:.2f} m^2 total)")
        if not raw_boxes:
            continue

        r_idx, c_idx = grid_indices(hall_xyz, x0, y0, args.cell_size)
        fixed_height = args.floor_z is not None and args.ceiling_z is not None

        if fixed_height or args.no_height_groups:
            # one height for the whole cluster: --floor-z/--ceiling-z were given
            # explicitly, or room grouping was turned off
            groups = [0] * len(raw_boxes)
            reason = "--floor-z/--ceiling-z given" if fixed_height else "--no-height-groups"
            print(f"  height: one per cluster ({reason}), "
                  f"floor_z={floor_z:.3f} ceiling_z={ceiling_z:.3f}")
        else:
            per_rect = estimate_rect_heights(hall_xyz, r_idx, c_idx, raw_boxes,
                                              args.floor_pct, args.ceiling_pct,
                                              args.min_box_points, (floor_z, ceiling_z))
            seed_cells = args.room_seed_width / args.cell_size
            is_seed = [min(r1 - r0 + 1, c1 - c0 + 1) >= seed_cells
                       for (r0, r1, c0, c1) in raw_boxes]
            if not any(is_seed):
                # nothing is wide enough to be a room on its own (a cluster that is all
                # sliver) -- let every rectangle seed, rather than lumping the lot together
                is_seed = [True] * len(raw_boxes)
                print(f"  rooms: no box is >= {args.room_seed_width} m wide "
                      f"(--room-seed-width), letting every box define its own room")
            groups = group_rects_into_rooms(raw_boxes, per_rect, args.height_group_tol, is_seed)
            print(f"  rooms: {len(raw_boxes)} box(es) ({sum(is_seed)} wide enough to seed a "
                  f"room, {len(is_seed) - sum(is_seed)} sliver(s) attached to a neighbour) "
                  f"-> {max(groups) + 1} room(s), height within {args.height_group_tol} m")

        n_groups = max(groups) + 1
        room_z = {}
        room_ids = {}
        for g in range(n_groups):
            member_rects = [raw_boxes[i] for i in range(len(raw_boxes)) if groups[i] == g]
            if fixed_height or args.no_height_groups:
                z0, z1, npts = floor_z, ceiling_z, len(hall_xyz)
            else:
                z0, z1, npts = consensus_room_height(hall_xyz, r_idx, c_idx, member_rects,
                                                      args.floor_pct, args.ceiling_pct,
                                                      (floor_z, ceiling_z))
            room_z[g] = (z0, z1)
            room_ids[g] = next_room_id
            next_room_id += 1
            print(f"    room {room_ids[g]}: {len(member_rects)} box(es), "
                  f"floor_z={z0:.3f} ceiling_z={z1:.3f} height={z1 - z0:.2f} m "
                  f"({npts} points)")
            hall_summaries.append({
                "hall": room_ids[g], "cluster": h_idx, "points": npts,
                "floor_z": z0, "ceiling_z": z1, "height_m": z1 - z0,
                "box_count": len(member_rects),
            })

        box_z = [room_z[groups[i]] for i in range(len(raw_boxes))]
        hall_boxes = boxes_to_geometry(raw_boxes, x0, y0, args.cell_size, box_z, args.yaw_deg)
        for i, b in enumerate(hall_boxes):
            b["id"] = next_box_id
            next_box_id += 1
            b["hall"] = room_ids[groups[i]]
            b["cluster"] = h_idx
            print(f"    box {b['id']} (room {b['hall']}): {b['width_m']:.2f} x "
                  f"{b['depth_m']:.2f} x {b['height_m']:.2f} m "
                  f"({b['area_m2']:.1f} m^2, {b['cell_count']} cells)")

        all_boxes.extend(hall_boxes)
        total_leftover_cells += leftover_cells

    print(f"\ntotal: {len(all_boxes)} box(es) across {len(hall_summaries)} room(s) "
          f"in {len(halls)} connected cluster(s), {total_leftover_cells} leftover "
          f"occupied cell(s) ({total_leftover_cells * args.cell_size ** 2:.2f} m^2)")
    if hall_summaries:
        heights = sorted({round(h["height_m"], 2) for h in hall_summaries})
        print(f"room heights: {heights} m")

    if args.min_volume > 0:
        kept = [b for b in all_boxes if b["volume_m3"] >= args.min_volume]
        dropped = len(all_boxes) - len(kept)
        if dropped:
            print(f"--min-volume {args.min_volume} m^3: dropped {dropped} box(es), "
                  f"{len(kept)} remain")
        all_boxes = kept
        for i, b in enumerate(all_boxes):
            b["id"] = i
        # hall_summaries' box_count was set during tiling, above -- keep it in sync with
        # what actually survived the filter (each hall's summary is otherwise used as-is
        # by downstream tools, e.g. to_openstudio.py's per-hall Space grouping).
        for hs in hall_summaries:
            hs["box_count"] = sum(1 for b in all_boxes if b["hall"] == hs["hall"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": str(source),
        "topic": args.topic if args.bag else None,
        "cell_size_m": args.cell_size,
        "yaw_deg": args.yaw_deg,
        "halls": hall_summaries,
        "boxes": all_boxes,
        "leftover_cells": total_leftover_cells,
        "leftover_area_m2": total_leftover_cells * args.cell_size ** 2,
    }, indent=2))
    print(f"wrote {len(all_boxes)} box(es) to {args.out}")


if __name__ == "__main__":
    main()
