"""Fast voxel sampling of a point cloud (numpy).

Each point is mapped to an integer 3-D cell index
    idx = floor((xyz - origin) / voxel_size)
and points sharing a cell collapse to a single voxel. For each occupied voxel
we return its cell-center coordinate, the majority semantic class of the points
it contains, and the number of points it absorbed.

Two lattices are provided:
- `voxelize`        : origin at the cloud's min corner (a plain uniform grid).
- `voxelize_octree` : origin at the centered bounding *cube*, edge
                      root_extent / 2**depth, with the far boundary folded into
                      the last cell. This reproduces exactly the set of occupied
                      leaves of a complete octree at that depth (see octree.py),
                      and is what the viewer draws.

Copied from PointCloudElaboration/OcTree/octree/voxelizer.py (OcTreeVoxel is
self-contained, no cross-import). The original's `labels`/majority-class
plumbing depended on OcTree/octree/classes.py (TUM-FACADE semantic ids), which
was intentionally not copied here -- this module is geometry-only (no
per-point classification yet). MAX_CLASS_ID is inlined as 0 (single dummy
class); callers pass an all-zeros `labels` array and the `labels` field of the
returned VoxelGrid is unused.
"""

from dataclasses import dataclass

import numpy as np

MAX_CLASS_ID = 0  # geometry-only here; see module docstring


@dataclass
class VoxelGrid:
    centers: np.ndarray  # (M, 3) float, voxel cell centers
    labels: np.ndarray  # (M,) int, majority class per voxel
    counts: np.ndarray  # (M,) int, points per voxel
    voxel_size: float
    origin: np.ndarray  # (3,) float, lower corner used for indexing

    def __len__(self) -> int:
        return len(self.centers)


def _grid_from_index(idx, labels, voxel_size, origin) -> VoxelGrid:
    """Collapse integer cell indices into a VoxelGrid (center + majority class)."""
    # Pack the (i, j, k) cell index into a single int64 key so uniqueness is
    # found with a 1-D sort instead of np.unique(axis=0)'s multi-column
    # lexsort -- an order of magnitude faster on multi-million-point clouds.
    lo = idx.min(axis=0)
    shifted = idx - lo
    ranges = shifted.max(axis=0) + 1
    key = (
        shifted[:, 0].astype(np.int64)
        + shifted[:, 1].astype(np.int64) * ranges[0]
        + shifted[:, 2].astype(np.int64) * ranges[0] * ranges[1]
    )

    _, first, inverse, counts = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    n_vox = len(first)

    centers = (idx[first] + 0.5) * voxel_size + origin

    n_classes = MAX_CLASS_ID + 1
    clipped = np.clip(labels, 0, MAX_CLASS_ID).astype(np.int64)
    flat = inverse * n_classes + clipped
    hist = np.bincount(flat, minlength=n_vox * n_classes).reshape(n_vox, n_classes)
    majority = hist.argmax(axis=1).astype(np.int32)

    return VoxelGrid(
        centers=centers,
        labels=majority,
        counts=counts,
        voxel_size=float(voxel_size),
        origin=np.asarray(origin, float),
    )


def voxelize(
    points: np.ndarray,
    labels: np.ndarray,
    voxel_size: float,
    origin: np.ndarray | None = None,
) -> VoxelGrid:
    """Sample `points` into a uniform voxel grid of edge `voxel_size`.

    origin defaults to points.min(axis=0) (min-corner-anchored lattice).
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    origin = points.min(axis=0) if origin is None else np.asarray(origin, float)

    idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    return _grid_from_index(idx, labels, voxel_size, origin)


def voxelize_octree(points: np.ndarray, labels: np.ndarray, depth: int) -> VoxelGrid:
    """Voxels of a complete octree at `depth` (centered cube lattice).

    Equivalent to the occupied leaves of build_octree(points, depth): same cube
    origin, edge root_extent/2**depth, with points on the far face folded into
    the last cell (matching the octree's `>= center` octant test).
    """
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)

    origin = cube_origin(points)
    extent = root_extent(points)
    n = 2**depth
    voxel_size = extent / n

    idx = np.floor((points - origin) / voxel_size).astype(np.int64)
    np.clip(idx, 0, n - 1, out=idx)  # fold the far boundary into the last cell
    return _grid_from_index(idx, labels, voxel_size, origin)


def filter_by_count(grid: VoxelGrid, min_count: int) -> VoxelGrid:
    """Keep only voxels with at least `min_count` points (a density filter).

    Drops sparse voxels — often isolated LiDAR returns or scan noise that show
    up as disconnected boxes in the viewer. This is a per-voxel threshold only:
    it does not check spatial connectivity, so a voxel with min_count+ points
    that has no occupied neighbours can still remain isolated after filtering.
    """
    if min_count <= 1:
        return grid
    mask = grid.counts >= min_count
    return VoxelGrid(
        centers=grid.centers[mask],
        labels=grid.labels[mask],
        counts=grid.counts[mask],
        voxel_size=grid.voxel_size,
        origin=grid.origin,
    )


def verify_nonempty(grid: "VoxelGrid", n_points: int) -> tuple[bool, int, int]:
    """Check the voxel invariant after a (re)voxelization.

    Returns (ok, n_empty, n_binned):
    - every voxel must hold at least one point (n_empty == 0), and
    - every input point must land in exactly one voxel (n_binned == n_points).
    Both always hold by construction; this makes that explicit so a broken
    change (or a stray voxel) is caught the moment the voxel size changes.
    """
    n_empty = int((grid.counts < 1).sum())
    n_binned = int(grid.counts.sum())
    ok = n_empty == 0 and n_binned == n_points
    return ok, n_empty, n_binned


def root_extent(points: np.ndarray) -> float:
    """Edge length of the cubic bounding box of the cloud."""
    points = np.asarray(points, dtype=np.float64)
    return float((points.max(axis=0) - points.min(axis=0)).max())


def cube_origin(points: np.ndarray) -> np.ndarray:
    """Lower corner of the centered cubic bounding box (matches the octree root)."""
    points = np.asarray(points, dtype=np.float64)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    return center - extent / 2.0


def voxel_size_for_depth(points: np.ndarray, depth: int) -> float:
    """Voxel edge of a complete octree at `depth`: root_extent / 2**depth."""
    return root_extent(points) / (2**depth)
