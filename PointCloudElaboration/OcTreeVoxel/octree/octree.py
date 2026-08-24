"""Octree spatial subdivision of a point cloud.

The root is the cubic bounding box of the cloud. Each node is split into 8
octants; only *occupied* octants (those containing points) are created, and
subdivision stops at `max_depth`. The occupied leaves at depth d are the same
voxels the fast voxelizer produces at edge = root_extent / 2**d — octree.py is
the explicit hierarchical structure (useful for reporting per-level node counts
and for understanding the sampling), while voxelizer.py is the fast path used
by the interactive viewer.

Copied verbatim from PointCloudElaboration/OcTree/octree/octree.py (OcTreeVoxel
is self-contained, no cross-import). No dependencies beyond numpy, so no
adaptation was needed.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OctreeNode:
    center: np.ndarray  # (3,) cube center
    size: float  # cube edge length
    depth: int
    point_indices: np.ndarray | None = None  # set only at leaves
    children: list["OctreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def _cube_root(points: np.ndarray) -> tuple[np.ndarray, float]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2.0
    size = float((hi - lo).max())
    return center, size


def build_octree(points: np.ndarray, max_depth: int) -> OctreeNode:
    """Build an octree over `points`, subdividing occupied octants to max_depth."""
    points = np.asarray(points, dtype=np.float64)
    center, size = _cube_root(points)
    root = OctreeNode(center=center, size=size, depth=0)
    _subdivide(root, points, np.arange(len(points)), max_depth)
    return root


def _subdivide(
    node: OctreeNode, points: np.ndarray, indices: np.ndarray, max_depth: int
) -> None:
    if node.depth >= max_depth or len(indices) == 0:
        node.point_indices = indices
        return

    # Assign each point to one of 8 octants by sign relative to the node center.
    pts = points[indices]
    offset = (pts >= node.center).astype(np.int64)  # (k, 3) each in {0,1}
    octant = offset[:, 0] * 4 + offset[:, 1] * 2 + offset[:, 2]  # 0..7

    child_size = node.size / 2.0
    quarter = child_size / 2.0
    for oct_id in range(8):
        mask = octant == oct_id
        if not mask.any():
            continue  # empty octant -> not created (sparse octree)
        sx, sy, sz = (oct_id >> 2) & 1, (oct_id >> 1) & 1, oct_id & 1
        sign = np.array([sx, sy, sz]) * 2 - 1  # {0,1} -> {-1,+1}
        child_center = node.center + sign * quarter
        child = OctreeNode(center=child_center, size=child_size, depth=node.depth + 1)
        node.children.append(child)
        _subdivide(child, points, indices[mask], max_depth)


def level_counts(root: OctreeNode) -> list[int]:
    """Number of occupied nodes at each depth, root first."""
    counts: list[int] = []
    frontier = [root]
    while frontier:
        counts.append(len(frontier))
        nxt: list[OctreeNode] = []
        for node in frontier:
            nxt.extend(node.children)
        frontier = nxt
    return counts


def leaf_voxels(root: OctreeNode) -> tuple[np.ndarray, float]:
    """Centers of the deepest occupied leaves and their (uniform) edge size."""
    leaves: list[OctreeNode] = []

    def walk(node: OctreeNode) -> None:
        if node.is_leaf:
            leaves.append(node)
        else:
            for c in node.children:
                walk(c)

    walk(root)
    centers = np.array([leaf.center for leaf in leaves], dtype=np.float64)
    size = leaves[0].size if leaves else 0.0
    return centers, size
