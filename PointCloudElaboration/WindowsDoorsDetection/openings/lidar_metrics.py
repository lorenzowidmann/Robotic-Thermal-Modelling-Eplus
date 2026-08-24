"""Metric size of a masked region, measured from the LiDAR scan of that frame.

Why this now exists at stage 1
------------------------------
geometry.py's original header says, correctly, that a metric gate on a WINDOW
mask cannot work: glazing returns nothing, and a window mask spans a
foreground mullion and a background seen through the glass, so
`pixel_height * median_depth / fy` measured 2.4-3.2 m for ordinary openings.

None of that argument applies to a DOOR. A door leaf is opaque, it is one
surface at one depth, and the LiDAR hits it. The pixel rules that decide
whether a door candidate survives -- minimum width, floor contact, "not taller
than a glazed wall" -- are all depth-dependent thresholds masquerading as
constants: the same door at 4 m and at 12 m differs by 3x in pixels. That is
why a real door at the far end of the corridor is 87 px tall and gets thrown
out by a rule written for one 5 m away.

So the LiDAR is consulted BEFORE a door candidate is discarded, and only for
doors. What it can do:

  * RESCUE a candidate a pixel rule wants to kill, when the metric size is
    door-like.
  * REJECT a candidate the pixel rules would have kept, when the metric size
    is definitively not door-like.
  * ABSTAIN, which is the common case and the important one. Too few points,
    or points spread over too many depths, means there is no measurement --
    and no measurement must never be read as a failed measurement. The pixel
    rules then decide alone, exactly as before.

The depth-spread guard is what makes abstention honest. A mask that straddles
a near mullion and a far wall has a bimodal depth histogram, and its median is
a number that describes no surface at all; `nearest_clouds_for_targets` will
happily hand one over. p95/p05 > `max_depth_ratio` is the check, and it is the
same failure the README documents for the densified-depth experiment (a
4.8-17.5 m region reporting a 4.4 m median).

Venv note: this module imports rosbags, so stage 1 now needs it too. It is
imported lazily, from load_clouds() only, so a run without --bag stays on the
old torch-only dependency set.
"""

import numpy as np

# A door OPENING, generously. Not a door leaf: the mask is the whole opening
# -- leaf plus frame plus reveal -- and this pipeline has to accept passages
# and double doors as well as single leaves, because opening_table.csv has one
# `door` class to express all three.
#
# The width ceiling was 2.20 and it was wrong. Session 9's corridor-end opening
# measures 2.55 m wide by 2.77 m tall (@34.4 m, consistently across frames) and
# is a real passage; 2.20 rejected it as `door_metric_dims`. Raised to 2.80,
# with the height ceiling to 3.00 for the same reason. The FLOOR is what does
# the real work here: session 9's persistent false positive, a dark recess
# behind a radiator, measures 0.37-0.83 m tall by ~0.43 m wide and is nowhere
# near either bound.
MIN_DOOR_H_M = 1.60
MAX_DOOR_H_M = 3.00
MIN_DOOR_W_M = 0.50
MAX_DOOR_W_M = 2.80
# Below this many LiDAR returns inside the mask there is no measurement.
MIN_POINTS = 12
# p95/p05 depth ratio above which the mask is not one surface, so its median
# depth describes nothing.
#
# 1.35 was a guess and it sat just below reality: on session 9 frames 1-5, FOUR
# of six door candidates abstained with `multi_depth` at 1.36, 1.40, 1.41 and
# 1.42 -- i.e. the check was switching itself off on almost every real
# candidate, and the same object flipped between `unknown` and `bad` from one
# frame to the next purely on scan jitter around the cutoff. 1.50 is measured
# rather than guessed: it admits the oblique recess that spans 4.74-6.74 m
# across its mask (a genuine surface seen at a steep angle) while still
# rejecting the corridor-run glazing case the README measured at 4.8-17.5 m
# (ratio 3.6), which is the failure this guard exists for.
MAX_DEPTH_RATIO = 1.50


DEFAULT_CLOUD_SOURCE = "raw"
RAW_LIDAR_TOPIC = "/livox/lidar"
REGISTERED_TOPIC = "/cloud_registered"
POSE_TOPIC = "/Odometry"


def load_clouds(bag, timestamps, store: str = "ROS2_HUMBLE",
                source: str = DEFAULT_CLOUD_SOURCE,
                lidar_topic: str = RAW_LIDAR_TOPIC,
                registered_topic: str = REGISTERED_TOPIC,
                pose_topic: str = POSE_TOPIC,
                loader_dir=None, traj=None):
    """The nearest world cloud per timestamp, from one of two sources.

    Both stages call this, so they cannot silently read different clouds --
    which matters, because stage 1's door measurements and stage 2's votes have
    to agree about what the sensor saw.

    source="raw" (the DEFAULT)
        ../../LivoxLidarOdometryLoader: the untouched `/livox/lidar` returns,
        placed in world by `/Odometry`'s poses. 83k points per scan over
        +-60 deg from 1.02 m.

    source="registered"
        FAST-LIO's `/cloud_registered`, via project_to_flir.py's reader. What
        this pipeline used to read, kept for comparison only.

    `raw` is the default because `registered` is measurably broken for this
    purpose on session 9: FAST-LIO's preprocessing drops it to 6.5k points
    inside a +-17 deg cone starting at 4.05 m, which projects to ~8% of the ZED
    frame. Both window bays fall outside it and collect ZERO returns even
    inside their bounding boxes, so no window voxel can exist regardless of
    what stage 1 decided. On the raw cloud the same bays get ~5 800 returns
    each at 1.6-5.3 m.

    The poses are FAST-LIO's either way -- this changes which POINTS are
    carried, not where the rig thought it was. If the crop damaged the
    trajectory, re-running FAST-LIO is the only fix and this is not it.

    `traj` overrides where the poses come from (source="raw" only): pass a
    LivoxLidarOdometryLoader Trajectory and the points are placed by THAT
    instead of by this bag's own `/Odometry`. This is how the drift-corrected
    trajectory out of 3DModelPointCloudExtraction/BagFilter*.m gets used -- the
    scans still come from the original bag, because the correction only ever
    touched the poses.
    """
    if source == "raw":
        import sys
        if loader_dir is not None and str(loader_dir) not in sys.path:
            sys.path.insert(0, str(loader_dir))
        import livox_odometry_loader as lol

        return lol.nearest_clouds_for_targets(
            bag, timestamps, lidar_topic, store, odom_topic=pose_topic, traj=traj)
    if source != "registered":
        raise ValueError(f"unknown cloud source {source!r}; use 'raw' or 'registered'")
    return _load_registered(bag, timestamps, registered_topic, store)


def _load_registered(bag, timestamps, topic: str, store: str):
    """One pass over the bag -> the nearest `/cloud_registered` per timestamp.

    Reuses EmissivityCalculation/project_to_flir.py's reader rather than
    re-implementing it, so stage 1 and stage 2 read the bag identically. The
    caller is responsible for having imported SensorFusionLoader FIRST -- see
    the import-order note in opening_voxel_consensus.py's header.
    """
    from project_to_flir import nearest_clouds_for_targets

    return nearest_clouds_for_targets(bag, timestamps, topic, store)


class FrameDepth:
    """The LiDAR returns of one frame, as pixel coordinates plus depth.

    Built once per frame and queried once per door candidate. Points outside
    the image, or behind the camera, are already dropped by
    project_lidar_to_camera's `valid` mask.
    """

    def __init__(self, uv, depth, valid, shape):
        h, w = shape
        px = np.round(uv[valid]).astype(int)
        px[:, 0] = np.clip(px[:, 0], 0, w - 1)
        px[:, 1] = np.clip(px[:, 1], 0, h - 1)
        self.col = px[:, 0]
        self.row = px[:, 1]
        self.depth = depth[valid].astype(float)
        self.shape = shape

    def in_region(self, region: np.ndarray) -> np.ndarray:
        """Depths of the returns whose pixel falls inside the MASK.

        The mask, not the bbox: a door candidate's box overlaps the wall beside
        it and the floor below it, and both would drag the median.
        """
        if self.col.size == 0:
            return self.depth[:0]
        return self.depth[region[self.row, self.col]]


def measure_region(region: np.ndarray, frame_depth: "FrameDepth", K,
                   min_points: int = MIN_POINTS,
                   max_depth_ratio: float = MAX_DEPTH_RATIO) -> dict:
    """Metric height/width of a masked region.

    ALWAYS returns a dict, never None, and the dict always says whether it
    holds a measurement (`usable`) and, when it does not, `reason` -- one of
    `few_points` or `multi_depth`. That matters more than it sounds: an
    abstention is the common outcome, and a run that reports 4 abstentions
    without saying which kind is a run nobody can act on. `few_points` says
    the surface returned nothing (glazing, distance, occlusion); `multi_depth`
    says the mask is not one surface and the fix is upstream, in what got
    merged into it.

    The size estimate is the pinhole one -- `extent_px * depth / focal` --
    deliberately, because it uses the MASK's full pixel extent and only borrows
    the scale from the LiDAR. Measuring the 3-D extent of the returns
    themselves would systematically under-report: LiDAR covers a door leaf
    densely but stops short of the top edge and the reveal, and returns nothing
    at all off a glazed panel in the door.
    """
    d = frame_depth.in_region(region)
    if d.size < min_points:
        return {"usable": False, "reason": "few_points", "n_points": int(d.size)}
    p05, p50, p95 = np.percentile(d, [5, 50, 95])
    ratio = float(p95 / max(1e-6, p05))
    if ratio > max_depth_ratio:
        return {"usable": False, "reason": "multi_depth", "n_points": int(d.size),
                "depth_m": round(float(p50), 2), "depth_ratio": round(ratio, 2),
                "depth_p05_m": round(float(p05), 2), "depth_p95_m": round(float(p95), 2)}
    ys, xs = np.nonzero(region)
    h_px = float(ys.max() - ys.min() + 1)
    w_px = float(xs.max() - xs.min() + 1)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    return {
        "usable": True,
        "n_points": int(d.size),
        "depth_m": round(float(p50), 2),
        "depth_ratio": round(ratio, 2),
        "height_m": round(h_px * float(p50) / fy, 2),
        "width_m": round(w_px * float(p50) / fx, 2),
    }


def door_verdict(metric: dict | None,
                 min_h: float = MIN_DOOR_H_M, max_h: float = MAX_DOOR_H_M,
                 min_w: float = MIN_DOOR_W_M, max_w: float = MAX_DOOR_W_M) -> str:
    """'ok' | 'bad' | 'unknown'.

    'unknown' covers both "no --bag was given" (metric is None) and "the bag
    was read but this region has no usable measurement", and it is NOT a soft
    'bad' -- callers must fall through to the pixel rules on it. Conflating the
    two would make every glazed or distant candidate fail for lack of evidence,
    which is the exact mistake geometry.py's header warns about.
    """
    if metric is None or not metric.get("usable"):
        return "unknown"
    h, w = metric["height_m"], metric["width_m"]
    return "ok" if (min_h <= h <= max_h and min_w <= w <= max_w) else "bad"
