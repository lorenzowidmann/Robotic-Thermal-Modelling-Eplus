"""
For each pose window of a rosbag2 LiDAR recording, print a depth (X) histogram
of the points and suggest tight px/py/pz passthrough bounds for
livox_hap_pattern.launch, instead of relying on one generic range for every
pose.

Reuses the pure-stdlib rosbag2 .db3 CustomMsg/PointCloud2 reader already
written and used in RecordCheck/detect_board_poses.py (same struct parsing,
no ROS install needed) -- these two folders must stay siblings.

Method:
  0. BACKGROUND SUBTRACTION (across all --windows, done once up front): the
     rover doesn't move between poses, only the board (+ the person holding
     it) does. A flat wall or floor facing the sensor produces the exact
     same kind of narrow, isolated depth peak as the board -- except it
     covers most of the LiDAR's FOV, so it contributes orders of magnitude
     more points (tens of thousands vs. a few hundred/thousand for the
     0.7x1.0 m board). Left unfiltered, the depth histogram below always
     locks onto the nearest big static surface, not the board -- which is
     why naive peak-picking on raw points returns near-identical px stuck
     at --x-lo on every pose. Fix: voxelize (--voxel, default 0.20 m) and
     mark a voxel as background if it's occupied in >= --bg-frac (default
     0.6) of the sampled chunks; points in a background voxel are dropped
     before anything else runs.

     WHICH chunks decides whether this works at all. The model is built from
     --bg-blocks (default 60) short samples spread across the WHOLE bag, NOT
     from the pose windows -- see background_blocks(). Modelling it from the
     pose windows alone (the original behaviour, still available as
     --bg-blocks 0) deletes the board itself whenever the board sits in
     roughly the same place each pose, which is the normal case for a
     stand-mounted target: measured here, it stayed within x 4.11-4.22 m
     across all 9 poses, i.e. inside one voxel. That left nothing but
     furniture residue to lock onto, and no amount of scoring downstream
     could recover from it.

     CAVEAT, checked against this rig's actual data: this room also has a
     white double-door cabinet with a window cut into one leaf (visible
     behind the board in the ZED photos of the same session) -- a static
     structure whose door/window panels happen to be close to the board's
     own ~0.7x1.0 m size. Background subtraction removes the *walls* and
     *floor* reliably (verified: without it every pose lands right at
     --x-lo with a ~4.5 m py spread; with it, spans shrink to well under
     1 m), but a same-sized static panel can still survive it and pass the
     footprint check below -- no purely geometric/statistical method can
     tell "board" from "cabinet panel of about the same size" apart on
     depth+size alone. That's why step 6 below prints several ranked
     candidates, not just one: cross-check them against the ZED photo of
     that pose (ground truth for where the board actually was) rather than
     trusting the auto-pick blind.
  1. Sample a handful of scans spread across the window, merge their points
     (also used for the background model above).
  2. Restrict the foreground points to a generous plausible band (Y, Z) wide
     enough to contain a hand-held board anywhere a person could
     realistically hold it, but narrow enough to drop obvious floor/ceiling
     returns.
  3. Histogram the remaining points' X (depth) in 0.1 m bins and print it, so
     the peak corresponding to the board is visible by eye.
  4. Auto-suggest px_min/px_max as the contiguous run of bins around the
     highest peak whose count is above --peak-frac of the max (default 0.4),
     expanded by --margin (default 0.3 m); then py/pz as the
     --pct/100-(100-pct) percentile range of the points falling inside that
     X range.
  5. Sanity-check the resulting py/pz footprint against the known board size
     (--board-w/--board-h, default 0.70/1.00 m, +/- --board-tol): background
     removal isolates "things that moved" (board + operator), not the board
     alone, so a pose whose footprint is way off ~0.7x1.0 m is probably
     picking up mostly the person -- printed as a WARNING, not silently
     trusted.
  6. Alongside the auto-pick, print up to --n-candidates (default 3) ranked
     alternate depth peaks with their own footprint, so a pose that picked
     the cabinet instead of the board is fixable from THIS run's output --
     find the candidate matching the ZED photo of that pose and feed its
     range straight to --fix-x, no need to rerun just to see the histogram
     again.
  7. --tune (off by default): peak-picking (step 4) always centers on the
     tallest depth bin, which is the cabinet whenever the board's own return
     is weaker than the background residue left behind it -- no amount of
     --peak-frac tuning fixes that, since it never even looks at candidate
     windows that aren't built around a tall bin. --tune instead SLIDES a
     window of every width in [--tune-width-min, --tune-width-max] across
     [--x-lo, x_hi] in --tune-step increments and keeps whichever position+
     width minimizes |width_y - board_w| + |height_z - board_h| directly --
     i.e. it optimizes for "looks like the board" instead of "has the most
     points", so it finds real board windows the coarse peak-picking misses
     entirely. Much slower (a full 2D scan per pose instead of one histogram
     argmax), so it's opt-in; the ranked candidates from step 6 are still
     printed alongside it for comparison, and the tuned window is only used
     as the pose's result if it beats the coarse auto-pick.

     FOOTPRINT ALONE IS NOT ENOUGH, and that is the whole reason for the
     gates below. A sliding window is free to stop anywhere, and "anywhere"
     includes the sparse VALLEY between two real depth peaks, where a
     hundred stray returns scatter across roughly board-sized Y/Z and score
     near-perfectly. Measured on the NewCalibration bag: 5 of 9 poses came
     back with 63-196 points while poses that genuinely hit the board
     returned 947-1456. So every candidate window must now pass, cheapest
     test first (see window_quality):
       --tune-min-pts       raw point count (default: auto, scaled from
                            --n-scans/--point-stride -- 250 at defaults,
                            where a real board yields ~1000-2000)
       --board-tol          Y/Z footprint vs. --board-w x --board-h
       --tune-max-thickness PLANARITY: RMS distance to the best-fit plane.
                            The board is a flat panel (a few cm thick at
                            these ranges); free-space scatter, or a window
                            straddling two objects, is tens of cm. This is
                            the single strongest discriminator.
       --tune-min-fill      OCCUPANCY: fraction of 5 cm cells in the Y/Z box
                            that actually contain a point. The board is a
                            FILLED rectangle; a hollow scatter is mostly
                            holes even when it happens to be coplanar.
     Plus one RELATIVE gate, --tune-min-pts-frac: within a pose the board is
     by far the densest board-shaped planar patch, so survivors with fewer
     than that fraction of the best survivor's points are dropped. Absolute
     thresholds have to stay loose enough for the farthest pose; this
     catches what they let through.

     Note what the gates do NOT prove: that the panel is THE board. A static
     cabinet door of about the same size is equally flat, filled and
     board-sized -- see the CAVEAT in step 0. The cross-pose warning at the
     end of the run (same px range recurring in 3+ poses = something that
     didn't move = not the board) is the check for that, and it stays a
     warning for a human with the ZED photo to settle.

Usage:
    py find_board_limits.py --bag <path> --windows 1:0:106 3:190:97 4:299:94 \\
        6:506:95 9:791:113 10:914:91 12:1151:98 15:1526:91 19:1980:98

    (each window is  label:start_s:duration_s , same numbers used for
    `rosbag play -s <start_s> -u <duration_s>`)

    # after eyeballing the histogram of one pose, force its X range and
    # recompute py/pz only:
    py find_board_limits.py --bag <path> --windows 9:791:113 --fix-x 3.2 4.1
"""

import argparse
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "RecordCheck"))
from detect_board_poses import _resolve_db3, _CUSTOM_MSG_TYPE, _POINT_CLOUD2_TYPE  # noqa: E402
from detect_board_poses import _custom_msg_xyz, _pointcloud2_xyz               # noqa: E402
from detect_board_poses import _voxel_keys                                    # noqa: E402


def sample_points(db3, start_s, dur_s, n_scans, point_stride):
    conn = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT id, name, type FROM topics WHERE type IN (?, ?)",
            (_CUSTOM_MSG_TYPE, _POINT_CLOUD2_TYPE),
        ).fetchall()
        if not rows:
            sys.exit("No LiDAR topic found in the bag.")
        topic_id, topic_name, msg_type = rows[0]
        parse = _custom_msg_xyz if msg_type == _CUSTOM_MSG_TYPE else _pointcloud2_xyz

        idx = cur.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id=? ORDER BY id",
            (topic_id,),
        ).fetchall()
        t0_ns = idx[0][1]
        lo_ns = t0_ns + int(start_s * 1e9)
        hi_ns = t0_ns + int((start_s + dur_s) * 1e9)
        window = [(mid, ns) for mid, ns in idx if lo_ns <= ns <= hi_ns]
        if not window:
            sys.exit(f"No scans in [{start_s}, {start_s + dur_s}]s -- check the offsets.")
        pick = window[::max(1, len(window) // n_scans)][:n_scans]

        pts = []
        for mid, _ in pick:
            blob = cur.execute("SELECT data FROM messages WHERE id=?", (mid,)).fetchone()[0]
            pts.append(parse(bytes(blob), point_stride))
        return np.vstack(pts) if pts else np.zeros((0, 3), dtype=np.float32), topic_name
    finally:
        conn.close()


def background_blocks(db3, n_blocks, n_scans, point_stride, block_span=3.0):
    """Sample n_blocks short chunks spread evenly over the WHOLE bag, to feed
    the background model.

    Why not just the pose windows (which is what this used to do): in a
    calibration session the board barely moves between poses. Measured on this
    rig's NewCalibration bag, across all 9 poses it stays inside
    x 4.11-4.22 m, y -0.27..+0.10 -- a span smaller than the 0.20 m background
    voxel in X. So "occupied in >= bg-frac of the POSES" is true of the board's
    own voxels, and background subtraction deleted THE BOARD along with the
    room, leaving only furniture residue for the peak-picker and the tuner to
    lock onto. That, not the scoring, is why every pose returned a static
    object or sparse scatter.

    Sampling the whole bag fixes it because the stretches BETWEEN poses -- when
    the board is being carried, re-clamped or out of frame -- are in the model
    too. Now the empty room is what recurs and the board doesn't. Verified: the
    same pose windows that yielded 0.30-0.56 m fragments under the old model
    yield a clean 0.96x0.46 m panel (plane RMS 0.017-0.04 m, fill 0.94-0.97)
    under this one."""
    conn = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        rows = cur.execute(
            "SELECT id, name, type FROM topics WHERE type IN (?, ?)",
            (_CUSTOM_MSG_TYPE, _POINT_CLOUD2_TYPE),
        ).fetchall()
        if not rows:
            sys.exit("No LiDAR topic found in the bag.")
        topic_id, _, msg_type = rows[0]
        parse = _custom_msg_xyz if msg_type == _CUSTOM_MSG_TYPE else _pointcloud2_xyz
        idx = cur.execute(
            "SELECT id, timestamp FROM messages WHERE topic_id=? ORDER BY id",
            (topic_id,),
        ).fetchall()
        t0_ns, dur_s = idx[0][1], (idx[-1][1] - idx[0][1]) / 1e9
        out = {}
        for b in range(n_blocks):
            lo = t0_ns + int(b * dur_s / n_blocks * 1e9)
            hi = lo + int(block_span * 1e9)
            pick = [m for m, ns in idx if lo <= ns <= hi][:n_scans]
            if not pick:
                continue
            out[b] = np.vstack([
                parse(bytes(cur.execute("SELECT data FROM messages WHERE id=?",
                                        (m,)).fetchone()[0]), point_stride)
                for m in pick])
        return out
    finally:
        conn.close()


def background_voxels(pose_points, voxel, bg_frac):
    """Voxels occupied in >= bg_frac of the poses -- the static room (walls,
    floor, rover), which the board and the person holding it never sit still
    in across poses. See the module docstring's step 0 for why this matters:
    without it, the walls/floor -- covering most of the FOV -- swamp the
    board's much smaller point count and the depth-histogram peak below
    always locks onto them instead."""
    counts = {}
    for pts in pose_points.values():
        for k in _voxel_keys(pts, voxel):
            counts[k] = counts.get(k, 0) + 1
    thr = max(1, int(np.ceil(bg_frac * len(pose_points))))
    return {k for k, c in counts.items() if c >= thr}


def foreground_points(pts, voxel, bg):
    """Points whose voxel is NOT in the background set -- i.e. things that
    moved between poses (the board + the operator holding it)."""
    if pts.shape[0] == 0:
        return pts
    q = np.floor(pts / voxel).astype(np.int64) + (1 << 19)
    keys = (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]
    mask = np.array([k not in bg for k in keys])
    return pts[mask]


def peak_ranges(hist, edges, peak_frac, margin, x_lo, x_hi, k=1):
    """Up to k non-overlapping (px_min, px_max) runs around the k tallest
    local maxima of hist, each run being the contiguous bins around its peak
    whose count is >= peak_frac of that peak's height, ranked tallest first.
    Same logic that used to pick only the single winner; now shared with the
    candidate table in main() so the auto-pick is always candidate #1."""
    order = np.argsort(-hist)
    used = []
    out = []
    for peak_i in order:
        if hist[peak_i] == 0 or len(out) >= k:
            break
        if any(lo <= peak_i <= hi for lo, hi in used):
            continue
        thr = hist[peak_i] * peak_frac
        lo = peak_i
        while lo > 0 and hist[lo - 1] >= thr:
            lo -= 1
        hi = peak_i
        while hi < len(hist) - 1 and hist[hi + 1] >= thr:
            hi += 1
        used.append((lo, hi))
        px_min = max(x_lo, edges[lo] - margin)
        px_max = min(x_hi, edges[hi + 1] + margin)
        out.append((px_min, px_max))
    return out


def suggest_bounds(pts, y_band, z_band, peak_frac, margin, pct, x_lo=0.3, x_hi=10.0):
    m = ((pts[:, 1] >= y_band[0]) & (pts[:, 1] <= y_band[1]) &
        (pts[:, 2] >= z_band[0]) & (pts[:, 2] <= z_band[1]) &
        (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi))
    p = pts[m]
    if len(p) < 20:
        # Same (hist, edges, banded) shape as the success path -- main()
        # unpacks it unconditionally, and --tune still wants the banded
        # points even when peak-picking found nothing to center on.
        return None, None, (None, None, p)

    bins = np.arange(x_lo, x_hi + 0.1, 0.1)
    hist, edges = np.histogram(p[:, 0], bins=bins)
    top = peak_ranges(hist, edges, peak_frac, margin, x_lo, x_hi, k=1)
    if not top:
        # Same (hist, edges, banded) shape as the success path -- main()
        # unpacks it unconditionally, and --tune still wants the banded
        # points even when peak-picking found nothing to center on.
        return None, None, (None, None, p)
    px_min, px_max = top[0]

    sl = p[(p[:, 0] >= px_min) & (p[:, 0] <= px_max)]
    return (px_min, px_max), sl, (hist, edges, p)


def _yz_box(pts, pct):
    """(y_lo, y_hi, z_lo, z_hi) of the pct/100-(100-pct) percentile spread."""
    ylo, yhi = np.percentile(pts[:, 1], [pct, 100 - pct])
    zlo, zhi = np.percentile(pts[:, 2], [pct, 100 - pct])
    return float(ylo), float(yhi), float(zlo), float(zhi)


def footprint(pts_in_range, pct):
    """(width_y, height_z) of the pct/100-(100-pct) percentile spread --
    same trimming as the py/pz suggestion, just without the added --margin,
    for comparing candidates against --board-w/--board-h."""
    ylo, yhi, zlo, zhi = _yz_box(pts_in_range, pct)
    return yhi - ylo, zhi - zlo


def plane_rms(pts):
    """RMS distance from the points to their best-fit plane -- i.e. how THICK
    the point set is (PCA: sqrt of the smallest covariance eigenvalue, whose
    eigenvector is the plane normal).

    The board is a flat panel, so its return is thin: sensor noise plus a bit
    of tilt, a few cm at these ranges. A window holding a thin scatter of
    stray returns spread through free space, or straddling two objects at
    different depths, is tens of cm thick. Footprint alone cannot tell those
    apart from the board -- 60 stray points span 1.0x0.7 m just as readily as
    1400 board points do -- which is exactly how the sliding search below
    used to settle in the valley BETWEEN the two real depth peaks."""
    if len(pts) < 3:
        return float("inf")
    d = pts - pts.mean(axis=0)
    ev = np.linalg.eigvalsh((d.T @ d) / len(d))     # ascending
    return float(np.sqrt(max(ev[0], 0.0)))


def fill_ratio(pts, box, cell):
    """Fraction of cell x cell bins that actually contain a point, over the
    trimmed Y/Z box. The board is a FILLED rectangle -- most cells get a
    return; a scatter is mostly holes. Complements plane_rms: a sparse
    scatter that happens to be coplanar (a few points off one far wall, say)
    passes the planarity test but fails this one."""
    ylo, yhi, zlo, zhi = box
    if yhi - ylo < cell or zhi - zlo < cell:
        return 0.0
    m = ((pts[:, 1] >= ylo) & (pts[:, 1] <= yhi) &
        (pts[:, 2] >= zlo) & (pts[:, 2] <= zhi))
    p = pts[m]
    ny = max(1, int(np.ceil((yhi - ylo) / cell)))
    nz = max(1, int(np.ceil((zhi - zlo) / cell)))
    iy = np.clip(((p[:, 1] - ylo) / cell).astype(np.int64), 0, ny - 1)
    iz = np.clip(((p[:, 2] - zlo) / cell).astype(np.int64), 0, nz - 1)
    return float(len(np.unique(iy * nz + iz))) / float(ny * nz)


def window_quality(sl, pct, gates):
    """Does this depth window look like the board? Returns a dict with n,
    footprint (wy, hz), thickness, fill, and `why` -- empty when every gate
    passed, otherwise the first gate it failed.

    Gates run cheapest-first, because the sliding search calls this a few
    thousand times per pose and the expensive geometry is worth running only
    on windows that are already board-sized:
      n         -- a real board at ~4.5 m over the sampled scans yields
                   ~1000-2000 points; a few dozen is scatter, not a panel.
      footprint -- Y/Z spread within board_tol of board_w x board_h.
      thickness -- planar (plane_rms).
      fill      -- filled rectangle, not hollow (fill_ratio).
    `gates` carries board_w, board_h, board_tol, min_pts, max_thick,
    min_fill and cell."""
    n = len(sl)
    q = {"n": n, "wy": 0.0, "hz": 0.0, "thick": float("nan"),
        "fill": float("nan"), "why": ""}
    if n < 3:
        q["why"] = "too few points"
        return q
    box = _yz_box(sl, pct)
    q["wy"], q["hz"] = box[1] - box[0], box[3] - box[2]
    if n < gates["min_pts"]:
        q["why"] = "too few points"
        return q
    if (abs(q["wy"] - gates["board_w"]) > gates["board_tol"] or
            abs(q["hz"] - gates["board_h"]) > gates["board_tol"]):
        q["why"] = "footprint mismatch"
        return q
    q["thick"] = plane_rms(sl)
    if q["thick"] > gates["max_thick"]:
        q["why"] = "not planar"
        return q
    q["fill"] = fill_ratio(sl, box, gates["cell"])
    if q["fill"] < gates["min_fill"]:
        q["why"] = "hollow (low fill)"
        return q
    return q


def plane_normal(pts):
    """Unit normal of the best-fit plane (PCA smallest eigenvector), flipped to
    face the sensor: +X is depth, so a surface the LiDAR can see has a normal
    with a negative X component."""
    d = pts - pts.mean(axis=0)
    _, evec = np.linalg.eigh((d.T @ d) / len(d))
    n = evec[:, 0]
    return -n if n[0] > 0 else n


def normal_angles(n):
    """(azimuth, elevation) of a plane normal, in degrees. Azimuth is the yaw
    off the sensor's line of sight, elevation the tilt up/down -- i.e. how the
    board was turned for that pose. (0, 0) means square-on to the LiDAR."""
    az = float(np.degrees(np.arctan2(n[1], -n[0])))
    el = float(np.degrees(np.arcsin(np.clip(n[2], -1.0, 1.0))))
    return az, el


def grow_to_plane(pts_band, x0, x1, pct, tol=0.05, pad=0.10, max_gap=0.15):
    """Extend a chosen depth window to the full extent of the panel it sits on.

    Why this is needed: the tuner scores the Y/Z FOOTPRINT, and a narrow slice
    of a yawed board satisfies that exactly as well as the whole panel -- a
    1.0 m board at 45 deg yaw is 0.7 m deep, but any 0.35 m slice through it
    still shows ~the right width and height, and wins on the width penalty.
    Measured on this bag: pose 6 kept 3236 of the panel's 7778 points and
    pose 3 only 2503 of 7486. For a passthrough crop box that is a real
    error -- half the board is thrown away before calibration ever sees it.

    So: fit the plane to the chosen window, keep the points within tol of that
    plane and inside the window's Y/Z box (padded by pad), then take the
    contiguous run in X containing the window, breaking at any gap wider than
    max_gap. The gap rule is what stops a coplanar wall further back from
    being swallowed; the Y/Z box is what stops the floor from being.

    Returns the widened (px_min, px_max), or the input unchanged when the
    window is too small to fit a plane to."""
    sl = pts_band[(pts_band[:, 0] >= x0) & (pts_band[:, 0] <= x1)]
    if len(sl) < 10:
        return x0, x1
    c = sl.mean(axis=0)
    d = sl - c
    _, evec = np.linalg.eigh((d.T @ d) / len(d))
    n = evec[:, 0]                                   # plane normal
    ylo, yhi, zlo, zhi = _yz_box(sl, pct)
    m = ((pts_band[:, 1] >= ylo - pad) & (pts_band[:, 1] <= yhi + pad) &
        (pts_band[:, 2] >= zlo - pad) & (pts_band[:, 2] <= zhi + pad) &
        (np.abs((pts_band - c) @ n) <= tol))
    xs = np.sort(pts_band[m][:, 0])
    if len(xs) < 10:
        return x0, x1
    inside = np.where((xs >= x0) & (xs <= x1))[0]
    if len(inside) == 0:
        return x0, x1
    i, j = int(inside[0]), int(inside[-1])
    while i > 0 and xs[i] - xs[i - 1] <= max_gap:
        i -= 1
    while j < len(xs) - 1 and xs[j + 1] - xs[j] <= max_gap:
        j += 1
    return float(min(x0, xs[i])), float(max(x1, xs[j]))


def tune_window(pts_band, x_lo, x_hi, width_min, width_max, step, pct, gates,
                width_penalty=0.15, min_pts_frac=0.2):
    """Slide a depth window of every width in [width_min, width_max] across
    [x_lo, x_hi] and keep whichever (start, width) passes every gate in
    window_quality AND minimizes
    |width_y - board_w| + |height_z - board_h| + width_penalty*width.

    Unlike peak_ranges (which always centers on the tallest depth bin -- the
    cabinet, whenever the board's own return is weaker than the background
    residue), this optimizes directly for "looks like the board", so it also
    considers windows that were never a histogram peak at all.
    O((width_max-width_min)/step * (x_hi-x_lo)/step) evaluations, a few
    thousand for typical ranges; points are pre-sorted by X so each window is
    two searchsorted calls and a view, not a full boolean mask.

    The gates are what make this search safe rather than reckless -- see
    window_quality and step 7 of the module docstring: scored on footprint
    alone it happily lands on a hundred stray points in the valley between
    two real depth peaks.

    On top of the absolute gates there's a RELATIVE one: after the sweep,
    survivors with fewer than min_pts_frac of the best survivor's point count
    are dropped. Absolute thresholds have to stay loose enough for the
    farthest pose, so this catches what they let through -- within one pose
    the board is by far the densest board-shaped planar patch.

    The width_penalty term matters too: a real board is a thin flat surface,
    so its true depth extent is small (a few cm to maybe 0.3-0.4 m at a steep
    tilt); a WIDE window spanning a meter or more of depth can still pass a
    loose 2-98 percentile check by just trimming away whatever doesn't fit
    -- it isn't "the board", it's an unprincipled grab-bag. Without the
    penalty this was observed to happily pick 1.5-1.8 m wide windows that
    scored well on Y/Z spread alone; the penalty biases ties toward the
    narrowest window that still matches, which is what a real board return
    looks like.

    Returns (best, info): best is (px_min, px_max, quality, score) or None if
    nothing passed; info carries the per-gate rejection counts and the window
    a footprint-only score WOULD have picked, so a pose that finds nothing
    reports why instead of just failing."""
    xs = np.ascontiguousarray(pts_band[np.argsort(pts_band[:, 0], kind="stable")])
    xcol = xs[:, 0]

    kept = []
    rej = {"too few points": 0, "footprint mismatch": 0,
        "not planar": 0, "hollow (low fill)": 0}
    loose = None
    width = width_min
    while width <= width_max + 1e-9:
        x0 = x_lo
        while x0 + width <= x_hi + 1e-9:
            i0 = int(np.searchsorted(xcol, x0, "left"))
            i1 = int(np.searchsorted(xcol, x0 + width, "right"))
            if i1 - i0 >= 3:
                q = window_quality(xs[i0:i1], pct, gates)
                score = (abs(q["wy"] - gates["board_w"]) +
                         abs(q["hz"] - gates["board_h"]) + width_penalty * width)
                if loose is None or score < loose[3]:
                    loose = (x0, x0 + width, q, score)
                if q["why"]:
                    rej[q["why"]] += 1
                else:
                    kept.append((x0, x0 + width, q, score))
            x0 += step
        width += step

    info = {"rejected": rej, "loose": loose, "dropped_sparse": 0, "n_best": 0}
    if not kept:
        return None, info
    info["n_best"] = max(k[2]["n"] for k in kept)
    dense = [k for k in kept if k[2]["n"] >= min_pts_frac * info["n_best"]]
    info["dropped_sparse"] = len(kept) - len(dense)
    return min(dense, key=lambda k: k[3]), info


def print_histogram(hist, edges, width=60):
    hmax = hist.max() if hist.max() > 0 else 1
    for i, c in enumerate(hist):
        if c == 0:
            continue
        bar = "#" * max(1, int(width * c / hmax))
        print(f"  {edges[i]:5.1f}-{edges[i+1]:<5.1f}m  {c:>5}  {bar}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="rosbag2 folder or .db3 file")
    ap.add_argument("--windows", nargs="+", required=True,
                    help="label:start_s:duration_s, one per pose "
                         "(same numbers as 'rosbag play -s ... -u ...')")
    ap.add_argument("--n-scans", type=int, default=10,
                    help="Scans sampled per window (default 10 -- also feeds the "
                         "background model, so needs more coverage than a single "
                         "pose's peak-picking alone would)")
    ap.add_argument("--point-stride", type=int, default=2,
                    help="Use every Nth point per scan (default 2)")
    ap.add_argument("--voxel", type=float, default=0.20,
                    help="Voxel size (m) for the background model (default 0.20 -- "
                         "HAP's non-repetitive scan pattern means a smaller voxel "
                         "doesn't reliably re-hit the same cells on a far wall across "
                         "only a handful of scans per window, leaving background "
                         "residue in the histogram)")
    ap.add_argument("--bg-blocks", type=int, default=60,
                    help="Short chunks sampled evenly across the WHOLE bag to build "
                         "the background model (default 60). The stretches between "
                         "poses -- board being carried or out of frame -- are what "
                         "make the room, and only the room, recur. 0 falls back to "
                         "the old behaviour of modelling the background from the "
                         "pose windows alone, which deletes the board itself "
                         "whenever it sits in roughly the same place each pose.")
    ap.add_argument("--bg-scans", type=int, default=2,
                    help="Scans per --bg-blocks chunk (default 2)")
    ap.add_argument("--bg-frac", type=float, default=0.6,
                    help="A voxel is background if occupied in >= this fraction of "
                         "the poses (default 0.6)")
    ap.add_argument("--n-candidates", type=int, default=3,
                    help="Ranked alternate depth peaks to print per pose, in case "
                         "the auto-pick is a static object, not the board (default "
                         "3; see the CAVEAT in the module docstring)")
    ap.add_argument("--y-band", type=float, nargs=2, default=[-2.5, 2.5],
                    help="Plausible Y band before peak-finding (default -2.5 2.5)")
    ap.add_argument("--z-band", type=float, nargs=2, default=[0.15, 2.2],
                    help="Plausible Z band before peak-finding (default 0.15 2.2 -- "
                         "excludes the floor return right in front of the sensor)")
    ap.add_argument("--x-lo", type=float, default=1.5,
                    help="Minimum depth considered (default 1.5 m -- excludes rover-"
                         "mounted hardware/cabling close to the sensor)")
    ap.add_argument("--board-w", type=float, default=0.70,
                    help="Known board width in m, for the footprint sanity check "
                         "(default 0.70)")
    ap.add_argument("--board-h", type=float, default=1.00,
                    help="Known board height in m, for the footprint sanity check "
                         "(default 1.00)")
    ap.add_argument("--board-tol", type=float, default=0.35,
                    help="Tolerance (m) on the footprint sanity check before "
                         "warning (default 0.35)")
    ap.add_argument("--peak-frac", type=float, default=0.4,
                    help="Bins >= this fraction of the peak count are kept as "
                         "the board's X range (default 0.4)")
    ap.add_argument("--margin", type=float, default=0.3,
                    help="Extra margin added to px_min/px_max (default 0.3 m)")
    ap.add_argument("--pct", type=float, default=2.0,
                    help="Percentile trimmed off each side of Y/Z inside the "
                         "picked X range (default 2.0)")
    ap.add_argument("--fix-x", type=float, nargs=2, default=None,
                    help="Skip peak-finding, use this px_min px_max directly "
                         "(after eyeballing a previous histogram)")
    ap.add_argument("--tune", action="store_true",
                    help="Also run a fine sliding-window search that directly "
                         "minimizes the footprint mismatch instead of picking the "
                         "tallest depth bin -- finds real board windows the coarse "
                         "peak-picking misses (step 7 in the module docstring); "
                         "used as the pose's result when it beats the coarse "
                         "auto-pick's footprint score. Slower -- a few hundred ms "
                         "per pose instead of near-instant.")
    ap.add_argument("--tune-width-min", type=float, default=0.35,
                    help="--tune: smallest window width tried, in m (default 0.35)")
    ap.add_argument("--tune-width-max", type=float, default=1.80,
                    help="--tune: largest window width tried, in m (default 1.80)")
    ap.add_argument("--tune-step", type=float, default=0.05,
                    help="--tune: step for both position and width, in m (default 0.05)")
    ap.add_argument("--tune-min-pts", type=int, default=None,
                    help="--tune: minimum points a window must have to be considered. "
                         "Default: auto, 50*--n-scans/--point-stride (250 at default "
                         "sampling), since a real board at ~4.5 m yields ~1000-2000 "
                         "points there -- a raw count in the dozens is stray scatter. "
                         "Scales with the sampling flags so changing them doesn't "
                         "silently disarm the gate.")
    ap.add_argument("--tune-max-thickness", type=float, default=0.06,
                    help="--tune: reject a window whose RMS distance to its best-fit "
                         "plane exceeds this, in m (default 0.06). The board is flat; "
                         "free-space scatter and windows straddling two objects are "
                         "not. Strongest of the gates -- raise it only for a heavily "
                         "tilted board or a noisier sensor.")
    ap.add_argument("--tune-min-fill", type=float, default=0.35,
                    help="--tune: reject a window unless at least this fraction of the "
                         "--tune-fill-cell cells inside its Y/Z box contain a point "
                         "(default 0.35). The board is a filled rectangle; a hollow "
                         "scatter that happens to span the right size is not.")
    ap.add_argument("--tune-fill-cell", type=float, default=0.05,
                    help="--tune: cell size (m) of the occupancy grid behind "
                         "--tune-min-fill (default 0.05)")
    ap.add_argument("--tune-min-pts-frac", type=float, default=0.2,
                    help="--tune: after the sweep, drop surviving windows with fewer "
                         "than this fraction of the densest survivor's point count "
                         "(default 0.2). Within one pose the board is by far the "
                         "densest board-shaped planar patch, so this catches sparse "
                         "look-alikes that the absolute gates -- which must stay loose "
                         "enough for the farthest pose -- let through.")
    ap.add_argument("--min-sep", type=float, default=10.0,
                    help="Two poses count as the same board orientation when their "
                         "plane normals are within this angle, in deg (default 10). "
                         "Drives the orientation-diversity report at the end of the "
                         "run -- what actually conditions a plane-based calibration.")
    ap.add_argument("--no-tune-grow", dest="tune_grow", action="store_false",
                    help="--tune: don't widen the chosen window to the full extent of "
                         "the plane it sits on. The widening is on by default because "
                         "the footprint score is equally happy with a narrow slice "
                         "through a yawed board as with the whole panel, which crops "
                         "away half the board (see grow_to_plane).")
    ap.add_argument("--tune-grow-tol", type=float, default=0.05,
                    help="--tune: max distance from the fitted plane for a point to "
                         "count when widening the window, in m (default 0.05)")
    ap.add_argument("--tune-width-penalty", type=float, default=0.15,
                    help="--tune: added to the score as width_penalty*window_width, "
                         "so ties favor the narrowest matching window instead of a "
                         "wide grab-bag that only passes on trimmed percentiles "
                         "(default 0.15; also applied to the coarse auto-pick's "
                         "width when deciding which one wins)")
    args = ap.parse_args()

    db3 = _resolve_db3(args.bag)
    print(f"Bag: {db3}\n")

    # Everything --tune uses to decide "is this window the board?" -- see
    # window_quality. min_pts defaults to a value scaled from the sampling
    # flags, because "how many points a board is worth" is set by how many
    # scans we merged and how hard we strided them, not by a magic constant.
    min_pts = args.tune_min_pts
    if min_pts is None:
        min_pts = max(60, int(50 * args.n_scans / max(1, args.point_stride)))
    gates = {"board_w": args.board_w, "board_h": args.board_h,
        "board_tol": args.board_tol, "min_pts": min_pts,
        "max_thick": args.tune_max_thickness, "min_fill": args.tune_min_fill,
        "cell": args.tune_fill_cell}
    if args.tune:
        print(f"--tune gates: n >= {min_pts}"
              f"{' (auto)' if args.tune_min_pts is None else ''}, "
              f"footprint {args.board_w:.2f}x{args.board_h:.2f}m +/-{args.board_tol:.2f}, "
              f"plane RMS <= {args.tune_max_thickness:.3f}m, "
              f"fill >= {args.tune_min_fill:.2f} @ {args.tune_fill_cell:.2f}m cells, "
              f"then >= {args.tune_min_pts_frac:.0%} of the pose's densest survivor.\n")

    # --- pass 1: sample every window up front, build the background model ---
    # (step 0 in the docstring -- the rover is static, only the board+operator
    # move between poses, so a voxel seen in most poses is the room, not the
    # board)
    windows = []
    pose_pts = {}
    for w in args.windows:
        label, start_s, dur_s = w.split(":")
        start_s, dur_s = float(start_s), float(dur_s)
        pts, topic = sample_points(db3, start_s, dur_s, args.n_scans, args.point_stride)
        windows.append((label, start_s, dur_s, topic))
        pose_pts[label] = pts

    bg = set()
    if args.bg_blocks > 0:
        # Default path: build the background from the WHOLE bag, not from the
        # pose windows -- see background_blocks() for why the pose-window model
        # deleted the board itself.
        blocks = background_blocks(db3, args.bg_blocks, args.bg_scans,
                                   args.point_stride)
        bg = background_voxels(blocks, args.voxel, args.bg_frac)
        print(f"Background model: {len(bg)} voxels (size {args.voxel} m, "
              f">= {args.bg_frac:.0%} of {len(blocks)} blocks sampled across the "
              f"whole bag) -- subtracted from every window below.\n")
    elif len(pose_pts) >= 3:
        bg = background_voxels(pose_pts, args.voxel, args.bg_frac)
        print(f"Background model: {len(bg)} voxels (size {args.voxel} m, "
              f">= {args.bg_frac:.0%} of {len(pose_pts)} poses) -- "
              f"subtracted from every window below.\n"
              "  NOTE: --bg-blocks 0 builds the model from the pose windows "
              "only. If the board sits in roughly the same spot in most poses, "
              "that deletes the board too.\n")
    else:
        print("Only 1-2 windows given -- can't build a reliable background "
              "model (need several poses to tell 'moved' from 'static'), "
              "skipping background subtraction.\n")

    print(f"{'pose':<6}{'px_min':>8}{'px_max':>8}{'py_min':>8}{'py_max':>8}"
          f"{'pz_min':>8}{'pz_max':>8}   n_pts")
    print("-" * 70)
    results = []
    for label, start_s, dur_s, topic in windows:
        pts = pose_pts[label]
        fg = foreground_points(pts, args.voxel, bg) if bg else pts
        print(f"\n=== pose {label}  (-s {start_s:g} -u {dur_s:g}, {topic}, "
              f"{len(pts)} points sampled, {len(fg)} after background removal) ===")

        if args.fix_x is not None:
            xr = tuple(args.fix_x)
            m = ((fg[:, 1] >= args.y_band[0]) & (fg[:, 1] <= args.y_band[1]) &
                (fg[:, 2] >= args.z_band[0]) & (fg[:, 2] <= args.z_band[1]) &
                (fg[:, 0] >= xr[0]) & (fg[:, 0] <= xr[1]))
            sl = fg[m]
            hist = edges = banded = None
        else:
            xr, sl, hinfo = suggest_bounds(fg, args.y_band, args.z_band,
                                           args.peak_frac, args.margin, args.pct,
                                           x_lo=args.x_lo)
            hist, edges, banded = hinfo
            if hist is not None:
                print_histogram(hist, edges)

        # Ranked alternate candidates -- see the CAVEAT in the module docstring:
        # a same-sized static cabinet panel can outrank or pass alongside the
        # real board, so list several and let the ZED photo of this pose settle it.
        if hist is not None and args.n_candidates > 1:
            cands = peak_ranges(hist, edges, args.peak_frac, args.margin,
                                args.x_lo, 10.0, k=args.n_candidates)
            print(f"  candidates (rank: px range, n points, footprint vs. "
                  f"~{args.board_w:.2f}x{args.board_h:.2f}m board, plane RMS, fill):")
            for i, (c0, c1) in enumerate(cands, start=1):
                csl = banded[(banded[:, 0] >= c0) & (banded[:, 0] <= c1)]
                if len(csl) < 10:
                    continue
                # Printed unconditionally here (unlike window_quality, which
                # short-circuits on the first failed gate) so the table stays
                # comparable across candidates when picking one by eye.
                cw, ch = footprint(csl, args.pct)
                cth = plane_rms(csl)
                cfl = fill_ratio(csl, _yz_box(csl, args.pct), args.tune_fill_cell)
                ok = (abs(cw - args.board_w) <= args.board_tol and
                     abs(ch - args.board_h) <= args.board_tol)
                mark = "auto-pick" if i == 1 else ""
                print(f"    {i}. px=[{c0:.2f},{c1:.2f}]  n={len(csl):5d}  "
                      f"{cw:.2f}x{ch:.2f}m  thick={cth:.3f}m  fill={cfl:.2f}  "
                      f"{'OK' if ok else 'mismatch':8s} {mark}")

        # --tune: direct footprint-minimizing search (step 7) -- catches real
        # board windows the peak-picking above never even considered, because
        # they were never the tallest bin. Only replaces xr/sl if it actually
        # scores better than the coarse auto-pick.
        tune_failed = False
        if args.tune and banded is not None:
            coarse_q = coarse_score = None
            if xr is not None and sl is not None and len(sl) >= 10:
                coarse_q = window_quality(sl, args.pct, gates)
                coarse_score = (abs(coarse_q["wy"] - args.board_w) +
                                abs(coarse_q["hz"] - args.board_h) +
                                args.tune_width_penalty * (xr[1] - xr[0]))
            tuned, info = tune_window(banded, args.x_lo, 10.0,
                                      args.tune_width_min, args.tune_width_max,
                                      args.tune_step, args.pct, gates,
                                      width_penalty=args.tune_width_penalty,
                                      min_pts_frac=args.tune_min_pts_frac)
            if tuned is None:
                tune_failed = True
                why = ", ".join(f"{v} {k}" for k, v in info["rejected"].items() if v)
                print(f"  --tune: no window passed the gates "
                      f"({why or 'no window had >= 3 points'}).")
                if info["loose"] is not None:
                    l0, l1, lq, _ = info["loose"]
                    # What the old footprint-only score would have returned --
                    # printed with its rejection reason so a genuinely tight
                    # gate is distinguishable from a genuinely absent board.
                    print(f"          footprint-only best was px=[{l0:.2f},{l1:.2f}] "
                          f"n={lq['n']} {lq['wy']:.2f}x{lq['hz']:.2f}m, rejected: "
                          f"{lq['why']}. If that IS the board, loosen "
                          f"--tune-min-pts/--tune-max-thickness/--tune-min-fill.")
            else:
                t0, t1, tq, tscore = tuned
                # A coarse auto-pick that fails the gates never wins on score
                # alone -- scoring a non-board window at all is the bug this
                # whole gate set exists to fix.
                better = (coarse_score is None or bool(coarse_q["why"]) or
                          tscore < coarse_score)
                if better:
                    verdict = "used, replaces auto-pick"
                    if coarse_q is not None and coarse_q["why"]:
                        verdict += f"; auto-pick rejected: {coarse_q['why']}"
                else:
                    verdict = "worse than auto-pick, kept coarse"
                print(f"  --tune: best window px=[{t0:.2f},{t1:.2f}]  n={tq['n']:5d}  "
                      f"{tq['wy']:.2f}x{tq['hz']:.2f}m  thick={tq['thick']:.3f}m  "
                      f"fill={tq['fill']:.2f}  score={tscore:.3f}  ({verdict})")
                if info["dropped_sparse"]:
                    print(f"          ({info['dropped_sparse']} other board-shaped planar "
                          f"windows dropped: under {args.tune_min_pts_frac:.0%} of this "
                          f"pose's densest survivor, {info['n_best']} points)")
                if better:
                    if args.tune_grow:
                        g0, g1 = grow_to_plane(banded, t0, t1, args.pct,
                                               tol=args.tune_grow_tol)
                        if (g1 - g0) - (t1 - t0) > 1e-6:
                            n_before = tq["n"]
                            t0, t1 = g0, g1
                            n_after = int(((banded[:, 0] >= t0) &
                                           (banded[:, 0] <= t1)).sum())
                            print(f"          grown along its own plane to "
                                  f"px=[{t0:.2f},{t1:.2f}]  n={n_before} -> {n_after} "
                                  f"(the tuner's window was a slice of a yawed panel)")
                    xr = (t0, t1)
                    sl = banded[(banded[:, 0] >= t0) & (banded[:, 0] <= t1)]

        # --tune found nothing board-like: do NOT quietly fall back to the
        # coarse auto-pick and print it as this pose's answer. The coarse pick
        # is the tallest depth peak, which is exactly the static furniture the
        # gates just rejected -- emitting it as px/py/pz is how a pose that
        # never saw the board ends up in a calibration launch file. Fail loudly
        # instead; the candidate table above plus --fix-x is the way out.
        if tune_failed:
            print("  -> FAILED: nothing in this window looks like the board "
                  "(see the rejections above). Cross-check the candidate table "
                  "against this pose's ZED photo, then rerun with --fix-x.")
            results.append((label, None, None, None))
            continue

        if xr is None or sl is None or len(sl) < 10:
            print("  WARNING: too few points in the plausible band -- "
                  "try widening --y-band/--z-band or check the window.")
            results.append((label, None, None, None))
            continue

        # Measure the footprint BEFORE padding. Deriving it back out of the
        # padded bounds (span - 2*margin) is wrong whenever pz_min clamps at
        # 0: the clamp shortens the span, the full 2*margin still comes off,
        # and the height reads ~0.13 m short -- enough to fire the mismatch
        # warning below on poses whose footprint was in fact fine.
        w_got, h_got = footprint(sl, args.pct)
        py = tuple(np.percentile(sl[:, 1], [args.pct, 100 - args.pct]))
        pz = tuple(np.percentile(sl[:, 2], [args.pct, 100 - args.pct]))
        py = (py[0] - args.margin, py[1] + args.margin)
        pz = (max(0.0, pz[0] - args.margin), pz[1] + args.margin)

        print(f"  -> px [{xr[0]:.2f}, {xr[1]:.2f}]  py [{py[0]:.2f}, {py[1]:.2f}]  "
              f"pz [{pz[0]:.2f}, {pz[1]:.2f}]   ({len(sl)} points in range)")

        # Background removal isolates "things that moved" (board + operator),
        # not the board alone -- flag a footprint that doesn't look like the
        # ~board_w x board_h board instead of silently trusting it (probably
        # mostly the person, or the wrong peak).
        if (abs(w_got - args.board_w) > args.board_tol or
                abs(h_got - args.board_h) > args.board_tol):
            print(f"  WARNING: footprint {w_got:.2f}x{h_got:.2f}m doesn't match "
                  f"the ~{args.board_w:.2f}x{args.board_h:.2f}m board -- likely "
                  "person+board mixed in, or the wrong peak; inspect the "
                  "histogram above / narrow --y-band,--z-band for this pose.")

        nrm = plane_normal(sl)
        az, el = normal_angles(nrm)
        print(f"     board plane normal: azimuth {az:+.1f} deg, elevation "
              f"{el:+.1f} deg")
        results.append((label, (xr, py, pz), (w_got, h_got), nrm))

    print("\n" + "=" * 70)
    print("SUMMARY (px_min px_max py_min py_max pz_min pz_max)")
    print("-" * 70)
    for label, r, _, _ in results:
        if r is None:
            print(f"pose {label}: FAILED")
            continue
        (px0, px1), (py0, py1), (pz0, pz1) = r
        print(f"pose {label}:  px_min:={px0:.2f} px_max:={px1:.2f} "
              f"py_min:={py0:.2f} py_max:={py1:.2f} pz_min:={pz0:.2f} pz_max:={pz1:.2f}")

    # With --tune's gates in place this is the one ambiguity left: they prove a
    # window is a board-SIZED flat filled panel, not that it's THE board, and a
    # static cabinet door qualifies on all three.
    #
    # Matching on depth ALONE is too crude to say that, though, and on this rig
    # it cried wolf on a correct run: the board is stand-mounted and gets
    # rotated in place, so it legitimately sits at ~the same depth in every
    # pose while its apparent width swings between ~0.97 m (facing) and
    # ~0.68 m (yawed). Only a repeat of the full position AND the same apparent
    # size is real evidence of "this never moved" -- so require both, and say
    # plainly that a board left untouched on its stand looks identical to a
    # static object from here. A human with the ZED photo settles it.
    # What actually constrains a plane-based extrinsic calibration is the
    # SPREAD of the board's plane normals, not the number of poses recorded.
    # Nine poses that all present the board at two orientations give the
    # solver two independent constraints, and no amount of correct cropping
    # downstream fixes that -- so measure it here, where the planes are
    # already fitted, rather than letting it surface as a bad calibration.
    normals = [(lbl, nrm) for lbl, r, _, nrm in results if r is not None]
    if len(normals) >= 2:
        print("\n" + "-" * 70)
        print("BOARD ORIENTATION PER POSE (normal of the fitted plane)")
        print("-" * 70)
        for lbl, n in normals:
            az, el = normal_angles(n)
            print(f"  pose {lbl:>3}:  azimuth {az:+6.1f} deg   elevation {el:+6.1f} deg")

        # Greedy grouping: a new orientation is one more than --min-sep degrees
        # away from every orientation already counted.
        groups = []
        for lbl, n in normals:
            for g in groups:
                if np.degrees(np.arccos(np.clip(abs(float(n @ g[1])), -1, 1))) <= args.min_sep:
                    g[0].append(lbl)
                    break
            else:
                groups.append(([lbl], n))
        widest = 0.0
        for i in range(len(normals)):
            for j in range(i + 1, len(normals)):
                a = np.degrees(np.arccos(np.clip(
                    abs(float(normals[i][1] @ normals[j][1])), -1, 1)))
                widest = max(widest, a)
        print(f"\n  {len(groups)} distinct orientation(s) at a {args.min_sep:g} deg "
              f"threshold, widest separation {widest:.1f} deg:")
        for members, n in groups:
            az, el = normal_angles(n)
            print(f"    az{az:+6.1f}/el{el:+6.1f}: poses {members}")
        if len(groups) < 3 or widest < 25.0:
            print(f"\n  WARNING: only {len(groups)} distinct board orientation(s), "
                  f"widest separation {widest:.1f} deg. Plane-based extrinsic "
                  "calibration needs the board turned to several clearly "
                  "different angles -- poses that repeat an orientation add "
                  "points but no new constraint, and the solve stays "
                  "ill-conditioned however clean the crop boxes are. Consider "
                  "re-recording with the board deliberately yawed and tilted "
                  "over a wide range.")

    valid = [(lbl, r, wh) for lbl, r, wh, _ in results if r is not None]

    def _center(r):
        return tuple((lo + hi) / 2 for lo, hi in r)

    reported = set()
    for lbl_i, r_i, wh_i in valid:
        if lbl_i in reported:
            continue
        ci = _center(r_i)
        group = [lbl_j for lbl_j, r_j, wh_j in valid
                if all(abs(a - b) < 0.12 for a, b in zip(ci, _center(r_j)))
                and abs(wh_i[0] - wh_j[0]) < 0.08
                and abs(wh_i[1] - wh_j[1]) < 0.08]
        if len(group) >= 3:
            reported.update(group)
            print(f"\nWARNING: poses {group} landed on the same spot "
                  f"(~{ci[0]:.2f}, {ci[1]:.2f}, {ci[2]:.2f}) AND the same apparent "
                  f"size ({wh_i[0]:.2f}x{wh_i[1]:.2f}m, within 8 cm). Either a "
                  "static object that survived background subtraction, or the "
                  "board was left untouched on its stand for those poses -- "
                  "which is useless for calibration either way, since they add "
                  "no new geometry. Check them against the ZED photos.")


if __name__ == "__main__":
    main()
