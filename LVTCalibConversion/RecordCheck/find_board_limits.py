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
     at --x-lo on every pose. Fix: voxelize each window's points (--voxel,
     default 0.20 m) and mark a voxel as background if it's occupied in
     >= --bg-frac (default 0.6) of the poses; points falling in a
     background voxel are dropped before anything else runs.

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
     width has >= --tune-min-pts points AND minimizes
     |width_y - board_w| + |height_z - board_h| directly -- i.e. it
     optimizes for "looks like the board" instead of "has the most points",
     so it finds real board windows the coarse peak-picking misses entirely.
     Much slower (a full 2D scan per pose instead of one histogram argmax),
     so it's opt-in; the ranked candidates from step 6 are still printed
     alongside it for comparison, and the tuned window is only used as the
     pose's result if it beats the coarse auto-pick's footprint score.

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
        return None, None, p

    bins = np.arange(x_lo, x_hi + 0.1, 0.1)
    hist, edges = np.histogram(p[:, 0], bins=bins)
    top = peak_ranges(hist, edges, peak_frac, margin, x_lo, x_hi, k=1)
    if not top:
        return None, None, p
    px_min, px_max = top[0]

    sl = p[(p[:, 0] >= px_min) & (p[:, 0] <= px_max)]
    return (px_min, px_max), sl, (hist, edges, p)


def footprint(pts_in_range, pct):
    """(width_y, height_z) of the pct/100-(100-pct) percentile spread --
    same trimming as the py/pz suggestion, just without the added --margin,
    for comparing candidates against --board-w/--board-h."""
    wy = (np.percentile(pts_in_range[:, 1], 100 - pct) -
         np.percentile(pts_in_range[:, 1], pct))
    hz = (np.percentile(pts_in_range[:, 2], 100 - pct) -
         np.percentile(pts_in_range[:, 2], pct))
    return float(wy), float(hz)


def tune_window(pts_band, board_w, board_h, x_lo, x_hi, width_min, width_max,
                step, min_pts, pct, width_penalty=0.15):
    """Slide a depth window of every width in [width_min, width_max] across
    [x_lo, x_hi] and keep whichever (start, width) has >= min_pts points AND
    minimizes |width_y - board_w| + |height_z - board_h| + width_penalty*width.

    Unlike peak_ranges (which always centers on the tallest depth bin --
    the cabinet, whenever the board's own return is weaker than the
    background residue), this optimizes directly for "looks like the
    board", so it also considers windows that were never a histogram peak
    at all. O((width_max-width_min)/step * (x_hi-x_lo)/step) evaluations,
    each a boolean mask + 4 percentiles -- a few thousand for typical
    ranges, a few hundred ms per pose; that's why it's opt-in (--tune)
    rather than the default.

    The width_penalty term matters: a real board is a thin flat surface, so
    its true depth extent is small (a few cm to maybe 0.3-0.4 m at a steep
    tilt); a WIDE window spanning a meter or more of depth can still pass a
    loose 2-98 percentile check by just trimming away whatever doesn't fit
    -- it isn't "the board", it's an unprincipled grab-bag. Without the
    penalty this was observed to happily pick 1.5-1.8 m wide windows that
    scored well on Y/Z spread alone; the penalty biases ties toward the
    narrowest window that still matches, which is what a real board return
    looks like.

    Returns (px_min, px_max, n, width_y, height_z, score) or None if no
    window anywhere had >= min_pts points."""
    best = None
    width = width_min
    while width <= width_max + 1e-9:
        x0 = x_lo
        while x0 + width <= x_hi:
            sl = pts_band[(pts_band[:, 0] >= x0) & (pts_band[:, 0] <= x0 + width)]
            if len(sl) >= min_pts:
                wy, hz = footprint(sl, pct)
                score = abs(wy - board_w) + abs(hz - board_h) + width_penalty * width
                if best is None or score < best[-1]:
                    best = (x0, x0 + width, len(sl), wy, hz, score)
            x0 += step
        width += step
    return best


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
    ap.add_argument("--tune-min-pts", type=int, default=60,
                    help="--tune: minimum points a window must have to be considered "
                         "(default 60)")
    ap.add_argument("--tune-width-penalty", type=float, default=0.15,
                    help="--tune: added to the score as width_penalty*window_width, "
                         "so ties favor the narrowest matching window instead of a "
                         "wide grab-bag that only passes on trimmed percentiles "
                         "(default 0.15; also applied to the coarse auto-pick's "
                         "width when deciding which one wins)")
    args = ap.parse_args()

    db3 = _resolve_db3(args.bag)
    print(f"Bag: {db3}\n")

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
    if len(pose_pts) >= 3:
        bg = background_voxels(pose_pts, args.voxel, args.bg_frac)
        print(f"Background model: {len(bg)} voxels (size {args.voxel} m, "
              f">= {args.bg_frac:.0%} of {len(pose_pts)} poses) -- "
              f"subtracted from every window below.\n")
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
            hist, edges, banded = hinfo if hinfo is not None else (None, None, None)
            if hist is not None:
                print_histogram(hist, edges)

        # Ranked alternate candidates -- see the CAVEAT in the module docstring:
        # a same-sized static cabinet panel can outrank or pass alongside the
        # real board, so list several and let the ZED photo of this pose settle it.
        if hist is not None and args.n_candidates > 1:
            cands = peak_ranges(hist, edges, args.peak_frac, args.margin,
                                args.x_lo, 10.0, k=args.n_candidates)
            print(f"  candidates (rank: px range, n points, footprint vs. "
                  f"~{args.board_w:.2f}x{args.board_h:.2f}m board):")
            for i, (c0, c1) in enumerate(cands, start=1):
                csl = banded[(banded[:, 0] >= c0) & (banded[:, 0] <= c1)]
                if len(csl) < 10:
                    continue
                cw, ch = footprint(csl, args.pct)
                ok = (abs(cw - args.board_w) <= args.board_tol and
                     abs(ch - args.board_h) <= args.board_tol)
                mark = "auto-pick" if i == 1 else ""
                print(f"    {i}. px=[{c0:.2f},{c1:.2f}]  n={len(csl):5d}  "
                      f"{cw:.2f}x{ch:.2f}m  {'OK' if ok else 'mismatch':8s} {mark}")

        # --tune: direct footprint-minimizing search (step 7) -- catches real
        # board windows the peak-picking above never even considered, because
        # they were never the tallest bin. Only replaces xr/sl if it actually
        # scores better than the coarse auto-pick.
        if args.tune and banded is not None:
            coarse_score = None
            if xr is not None and sl is not None and len(sl) >= 10:
                cw0, ch0 = footprint(sl, args.pct)
                coarse_score = (abs(cw0 - args.board_w) + abs(ch0 - args.board_h) +
                                args.tune_width_penalty * (xr[1] - xr[0]))
            tuned = tune_window(banded, args.board_w, args.board_h, args.x_lo, 10.0,
                                args.tune_width_min, args.tune_width_max,
                                args.tune_step, args.tune_min_pts, args.pct,
                                width_penalty=args.tune_width_penalty)
            if tuned is None:
                print("  --tune: no window anywhere had enough points.")
            else:
                t0, t1, tn, tw, th, tscore = tuned
                better = coarse_score is None or tscore < coarse_score
                print(f"  --tune: best window px=[{t0:.2f},{t1:.2f}]  n={tn:5d}  "
                      f"{tw:.2f}x{th:.2f}m  score={tscore:.3f}"
                      f"{'  (used, replaces auto-pick)' if better else '  (worse than auto-pick, kept coarse)'}")
                if better:
                    xr = (t0, t1)
                    sl = banded[(banded[:, 0] >= t0) & (banded[:, 0] <= t1)]

        if xr is None or sl is None or len(sl) < 10:
            print("  WARNING: too few points in the plausible band -- "
                  "try widening --y-band/--z-band or check the window.")
            results.append((label, None))
            continue

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
        w_got = (py[1] - py[0]) - 2 * args.margin
        h_got = (pz[1] - pz[0]) - 2 * args.margin
        if (abs(w_got - args.board_w) > args.board_tol or
                abs(h_got - args.board_h) > args.board_tol):
            print(f"  WARNING: footprint {w_got:.2f}x{h_got:.2f}m doesn't match "
                  f"the ~{args.board_w:.2f}x{args.board_h:.2f}m board -- likely "
                  "person+board mixed in, or the wrong peak; inspect the "
                  "histogram above / narrow --y-band,--z-band for this pose.")

        results.append((label, (xr, py, pz)))

    print("\n" + "=" * 70)
    print("SUMMARY (px_min px_max py_min py_max pz_min pz_max)")
    print("-" * 70)
    for label, r in results:
        if r is None:
            print(f"pose {label}: FAILED")
            continue
        (px0, px1), (py0, py1), (pz0, pz1) = r
        print(f"pose {label}:  px_min:={px0:.2f} px_max:={px1:.2f} "
              f"py_min:={py0:.2f} py_max:={py1:.2f} pz_min:={pz0:.2f} pz_max:={pz1:.2f}")

    # The rover doesn't move between poses, the board does. If the same X range
    # (within 0.15 m) recurs across several poses, it's almost certainly
    # something fixed (rover, furniture), not the board -- warn instead of
    # failing silently.
    valid = [(lbl, r[0]) for lbl, r in results if r is not None]
    for i, (lbl_i, (a0, a1)) in enumerate(valid):
        dupes = [lbl_j for lbl_j, (b0, b1) in valid
                if lbl_j != lbl_i and abs(a0 - b0) < 0.15 and abs(a1 - b1) < 0.15]
        if len(dupes) >= 2:
            print(f"\nWARNING: pose {lbl_i} has the same px as {dupes} -- "
                  "likely a fixed object (not the board). Check the histogram "
                  "above: if the chosen peak looks too close/too wide in Y, "
                  "raise --x-lo or --z-band and rerun just those poses.")
            break


if __name__ == "__main__":
    main()
