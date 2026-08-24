"""Export ONE merged .pcd from a boxes.json -- applying the same per-session
alignment you dragged/rotated the boxes into in merge_boxes.py to the point clouds
those boxes came from.

merge_boxes.py aligns SESSIONS BY DRAGGING AND 90-DEGREE-ROTATING THEIR BOXES, but it
only ever writes the boxes: the clouds behind them stay in their original, unaligned
frames. This script closes that gap. It recovers the rotation and translation each
session was moved by, applies it to that session's cloud, keeps the points its boxes
actually contain, and writes the union as a single .pcd -- so the aligned layout you
built by eye becomes an actual merged point cloud, not just a set of rectangles.

Run it after merge_boxes.py (or directly on a single-session boxes.json / an edited
boxes_edited.json, in which case there is nothing to align and it simply exports the
points those boxes contain).

HOW THE PER-SESSION ALIGNMENT IS RECOVERED
merge_boxes.py does NOT record the drag/rotate anywhere -- it bakes the result into
each box's x_min/x_max/y_min/y_max and moves on. So the alignment is read back here by
differencing: every merged box carries "session" and "source_id", so it is matched to
the same box in that session's original file, and the (rotation, dx, dy) that explains
merged = rotate(original) + (dx, dy) is solved for.

A session's drag is translate-only in X/Y and by construction applies the SAME dx/dy to
every box in the session (see merge_boxes.py's docstring and its on_session_move); its
rotation is one of 0/90/180/270 degrees clockwise, applied to every box in the session
together (see its rotate_session). Composing any number of drags and 90-degree turns
still reduces to exactly one such (rotation, dx, dy) triple, so for each session this
script tries all 4 right angles, and for each one computes the dx/dy every one of the
session's boxes would need to agree on. Whichever angle gives the smallest disagreement
wins, and this script refuses to guess if even the best angle disagrees by more than a
millimetre -- disagreement would mean the file was not produced by drags/90-degree
turns alone, and silently averaging it would smear the cloud. z is never touched by
either operation, so the offset is always (dx, dy, 0).

The alternative -- teaching merge_boxes.py to write the alignment out -- was NOT taken
on purpose: differencing works on boxes_merged.json files you have ALREADY produced,
with no need to redo the alignment, and it stays correct if someone hand-edits a merged
file.

WHAT GETS KEPT
By default only the points INSIDE the boxes, since the boxes are what define the rooms
(clutter, stray outdoor returns and outlier streaks outside them are dropped). Pass
--all-points to keep every point of every session instead, still aligned, when you want
the full scan rather than just the modelled volume.

OVERLAP / DOUBLE COUNTING
Two sessions can legitimately resolve to the SAME source cloud (e.g. boxes.json and
boxes_edited.json both fitted from one .pcd). Their boxes then cover overlapping
volume and those points would be written twice. This is detected and reported, and
--voxel collapses the duplicates onto a single grid.

REUSES (does not reimplement) fit_boxes.py's SAVED_BOXES_DIR and its load_pcd_cloud
.pcd reader, so a cloud is read here exactly the way the fitting step read it.

Usage:
    python to_pcd.py [SavedBoxes/boxes_merged.json]
        [--out SavedBag/merged_cloud.pcd]
        [--all-points] [--margin 0.0] [--voxel 0.0]
        [--source SESSION_INDEX=path/to/cloud.pcd ...]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

from fit_boxes import SAVED_BOXES_DIR, load_pcd_cloud

_THIS_DIR = Path(__file__).resolve().parent
SAVED_BAG_DIR = _THIS_DIR / "SavedBag"

# Offsets are compared at this tolerance (metres). merge_boxes.py adds the identical
# float dx/dy to every box in a session, so real disagreement means the file did not
# come from that drag -- 1 mm leaves room for json round-tripping and nothing else.
OFFSET_TOL_M = 1e-3


def resolve_path(raw, extra_dirs=()):
    """Resolve a path recorded inside a boxes.json.

    Those paths are stored as typed on the command line ("SavedBag\\x.pcd"), so they
    are relative to wherever the pipeline was run from, not to this script. Try the
    literal path first, then the same path relative to the script directory and to the
    usual data folders, and normalise Windows separators so a file written on Windows
    still resolves if the tree is ever used from a POSIX shell.
    """
    if raw is None:
        return None
    candidates = [Path(raw)]
    norm = str(raw).replace("\\", "/")
    candidates.append(Path(norm))
    for d in (_THIS_DIR, SAVED_BAG_DIR, SAVED_BOXES_DIR, *extra_dirs):
        candidates.append(d / norm)
        candidates.append(d / Path(norm).name)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def load_boxes_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = data.get("boxes") or []
    if not boxes:
        raise SystemExit(f"{path} contains no boxes")
    return data, boxes


# The 4 rotations merge_boxes.py's 'r'/'R' keys can ever produce (clockwise degrees).
CANDIDATE_ROTATIONS_CW = (0, 90, 180, 270)


def _rotate_xy(x, y, deg):
    """Standard CCW-positive rotation about the origin, matching rotate_z's convention."""
    if deg == 0.0:
        return x, y
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return x * c - y * s, x * s + y * c


def _rotate_box_cw(b, cw_deg):
    """Rotate a box's (x_min,y_min)/(x_max,y_max) diagonal about the origin by cw_deg
    clockwise (merge_boxes.py's convention) and return the resulting axis-aligned
    x_min, x_max, y_min, y_max -- width/height swap on an odd number of 90s, same as
    rotate_session does to the box itself."""
    x0, y0 = _rotate_xy(b["x_min"], b["y_min"], -cw_deg)
    x1, y1 = _rotate_xy(b["x_max"], b["y_max"], -cw_deg)
    return min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)


def session_offsets(data, boxes, boxes_path):
    """Per-session (rotation_cw_deg, dx, dy) alignment, recovered by differencing against
    the original session files. Returns {session_index: (rotation_cw_deg, dx, dy)}.

    A single-session file (plain boxes.json / boxes_edited.json) has no "sessions" key
    and was never dragged or rotated, so its alignment is the identity.
    """
    sessions = data.get("sessions")
    if not sessions:
        return {0: (0, 0.0, 0.0)}

    offsets = {}
    by_session = defaultdict(list)
    for b in boxes:
        by_session[b.get("session", 0)].append(b)

    for s in sessions:
        idx = s["index"]
        merged_boxes = by_session.get(idx, [])
        if not merged_boxes:
            print(f"  session {idx}: no boxes survived into the merge, skipped")
            continue

        orig_path = resolve_path(s.get("file"), extra_dirs=(boxes_path.parent,))
        if orig_path is None:
            raise SystemExit(
                f"session {idx}: cannot find its original boxes file {s.get('file')!r}.\n"
                f"It is needed to recover how this session was dragged/rotated in "
                f"merge_boxes.py (that is not stored in the merged file).\n"
                f"Put the file back, or re-run merge_boxes.py from the folder it was "
                f"originally run from.")

        with open(orig_path, "r", encoding="utf-8") as f:
            orig_by_id = {b["id"]: b for b in (json.load(f).get("boxes") or [])}

        # Try every 90-degree turn merge_boxes.py could have applied; keep whichever one
        # gives the smallest disagreement across the session's boxes about dx/dy.
        best = None  # (max_spread, cw_deg, dx, dy, n_matched)
        for cw_deg in CANDIDATE_ROTATIONS_CW:
            deltas = []
            for b in merged_boxes:
                ob = orig_by_id.get(b.get("source_id"))
                if ob is None:
                    continue
                pxmin, _, pymin, _ = _rotate_box_cw(ob, cw_deg)
                deltas.append((b["x_min"] - pxmin, b["y_min"] - pymin))
            if not deltas:
                continue
            deltas = np.asarray(deltas, dtype=float)
            spread = deltas.max(axis=0) - deltas.min(axis=0)
            max_spread = float(spread.max())
            if best is None or max_spread < best[0]:
                dx, dy = deltas.mean(axis=0)
                best = (max_spread, cw_deg, float(dx), float(dy), len(deltas))

        if best is None:
            raise SystemExit(
                f"session {idx}: none of its merged boxes could be matched back to "
                f"{orig_path} by source_id, so its alignment cannot be recovered.")

        max_spread, cw_deg, dx, dy, n = best
        if max_spread > OFFSET_TOL_M:
            raise SystemExit(
                f"session {idx}: its boxes disagree about how the session moved even "
                f"under the best-fitting 90 deg rotation ({cw_deg} deg cw, spread "
                f"{max_spread * 1000:.1f} mm).\n"
                f"merge_boxes.py only ever drags a whole session by one identical dx/dy "
                f"and rotates it by 0/90/180/270 deg, so this file was not produced that "
                f"way (hand-edited?) and there is no single alignment to apply to its "
                f"cloud. Refusing to guess.")

        offsets[idx] = (cw_deg, dx, dy)
        rot_note = f", rotated {cw_deg} deg clockwise" if cw_deg else ""
        print(f"  session {idx}: dragged by dx={dx:+.3f} m, dy={dy:+.3f} m{rot_note} "
              f"({n} box(es) agree)")
    return offsets


def rotate_z(xyz, deg):
    """Rotate about Z. Boxes are written back in the original cloud frame, but their
    x_min/x_max/y_min/y_max (the numbers the crop and the drag both use) live in the
    frame fit_boxes.py gridded in, i.e. the cloud rotated by -yaw_deg. So for a session
    fitted with --yaw-deg != 0 the cloud has to be taken into that same frame before
    the axis-aligned test means anything."""
    if deg == 0.0:
        return xyz
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return xyz @ r.T


def points_in_boxes(xyz, boxes, margin):
    """Mask of points inside the union of the boxes (axis-aligned, inclusive)."""
    keep = np.zeros(len(xyz), dtype=bool)
    for b in boxes:
        keep |= (
            (xyz[:, 0] >= b["x_min"] - margin) & (xyz[:, 0] <= b["x_max"] + margin)
            & (xyz[:, 1] >= b["y_min"] - margin) & (xyz[:, 1] <= b["y_max"] + margin)
            & (xyz[:, 2] >= b["z_min"] - margin) & (xyz[:, 2] <= b["z_max"] + margin)
        )
    return keep


def load_cloud_with_colors(path):
    """xyz via fit_boxes.load_pcd_cloud (same reader the fitting step used), plus the
    colors if the file carries any, so a colored scan stays colored on the way out."""
    xyz = load_pcd_cloud(path)
    pcd = o3d.io.read_point_cloud(str(path))
    rgb = np.asarray(pcd.colors, dtype=np.float32)
    if len(rgb) != len(xyz):
        rgb = None
    return xyz, rgb


def parse_source_overrides(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--source expects SESSION_INDEX=path, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[int(k)] = Path(v)
        except ValueError:
            raise SystemExit(f"--source session index must be an integer, got {k!r}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boxes", nargs="?", type=Path,
                    default=SAVED_BOXES_DIR / "boxes_merged.json",
                    help="boxes json to export from (default: "
                         "SavedBoxes/boxes_merged.json). A merged file aligns each "
                         "session's cloud by the offset its boxes were dragged by; a "
                         "single-session boxes.json/boxes_edited.json is simply cropped.")
    ap.add_argument("--out", type=Path, default=SAVED_BAG_DIR / "merged_cloud.pcd",
                    help="output .pcd (default: SavedBag/merged_cloud.pcd)")
    ap.add_argument("--all-points", action="store_true",
                    help="keep every point of every session, aligned but NOT cropped to "
                         "the boxes (default keeps only what the boxes contain)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="grow every box by this many metres before cropping, to keep a "
                         "little context around each room (default 0)")
    ap.add_argument("--voxel", type=float, default=0.0,
                    help="voxel-downsample the merged result, in metres. Also the way to "
                         "collapse points written twice where sessions share a source "
                         "cloud (default 0 = off)")
    ap.add_argument("--source", action="append", metavar="IDX=PATH",
                    help="override the cloud for one session, e.g. --source 0=SavedBag/a.pcd "
                         "(repeatable). Use when the .pcd recorded in the boxes file has "
                         "moved or been renamed.")
    args = ap.parse_args()

    if not args.boxes.exists():
        raise SystemExit(f"{args.boxes} does not exist")

    data, boxes = load_boxes_file(args.boxes)
    print(f"{args.boxes}: {len(boxes)} box(es)")

    print("Recovering per-session alignment:")
    offsets = session_offsets(data, boxes, args.boxes.resolve())

    sessions = data.get("sessions")
    overrides = parse_source_overrides(args.source)
    if sessions:
        session_source = {s["index"]: s.get("source") for s in sessions}
        session_file = {s["index"]: s.get("file") for s in sessions}
    else:
        # single-session file: everything belongs to session 0
        session_source = {0: data.get("source")}
        session_file = {0: str(args.boxes)}

    by_session = defaultdict(list)
    for b in boxes:
        by_session[b.get("session", 0)].append(b)

    # A session's boxes were fitted in that session's gridding frame, so yaw comes from
    # its own file, not from the merged wrapper (whose yaw_deg is null by construction).
    session_yaw = {}
    for idx in by_session:
        yaw = 0.0
        orig = resolve_path(session_file.get(idx), extra_dirs=(args.boxes.resolve().parent,))
        if orig is not None:
            with open(orig, "r", encoding="utf-8") as f:
                yaw = json.load(f).get("yaw_deg") or 0.0
        elif not sessions:
            yaw = data.get("yaw_deg") or 0.0
        session_yaw[idx] = float(yaw)

    resolved = {}
    for idx in sorted(by_session):
        p = overrides.get(idx) or resolve_path(session_source.get(idx))
        if p is None or not Path(p).exists():
            raise SystemExit(
                f"session {idx}: cannot find its source cloud "
                f"{session_source.get(idx)!r}.\nPass it explicitly with "
                f"--source {idx}=path/to/cloud.pcd")
        resolved[idx] = Path(p)

    shared = defaultdict(list)
    for idx, p in resolved.items():
        shared[p].append(idx)
    for p, idxs in shared.items():
        if len(idxs) > 1:
            print(f"NOTE: sessions {idxs} all resolve to {p.name}. Where their boxes "
                  f"overlap the same points are written once per session; pass --voxel "
                  f"(e.g. --voxel 0.05) to collapse the duplicates.")

    chunks_xyz, chunks_rgb, any_color = [], [], False
    for idx in sorted(by_session):
        path = resolved[idx]
        cw_deg, dx, dy = offsets.get(idx, (0, 0.0, 0.0))
        yaw = session_yaw.get(idx, 0.0)
        print(f"\nsession {idx}: {path}")

        xyz, rgb = load_cloud_with_colors(path)
        any_color = any_color or rgb is not None

        # Order matters: into the gridding frame, then the merge_boxes.py rotation, THEN
        # translate -- the drag offset was measured on x_min/y_min in the already-rotated
        # frame (see _rotate_box_cw), so the cloud has to be rotated the same way first.
        xyz = rotate_z(xyz, -yaw)
        if yaw:
            print(f"  rotated by {-yaw:.2f} deg about Z into this session's fitting frame")
        if cw_deg:
            xyz = rotate_z(xyz, -cw_deg)
            print(f"  rotated by {cw_deg} deg clockwise about Z (merge_boxes.py alignment)")
        xyz = xyz + np.array([dx, dy, 0.0], dtype=xyz.dtype)

        if args.all_points:
            mask = np.ones(len(xyz), dtype=bool)
            print(f"  keeping all {len(xyz)} point(s) (--all-points)")
        else:
            mask = points_in_boxes(xyz, by_session[idx], args.margin)
            print(f"  inside {len(by_session[idx])} box(es): "
                  f"{int(mask.sum())} / {len(xyz)} point(s)")
            if not mask.any():
                print("  WARNING: no points fell inside this session's boxes -- is this "
                      "the cloud the boxes were fitted from?")

        chunks_xyz.append(xyz[mask])
        chunks_rgb.append(rgb[mask] if rgb is not None else None)

    merged = np.vstack(chunks_xyz)
    if len(merged) == 0:
        raise SystemExit("merged result is empty -- nothing to write")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged.astype(np.float64))
    if any_color:
        # Sessions without color get neutral grey so the arrays stay aligned rather than
        # silently dropping color for everyone.
        filled = [c if c is not None else np.full((len(x), 3), 0.5, dtype=np.float32)
                  for c, x in zip(chunks_rgb, chunks_xyz)]
        pcd.colors = o3d.utility.Vector3dVector(np.vstack(filled).astype(np.float64))

    print(f"\nmerged: {len(merged)} point(s) from {len(chunks_xyz)} session(s)")
    if args.voxel > 0:
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(args.voxel)
        print(f"voxel {args.voxel} m: {before} -> {len(pcd.points)} point(s)")

    lo = np.asarray(pcd.points).min(axis=0)
    hi = np.asarray(pcd.points).max(axis=0)
    print(f"extent  X {lo[0]:8.2f} {hi[0]:8.2f}\n"
          f"        Y {lo[1]:8.2f} {hi[1]:8.2f}\n"
          f"        Z {lo[2]:8.2f} {hi[2]:8.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.out), pcd, write_ascii=False):
        raise SystemExit(f"open3d failed to write {args.out}")
    print(f"\nwrote {args.out}")
    print("View it with ViewPCD.m, or feed it back to fit_boxes.py --pcd to re-fit "
          "boxes on the merged cloud.")


if __name__ == "__main__":
    main()
