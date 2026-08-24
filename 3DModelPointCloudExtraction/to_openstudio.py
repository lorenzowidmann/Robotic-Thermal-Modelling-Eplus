"""Build an OpenStudio model (.osm) from fit_boxes.py's boxes.json.

Replaces the old fit_planes.py/planes.json version of this script: input is
now a set of axis-aligned BOXES (one Space per room, tiled from several
touching boxes), not individually RANSAC-fit wall/floor/ceiling planes.

GROUPING: boxes share a room via their "hall" field (fit_boxes.py's room
id, preserved through interactive_boxes.py edits) -- every box with the
same "hall" becomes surfaces inside ONE OpenStudio Space (+ ThermalZone).
A box with no "hall" field falls back to being its own single-box room
(printed as a warning -- shouldn't happen from fit_boxes.py's own output).

TOUCHING *OR OVERLAPPING* BOXES ARE OPEN TO EACH OTHER, WHETHER OR NOT
THEY'RE THE SAME ROOM -- no Wall surface is ever created at an edge where
another box is right there, regardless of whether that other box is in the
same "hall" or a different one, and regardless of whether it merely touches
that face or straddles it (boxes that interpenetrate). This matters even
across rooms: fit_boxes.py can split
one physically continuous corridor into two "rooms" over nothing more than
a slight, spurious ceiling-height reading at one point (see its
--height-group-tol) -- inserting a real wall there would seal off an open
passage that doesn't exist in the building. So the rule for every box's 4
vertical edges is simply:

  - edge covered by ANY other box's volume just outside it (same room or
    not, touching or overlapping) -> no surface at all; the two spaces
    stay physically open to each other there.
  - edge covered by nothing (no other box reaches it at all) -> a normal
    exterior Wall (Outdoors/SunExposed/WindExposed).

Rooms therefore only affect which OpenStudio Space/ThermalZone a surface
belongs to (for floor area/volume bookkeeping and later HVAC assignment in
OpenStudio) -- there is no thermal separation between touching rooms
unless you edit the model afterwards (or remove/re-tile boxes to actually
close that opening with real geometry).

Every box also gets a Floor (z_min, "Ground") and RoofCeiling (z_max,
"Outdoors") surface, EXCEPT when that slab lies strictly inside another
box's volume (see is_buried_slab) -- an overlapping box would otherwise
leave a floor/ceiling plate cutting through the middle of an open space.
Coplanar slabs of neighbouring boxes are NOT filtered: OpenStudio has no
problem with a room's floor made of several coplanar rectangles, and
skipping that filtering avoids needing full polygon-union geometry for
possibly non-rectangular (L/T-shaped) room footprints.

Usage:
    python to_openstudio.py [--boxes SavedBoxes/boxes.json]
        [--out OpenStudioModel/<timestamp>.osm] [--name Building]
        [--eps 0.03] [--no-snap]

Output goes to OpenStudioModel/ next to this script by default, named after
the moment it's saved (yy-MM-dd - HH-mm-ss.osm, same convention as
LoopClosure_vFinal's SavedBag/ and SavedBoxes/), so repeated runs never
overwrite each other; the directory is created if missing.

SNAPPING (on by default, see snap_box_coordinates): edges dragged by hand
in interactive_boxes.py rarely land back on fit_boxes.py's exact
--cell-size grid, so two boxes meant to touch can end up with a hairline
gap or overlap between them instead of sharing an exact coordinate. Before
building anything, every box edge within --eps of another gets nudged onto
one shared value -- so boxes that were meant to touch DO touch exactly:
no sliver gap in the floor/ceiling, and build_wall_segments' own --eps
tolerance (used for wall-adjacency detection, same value) has less to
paper over. --no-snap uses --boxes's coordinates completely as-is.

Venv: C:\\venvs\\planefit (same as fit_boxes.py; `pip install openstudio`
added there -- openstudio 3.9+ has cp312/cp313 wheels, unlike open3d).
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import openstudio

from fit_boxes import SAVED_BOXES_DIR

SIDES = ("left", "right", "bottom", "top")

# Default home for every .osm, resolved against THIS FILE's location (not the cwd), same
# convention as fit_boxes.py's SAVED_BOXES_DIR and the MATLAB scripts' SavedBag/.
OPENSTUDIO_MODEL_DIR = Path(__file__).resolve().parent / "OpenStudioModel"


def fix_winding(corners, outward_ref):
    """Reverse the vertex order if the polygon's implied normal (from its
    own point order) points opposite to outward_ref."""
    c = np.array(corners, dtype=float)
    implied = np.cross(c[1] - c[0], c[3] - c[0])
    if np.dot(implied, outward_ref) < 0:
        return corners[::-1]
    return corners


def rect_corners(x0, x1, y0, y1, z, normal_sign):
    """Horizontal rectangle at height z. normal_sign=+1 for ceiling (normal
    +Z), -1 for floor (normal -Z)."""
    pts = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
    return fix_winding(pts, (0.0, 0.0, float(normal_sign)))


def wall_corners(side, fixed, lo, hi, z0, z1):
    """Vertical rectangle for one wall segment. `side` fixes which axis is
    constant (fixed) and which spans (lo, hi); returns (corners, outward)."""
    if side in ("left", "right"):
        x = fixed
        pts = [(x, lo, z0), (x, hi, z0), (x, hi, z1), (x, lo, z1)]
        outward = (1.0, 0.0, 0.0) if side == "right" else (-1.0, 0.0, 0.0)
    else:
        y = fixed
        pts = [(lo, y, z0), (hi, y, z0), (hi, y, z1), (lo, y, z1)]
        outward = (0.0, 1.0, 0.0) if side == "top" else (0.0, -1.0, 0.0)
    return fix_winding(pts, outward), outward


def interval_intersect(a, b, eps):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if hi - lo > eps else None


def subtract_rect_2d(rect, cut, eps):
    """rect, cut: (span_lo, span_hi, z_lo, z_hi). Returns up to 4 leftover
    axis-aligned rectangles covering rect minus cut (standard "cross"
    decomposition: two full-height side strips + a bottom/top strip over
    just the cut's own span). Only cuts where cut actually overlaps rect
    in BOTH span and z remove anything -- this is what keeps two boxes
    that touch in plan but don't overlap in height (a shorter room next
    to a taller one) from opening a hole above where the short room's
    ceiling doesn't reach."""
    a0, a1, b0, b1 = rect
    c0, c1, d0, d1 = cut
    cc0, cc1 = max(a0, c0), min(a1, c1)
    cd0, cd1 = max(b0, d0), min(b1, d1)
    if cc1 - cc0 <= eps or cd1 - cd0 <= eps:
        return [rect]   # no overlap in span, or no overlap in z -- cut misses entirely
    out = []
    if cc0 - a0 > eps:
        out.append((a0, cc0, b0, b1))
    if a1 - cc1 > eps:
        out.append((cc1, a1, b0, b1))
    if cd0 - b0 > eps:
        out.append((cc0, cc1, b0, cd0))
    if b1 - cd1 > eps:
        out.append((cc0, cc1, cd1, b1))
    return out


def subtract_rects_2d(rect, cuts, eps):
    pieces = [rect]
    for cut in cuts:
        next_pieces = []
        for p in pieces:
            next_pieces.extend(subtract_rect_2d(p, cut, eps))
        pieces = next_pieces
    return [p for p in pieces if p[1] - p[0] > eps and p[3] - p[2] > eps]


def side_info(box, side):
    """(fixed coordinate of this side, (span_lo, span_hi) along it, axis
    letter the fixed coordinate lives on, outward sign along that axis)."""
    if side == "left":
        return box["x_min"], (box["y_min"], box["y_max"]), "x", -1.0
    if side == "right":
        return box["x_max"], (box["y_min"], box["y_max"]), "x", +1.0
    if side == "bottom":
        return box["y_min"], (box["x_min"], box["x_max"]), "y", -1.0
    return box["y_max"], (box["x_min"], box["x_max"]), "y", +1.0   # "top"


def covering_boxes(box, side, all_boxes, eps):
    """Every OTHER box sitting just OUTSIDE `box`'s `side`, with the
    overlap of their spans along that edge.

    Coverage is tested by probing a point eps outside the face rather than
    by matching the neighbour's opposite face to it, so this catches both
    cases that must not get a wall:

      - boxes that merely TOUCH (neighbour's opposite face == this face),
      - boxes that OVERLAP / interpenetrate (neighbour's volume straddles
        this face -- e.g. two boxes fit over the same corridor with a bit
        of intersection). The old coplanar-face test missed these, leaving
        a wall buried inside the neighbouring space, visible in
        OpenStudio as a panel sticking into an open room.
    """
    fixed, span, axis, sign = side_info(box, side)
    span_axis = "y" if axis == "x" else "x"
    probe = fixed + sign * eps
    out = []
    for ob in all_boxes:
        if ob is box:
            continue
        if not (ob[f"{axis}_min"] <= probe <= ob[f"{axis}_max"]):
            continue
        ov = interval_intersect(span, (ob[f"{span_axis}_min"], ob[f"{span_axis}_max"]), eps)
        if ov is not None:
            out.append((ob, ov))
    return out


def is_buried_slab(box, z, all_boxes, eps):
    """True if the horizontal rectangle of `box` at height z lies strictly
    inside ANOTHER box's volume -- an interior floor/ceiling slab cutting
    through an open space (happens when boxes overlap, e.g. a low box
    tiled inside a taller one). Coplanar slabs of neighbouring boxes at
    the same height are NOT buried (z is on that box's own boundary), so
    multi-box room floors/ceilings are kept as before."""
    for ob in all_boxes:
        if ob is box:
            continue
        if (ob["x_min"] <= box["x_min"] + eps and ob["x_max"] >= box["x_max"] - eps
                and ob["y_min"] <= box["y_min"] + eps and ob["y_max"] >= box["y_max"] - eps
                and ob["z_min"] <= z - eps and ob["z_max"] >= z + eps):
            return True
    return False


def build_wall_segments(box, all_boxes, eps):
    """For every side of `box`: the (side, fixed, lo, hi, z_lo, z_hi)
    exterior wall segments to turn into Surfaces -- the part of that whole
    face (its span x its own [z_min, z_max]) not covered by any OTHER
    box's own footprint AND height just outside that face, same room or
    not (see the module docstring: touching -- or overlapping -- boxes are
    always left open to each other, never separated by a wall, regardless
    of which room each is in). Z matters here, not just span: a box only
    opens up the portion of its neighbour's wall it actually reaches -- a
    shorter room touching a taller one in plan does NOT open up the taller
    room's wall above the shorter room's own ceiling, since nothing is
    there to be open into."""
    segments = []
    for side in SIDES:
        fixed, span, _, _ = side_info(box, side)
        face = (span[0], span[1], box["z_min"], box["z_max"])
        cuts = []
        for ob, ov in covering_boxes(box, side, all_boxes, eps):
            z_lo, z_hi = max(box["z_min"], ob["z_min"]), min(box["z_max"], ob["z_max"])
            if z_hi - z_lo > eps:
                cuts.append((ov[0], ov[1], z_lo, z_hi))
        for lo, hi, z_lo, z_hi in subtract_rects_2d(face, cuts, eps):
            segments.append((side, fixed, lo, hi, z_lo, z_hi))
    return segments


def openings_in_segment(openings, box_id, side, lo, hi, z_lo, z_hi, inset):
    """The openings that belong to one wall segment, clipped and inset.

    A SubSurface has to lie inside its parent Surface. An opening that straddles
    the seam between two wall segments of the same face belongs to neither, and
    an opening flush with the parent's own edge is a degenerate parent polygon
    once the hole is cut. Clipping to the segment and insetting keeps every
    emitted rectangle strictly inside exactly one parent.

    Yields (u0, u1, z0, z1), opening.
    """
    for op in openings:
        if op["box_id"] != box_id or op["side"] != side:
            continue
        u0, u1 = max(op["u_min"], lo + inset), min(op["u_max"], hi - inset)
        v0, v1 = max(op["z_min"], z_lo + inset), min(op["z_max"], z_hi - inset)
        if u1 - u0 <= inset or v1 - v0 <= inset:
            continue
        yield (u0, u1, v0, v1), op


def cluster_values(values, eps):
    """Group nearby scalar values into clusters (greedy: sorted, a new
    value joins the current cluster if it's within eps of the cluster's
    LAST member) and return {original_value: cluster_mean}. Chaining means
    a long run of values each eps apart can end up spanning more than eps
    end-to-end -- fine for real box coordinates (edges cluster tightly
    around actual wall positions, not in a dense continuous spread), but
    keep --eps modest for that reason."""
    order = sorted(set(values))
    if not order:
        return {}
    clusters = [[order[0]]]
    for v in order[1:]:
        if v - clusters[-1][-1] <= eps:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    mapping = {}
    for c in clusters:
        rep = sum(c) / len(c)
        for v in c:
            mapping[v] = rep
    return mapping


def snap_box_coordinates(boxes, eps):
    """Nudge every box's x_min/x_max/y_min/y_max onto a shared snapped
    grid: two edges (from different boxes, or even the same box) within
    eps of each other collapse to the exact same coordinate. This is what
    makes "touching within tolerance" (build_wall_segments already treats
    them as open) ALSO touch perfectly in the geometry itself -- no sliver
    gap or overlap left between two boxes nudged apart by hand in
    interactive_boxes.py's drag handles, in the floor/ceiling rectangles
    or anywhere else. Mutates `boxes` in place."""
    xs = [b["x_min"] for b in boxes] + [b["x_max"] for b in boxes]
    ys = [b["y_min"] for b in boxes] + [b["y_max"] for b in boxes]
    xmap, ymap = cluster_values(xs, eps), cluster_values(ys, eps)
    n_moved = 0
    for b in boxes:
        for key, mapping in (("x_min", xmap), ("x_max", xmap),
                              ("y_min", ymap), ("y_max", ymap)):
            snapped = mapping[b[key]]
            if abs(snapped - b[key]) > 1e-9:
                n_moved += 1
            b[key] = snapped
    if n_moved:
        print(f"snap: nudged {n_moved} box edge coordinate(s) onto a shared grid "
              f"(within {eps} m) so touching boxes align exactly")
    return boxes


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=Path, default=SAVED_BOXES_DIR / "boxes.json",
                    help="boxes.json to build the model from (default: SavedBoxes/boxes.json "
                         "next to this script)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .osm path -- default: OpenStudioModel/<timestamp>.osm next "
                         "to this script (yy-MM-dd - HH-mm-ss.osm, the moment it's saved, "
                         "so repeated runs never overwrite each other; directory created if "
                         "missing)")
    ap.add_argument("--name", default="Building", help="Building/Space-prefix name")
    ap.add_argument("--eps", type=float, default=0.03,
                    help="m -- tolerance for treating two box edges as touching/aligned, "
                         "and (unless --no-snap) for snapping them to align exactly")
    ap.add_argument("--openings", type=Path, default=None,
                    help="openings.json from WindowsDoorsDetection/fit_openings.py -- each "
                         "rectangle becomes a Door or FixedWindow SubSurface on the wall of "
                         "the box face it was fitted to. Without this the model has no "
                         "openings at all, which is what it had before.")
    ap.add_argument("--opening-inset", type=float, default=0.05,
                    help="m -- keep each SubSurface this far inside its parent wall segment "
                         "(default 0.05). An opening flush with the wall edge leaves a "
                         "degenerate parent polygon once the hole is cut.")
    ap.add_argument("--no-snap", action="store_true",
                    help="skip snapping near-touching box edges to align exactly (see "
                         "snap_box_coordinates) -- use the coordinates from --boxes as-is")
    args = ap.parse_args()

    data = json.loads(args.boxes.read_text())
    boxes = data.get("boxes", [])
    if not boxes:
        raise SystemExit(f"no boxes in {args.boxes} -- nothing to build a model from")

    if not args.no_snap:
        boxes = snap_box_coordinates(boxes, args.eps)

    n_no_hall = sum(1 for b in boxes if "hall" not in b)
    if n_no_hall:
        print(f"WARNING: {n_no_hall}/{len(boxes)} box(es) have no 'hall' field -- each is "
              f"treated as its own single-box room. This shouldn't happen from fit_boxes.py's "
              f"own output; if this came out of interactive_boxes.py, re-save with the "
              f"current version (it now preserves 'hall' through edits).")

    def room_of(b):
        return b.get("hall", b["id"])

    rooms = defaultdict(list)
    for b in boxes:
        rooms[room_of(b)].append(b)

    # translate to a local origin near (0,0,0) -- purely cosmetic, real-world coordinates
    # are recoverable by adding `origin` back
    origin = np.array([
        min(b["x_min"] for b in boxes), min(b["y_min"] for b in boxes),
        min(b["z_min"] for b in boxes),
    ], dtype=float)
    print(f"translating geometry by {(-origin).round(3).tolist()} "
          f"(original min corner {origin.round(3).tolist()})")

    model = openstudio.model.Model()
    spaces = {}
    for room_id in sorted(rooms, key=str):
        space = openstudio.model.Space(model)
        space.setName(f"{args.name} Room {room_id}")
        zone = openstudio.model.ThermalZone(model)
        zone.setName(f"{args.name} Room {room_id} Zone")
        space.setThermalZone(zone)
        spaces[room_id] = space

    def to_os_points(corners):
        # explicit float(): a numpy int64/float32 scalar (e.g. from integer coordinates in
        # a hand-edited boxes.json) doesn't match Point3d's SWIG binding, which wants a
        # plain Python float for each of x/y/z
        return openstudio.Point3dVector(
            [openstudio.Point3d(*(float(v) for v in (np.array(c, dtype=float) - origin)))
             for c in corners])

    openings = []
    if args.openings:
        openings = json.loads(args.openings.read_text(encoding="utf-8")).get("openings", [])
        print(f"{len(openings)} opening(s) from {args.openings.name}")

    n_floor = n_roof = n_ext_wall = n_buried = 0
    n_sub = defaultdict(int)
    placed = set()

    for box in boxes:
        space = spaces[room_of(box)]
        x0, x1, y0, y1 = box["x_min"], box["x_max"], box["y_min"], box["y_max"]
        z0, z1 = box["z_min"], box["z_max"]

        if is_buried_slab(box, z0, boxes, args.eps):
            n_buried += 1
        else:
            floor = openstudio.model.Surface(
                to_os_points(rect_corners(x0, x1, y0, y1, z0, -1)), model)
            floor.setSpace(space)
            floor.setName(f"box_{box['id']}_floor")
            floor.setSurfaceType("Floor")
            floor.setOutsideBoundaryCondition("Ground")
            floor.setSunExposure("NoSun")
            floor.setWindExposure("NoWind")
            n_floor += 1

        if is_buried_slab(box, z1, boxes, args.eps):
            n_buried += 1
        else:
            roof = openstudio.model.Surface(
                to_os_points(rect_corners(x0, x1, y0, y1, z1, +1)), model)
            roof.setSpace(space)
            roof.setName(f"box_{box['id']}_roof")
            roof.setSurfaceType("RoofCeiling")
            roof.setOutsideBoundaryCondition("Outdoors")
            roof.setSunExposure("SunExposed")
            roof.setWindExposure("WindExposed")
            n_roof += 1

        for side, fixed, lo, hi, sz0, sz1 in build_wall_segments(box, boxes, args.eps):
            corners, _ = wall_corners(side, fixed, lo, hi, sz0, sz1)
            wall = openstudio.model.Surface(to_os_points(corners), model)
            wall.setSpace(space)
            wall.setSurfaceType("Wall")
            wall.setName(f"box_{box['id']}_wall_{side}_{n_ext_wall}")
            wall.setOutsideBoundaryCondition("Outdoors")
            wall.setSunExposure("SunExposed")
            wall.setWindExposure("WindExposed")
            n_ext_wall += 1

            # Openings go on AFTER the parent exists and only on exterior walls:
            # a face covered by a neighbouring box got no Surface at all (see
            # the module docstring), and an opening there is already an open
            # passage in the geometry -- there is nothing to cut a hole in.
            for rect, op in openings_in_segment(openings, box["id"], side, lo, hi,
                                                sz0, sz1, args.opening_inset):
                u0, u1, v0, v1 = rect
                sub_corners, _ = wall_corners(side, fixed, u0, u1, v0, v1)
                sub = openstudio.model.SubSurface(to_os_points(sub_corners), model)
                sub.setSurface(wall)
                sub.setSubSurfaceType(op.get("subsurface_type",
                                             "Door" if op["class"] == "door"
                                             else "FixedWindow"))
                sub.setName(f"{op['class']}_{op.get('id', len(placed))}_"
                            f"box{box['id']}_{side}")
                n_sub[op["class"]] += 1
                placed.add(op.get("id"))

    print(f"{len(boxes)} box(es) -> {len(spaces)} room(s): {n_floor} floor, {n_roof} roof, "
          f"{n_ext_wall} exterior wall(s) (no walls between any touching or overlapping "
          f"boxes, same room or not -- see the module docstring)")
    if n_buried:
        print(f"  skipped {n_buried} floor/ceiling slab(s) buried inside another box's volume")
    if openings:
        print(f"  {sum(n_sub.values())} sub-surface(s): " +
              ", ".join(f"{c}={n}" for c, n in sorted(n_sub.items())))
        missed = [o for o in openings if o.get("id") not in placed]
        if missed:
            print(f"  {len(missed)} opening(s) placed nowhere -- their box face has no "
                  f"exterior wall there (another box covers it), or they fell outside every "
                  f"wall segment: " +
                  ", ".join(f"#{o.get('id')} {o['class']} box {o['box_id']} {o['side']}"
                            for o in missed))

    for room_id, space in spaces.items():
        try:
            print(f"  room {room_id}: floor area {space.floorArea():.2f} m2, "
                  f"volume {space.volume():.2f} m3")
        except Exception as e:
            print(f"  room {room_id}: area/volume n/a ({e})")

    out = args.out
    if out is None:
        stamp = datetime.now().strftime("%y-%m-%d - %H-%M-%S")   # "/" and ":" aren't valid
        out = OPENSTUDIO_MODEL_DIR / f"{stamp}.osm"               # in a Windows filename
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(openstudio.path(str(out)), True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
