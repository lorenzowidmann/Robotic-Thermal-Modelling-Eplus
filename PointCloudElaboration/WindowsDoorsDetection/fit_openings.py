"""Turn labelled opening points into wall-mounted rectangles an OpenStudio
SubSurface can be built from, and draw them on the boxes.

This is the step between WindowsDoorsDetection and 3DModelPointCloudExtraction:

    paint_openings.py --out-csv   ->  x,y,z,opening_class
    fit_boxes.py                  ->  boxes.json (the rooms)
    THIS                          ->  openings.json (rectangles ON box faces)
    to_openstudio.py --openings   ->  .osm with Door / FixedWindow sub-surfaces

Why a rectangle on a box face and not a polygon fit to the points: OpenStudio
does not accept a free-floating opening. A SubSurface must be **coplanar with,
and inside, its parent Surface** -- the wall polygon `to_openstudio.py` already
builds from a box side. So the useful output is not "where are the points", it
is "which face, and which rectangle of it". The points only have to be good
enough to pick the face and bound the rectangle, which is a much weaker demand
than fitting glass that returns almost nothing (see the README section "The door
is not where the door is").

How a point becomes a rectangle:

1. **Face assignment.** Each point goes to the nearest vertical box face whose
   span and height it falls within, if that face is closer than
   --max-face-dist. Points near no face are dropped and counted -- clutter in
   the middle of the room is not an opening.
2. **Clustering** in that face's own 2-D coordinates (u along the face, z up),
   by occupied --cluster-cell cells joined 8-connected. Two windows on the same
   wall separate as long as the gap between them is wider than one cell.
3. **Rectangle** = the cluster's extent in (u, z), clipped to the face, then
   filtered by --min-width / --min-height / --min-points.

Every rectangle carries the box id, the side, and the count and spread of the
points behind it, so a bad one can be traced rather than just deleted.

Usage:
    C:\\venvs\\planefit\\Scripts\\python.exe fit_openings.py ^
        --points openings_pts.csv ^
        --boxes ...\\3DModelPointCloudExtraction\\SavedBoxes\\boxes_edited2.json ^
        --out openings.json --screenshot openings_model.png

    ... --points-in-view      :: draw the source points too
    ... --no-show             :: just write the json

    :: a repeated bay: snap the copies to one size on one pitch, and fill the
    :: slots the scan missed (those are INFERRED -- drawn hollow, n_points 0)
    ... --regularize-fill --regularize-size-tol 2.5

Venv: C:\\venvs\\planefit.
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

def _find_root(start: Path) -> Path:
    """Walk up until the directory holding SensorFusionLoader/ is found."""
    for d in [start, *start.parents]:
        if (d / "SensorFusionLoader").is_dir():
            return d
    raise RuntimeError(
        f"SensorFusionLoader/ not found in any parent of {start} -- it holds "
        "rig_calibration.py/.yaml and projection.py, which --masks needs.")


_ROOT = _find_root(Path(__file__).resolve().parent)

# The door band is stage 1B's, imported rather than restated so the two cannot
# drift: a candidate the metric check accepted in 2-D must not be rejected here
# on different numbers. See openings/lidar_metrics.py for how each was measured.
sys.path.insert(0, str(Path(__file__).resolve().parent / "openings"))
from lidar_metrics import (MIN_DOOR_H_M, MAX_DOOR_H_M,  # noqa: E402
                           MIN_DOOR_W_M, MAX_DOOR_W_M)

CLASS_COLORS = {"door": (220, 40, 40), "window": (40, 110, 230)}
BOX_COLOR = (90, 90, 100)
SIDES = ("left", "right", "bottom", "top")
# OpenStudio's own vocabulary, so to_openstudio.py can pass it straight through.
OS_SUBSURFACE_TYPE = {"door": "Door", "window": "FixedWindow"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Fit wall-mounted opening rectangles to labelled opening points")
    p.add_argument("--points", required=True, metavar="CSV",
                   help="x,y,z,opening_class -- paint_openings.py --out-csv, or stage 2's "
                        "door_window_voxels.csv (same columns plus extras).")
    p.add_argument("--boxes", required=True, metavar="JSON",
                   help="fit_boxes.py's boxes.json for the same session.")
    p.add_argument("--out", default=None, metavar="JSON",
                   help="Write the rectangles here (default <points dir>/openings.json).")

    p.add_argument("--max-face-dist", type=float, default=0.60, metavar="M",
                   help="A point farther than this from every box face is not an opening "
                        "(default 0.60). The glazing sits in a reveal and the returns are "
                        "off the mullions, so this cannot be tight.")
    p.add_argument("--span-margin", type=float, default=0.25, metavar="M",
                   help="Slack when testing whether a point falls within a face's span and "
                        "height (default 0.25).")
    p.add_argument("--cluster-cell", type=float, default=0.30, metavar="M",
                   help="Cell size for the on-face clustering (default 0.30). Two openings "
                        "closer than this on the same wall merge into one.")
    p.add_argument("--min-points", type=int, default=60, metavar="N",
                   help="Drop a cluster with fewer points than this (default 60).")
    p.add_argument("--min-width", type=float, default=0.50, metavar="M",
                   help="Applies to every class (default 0.50); doors get the tighter band "
                        "below on top of it.")
    p.add_argument("--min-height", type=float, default=0.50, metavar="M")
    p.add_argument("--max-width", type=float, default=0.0, metavar="M",
                   help="Drop a cluster wider than this, 0 = no limit (default). A whole "
                        "glazed corridor wall is one legitimate cluster many metres long.")

    g = p.add_argument_group(
        "plausibility -- where an opening can be, not just how big it is")
    g.add_argument("--door-min-w", type=float, default=MIN_DOOR_W_M, metavar="M")
    g.add_argument("--door-max-w", type=float, default=MAX_DOOR_W_M, metavar="M")
    g.add_argument("--door-min-h", type=float, default=MIN_DOOR_H_M, metavar="M",
                   help=f"Door size band, default {MIN_DOOR_W_M}-{MAX_DOOR_W_M} m wide by "
                        f"{MIN_DOOR_H_M}-{MAX_DOOR_H_M} m tall -- stage 1B's own measured "
                        f"values, imported from openings/lidar_metrics.py.")
    g.add_argument("--door-max-h", type=float, default=MAX_DOOR_H_M, metavar="M")
    g.add_argument("--door-head-clearance", type=float, default=0.12, metavar="M",
                   help="Keep at least this much wall above a door's head (default 0.12, "
                        "0 = off). A door never reaches the ceiling -- there is a lintel. "
                        "The mask does though: it runs to the top of the opening and the "
                        "rectangle then lands exactly on the box's z_max, which reads as a "
                        "full-height hole. The head is CLAMPED, not rejected: the door is "
                        "real, only its top edge is wrong.")
    g.add_argument("--door-floor-tol", type=float, default=0.30, metavar="M",
                   help="A door has to reach the floor: reject one whose sill is more than "
                        "this above its box's z_min (default 0.30, 0 = off). This is what "
                        "kills the fragments a door mask leaves on the side walls beside "
                        "the real opening -- they float.")
    g.add_argument("--window-max-sill", type=float, default=0.0, metavar="M",
                   help="Reject a window whose sill is higher than this above the floor. "
                        "0 = off (default). Deliberately NOT a MIN sill: ADE `windowpane` "
                        "swallows the radiators under the glass, so every window here "
                        "measures down to the floor and a minimum-sill rule would reject "
                        "all of them. See the README.")
    g.add_argument("--exterior-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep openings only on faces that will actually become a wall "
                        "(default on). A face another box sits against gets no Surface in "
                        "to_openstudio.py -- the two spaces are open to each other there -- "
                        "so an opening on it can never be built.")
    g.add_argument("--cover-eps", type=float, default=0.03, metavar="M",
                   help="Probe distance for the face-covered test, to_openstudio.py's --eps "
                        "(default 0.03).")
    m = p.add_argument_group(
        "masks -- take the outline from the camera, the plane from the LiDAR")
    m.add_argument("--masks", action="store_true",
                   help="Fit each opening from its MASK projected onto the wall plane, "
                        "instead of from the extent of its LiDAR points. The laser reaches "
                        "the bottom edge of a window and stops ~57%% of the way up it (the "
                        "HAP's 25 deg vertical against the ZED's 54 deg), so a point extent "
                        "measures the sensor, not the opening. Needs --session-dir and a "
                        "csv from paint_openings.py, which carries the frame and segment id "
                        "each point came from. One rectangle per (frame, segment), so bays "
                        "stay separate instances and nothing has to be re-split by "
                        "clustering.")
    m.add_argument("--session-dir", default=None, metavar="DIR",
                   help="Session holding sync_manifest.json, for --masks.")
    m.add_argument("--opening-map-dir", default=None, metavar="DIR",
                   help="Stage 1 output with labels.npy (default <session>/opening_map).")
    m.add_argument("--calibration", default=None, metavar="YAML")
    m.add_argument("--mask-min-points", type=int, default=20, metavar="N",
                   help="A segment needs this many LiDAR points on one face before its "
                        "mask is projected there (default 20). Below it, nothing says "
                        "WHICH wall the mask is on, and projecting onto a guess is worse "
                        "than not projecting.")
    m.add_argument("--mask-min-cos", type=float, default=0.20, metavar="C",
                   help="Refuse a ray whose |cos| against the wall normal is under this "
                        "(default 0.20, ~78 deg off face-on). Near-parallel rays slide the "
                        "intersection metres for one pixel of noise.")
    m.add_argument("--mask-max-depth", type=float, default=25.0, metavar="M",
                   help="Refuse a projection landing further than this from the camera "
                        "(default 25, 0 = no limit).")
    m.add_argument("--merge-touch-gap", type=float, default=0.0, metavar="M",
                   help="Fuse openings on the same face that TOUCH within this gap into one "
                        "rectangle (0 = off). The result is the group's axis-aligned "
                        "bounding box, so it is always a rectangle, never an L: only "
                        "side-by-side or stacked neighbours merge, never diagonal ones, "
                        "whose bounding box would swallow the empty corner between them. "
                        "Use it when the segmentation cuts one run of glazing at its "
                        "mullions and the model should carry one SubSurface.")
    m.add_argument("--mask-merge-iou", type=float, default=0.50, metavar="F",
                   help="Views of one opening merge when their rectangles overlap by at "
                        "least this fraction OF THE SMALLER of the two (default 0.50). "
                        "Containment, not IoU: a view that caught only part of a bay is "
                        "largely inside the fuller view, so its IoU is low while its "
                        "containment is near 1 -- scoring by IoU left three overlapping "
                        "rectangles where there is one bay. Each edge of the merged "
                        "rectangle is the MEDIAN across views, never the union, which "
                        "would grow with every extra view and every bad one.")

    v = p.add_argument_group(
        "void evidence -- glass returns nothing, so a hole in the wall IS the window")
    v.add_argument("--void-evidence", action="store_true",
                   help="Rescue window segments that have too few LiDAR returns to name "
                        "their wall (--mask-min-points), by testing whether they sit on a "
                        "HOLE in the wall's coverage. Glazing reflects almost nothing back, "
                        "so the better glazed a bay is the fewer points it has -- the normal "
                        "route fails hardest exactly where the window is most real. Needs "
                        "--scene-points. Windows only: a door is opaque and a hole where a "
                        "door should be is a missing scan, not a door.")
    v.add_argument("--scene-points", default=None, metavar="CSV",
                   help="The NON-opening returns, from paint_openings.py --out-scene-csv. "
                        "These are the wall around the hole, and they are what make the "
                        "hole mean something.")
    v.add_argument("--void-slab", type=float, default=0.35, metavar="M",
                   help="A scene point counts as 'on this wall' within this distance of the "
                        "face plane (default 0.35, matching the reveal depth --max-face-dist "
                        "allows).")
    v.add_argument("--void-ring", type=float, default=0.40, metavar="M",
                   help="Width of the band around the rectangle whose density the inside is "
                        "compared against (default 0.40).")
    v.add_argument("--void-max-ratio", type=float, default=0.35, metavar="F",
                   help="The inside must be at most this fraction as dense as the ring "
                        "(default 0.35). Above it the surface is returning light like a "
                        "wall does, so it is a wall.")
    v.add_argument("--image-only-windows", action="store_true",
                   help="Windows are decided by the CAMERA alone. The LiDAR returns that "
                        "land inside a window play no part in confirming it, choosing its "
                        "wall, or sizing it -- glass returns almost nothing, so those points "
                        "describe the radiator and the mullions in front of the glass, not "
                        "the opening. A `windowpane` mask of at least --window-min-mask-px "
                        "is taken as a window; its wall is picked by ray; its size is the "
                        "mask intersected with that wall's plane. The plane still comes from "
                        "the LiDAR, but from the WALL's returns (via fit_boxes.py), which are "
                        "dense and reliable. Doors keep the point-based path: a door is "
                        "opaque and its returns are trustworthy.")
    v.add_argument("--window-min-mask-px", type=int, default=8000, metavar="N",
                   help="With --image-only-windows, a mask this big (default 8000 px, ~0.4%% "
                        "of a 1920x1080 frame) is confirmed as a window. Below it the region "
                        "is too small to trust as an opening rather than a reflection or a "
                        "sliver of one seen edge-on.")
    v.add_argument("--mask-face-min-frac", type=float, default=0.60, metavar="F",
                   help="When the wall is chosen by ray (void route), at least this "
                        "fraction of the mask's boundary must land on the candidate face "
                        "(default 0.60). One ray through the centroid says nothing about "
                        "whether the rest of the mask reaches that face at all.")
    v.add_argument("--void-min-ring-points", type=int, default=40, metavar="N",
                   help="The ring needs at least this many returns before the hole is "
                        "believed (default 40). Without it, an unscanned patch of wall "
                        "passes the density test trivially -- nothing inside, nothing "
                        "around, ratio 0/0 -- and every gap in the scan becomes a window.")

    r = p.add_argument_group(
        "regularity -- a repeated bay is evidence, and a missed one is a gap")
    r.add_argument("--regularize", action="store_true",
                   help="Per wall face and class: detect a regular lattice of openings and "
                        "snap them all to one size and one pitch. A facade bay repeats; the "
                        "LiDAR's view of each copy does not, so the copies come out as "
                        "rectangles of slightly different size at slightly wrong spacing. "
                        "OFF by default -- it replaces measurements with a model, and the "
                        "run prints every change it makes.")
    r.add_argument("--regularize-fill", action="store_true",
                   help="Also CREATE an opening at each empty slot of a detected lattice "
                        "(implies --regularize). These are inferred, not measured: they "
                        "carry n_points 0 and \"synthetic\": true in openings.json, and the "
                        "viewer draws them hollow. A bay the scan simply never saw from a "
                        "good angle is the case this exists for.")
    r.add_argument("--regularize-extend", action="store_true",
                   help="Continue a detected lattice to BOTH ENDS of the wall face, not just "
                        "between the openings actually found (implies --regularize-fill). "
                        "Use it when a bay demonstrably repeats along a facade and the scan "
                        "only covered the middle of it. Everything it adds is synthetic, and "
                        "it stops at the face's own span -- a lattice cannot cross onto "
                        "another box's wall, because that is a different face.")
    r.add_argument("--bays-from-wall", action="store_true",
                   help="Find the openings in the WALL -- position, width, sill and head from "
                        "its own bay rhythm read in plan view at every height -- and use the "
                        "camera only to say WHAT IS IN each one: a window, a door, or nothing "
                        "seen. Neither the points inside an opening nor a mask's own extent "
                        "set its geometry: the mask only has to overlap a bay. Every window "
                        "on one wall then comes out identical, which is what a repeated bay "
                        "is, while a door keeps its own width and is taken down to the floor "
                        "(--bay-doors, --bay-door-width, --bay-door-to-floor). Needs "
                        "--scene-points.")
    r.add_argument("--bay-trough-frac", type=float, default=0.35, metavar="F",
                   help="A bin belongs to a trough while it holds under this fraction of the "
                        "LOCAL pier level, default 0.35. This is what sets the bay width. "
                        "Local, not global: the rover passed close to one end of the corridor "
                        "and far from the other, so the same pier returns 60 points at one end "
                        "and 700 at the other and one threshold cannot serve both.")
    r.add_argument("--bay-uniform-size", action=argparse.BooleanOptionalAction, default=True,
                   help="Every window bay on one face comes out the SAME size, height "
                        "included (default on). A bay is one window built many times, so its "
                        "copies are many measurements of one size, not many sizes. The width "
                        "already arrives from the wall as a single number; this gives the "
                        "sill and head the same standing, which they otherwise lack because "
                        "each bay's vertical scan is stopped somewhere different by a "
                        "radiator, a curtain or the rover's line of sight. Doors are exempt: "
                        "a door does not repeat.")
    r.add_argument("--bay-size-nstd", type=float, default=1.0, metavar="N",
                   help="How far from the mean a bay may sit and still count towards the "
                        "shared size (default 1.0 standard deviation). The spread of the "
                        "copies IS the estimate of how far a copy may be off: inside it they "
                        "are one window measured repeatedly, outside it -- a bay read half "
                        "height, a doorway read as a window -- they are a measurement of "
                        "something else and are excluded from the average, not from the "
                        "output. Every exclusion is printed. Applies to the width too.")
    r.add_argument("--bay-min-pier", type=float, default=0.35, metavar="M",
                   help="The narrowest run of returns that counts as a PIER (default 0.35). "
                        "Anything narrower inside a bay -- a mullion, a window frame, the "
                        "reveal -- is erased before the bay's edges are measured. Without it "
                        "the edge walk stops at the mullion and the bay comes out a third of "
                        "its true width: session 9 measured 1.50 m bays this way against 2.33 "
                        "m once the single-bin spikes are opened out. 0 = off.")
    r.add_argument("--bay-snap-phase", action="store_true",
                   help="Put every bay exactly on the fitted lattice slot instead of on its "
                        "own measured trough centre. Off by default: session 9's south wall "
                        "is 0.33 m out of phase before x=21 m and in phase after it, so one "
                        "offset misplaces five of nine bays. The lattice still decides HOW "
                        "MANY bays there are and they still share one width -- only their "
                        "centres are measured.")
    r.add_argument("--bay-floor-tol", type=float, default=0.35, metavar="M",
                   help="A bay whose opening runs to within this of the floor is flagged "
                        "`reaches_floor` (default 0.35) -- the wall's own evidence that the "
                        "opening is a doorway rather than a window over a spandrel.")
    r.add_argument("--bay-doors", action=argparse.BooleanOptionalAction, default=True,
                   help="Let a `door` detection claim a bay, not only `windowpane` (default "
                        "on). The bay says there is an opening; the camera says what is in "
                        "it. Class is decided by which class's masks cover the most of the "
                        "bay. --no-bay-doors leaves doors exactly as measured.")
    r.add_argument("--bay-door-width", choices=("mask", "bay"), default="mask",
                   help="A door claiming a bay keeps its own measured width (default `mask`) "
                        "or is widened to the bay (`bay`). Session 9's bays are 2.3 m and its "
                        "doors under 1.1 m -- the door stands IN the structural opening, it "
                        "is not the same size as it.")
    r.add_argument("--bay-door-to-floor", action=argparse.BooleanOptionalAction, default=True,
                   help="Take a bay-confirmed door's sill to the floor (default on). A door "
                        "reaches the floor by definition, and the vertical scan cannot see "
                        "the bottom of the opening past whatever is standing in it.")
    r.add_argument("--bay-z-band", type=float, default=0.20, metavar="M",
                   help="Height of each slice in the vertical scan (default 0.20).")
    r.add_argument("--bay-z-step", type=float, default=0.10, metavar="M",
                   help="Step between slices in the vertical scan (default 0.10).")
    r.add_argument("--bay-z-min-points", type=int, default=150, metavar="N",
                   help="Returns needed in a slice before its contrast is trusted "
                        "(default 150).")
    r.add_argument("--bay-min-contrast", type=float, default=0.35, metavar="F",
                   help="Pier-vs-trough contrast a slice must show to count as cutting "
                        "through glazing (default 0.35). Solid wall gives ~0 contrast; the "
                        "sill and head are where the contrast starts and stops.")
    r.add_argument("--bay-height-from-mask", action="store_true",
                   help="Take each bay's SILL and HEAD from the camera masks that confirmed "
                        "it, keeping only position and width from the wall. The vertical "
                        "scan cannot see past a radiator -- it reads the top of the radiator "
                        "as the bottom of the wall and puts the sill there (0.87 m on "
                        "session 9, against -0.05 m from the masks). The masks see the whole "
                        "opening down to the floor. Head is the higher of the two, since the "
                        "wall rhythm survives above where a single mask gets clipped.")
    r.add_argument("--bay-confirm-overlap", type=float, default=0.20, metavar="M",
                   help="A `windowpane` detection confirms a bay when it overlaps it by at "
                        "least this much along the wall (default 0.20).")
    r.add_argument("--pitch-from-wall", action="store_true",
                   help="Measure the bay pitch from the WALL's returns in plan view "
                        "(autocorrelation of a horizontal band histogrammed along the "
                        "wall), and give it to the regularizer instead of fitting a pitch "
                        "to the openings' centres. The centres carry whatever drift the "
                        "merge left in them; the wall does not. Needs --scene-points. "
                        "Session 9 reads 3.20 m this way from both walls and every height "
                        "band, against 3.35 m (refused, RMS 0.41) and a degenerate 1.40 m "
                        "from the centre fit.")
    r.add_argument("--pitch-bands", action=argparse.BooleanOptionalAction, default=True,
                   help="Measure the pitch from EVERY height band and average their "
                        "autocorrelations before picking the peak (default on). The rhythm "
                        "belongs to the structure, so it shows in every plan view that cuts "
                        "the wall: on session 9, 19 of 21 bands from the floor to 1.87 m read "
                        "3.20 m on both walls. One band can be wrong -- the north wall reads "
                        "1.60 m at z 0.07-0.27, off the skirting and the radiator feet -- and "
                        "one bad band out of twenty cannot move the stack. --no-pitch-bands "
                        "uses the single --pitch-z band.")
    r.add_argument("--pitch-min-bands", type=int, default=3, metavar="N",
                   help="Bands needed before the stack is used at all (default 3); below it "
                        "the measurement falls back to the single --pitch-z band.")
    r.add_argument("--pitch-z", type=float, nargs=2, default=(0.90, 1.70),
                   metavar=("ZLO", "ZHI"),
                   help="Height band of wall used for the bay WIDTH and phase, and for the "
                        "pitch under --no-pitch-bands (default 0.90-1.70 m). Chosen ABOVE the "
                        "radiators: lower down they fill the troughs with returns and the "
                        "contrast drops (r 0.44-0.51 against 0.60-0.62 for this band).")
    r.add_argument("--pitch-bin", type=float, default=0.10, metavar="M",
                   help="Bin width of the wall profile (default 0.10) -- also the "
                        "resolution the pitch is reported at.")
    r.add_argument("--pitch-min", type=float, default=1.50, metavar="M",
                   help="Shortest lag considered a pitch (default 1.50).")
    r.add_argument("--pitch-max", type=float, default=8.00, metavar="M",
                   help="Longest lag considered a pitch (default 8.00).")
    r.add_argument("--pitch-min-corr", type=float, default=0.30, metavar="R",
                   help="Minimum autocorrelation at the peak before the pitch is believed "
                        "(default 0.30). Below it the wall has no rhythm to find.")
    r.add_argument("--pitch-min-points", type=int, default=500, metavar="N",
                   help="Wall returns needed in the band before the profile is trusted "
                        "(default 500).")
    r.add_argument("--regularize-min-pier", type=float, default=1.20, metavar="F",
                   help="A detected pitch must be at least this many times the openings' "
                        "own median width (default 1.20), i.e. at least 20%% of a width of "
                        "PIER between them. A pitch equal to the width means the openings "
                        "touch edge to edge -- that is one continuous run of glazing, a "
                        "shape rather than a rhythm, and fitting a lattice to it imposes a "
                        "repeat the wall does not have.")
    r.add_argument("--regularize-min-count", type=int, default=3, metavar="N",
                   help="A lattice needs at least this many openings on one face to be "
                        "claimed at all (default 3). Two is a coincidence.")
    r.add_argument("--regularize-tol", type=float, default=0.35, metavar="M",
                   help="Max RMS residual of the fitted lattice (default 0.35). Above it the "
                        "spacing is not regular and the face is left exactly as measured.")
    r.add_argument("--regularize-size-tol", type=float, default=0.60, metavar="M",
                   help="Max spread (p95-p05) of the openings' widths on a face before they "
                        "are refused as 'not the same window repeated' (default 0.60).")
    g.add_argument("--merge-corner-dist", type=float, default=1.0, metavar="M",
                   help="Same-class rectangles on DIFFERENT faces closer than this collapse "
                        "to the one with the most points (default 1.0, 0 = off). Measured "
                        "box-to-box, not centre-to-centre: an opening at the end of a "
                        "corridor is in reach of three faces at once -- the end wall and "
                        "both side walls -- and its two fits nearly touch at the corner "
                        "while their centres are over 1.5 m apart.")
    p.add_argument("--points-in-view", action="store_true",
                   help="Draw the source points as well as the fitted rectangles.")
    p.add_argument("--screenshot", default=None, metavar="PNG")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--window-size", default="1600,1000", metavar="W,H")
    return p.parse_args()


def load_points(path: Path):
    """stdlib csv -- works on paint_openings.py's output and on stage 2's
    door_window_voxels.csv, which has the same four columns plus more.

    `frame` and `segment_id` come back too when present (paint_openings.py
    writes them); they are empty strings on a csv that has no provenance, which
    is what --masks checks before refusing to run.
    """
    xyz, cls, prov = [], [], []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            xyz.append((float(r["x"]), float(r["y"]), float(r["z"])))
            cls.append(r["opening_class"])
            prov.append((r.get("frame", ""), r.get("segment_id", "")))
    return (np.asarray(xyz, dtype=float), np.asarray(cls, dtype=object),
            np.asarray(prov, dtype=object))


def load_scene(path: Path) -> np.ndarray:
    """x,y,z of the non-opening returns -- paint_openings.py --out-scene-csv."""
    xyz = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            xyz.append((float(r["x"]), float(r["y"]), float(r["z"])))
    return np.asarray(xyz, dtype=float)


# --- mask projection --------------------------------------------------------
# The LiDAR reaches the bottom edge of a window mask and stops roughly halfway
# up it (measured: 56.8% of the mask's height, median over 11 segments, with the
# gap always at the TOP -- the HAP's 25 deg vertical against the ZED's 54 deg).
# So the extent of the points is the sensor's reach, not the opening. These
# functions take the outline from the mask instead and put it on the plane the
# points identified: the camera measures the shape, the LiDAR measures which
# wall it is on and how far away.

def camera_rays(px: np.ndarray, cal, triplet, width: int, height: int):
    """Pixels -> (camera centre in world, unit ray directions in world).

    Runs projection.project_lidar_to_camera's chain backwards:
        world -> lidar   p_l = R^T (p_w - t)
        lidar -> camera  p_c = R_lc p_l + t_lc
    so a camera-frame direction goes back as d_w = R (R_lc^T d_c), and the
    camera centre is the world point whose camera coordinates are zero.
    """
    import cv2
    from projection import quat_to_rotation_matrix

    R = quat_to_rotation_matrix(*np.asarray(triplet["lidar"]["orientation"], dtype=float))
    t = np.asarray(triplet["lidar"]["position"], dtype=float)
    T = cal.T_lidar_to_zed
    R_lc, t_lc = T[:3, :3], T[:3, 3]

    # p_c = 0  ->  p_l = -R_lc^T t_lc  ->  p_w = R p_l + t
    origin = R @ (-R_lc.T @ t_lc) + t

    K = cal.zed_K_for(width, height)
    norm = cv2.undistortPoints(px.astype(np.float64).reshape(-1, 1, 2),
                               K, cal.zed_calib.dist).reshape(-1, 2)
    d_c = np.column_stack([norm, np.ones(len(norm))])
    d_w = (R @ (R_lc.T @ d_c.T)).T
    d_w /= np.linalg.norm(d_w, axis=1, keepdims=True)
    return origin, d_w


def project_mask_to_plane(mask_px, cal, triplet, width, height,
                          fix_ax, plane_coord, min_cos, max_depth):
    """Mask pixels -> their intersections with the wall plane, in world.

    Returns (points, n_grazing, n_far). A ray nearly parallel to the wall is
    refused, not clamped: near parallel the intersection slides metres for one
    pixel of noise, which is exactly the far end of a corridor seen edge-on.
    """
    origin, dirs = camera_rays(mask_px, cal, triplet, width, height)
    denom = dirs[:, fix_ax]
    # |denom| is |cos| between the ray and the plane NORMAL, the axis being
    # constant on the face -- 1 is face-on, 0 is parallel to the wall.
    ok = np.abs(denom) >= min_cos
    n_grazing = int((~ok).sum())
    tt = np.full(len(dirs), np.nan)
    tt[ok] = (plane_coord - origin[fix_ax]) / denom[ok]
    ok &= tt > 0                       # the wall must be in front of the camera
    if max_depth > 0:
        far = ok & (tt > max_depth)
        n_far = int(far.sum())
        ok &= ~far
    else:
        n_far = 0
    if not ok.any():
        return np.empty((0, 3)), n_grazing, n_far
    return origin + tt[ok, None] * dirs[ok], n_grazing, n_far


def face_of(box, side):
    """(fixed coordinate, span axis index, (span_lo, span_hi), fixed axis index).

    Same side vocabulary as to_openstudio.side_info, so the rectangles this
    writes land on the surfaces that script builds."""
    if side == "left":
        return box["x_min"], 1, (box["y_min"], box["y_max"]), 0
    if side == "right":
        return box["x_max"], 1, (box["y_min"], box["y_max"]), 0
    if side == "bottom":
        return box["y_min"], 0, (box["x_min"], box["x_max"]), 1
    return box["y_max"], 0, (box["x_min"], box["x_max"]), 1   # "top"


def assign_faces(pts: np.ndarray, boxes, max_dist: float, margin: float):
    """Nearest vertical face per point. Returns (index into `faces`, distance),
    with -1 where nothing was close enough."""
    faces = [(b, s) for b in boxes for s in SIDES]
    best = np.full(len(pts), -1, dtype=np.int64)
    best_d = np.full(len(pts), np.inf)
    for fi, (box, side) in enumerate(faces):
        fixed, span_ax, span, fix_ax = face_of(box, side)
        d = np.abs(pts[:, fix_ax] - fixed)
        ok = (d < max_dist) & (d < best_d)
        ok &= (pts[:, span_ax] >= span[0] - margin) & (pts[:, span_ax] <= span[1] + margin)
        ok &= (pts[:, 2] >= box["z_min"] - margin) & (pts[:, 2] <= box["z_max"] + margin)
        best[ok] = fi
        best_d[ok] = d[ok]
    return faces, best, best_d


def rect_is_covered(rec, box, boxes, eps):
    """True if another box sits against the face right where this rectangle is.

    Tested at the rectangle's own centre, not over the whole face:
    to_openstudio.build_wall_segments subtracts only the covered *part* of a
    face and still builds walls either side of it, so a face that is covered
    somewhere is not a face without walls. Rejecting on the whole face
    discarded 11 of 13 real openings along this corridor, where parallel boxes
    cover each other's long sides over part of their length.

    The probe is to_openstudio.covering_boxes': a point eps outside the face,
    tested for being inside any other box. Duplicated rather than imported --
    that module pulls in `openstudio` at import time -- but it has to stay the
    same test, or this keeps openings the model cannot build.
    """
    fixed, span_ax, _span, fix_ax = face_of(box, rec["side"])
    sign = -1.0 if rec["side"] in ("left", "bottom") else 1.0
    probe = np.zeros(3)
    probe[fix_ax] = fixed + sign * eps
    probe[span_ax] = 0.5 * (rec["u_min"] + rec["u_max"])
    probe[2] = 0.5 * (rec["z_min"] + rec["z_max"])
    for ob in boxes:
        if ob is box:
            continue
        if (ob["x_min"] <= probe[0] <= ob["x_max"]
                and ob["y_min"] <= probe[1] <= ob["y_max"]
                and ob["z_min"] <= probe[2] <= ob["z_max"]):
            return True
    return False


def aabb_distance(a, b):
    """Distance between two axis-aligned boxes, 0 if they touch or overlap."""
    gap = np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))
    return float(np.linalg.norm(gap))


def rect_aabb(rec):
    c = rect_corners(rec)
    return c.min(axis=0), c.max(axis=0)


def clamp_head(rec, box, clearance):
    """Pull a door's head down off the ceiling, in place.

    The mask runs to the top of the opening, so the rectangle lands on the box's
    own z_max and the model gets a full-height hole where the building has a
    lintel. Clamped rather than rejected: the door is real, only its top edge
    is. Windows are left alone -- a clerestory legitimately sits high.
    """
    if clearance <= 0 or rec["class"] != "door":
        return
    top = box["z_max"] - clearance
    if rec["z_max"] <= top:
        return
    rec["head_clamped_from"] = rec["z_max"]
    rec["z_max"] = round(float(top), 4)
    rec["height_m"] = round(float(rec["z_max"] - rec["z_min"]), 4)


def plausibility(rec, box, args):
    """Why this rectangle is not an opening where it sits, or None."""
    w, h = rec["width_m"], rec["height_m"]
    # --min-points is meaningless for a void-sourced rectangle: its whole premise
    # is that the surface returned nothing, so counting its returns and rejecting
    # it for having few is circular. Its quality bar is the void test's own --
    # --void-min-ring-points and --void-max-ratio -- which it has already passed.
    if rec.get("evidence") not in ("void", "image") and rec["n_points"] < args.min_points:
        return f"below --min-points ({rec['n_points']})"
    if w < args.min_width:
        return f"below --min-width ({w:.2f} m)"
    if h < args.min_height:
        return f"below --min-height ({h:.2f} m)"
    if args.max_width > 0 and w > args.max_width:
        return f"above --max-width ({w:.2f} m)"

    if rec["class"] == "door":
        if not (args.door_min_w <= w <= args.door_max_w):
            return f"door width {w:.2f} m outside {args.door_min_w}-{args.door_max_w} m"
        if not (args.door_min_h <= h <= args.door_max_h):
            return f"door height {h:.2f} m outside {args.door_min_h}-{args.door_max_h} m"
        if args.door_floor_tol > 0:
            sill = rec["z_min"] - box["z_min"]
            if sill > args.door_floor_tol:
                return f"door does not reach the floor (sill {sill:.2f} m)"
    elif rec["class"] == "window" and args.window_max_sill > 0:
        sill = rec["z_min"] - box["z_min"]
        if sill > args.window_max_sill:
            return f"window sill {sill:.2f} m above --window-max-sill"
    return None


def cluster_on_face(uv: np.ndarray, cell: float):
    """8-connected components over occupied cells. Returns a label per point.

    Grid flood-fill rather than DBSCAN so this needs no sklearn -- the venv that
    has pyvista does not necessarily have it, and the geometry here is a plane,
    where a grid is exactly as good.
    """
    keys = np.floor(uv / cell).astype(np.int64)
    cells = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)

    labels = np.full(len(uv), -1, dtype=np.int64)
    seen, lab = set(), 0
    for start in cells:
        if start in seen:
            continue
        queue, group = deque([start]), []
        seen.add(start)
        while queue:
            c = queue.popleft()
            group.append(c)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    n = (c[0] + du, c[1] + dv)
                    if n in cells and n not in seen:
                        seen.add(n)
                        queue.append(n)
        for c in group:
            labels[cells[c]] = lab
        lab += 1
    return labels


def _mask_boundary(ys, xs):
    """The mask's outline pixels: per row the leftmost and rightmost, per column
    the topmost and bottommost. Bounds the region exactly, at a fraction of the
    pixels, without projecting its whole interior."""
    keep = np.zeros(len(ys), dtype=bool)
    for arr, key in ((ys, xs), (xs, ys)):
        order = np.lexsort((key, arr))
        first = np.ones(len(order), dtype=bool)
        first[1:] = arr[order][1:] != arr[order][:-1]
        keep[order[first]] = True
        last = np.ones(len(order), dtype=bool)
        last[:-1] = arr[order][:-1] != arr[order][1:]
        keep[order[last]] = True
    return np.column_stack([xs[keep], ys[keep]]).astype(float)


def point_on_face_is_covered(hit, box, side, boxes, eps):
    """True if another box sits immediately outside the face at this point.

    Same probe as rect_is_covered, but for a single ray hit rather than a
    finished rectangle -- needed during face SELECTION, not after it.
    """
    _fixed, _span_ax, _span, fix_ax = face_of(box, side)
    sign = -1.0 if side in ("left", "bottom") else 1.0
    probe = np.array(hit, dtype=float)
    probe[fix_ax] += sign * eps
    for ob in boxes:
        if ob is box:
            continue
        if (ob["x_min"] <= probe[0] <= ob["x_max"]
                and ob["y_min"] <= probe[1] <= ob["y_max"]
                and ob["z_min"] <= probe[2] <= ob["z_max"]):
            return True
    return False


def face_by_ray(boundary_px, boxes, cal, triplet, width, height,
                min_cos, margin, cover_eps, min_frac=0.60):
    """Which wall a mask sits on, decided by RAYS not by points.

    The normal route -- majority vote of the segment's own LiDAR returns -- fails
    exactly where it matters most: glass returns nothing, so the better glazed a
    bay is, the fewer points it has to name its own wall with.

    Three things this has to get right, each of which a single centroid ray got
    wrong on session 9:

    * **Use the whole boundary, not one ray.** A face is only a candidate if at
      least `min_frac` of the mask's boundary rays land on it. One ray through
      the centroid says nothing about whether the rest of the mask even reaches
      that face.
    * **Skip faces that will never be a wall.** The nearest hit is often an
      INTERNAL face -- box0's top against box1's bottom, say -- where
      to_openstudio builds no Surface at all because the two spaces are open to
      each other. An opening there cannot be built. The ray has to carry on to
      the exterior wall behind it, which is the one the window is actually in.
    * **Nearest of what survives.** Among faces that pass both tests, the
      closest one is the surface the camera actually sees; anything further is
      behind it.

    Returns (face index, distance along the centroid ray, fraction landed) or
    (-1, inf, 0.0).
    """
    faces = [(b, s) for b in boxes for s in SIDES]
    px = np.asarray(boundary_px, dtype=float)
    origin, dirs = camera_rays(px, cal, triplet, width, height)
    centre = px.mean(axis=0)
    _o, cdirs = camera_rays(np.asarray([centre]), cal, triplet, width, height)
    cd = cdirs[0]

    best, best_t, best_frac = -1, np.inf, 0.0
    for fi, (box, side) in enumerate(faces):
        fixed, span_ax, span, fix_ax = face_of(box, side)

        denom_c = cd[fix_ax]
        if abs(denom_c) < min_cos:
            continue
        t_c = (fixed - origin[fix_ax]) / denom_c
        if t_c <= 0 or t_c >= best_t:
            continue
        hit_c = origin + t_c * cd
        if not (span[0] - margin <= hit_c[span_ax] <= span[1] + margin):
            continue
        if not (box["z_min"] - margin <= hit_c[2] <= box["z_max"] + margin):
            continue
        if point_on_face_is_covered(hit_c, box, side, boxes, cover_eps):
            continue

        denom = dirs[:, fix_ax]
        ok = np.abs(denom) >= min_cos
        tt = np.full(len(dirs), np.nan)
        tt[ok] = (fixed - origin[fix_ax]) / denom[ok]
        ok &= tt > 0
        if not ok.any():
            continue
        hits = origin + tt[ok, None] * dirs[ok]
        landed = ((hits[:, span_ax] >= span[0] - margin)
                  & (hits[:, span_ax] <= span[1] + margin)
                  & (hits[:, 2] >= box["z_min"] - margin)
                  & (hits[:, 2] <= box["z_max"] + margin))
        frac = float(landed.sum()) / len(dirs)
        if frac < min_frac:
            continue

        best, best_t, best_frac = fi, t_c, frac
    return best, best_t, best_frac


def void_stats(rect, box, side, scene_uv, ring_m):
    """Point density inside the rectangle vs in a ring around it, on the face.

    Glazing returns nothing, so a real window is a HOLE in the wall's coverage.
    A hole on its own means nothing though -- it is equally the signature of a
    patch the sensor never swept. The ring is what separates the two: returns
    all around and none inside is glass; nothing anywhere is simply unscanned,
    and this refuses to call that a window.

    Returns (n_inside, n_ring, density_inside, density_ring) in points/m^2.
    """
    u0, u1, v0, v1 = rect
    if scene_uv is None or not len(scene_uv):
        return 0, 0, 0.0, 0.0
    u, v = scene_uv[:, 0], scene_uv[:, 1]
    inside = (u >= u0) & (u <= u1) & (v >= v0) & (v <= v1)
    outer = ((u >= u0 - ring_m) & (u <= u1 + ring_m)
             & (v >= v0 - ring_m) & (v <= v1 + ring_m))
    ring = outer & ~inside

    a_in = max(1e-6, (u1 - u0) * (v1 - v0))
    a_out = max(1e-6, (u1 - u0 + 2*ring_m) * (v1 - v0 + 2*ring_m))
    a_ring = max(1e-6, a_out - a_in)
    n_in, n_ring = int(inside.sum()), int(ring.sum())
    return n_in, n_ring, n_in / a_in, n_ring / a_ring


def face_scene_uv(scene_pts, box, side, slab):
    """Scene points lying within `slab` of this face, in the face's own (u, z)."""
    if scene_pts is None or not len(scene_pts):
        return np.empty((0, 2))
    fixed, span_ax, span, fix_ax = face_of(box, side)
    near = np.abs(scene_pts[:, fix_ax] - fixed) <= slab
    if not near.any():
        return np.empty((0, 2))
    sel = scene_pts[near]
    return np.column_stack([sel[:, span_ax], sel[:, 2]])


def fit_from_masks(pts, cls, prov, boxes, args):
    """One rectangle per (frame, segment): the MASK on the plane its points name.

    The points are used only to answer two questions they answer well -- which
    box face this segment sits on, and how far away it is. The outline comes
    from the mask, at full camera resolution, including the upper half of the
    window no laser ever reached.

    Instance identity comes free with it. Each (frame, segment) is one window in
    one view, so nothing has to be re-separated by clustering afterwards -- which
    is what turned a run of bays into one 15.6 m rectangle before.
    """
    import sys as _sys
    _sys.path.insert(0, str(_ROOT / "SensorFusionLoader"))
    from rig_calibration import load_rig_calibration

    session = Path(args.session_dir)
    opening_dir = (Path(args.opening_map_dir) if args.opening_map_dir
                   else session / "opening_map")
    cal = load_rig_calibration(args.calibration
                               or (_ROOT / "SensorFusionLoader" / "rig_calibration.yaml"))
    manifest = json.loads((session / "sync_manifest.json").read_text(encoding="utf-8"))
    by_stem = {Path(t["flir"]["file"]).stem: t for t in manifest["triplets"]}

    faces, fidx, fdist = assign_faces(pts, boxes, args.max_face_dist, args.span_margin)
    n_orphan = int((fidx < 0).sum())

    # (frame, segment) -> the faces its points voted for, and how many
    groups = defaultdict(Counter)
    seen_frames = set()
    for i in np.nonzero(fidx >= 0)[0]:
        frame, sid = prov[i]
        if not frame:
            sys.exit("--masks needs the `frame` and `segment_id` columns; this csv has "
                     "none. Re-run paint_openings.py --out-csv to produce them.")
        groups[(frame, int(sid), cls[i])][int(fidx[i])] += 1
        seen_frames.add(frame)

    # Window segments with ZERO returns never appear in the points csv at all --
    # it only holds points, and they have none. They are the purest case of what
    # --void-evidence is for: a bay so well glazed that nothing came back. Pull
    # them in from stage 1's own segment lists, with an empty face vote, so the
    # void test downstream gets a chance to place them.
    n_zero_pt = 0
    if args.void_evidence or args.image_only_windows:
        for frame in sorted(seen_frames):
            seg_path = opening_dir / frame / "segments.json"
            if not seg_path.is_file():
                continue
            doc = json.loads(seg_path.read_text(encoding="utf-8"))
            for s in doc["segments"]:
                if s["top_class"] != "window":
                    continue
                key = (frame, int(s["id"]), "window")
                if key not in groups:
                    groups[key] = Counter()
                    n_zero_pt += 1

    scene_pts = None
    if args.void_evidence or args.pitch_from_wall or args.bays_from_wall:
        if not args.scene_points:
            sys.exit("--void-evidence / --pitch-from-wall / --bays-from-wall need "
                     "--scene-points "
                     "(paint_openings.py --out-scene-csv): the wall's own returns are what "
                     "prove a hole is glass, and what carry the bay rhythm.")
        scene_pts = load_scene(Path(args.scene_points))
        print(f"  scene: {len(scene_pts)} non-opening point(s) for the void test")

    openings, rejected = [], []
    n_grazing = n_far = n_noface = 0
    n_void_ok = n_void_no_ring = n_void_too_dense = n_void_noface = 0
    n_img_small = n_img_noface = n_img_ok = 0
    labels_cache = {}
    face_uv_cache = {}
    for (frame, sid, klass), face_votes in sorted(groups.items()):
        top = face_votes.most_common(1)
        fi, n_on_face = top[0] if top else (-1, 0)
        by_void = False
        by_image = args.image_only_windows and klass == "window"
        if by_image:
            # The camera decides. Interior returns are not consulted for the
            # confirmation, the wall, or the size -- see --image-only-windows.
            pass
        elif n_on_face < args.mask_min_points:
            # Too few returns to name a wall the usual way. For a WINDOW that is
            # not a failure, it is the expected signature of glass -- so try the
            # void route instead of dropping it.
            if not (args.void_evidence and klass == "window"):
                n_noface += 1
                continue
            by_void = True

        if frame not in labels_cache:
            labels_cache[frame] = np.load(opening_dir / frame / "labels.npy")
        labels = labels_cache[frame]
        zh, zw = labels.shape
        ys, xs = np.nonzero(labels == sid)
        if not len(ys):
            continue

        if by_image:
            # Confirmed on image size alone, and its wall found by ray. No point
            # count is consulted: a well-glazed bay has the fewest returns of
            # anything on the wall, so requiring them selects against exactly the
            # windows that are most real.
            if len(ys) < args.window_min_mask_px:
                n_img_small += 1
                continue
            fi_ray, _t, _frac = face_by_ray(
                _mask_boundary(ys, xs), boxes, cal, by_stem[frame], zw, zh,
                args.mask_min_cos, args.span_margin, args.cover_eps,
                args.mask_face_min_frac)
            if fi_ray < 0:
                n_img_noface += 1
                continue
            fi = fi_ray
        elif by_void:
            # No usable point vote, so the wall is chosen geometrically -- from
            # the whole mask boundary, skipping faces that will never become a
            # Surface. The void test below is what then has to justify calling
            # the result a window.
            fi_ray, _t, _frac = face_by_ray(
                _mask_boundary(ys, xs), boxes, cal, by_stem[frame], zw, zh,
                args.mask_min_cos, args.span_margin, args.cover_eps,
                args.mask_face_min_frac)
            if fi_ray < 0:
                n_void_noface += 1
                continue
            fi = fi_ray
        box, side = faces[fi]
        fixed, span_ax, span, fix_ax = face_of(box, side)
        px = _mask_boundary(ys, xs)

        world, ng, nf = project_mask_to_plane(
            px, cal, by_stem[frame], zw, zh, fix_ax, fixed,
            args.mask_min_cos, args.mask_max_depth)
        n_grazing += ng
        n_far += nf
        if len(world) < 4:
            continue

        u0, u1 = float(world[:, span_ax].min()), float(world[:, span_ax].max())
        v0, v1 = float(world[:, 2].min()), float(world[:, 2].max())
        u0, u1 = max(u0, span[0]), min(u1, span[1])
        v0, v1 = max(v0, box["z_min"]), min(v1, box["z_max"])
        if u1 <= u0 or v1 <= v0:
            continue

        void = None
        if by_void:
            key = (box["id"], side)
            if key not in face_uv_cache:
                face_uv_cache[key] = face_scene_uv(scene_pts, box, side, args.void_slab)
            n_in, n_ring, d_in, d_ring = void_stats(
                (u0, u1, v0, v1), box, side, face_uv_cache[key], args.void_ring)
            # The ring must itself be populated, or "no points inside" says
            # nothing: an unscanned patch of wall looks identical to glass.
            if n_ring < args.void_min_ring_points:
                n_void_no_ring += 1
                continue
            if d_in > args.void_max_ratio * d_ring:
                n_void_too_dense += 1
                continue
            n_void_ok += 1
            void = {"n_inside": n_in, "n_ring": n_ring,
                    "density_inside": round(d_in, 2), "density_ring": round(d_ring, 2),
                    "ratio": round(d_in / max(1e-6, d_ring), 3)}

        rec = {
            "class": klass,
            "subsurface_type": OS_SUBSURFACE_TYPE.get(klass, "FixedWindow"),
            "box_id": box["id"], "hall": box.get("hall"), "side": side,
            "plane_coord": round(float(fixed), 4),
            "u_min": round(u0, 4), "u_max": round(u1, 4),
            "z_min": round(v0, 4), "z_max": round(v1, 4),
            "width_m": round(u1 - u0, 4), "height_m": round(v1 - v0, 4),
            "n_points": int(n_on_face),
            "evidence": "image" if by_image else "void" if by_void else "points",
            "void": void,
            "from_mask": {"frame": frame, "segment_id": sid,
                          "mask_px": int(len(ys)), "boundary_px": int(len(px)),
                          "projected_px": int(len(world))},
            "point_offset_p50": round(float(np.median(
                fdist[(fidx == fi)])), 4) if (fidx == fi).any() else None,
            "point_offset_max": None,
        }
        clamp_head(rec, box, args.door_head_clearance)
        why = plausibility(rec, box, args)
        if not why and args.exterior_only and rect_is_covered(rec, box, boxes,
                                                              args.cover_eps):
            why = "another box sits against the wall here -- no Surface to cut"
        if why:
            rec["rejected"] = why
            rejected.append(rec)
        else:
            if by_image:
                n_img_ok += 1
            openings.append(rec)

    notes = [f"{len(groups)} (frame, segment) pair(s) -> {len(openings)} kept, "
             f"{len(rejected)} rejected"]
    if n_noface:
        notes.append(f"{n_noface} segment(s) had fewer than --mask-min-points on any "
                     f"face, so no plane could be named for them")
    if args.image_only_windows:
        notes.append(f"image-only windows: {n_img_ok} confirmed from the mask alone "
                     f"(>= {args.window_min_mask_px} px), LiDAR inside them ignored")
        if n_img_small:
            notes.append(f"  {n_img_small} refused: mask below --window-min-mask-px")
        if n_img_noface:
            notes.append(f"  {n_img_noface} refused: no buildable wall under the mask")
    if args.void_evidence:
        notes.append(f"void evidence: {n_void_ok} window(s) rescued as a hole in the "
                     f"wall's coverage ({n_zero_pt} of the candidates had ZERO returns "
                     f"and were not in the points csv at all)")
        if n_void_noface:
            notes.append(f"  {n_void_noface} refused: the centroid ray hit no box face")
        if n_void_no_ring:
            notes.append(f"  {n_void_no_ring} refused: fewer than "
                         f"--void-min-ring-points around the hole, so the wall there was "
                         f"never scanned -- an absent measurement, not a window")
        if n_void_too_dense:
            notes.append(f"  {n_void_too_dense} refused: the inside returns like a wall "
                         f"(density above --void-max-ratio of the ring)")
    if n_grazing:
        notes.append(f"{n_grazing} boundary pixel(s) refused as grazing "
                     f"(|cos| < {args.mask_min_cos})")
    if n_far:
        notes.append(f"{n_far} boundary pixel(s) refused beyond "
                     f"{args.mask_max_depth:g} m")

    openings = merge_views(openings, args.mask_merge_iou)
    openings = merge_touching(openings, args.merge_touch_gap, notes)
    if args.bays_from_wall:
        openings = bays_to_openings(openings, boxes, scene_pts, args, notes)
    openings = merge_corners(openings, rejected, args.merge_corner_dist)
    if args.regularize or args.regularize_fill or args.regularize_extend:
        openings, rnotes = regularize(openings, rejected, boxes, args,
                                      scene_pts if args.pitch_from_wall else None)
        notes += rnotes
    openings.sort(key=lambda r: (r["class"], r["box_id"], r["side"], r["u_min"]))
    for i, r in enumerate(openings):
        r["id"] = i
    return openings, rejected, n_orphan, notes


def merge_views(openings, min_iou):
    """Collapse the several views of one physical opening into one rectangle.

    The same window is projected once per frame that saw it, so a bay observed
    from six positions arrives as six overlapping rectangles on the same face.
    They are merged by 2-D overlap in the face's own (u, z), and each edge is
    taken as the MEDIAN across the views rather than the union: a union grows
    with every extra view and with every bad one, a median does not.
    """
    if min_iou <= 0 or not openings:
        return openings

    def overlap(a, b):
        """Intersection over the SMALLER rectangle, not over the union.

        A view that caught only part of a bay -- the rover passing close, or the
        far half occluded by a pier -- produces a rectangle largely CONTAINED in
        the fuller view's. Containment is near 1 there while IoU is small, so
        scoring by IoU leaves the partial views behind as separate windows,
        which is exactly what a first pass at 0.30 IoU did: three overlapping
        rectangles where there is one bay.
        """
        du = min(a["u_max"], b["u_max"]) - max(a["u_min"], b["u_min"])
        dv = min(a["z_max"], b["z_max"]) - max(a["z_min"], b["z_min"])
        if du <= 0 or dv <= 0:
            return 0.0
        inter = du * dv
        aa = (a["u_max"]-a["u_min"]) * (a["z_max"]-a["z_min"])
        bb = (b["u_max"]-b["u_min"]) * (b["z_max"]-b["z_min"])
        return inter / min(aa, bb)

    groups = defaultdict(list)
    for op in openings:
        groups[(op["box_id"], op["side"], op["class"])].append(op)

    out = []
    for key, ops in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ops = sorted(ops, key=lambda o: -o["n_points"])
        used = [False] * len(ops)
        for i, seed in enumerate(ops):
            if used[i]:
                continue
            cluster = [seed]
            used[i] = True
            for j in range(i + 1, len(ops)):
                if not used[j] and overlap(seed, ops[j]) >= min_iou:
                    cluster.append(ops[j])
                    used[j] = True
            rec = dict(seed)
            for k in ("u_min", "u_max", "z_min", "z_max"):
                rec[k] = round(float(np.median([c[k] for c in cluster])), 4)
            rec["width_m"] = round(rec["u_max"] - rec["u_min"], 4)
            rec["height_m"] = round(rec["z_max"] - rec["z_min"], 4)
            rec["n_points"] = int(sum(c["n_points"] for c in cluster))
            rec["n_views"] = len(cluster)
            rec["from_mask"] = {"views": [c["from_mask"] for c in cluster]}
            # The seed is the view with the most points, so a void-supported view
            # would otherwise vanish into it silently. Record how many of the
            # merged views were placed by the void test: 'void' alone means the
            # opening exists ONLY because of it.
            n_void = sum(1 for c in cluster if c.get("evidence") == "void")
            rec["n_views_void"] = n_void
            rec["evidence"] = ("void" if n_void == len(cluster)
                               else "points+void" if n_void else "points")
            out.append(rec)
    return out


def fit(pts, cls, boxes, args):
    faces, fidx, fdist = assign_faces(pts, boxes, args.max_face_dist, args.span_margin)
    n_orphan = int((fidx < 0).sum())

    groups = defaultdict(list)
    for i in np.nonzero(fidx >= 0)[0]:
        groups[(int(fidx[i]), cls[i])].append(i)

    openings, rejected = [], []
    for (fi, klass), idx in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        box, side = faces[fi]
        fixed, span_ax, span, fix_ax = face_of(box, side)
        idx = np.asarray(idx)
        uv = np.column_stack([pts[idx, span_ax], pts[idx, 2]])
        labels = cluster_on_face(uv, args.cluster_cell)
        for lab in np.unique(labels):
            m = labels == lab
            sub, sub_uv = idx[m], uv[m]
            u0, v0 = sub_uv.min(axis=0)
            u1, v1 = sub_uv.max(axis=0)
            u0, u1 = max(u0, span[0]), min(u1, span[1])
            v0, v1 = max(v0, box["z_min"]), min(v1, box["z_max"])
            rec = {
                "class": klass,
                "subsurface_type": OS_SUBSURFACE_TYPE.get(klass, "FixedWindow"),
                "box_id": box["id"],
                "hall": box.get("hall"),
                "side": side,
                # The wall plane the rectangle lies in -- the box face, NOT the
                # mean of the points: the points sit in the reveal, up to
                # --max-face-dist off it, and the parent Surface is the face.
                "plane_coord": round(float(fixed), 4),
                "u_min": round(float(u0), 4), "u_max": round(float(u1), 4),
                "z_min": round(float(v0), 4), "z_max": round(float(v1), 4),
                "width_m": round(float(u1 - u0), 4),
                "height_m": round(float(v1 - v0), 4),
                "n_points": int(m.sum()),
                # How far off the face the supporting points actually sit. Big
                # numbers mean the rectangle is resting on clutter, not on the
                # wall, and the face assignment should be distrusted.
                "point_offset_p50": round(float(np.median(fdist[sub])), 4),
                "point_offset_max": round(float(fdist[sub].max()), 4),
            }
            clamp_head(rec, box, args.door_head_clearance)
            why = plausibility(rec, box, args)
            if not why and args.exterior_only and rect_is_covered(rec, box, boxes,
                                                                 args.cover_eps):
                why = "another box sits against the wall here -- no Surface to cut"
            if why:
                rec["rejected"] = why
                rejected.append(rec)
            else:
                openings.append(rec)

    openings = merge_corners(openings, rejected, args.merge_corner_dist)
    notes = []
    if args.regularize or args.regularize_fill or args.regularize_extend:
        openings, notes = regularize(openings, rejected, boxes, args)
    openings.sort(key=lambda r: (r["class"], r["box_id"], r["side"], r["u_min"]))
    for i, r in enumerate(openings):
        r["id"] = i
    return openings, rejected, n_orphan, notes


def fit_lattice(centres, min_pitch):
    """(offset, pitch, k per centre, RMS residual) for u = offset + k * pitch.

    Every observed gap, and that gap split into 2, 3 or 4, is tried as the
    pitch; each candidate assigns slot indices by rounding, gets offset and
    pitch least-squared against those indices, and the best RMS wins. Ties go to
    the LARGER pitch, which invents the fewest empty slots.

    Searched rather than taken from the median gap, because a missed bay makes
    the gaps multimodal and no single summary of them is the pitch: three bays
    at 5, 9 and 17 have gaps {4, 8}, whose median is 6 -- a pitch that fits
    none of the three. The true pitch, 4, is only the *smaller* gap here, and
    only the search finds it.

    min_pitch rejects candidates finer than the openings themselves; slots
    closer together than one opening is wide are not slots.
    """
    u = np.sort(np.asarray(centres, dtype=float))
    diffs = np.diff(u)
    if len(diffs) == 0 or diffs.min() <= 1e-6:
        return None
    candidates = sorted({round(float(d) / n, 4)
                         for d in diffs for n in (1, 2, 3, 4)
                         if d / n >= max(min_pitch, 1e-3)}, reverse=True)

    best = None
    for p in candidates:
        k = np.rint((u - u[0]) / p)
        if len(np.unique(k)) < len(k):        # two openings claiming one slot
            continue
        A = np.column_stack([np.ones_like(k), k])
        (offset, pitch), *_ = np.linalg.lstsq(A, u, rcond=None)
        if pitch < min_pitch:
            continue
        rms = float(np.sqrt(((u - (offset + pitch * k)) ** 2).mean()))
        # Strictly better only: candidates are walked largest pitch first, so an
        # equal fit keeps the coarser lattice.
        if best is None or rms < best[3] - 1e-9:
            best = (float(offset), float(pitch), k.astype(int), rms)
    return best


def rolling_pct(a, win, q):
    """Rolling percentile over `win` bins, edge-padded.

    The along-wall profile is not comparable with itself end to end: the rover
    passed close to some of the wall and far from the rest, so the same pier
    returns 60 points at one end of session 9's corridor and 700 at the other.
    A single global threshold therefore cuts the near half of the wall in the
    wrong place. This is the local level everything is measured against instead.
    """
    n = len(a)
    h = max(1, win // 2)
    pad = np.pad(a, h, mode="edge")
    return np.array([np.percentile(pad[i:i + 2 * h + 1], q) for i in range(n)])


def normalise_profile(prof, pitch, bin_m):
    """The profile as a fraction of the LOCAL pier level, so 1 is solid wall.

    The envelope is a rolling 90th percentile two pitches wide -- wide enough to
    always contain a pier, narrow enough to follow the scan's density along the
    corridor. Floored at the 60th percentile of the occupied bins so a stretch of
    wall that is entirely opening cannot normalise its own noise up to 1.
    """
    win = max(3, int(round(2.0 * pitch / bin_m)))
    pier = rolling_pct(prof, win, 90)
    if (prof > 0).any():
        pier = np.maximum(pier, np.percentile(prof[prof > 0], 60))
    return prof / np.maximum(pier, 1e-6)


def binary_open(mask, k):
    """Erode then dilate by `k` bins: runs shorter than k disappear, the rest
    keep their exact extent.

    This is the whole fix for bays that measured a third of their true width. A
    window bay is not empty in plan view -- its mullions, its frame and the
    reveal each return a single dense bin right in the middle of the opening
    (measured on session 9: 90-180 returns in one 0.10 m bin, against a pier
    level of 105). Walking outward from the bay centre until the profile rises
    therefore stopped at the mullion, 0.30 m from the centre, and called that the
    edge of the window. A pier is 1 m of wall; a mullion is one bin. Opening the
    occupancy mask by --bay-min-pier removes the second and leaves the first.
    """
    if k <= 1:
        return mask.copy()
    m = mask.astype(np.int16)
    n = len(m)
    cs = np.concatenate([[0], np.cumsum(m)])
    er = np.zeros(n, dtype=bool)
    for i in range(n - k + 1):
        er[i] = (cs[i + k] - cs[i]) == k
    out = np.zeros(n, dtype=bool)
    for i in np.nonzero(er)[0]:
        out[i:i + k] = True
    return out


def _threshold_crossing(y, ctr, i_in, i_out, thr):
    """Where between bins i_in and i_out the profile crosses `thr`.

    The bin is 0.10 m and an edge quantised to it accumulates half a bin of
    error on each side of every bay; linear interpolation between the last bin
    inside and the first bin outside costs nothing and removes it.
    """
    y0, y1 = y[i_in], y[i_out]
    if y1 == y0:
        return float(ctr[i_in])
    f = float(np.clip((thr - y0) / (y1 - y0), 0.0, 1.0))
    return float(ctr[i_in] + f * (ctr[i_out] - ctr[i_in]))


def trimmed_mean(vals, n_std):
    """(mean of the values within n_std standard deviations, keep mask).

    The one summary a repeated element deserves. A bay seen from a bad angle, or
    half occluded by a pier, measures short -- and it is exactly as much a
    measurement of the bay as the good ones are, so a plain mean is dragged down
    by it and a median throws away the precision of the copies that agree. The
    spread of the copies is itself the estimate of how far a copy may sit from
    the truth: everything inside one of those is the same window measured
    repeatedly, everything outside it is a different measurement of something
    else.

    Fewer than three values has no usable spread, so nothing is trimmed. A zero
    spread (every copy identical) keeps everything, rather than dividing by it.
    """
    a = np.asarray(list(vals), dtype=float)
    keep = np.ones(len(a), dtype=bool)
    if len(a) < 3:
        return (float(a.mean()) if len(a) else None), keep
    sd = float(a.std())
    if sd > 1e-9:
        keep = np.abs(a - a.mean()) <= n_std * sd
        if not keep.any():
            keep = np.ones(len(a), dtype=bool)
    return float(a[keep].mean()), keep


def _autocorr(prof):
    """Normalised autocorrelation of a profile, or None if it carries nothing."""
    f = prof - prof.mean()
    if not np.any(f):
        return None
    ac = np.correlate(f, f, mode="full")[len(f) - 1:]
    if ac[0] <= 0:
        return None
    return ac / ac[0]


def _ac_peak(ac, args):
    """(lag in metres, correlation) of the best peak in the allowed lag band.

    Sub-bin by a parabola through the three samples at the maximum. The bin is
    0.10 m and the peak is broad, so the integer lag alone would quantise the
    pitch badly enough to accumulate visible drift along a long wall: 3.20
    against a true 3.28 slips 0.5 m over six bays.
    """
    lo = int(round(args.pitch_min / args.pitch_bin))
    hi = min(len(ac) - 1, int(round(args.pitch_max / args.pitch_bin)))
    if hi <= lo + 1:
        return None, 0.0
    k = int(np.argmax(ac[lo:hi])) + lo
    shift = 0.0
    if 0 < k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            shift = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return (k + shift) * args.pitch_bin, float(ac[k])


def face_bands(pts, span_ax, edges, box, args):
    """The wall sliced into overlapping height bands, each histogrammed along it.

    One plan view per height, which is what the whole vertical story is read
    from: the pitch (stacked over every band that has enough returns), and then
    per bay how far up and down the opening actually goes. Bands with fewer than
    --bay-z-min-points are dropped rather than trusted thin.

    Returns a list of dicts with the band's z range, its raw profile and its
    point count, ordered from the floor up.
    """
    out = []
    z = box["z_min"]
    while z + args.bay_z_band <= box["z_max"] + 1e-9:
        m = (pts[:, 2] >= z) & (pts[:, 2] < z + args.bay_z_band)
        n = int(m.sum())
        if n >= args.bay_z_min_points:
            out.append({"z_lo": float(z), "z_hi": float(z + args.bay_z_band),
                        "z_mid": float(z + args.bay_z_band / 2.0), "n": n,
                        "prof": np.histogram(pts[m, span_ax],
                                             bins=edges)[0].astype(float)})
        else:
            out.append({"z_lo": float(z), "z_hi": float(z + args.bay_z_band),
                        "z_mid": float(z + args.bay_z_band / 2.0), "n": n,
                        "prof": None})
        z += args.bay_z_step
    return out


def pitch_from_wall(scene_pts, box, side, args):
    """The bay pitch, measured from the WALL's own returns in plan view.

    Fitting a pitch to the openings' centres fits whatever drift the merge left
    in them -- on session 9 that gave 3.35 m at RMS 0.41 m on one wall (refused)
    and a degenerate 1.40 m on the other. The wall itself is the better witness:
    glass returns nothing and pier returns strongly, so a horizontal band of
    wall, histogrammed along its length, is a square-ish wave whose period IS
    the bay pitch -- read off thousands of dense returns instead of a handful of
    rectangle centres.

    Autocorrelation of that profile, peak over lags in [--pitch-min,
    --pitch-max].

    Read at EVERY height, not one band. The rhythm is a property of the
    structure, so it should show up in every plan view that cuts the wall, and
    on session 9 it does: 3.20 m in 19 of 21 bands from the floor to 1.87 m, on
    both walls. Averaging their autocorrelations before picking the peak is what
    makes that redundancy pay -- a single band can be wrong (north wall,
    z 0.07-0.27, reads 1.60 m off the skirting and the radiator feet), and one
    bad band out of twenty cannot move the stack. The stacked peak lands at
    3.180 / 3.175 m against 3.184 / 3.168 from the single band, i.e. the same
    answer, but now it does not depend on --pitch-z having been chosen well.
    --no-pitch-bands restores the single --pitch-z band.

    Returns (pitch, correlation) or (None, r).
    """
    if scene_pts is None or not len(scene_pts):
        return None, 0.0
    fixed, span_ax, span, fix_ax = face_of(box, side)
    on_face = ((np.abs(scene_pts[:, fix_ax] - fixed) <= args.void_slab)
               & (scene_pts[:, span_ax] >= span[0])
               & (scene_pts[:, span_ax] <= span[1]))
    if int(on_face.sum()) < args.pitch_min_points:
        return None, 0.0
    pts = scene_pts[on_face]

    edges = np.arange(span[0], span[1] + args.pitch_bin, args.pitch_bin)
    if len(edges) < 8:
        return None, 0.0

    acs = []
    if args.pitch_bands:
        for b in face_bands(pts, span_ax, edges, box, args):
            if b["prof"] is None:
                continue
            ac = _autocorr(b["prof"])
            if ac is not None:
                acs.append(ac)
    if len(acs) < args.pitch_min_bands:
        # Not enough bands to stack -- fall back to the single band, which is
        # what --pitch-z is for.
        band = ((pts[:, 2] >= args.pitch_z[0]) & (pts[:, 2] <= args.pitch_z[1]))
        if int(band.sum()) < args.pitch_min_points:
            return None, 0.0
        ac = _autocorr(np.histogram(pts[band, span_ax], bins=edges)[0].astype(float))
        if ac is None:
            return None, 0.0
        acs = [ac]

    # Unweighted, deliberately. Weighting by point count hands the answer to the
    # floor bands, which hold half the returns of the whole wall and carry the
    # least rhythm -- the skirting runs straight past the openings.
    stack = np.mean(np.asarray(acs), axis=0)
    pitch, r = _ac_peak(stack, args)
    if pitch is None:
        return None, 0.0
    if r < args.pitch_min_corr:
        return None, r
    return pitch, r


def _runs_of(flags):
    """Contiguous True runs of a boolean array, as (start, end) index pairs."""
    runs, start = [], None
    for i, g in enumerate(flags):
        if g and start is None:
            start = i
        elif not g and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def slot_z_extent(bands, ctr, u0, u1, pitch, args):
    """How far up and down ONE bay is open, from the stack of plan views.

    Per height band, the bay's own occupancy is compared against the two piers
    flanking it in that same band. Both are read off the band's LOCALLY
    normalised profile, so a band the scan hit thinly is not penalised for being
    thin -- only for being as solid inside the bay as it is beside it.

    This is measured per bay rather than once for the whole wall because that is
    the question being asked of it: a wall of glazing and a door in the same run
    have completely different vertical extents, and averaging them over the wall
    produced one sill for both. A door's opening reaches the floor and its runs
    say so; a window's stops at the sill.

    Returns (sill, head, [(z_lo, z_hi, contrast_peak), ...] for every run).
    """
    w = u1 - u0
    inside = (ctr >= u0) & (ctr <= u1)
    pier = ((np.abs(ctr - (0.5 * (u0 + u1) - pitch / 2.0)) <= (pitch - w) / 2.0)
            | (np.abs(ctr - (0.5 * (u0 + u1) + pitch / 2.0)) <= (pitch - w) / 2.0))
    if not inside.any() or not pier.any():
        return None, None, []

    zs, cs = [], []
    for b in bands:
        zs.append(b["z_mid"])
        if b["prof"] is None:
            cs.append(np.nan)
            continue
        n = normalise_profile(b["prof"], pitch, args.pitch_bin)
        a, c = float(n[pier].mean()), float(n[inside].mean())
        cs.append((a - c) / (a + c) if (a + c) > 0 else 0.0)
    zs, cs = np.asarray(zs), np.asarray(cs)
    good = np.nan_to_num(cs, nan=-1.0) >= args.bay_min_contrast
    if not good.any():
        return None, None, []
    half = args.bay_z_band / 2.0
    runs = [(float(zs[a] - half), float(zs[b] + half),
             round(float(np.nanmax(cs[a:b + 1])), 3)) for a, b in _runs_of(good)]
    a, b = max(_runs_of(good), key=lambda ab: ab[1] - ab[0])
    return float(zs[a] - half), float(zs[b] + half), runs


def wall_bays(scene_pts, box, side, args):
    """The bay grid of one wall, read entirely from that wall's own returns.

    Everything an opening needs except its class:

    * **pitch** -- autocorrelation of the along-wall profile, stacked over every
      height band (see pitch_from_wall);
    * **phase** -- the slot offset that puts the slots on the TROUGHS, found by
      minimising occupancy at the slot centres. The wall says where its own
      gaps are. It only has to be good enough to land inside the right bay:
      each bay's position is then MEASURED, see below;
    * **width** -- each bay walked out from its slot to where the locally
      normalised profile crosses --bay-trough-frac, on an occupancy mask that
      has been opened by --bay-min-pier first so that a mullion inside the bay
      is not mistaken for its edge. The reported width is the MEDIAN across
      bays, because the bays are one repeated element and their individual
      edges are noisier than their common width;
    * **position** -- each bay's own measured trough centre, not the lattice
      slot. Session 9's south wall is 0.33 m out of phase before x=21 m and in
      phase after it -- the corridor is not one continuous build -- so a single
      offset put five of nine bays a third of a metre off the opening they
      describe. --bay-snap-phase forces the strict lattice back on;
    * **sill and head** -- per bay, from the stack of plan views. See
      slot_z_extent.

    Returns a dict, or None when the wall has no rhythm to read.
    """
    fixed, span_ax, span, fix_ax = face_of(box, side)
    near = np.abs(scene_pts[:, fix_ax] - fixed) <= args.void_slab
    on_face = (near
               & (scene_pts[:, span_ax] >= span[0])
               & (scene_pts[:, span_ax] <= span[1]))
    if int(on_face.sum()) < args.pitch_min_points:
        return None
    pts = scene_pts[on_face]

    pitch, r = pitch_from_wall(scene_pts, box, side, args)
    if not pitch:
        return None

    edges = np.arange(span[0], span[1] + args.pitch_bin, args.pitch_bin)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    band = (pts[:, 2] >= args.pitch_z[0]) & (pts[:, 2] <= args.pitch_z[1])
    prof = np.histogram(pts[band, span_ax], bins=edges)[0].astype(float)
    if not prof.any():
        return None
    norm = normalise_profile(prof, pitch, args.pitch_bin)

    # phase: slots on the troughs
    n_slot = int((span[1] - span[0]) / pitch) + 1
    best_ph, best_v = 0.0, np.inf
    for ph in np.linspace(0.0, pitch, 60, endpoint=False):
        xs = span[0] + ph + pitch * np.arange(n_slot)
        xs = xs[(xs >= span[0]) & (xs <= span[1])]
        if not len(xs):
            continue
        v = float(np.interp(xs, ctr, norm).mean())
        if v < best_v:
            best_ph, best_v = ph, v
    slots = span[0] + best_ph + pitch * np.arange(n_slot)
    slots = slots[(slots >= span[0]) & (slots <= span[1])]
    if len(slots) < 2:
        return None

    # Occupancy, with anything narrower than a pier erased -- see binary_open.
    occ = binary_open(norm > args.bay_trough_frac,
                      max(1, int(round(args.bay_min_pier / args.pitch_bin))))

    found = []
    for k, s in enumerate(slots):
        i = int(np.clip(np.searchsorted(ctr, s), 0, len(ctr) - 1))
        if occ[i]:
            continue                     # slot centre is wall, not an opening
        lo = i
        while lo > 0 and not occ[lo - 1]:
            lo -= 1
        hi = i
        while hi < len(ctr) - 1 and not occ[hi + 1]:
            hi += 1
        u0 = (_threshold_crossing(norm, ctr, lo, lo - 1, args.bay_trough_frac)
              if lo > 0 else float(ctr[lo]))
        u1 = (_threshold_crossing(norm, ctr, hi, hi + 1, args.bay_trough_frac)
              if hi < len(ctr) - 1 else float(ctr[hi]))
        w = u1 - u0
        if not (0 < w < pitch):
            continue                     # a gap wider than the pitch is not a bay
        found.append({"slot": int(k), "u_slot": float(s),
                      "u_centre": 0.5 * (u0 + u1),
                      "measured": (u0, u1), "measured_width": w})
    if len(found) < 2:
        return None
    # One width for the whole wall, from the copies that agree with each other.
    bay_w, w_keep = trimmed_mean((f["measured_width"] for f in found),
                                 args.bay_size_nstd)
    for f, k in zip(found, w_keep):
        f["width_outlier"] = not bool(k)

    bands = face_bands(pts, span_ax, edges, box, args)
    bays = []
    for f in found:
        c = f["u_slot"] if args.bay_snap_phase else f["u_centre"]
        u0, u1 = c - bay_w / 2.0, c + bay_w / 2.0
        sill, head, runs = slot_z_extent(bands, ctr, u0, u1, pitch, args)
        bays.append({**f, "u_min": u0, "u_max": u1, "width": bay_w,
                     "sill": sill, "head": head, "z_runs": runs,
                     "reaches_floor": bool(
                         sill is not None
                         and sill - box["z_min"] <= args.bay_floor_tol)})

    have = [b for b in bays if b["sill"] is not None]
    if not have:
        return None
    # And one sill and one head, the same way. A bay that runs to the floor is a
    # doorway, not a short window, and it falls outside the spread of the rest
    # rather than dragging the sill down with it.
    sill, s_keep = trimmed_mean((b["sill"] for b in have), args.bay_size_nstd)
    head, h_keep = trimmed_mean((b["head"] for b in have), args.bay_size_nstd)
    for b, ks, kh in zip(have, s_keep, h_keep):
        b["sill_outlier"] = not bool(ks)
        b["head_outlier"] = not bool(kh)
    for b in bays:                       # a bay the vertical scan lost keeps the wall's
        if b["sill"] is None:
            b["sill"], b["head"] = sill, head
            b["sill_outlier"] = b["head_outlier"] = False

    # The whole-wall contrast curve, kept for the diagnostics that plot it.
    zs = [b["z_mid"] for b in bands]
    inside = np.zeros(len(ctr), dtype=bool)
    pier = np.zeros(len(ctr), dtype=bool)
    for b in bays:
        inside |= (ctr >= b["u_min"]) & (ctr <= b["u_max"])
    for s in slots[:-1]:
        pier |= np.abs(ctr - (s + pitch / 2.0)) <= (pitch - bay_w) / 2.0
    contrasts = []
    for bd in bands:
        if bd["prof"] is None or not pier.any() or not inside.any():
            contrasts.append(None)
            continue
        n = normalise_profile(bd["prof"], pitch, args.pitch_bin)
        a, c = float(n[pier].mean()), float(n[inside].mean())
        contrasts.append(round((a - c) / (a + c), 3) if (a + c) > 0 else 0.0)

    return {"pitch": pitch, "corr": r, "slots": slots, "width": bay_w,
            "sill": sill, "head": head, "bays": bays,
            "contrast": {"z": zs, "r": contrasts}}


def bays_to_openings(openings, boxes, scene_pts, args, notes):
    """Give each opening the wall found its geometry, and the camera its class.

    The LiDAR is asked only what it answers well -- WHERE the wall has a hole
    and how big it is, from the piers either side of it, which are dense and
    opaque. The camera is asked only what it answers well -- WHAT is in that
    hole. Neither the returns inside an opening nor a mask's own outline set the
    rectangle: the mask only has to overlap a bay.

    Class comes from whichever detections overlap the bay, weighed by how much
    of it they cover, so a `door` detection in a bay makes it a door and a
    `windowpane` detection makes it a window. With --bay-doors off, only windows
    can claim a bay and doors are passed through as measured, which is the older
    behaviour.

    Height comes from the bay's own vertical scan (slot_z_extent), so the two
    classes no longer share one sill. A door additionally has its sill taken to
    the floor: a door reaches the floor by definition, and the vertical scan
    cannot see the bottom of the opening past whatever is standing in it.
    """
    by_box = {b["id"]: b for b in boxes}
    faces = defaultdict(list)
    for op in openings:
        faces[(op["box_id"], op["side"])].append(op)

    out = []
    for (box_id, side), ops in sorted(faces.items(), key=lambda kv: str(kv[0])):
        claimable = [o for o in ops
                     if o["class"] == "window" or (args.bay_doors and o["class"] == "door")]
        passthrough = [o for o in ops if o not in claimable]
        if not claimable:
            out.extend(ops)
            continue
        box = by_box[box_id]
        bays = wall_bays(scene_pts, box, side, args)
        if bays is None:
            notes.append(f"box {box_id} {side}: no bay rhythm readable from the wall, "
                         f"openings left as measured")
            out.extend(ops)
            continue

        _fixed, span_ax, span, _fix_ax = face_of(box, side)

        # --- pass 1: what is in each bay, and what would each one measure? ---
        found, claimed, tally = [], set(), Counter()
        for bay in bays["bays"]:
            u0, u1 = bay["u_min"], bay["u_max"]
            if u0 < span[0] or u1 > span[1]:
                continue
            seen = [o for o in claimable
                    if min(o["u_max"], u1) - max(o["u_min"], u0) > args.bay_confirm_overlap]
            if not seen:
                continue
            # Which class this bay holds: the one whose detections cover the most
            # of it. Overlap length, not detection count -- one mask across the
            # whole bay says more than three slivers of another class.
            votes = Counter()
            for o in seen:
                votes[o["class"]] += min(o["u_max"], u1) - max(o["u_min"], u0)
            klass = max(votes.items(), key=lambda kv: (kv[1], kv[0] == "door"))[0]
            same = [o for o in seen if o["class"] == klass]
            for o in seen:
                claimed.add(id(o))
            tally[klass] += 1

            z0 = bay["sill"] if bay["sill"] is not None else bays["sill"]
            z1 = bay["head"] if bay["head"] is not None else bays["head"]
            if args.bay_height_from_mask:
                # The masks reach the floor; the contrast scan stops at the
                # radiator. Take the lower sill from them, and the higher head
                # from whichever source saw further up.
                z0 = min(o["z_min"] for o in same)
                z1 = max(z1, max(o["z_max"] for o in same))
            found.append({"bay": bay, "class": klass, "same": same, "votes": votes,
                          "base": max(same, key=lambda o: o["n_points"]),
                          "z0": z0, "z1": z1})

        # --- pass 2: one size for the repeated element ---
        # A bay is one window built many times, so its copies are many
        # measurements of ONE size, not many sizes. Width already comes out of
        # the wall as a single number; the sill and head still vary per bay,
        # because each bay's vertical scan is stopped somewhere different by a
        # radiator, a curtain or the rover's own line of sight. Averaging the
        # copies that agree -- everything within --bay-size-nstd of the mean --
        # gives the height the same standing as the width, without letting one
        # bay measured half-height decide it for the rest.
        wins = [f for f in found if f["class"] == "window"]
        uniform = None
        if args.bay_uniform_size and len(wins) >= 2:
            z0u, k0 = trimmed_mean((f["z0"] for f in wins), args.bay_size_nstd)
            z1u, k1 = trimmed_mean((f["z1"] for f in wins), args.bay_size_nstd)
            uniform = (z0u, z1u)
            # Sill and head are trimmed independently -- a bay can be typical at
            # one edge and an outlier at the other -- so they are counted
            # separately. An intersection would report a bay as excluded when
            # only one of its two edges was.
            notes.append(
                f"box {box_id} {side}: {len(wins)} window bay(s) share one size "
                f"{bays['width']:.2f} x {z1u - z0u:.2f} m at z {z0u:.2f}..{z1u:.2f} "
                f"(mean over the bays within {args.bay_size_nstd:g} sd; "
                f"sill from {int(k0.sum())}/{len(wins)}, head from "
                f"{int(k1.sum())}/{len(wins)})")
            for f, a, b in zip(wins, k0, k1):
                if a and b:
                    continue
                which = ("sill and head" if not (a or b)
                         else "sill" if not a else "head")
                notes.append(
                    f"    slot {f['bay']['slot']:>2} z {f['z0']:.2f}..{f['z1']:.2f} -- "
                    f"{which} outside the spread, not averaged in")

        kept = []
        for f in found:
            bay, klass, same = f["bay"], f["class"], f["same"]
            u0, u1 = bay["u_min"], bay["u_max"]
            z0, z1 = f["z0"], f["z1"]
            if uniform is not None and klass == "window":
                z0, z1 = uniform
            if klass == "door" and args.bay_door_to_floor:
                z0 = box["z_min"]
            z0 = max(z0, box["z_min"])
            z1 = min(z1, box["z_max"])
            if z1 <= z0:
                continue

            # A door is the width of a door, not the width of the bay it stands
            # in: session 9's bays are 2.3 m and its doors are under 1.1 m. Only
            # the bay's presence and its head are taken from the wall.
            if klass == "door" and args.bay_door_width == "mask":
                du0 = min(o["u_min"] for o in same)
                du1 = max(o["u_max"] for o in same)
            else:
                du0, du1 = u0, u1

            rec = dict(f["base"])
            rec.update(**{"class": klass,
                          "subsurface_type": OS_SUBSURFACE_TYPE.get(klass, "FixedWindow")})
            rec.update(u_min=round(du0, 4), u_max=round(du1, 4),
                       z_min=round(z0, 4), z_max=round(z1, 4),
                       width_m=round(du1 - du0, 4), height_m=round(z1 - z0, 4),
                       evidence="wall_bay",
                       bay={"slot": bay["slot"], "pitch_m": round(bays["pitch"], 4),
                            "corr": round(bays["corr"], 3),
                            "u_min": round(u0, 4), "u_max": round(u1, 4),
                            "bay_width_m": round(bay["width"], 4),
                            "measured_width_m": round(bay["measured_width"], 4),
                            "width_outlier": bay.get("width_outlier", False),
                            "phase_offset_m": round(bay["u_centre"] - bay["u_slot"], 4),
                            "sill_from_wall": (None if bay["sill"] is None
                                               else round(bay["sill"], 4)),
                            "head_from_wall": (None if bay["head"] is None
                                               else round(bay["head"], 4)),
                            "z_before_uniform": [round(f["z0"], 4), round(f["z1"], 4)],
                            "uniform_size": bool(uniform is not None
                                                 and klass == "window"),
                            "reaches_floor": bay["reaches_floor"],
                            "z_runs": bay["z_runs"],
                            "class_votes": {k: round(v, 3) for k, v in f["votes"].items()},
                            "confirmed_by": len(same)})
            rec.pop("regularized", None)
            clamp_head(rec, box, args.door_head_clearance)
            kept.append(rec)

        # A door the bay scan never claimed is still a door: it may sit in a
        # stretch of wall that has no rhythm at all, which is not evidence
        # against it. An unclaimed window is dropped -- the wall is the witness
        # for glazing, and it did not see one there.
        orphan_doors = [o for o in claimable
                        if o["class"] == "door" and id(o) not in claimed]
        n_drop = sum(1 for o in claimable
                     if o["class"] != "door" and id(o) not in claimed)
        notes.append(
            f"box {box_id} {side}: {len(bays['bays'])} bay(s) on a {bays['pitch']:.2f} m "
            f"pitch (r={bays['corr']:.2f}), {bays['width']:.2f} m wide, z "
            f"{bays['sill']:.2f}..{bays['head']:.2f} m -> "
            + (", ".join(f"{n} {k}" for k, n in sorted(tally.items())) or "nothing")
            + f" confirmed by the camera, {len(claimable)} measured opening(s) replaced"
            + (f", {n_drop} unconfirmed window(s) dropped" if n_drop else "")
            + (f", {len(orphan_doors)} door(s) kept off the rhythm" if orphan_doors else ""))
        for b in bays["bays"]:
            notes.append(
                f"    slot {b['slot']:>2} u {b['u_min']:6.2f}..{b['u_max']:6.2f} "
                f"(measured {b['measured_width']:.2f} m, phase "
                f"{b['u_centre'] - b['u_slot']:+.2f} m)  z "
                + ("--" if b["sill"] is None else f"{b['sill']:.2f}..{b['head']:.2f}")
                + ("  reaches the floor" if b["reaches_floor"] else ""))
        out.extend(kept)
        out.extend(orphan_doors)
        out.extend(passthrough)
    return out


def phase_for_pitch(centres, pitch):
    """(offset, slot per centre, RMS) for a pitch that is already known.

    Only the phase is free. Slots come from rounding each centre onto the given
    pitch; the offset is then the mean residual, which is the least-squares
    answer when the spacing is fixed.
    """
    u = np.sort(np.asarray(centres, dtype=float))
    k = np.rint((u - u[0]) / pitch)
    offset = float(np.mean(u - pitch * k))
    resid = u - (offset + pitch * k)
    return offset, k.astype(int), float(np.sqrt((resid ** 2).mean()))


def regularize(openings, rejected, boxes, args, scene_pts=None):
    """Snap each face's repeated openings onto one size and one pitch.

    Only where the evidence says they ARE repeated: at least
    --regularize-min-count of them, widths spread by less than
    --regularize-size-tol, and a lattice residual under --regularize-tol. A face
    that fails any of those is left exactly as measured -- the point is to
    recover a bay the scan saw unevenly, not to impose regularity on a wall that
    has none.
    """
    by_box = {b["id"]: b for b in boxes}
    groups = defaultdict(list)
    for op in openings:
        groups[(op["box_id"], op["side"], op["class"])].append(op)

    out, notes = [], []
    for key, ops in sorted(groups.items(), key=lambda kv: str(kv[0])):
        box_id, side, klass = key
        box = by_box[box_id]
        if len(ops) < args.regularize_min_count:
            out.extend(ops)
            continue

        widths = np.array([o["width_m"] for o in ops])
        spread = float(np.percentile(widths, 95) - np.percentile(widths, 5))
        if spread > args.regularize_size_tol:
            notes.append(f"box {box_id} {side} {klass}: widths spread {spread:.2f} m "
                         f"> --regularize-size-tol, left as measured")
            out.extend(ops)
            continue

        ops = sorted(ops, key=lambda o: 0.5 * (o["u_min"] + o["u_max"]))
        centres = [0.5 * (o["u_min"] + o["u_max"]) for o in ops]
        # The pitch has to leave PIER between the openings. A pitch equal to the
        # opening width means they touch edge to edge, which is one continuous
        # run of glazing -- a shape, not a rhythm -- and calling it a lattice
        # would impose a repeat that the wall does not have.
        w_med = float(np.median(widths))
        from_wall = False
        # Prefer a pitch measured off the wall to one fitted to these centres:
        # the centres carry the merge's drift, the wall does not.
        wall_pitch, wall_r = pitch_from_wall(scene_pts, box, side, args)
        if wall_pitch and wall_pitch >= w_med * args.regularize_min_pier:
            offset, ks, rms = phase_for_pitch(centres, wall_pitch)
            lat = (offset, wall_pitch, ks, rms)
            from_wall = True
            notes.append(f"box {box_id} {side} {klass}: pitch {wall_pitch:.2f} m measured "
                         f"from the wall's own returns (autocorrelation r={wall_r:.2f}); "
                         f"only the phase was fitted")
        else:
            if wall_pitch:
                notes.append(f"box {box_id} {side} {klass}: wall pitch {wall_pitch:.2f} m "
                             f"leaves no pier against a {w_med:.2f} m opening, falling back "
                             f"to fitting the centres")
            lat = fit_lattice(centres, min_pitch=w_med * args.regularize_min_pier)
        if lat is None:
            out.extend(ops)
            continue
        offset, pitch, ks, rms = lat
        if from_wall:
            # The pitch was measured independently, off the wall. The residual
            # here is how far the OPENINGS have drifted from it -- the error
            # being corrected -- so vetoing on it would throw away the fix. What
            # must hold instead is that every opening lands unambiguously in one
            # slot: more than half a pitch out and it is being snapped to the
            # wrong bay, which is worse than leaving it alone.
            if len(set(ks.tolist())) < len(ks):
                # Two openings landing in one slot means the wall's rhythm is
                # finer than, or simply not, the spacing these openings have --
                # snapping would stack them on top of each other and silently
                # delete one.
                notes.append(f"box {box_id} {side} {klass}: {len(ks) - len(set(ks.tolist()))} "
                             f"opening(s) collide in one slot of the {pitch:.2f} m wall "
                             f"lattice -- they are not on that rhythm, left as measured")
                out.extend(ops)
                continue
            worst = float(np.max(np.abs(
                np.sort(np.asarray(centres, dtype=float)) - (offset + pitch * ks))))
            if worst > pitch / 2.0:
                notes.append(f"box {box_id} {side} {klass}: an opening sits {worst:.2f} m "
                             f"off the {pitch:.2f} m wall lattice, over half a pitch -- "
                             f"slot assignment is ambiguous, left as measured")
                out.extend(ops)
                continue
            notes.append(f"box {box_id} {side} {klass}: snapped to the wall lattice "
                         f"(RMS {rms:.2f} m, worst {worst:.2f} m of a {pitch:.2f} m pitch)")
        elif rms > args.regularize_tol:
            notes.append(f"box {box_id} {side} {klass}: lattice residual {rms:.2f} m "
                         f"> --regularize-tol, left as measured")
            out.extend(ops)
            continue

        w = float(np.median(widths))
        z0 = float(np.median([o["z_min"] for o in ops]))
        z1 = float(np.median([o["z_max"] for o in ops]))
        notes.append(f"box {box_id} {side} {klass}: {len(ops)} opening(s) on a "
                     f"{pitch:.2f} m pitch, RMS {rms:.2f} m -> {w:.2f} x {z1 - z0:.2f} m each")

        for op, k in zip(ops, ks):
            u = offset + pitch * k
            op["regularized"] = {
                "slot": int(k), "pitch_m": round(pitch, 4), "lattice_rms_m": round(rms, 4),
                "was": {kk: op[kk] for kk in ("u_min", "u_max", "z_min", "z_max")},
            }
            op.update(u_min=round(u - w / 2, 4), u_max=round(u + w / 2, 4),
                      z_min=round(z0, 4), z_max=round(z1, 4),
                      width_m=round(w, 4), height_m=round(z1 - z0, 4))
            # The median head can sit back on the ceiling even when every input
            # was clamped off it, so re-clamp whatever the lattice produced.
            clamp_head(op, box, args.door_head_clearance)
            out.append(op)

        if not (args.regularize_fill or args.regularize_extend):
            continue
        _fixed, _span_ax, span, _fix_ax = face_of(box, side)
        taken = set(int(k) for k in ks)
        k_lo, k_hi = int(ks.min()), int(ks.max())
        if args.regularize_extend:
            # Out to where a whole opening still fits on this face, no further.
            k_lo = int(np.ceil((span[0] + w / 2 - offset) / pitch))
            k_hi = int(np.floor((span[1] - w / 2 - offset) / pitch))
            k_lo, k_hi = min(k_lo, int(ks.min())), max(k_hi, int(ks.max()))
        n_new = 0
        for k in range(k_lo, k_hi + 1):
            if k in taken:
                continue
            u = offset + pitch * k
            rec = dict(ops[0])
            rec.pop("regularized", None)
            rec.update(u_min=round(u - w / 2, 4), u_max=round(u + w / 2, 4),
                       z_min=round(z0, 4), z_max=round(z1, 4),
                       width_m=round(w, 4), height_m=round(z1 - z0, 4),
                       n_points=0, synthetic=True,
                       point_offset_p50=None, point_offset_max=None,
                       regularized={"slot": int(k), "pitch_m": round(pitch, 4),
                                    "lattice_rms_m": round(rms, 4), "was": None})
            clamp_head(rec, box, args.door_head_clearance)
            if rec["u_min"] < span[0] or rec["u_max"] > span[1]:
                rec["rejected"] = "synthetic slot falls off the end of the wall"
                rejected.append(rec)
                continue
            if args.exterior_only and rect_is_covered(rec, box, boxes, args.cover_eps):
                rec["rejected"] = "synthetic slot sits where another box covers the wall"
                rejected.append(rec)
                continue
            out.append(rec)
            n_new += 1
        if n_new:
            notes.append(f"box {box_id} {side} {klass}: filled {n_new} empty slot(s) "
                         f"-- INFERRED, not measured")
    return out, notes


def merge_touching(openings, gap, notes):
    """Fuse openings that touch on the same face into one rectangle.

    A bay split across several views, or a run of glazing the segmentation cut
    at a mullion, arrives as neighbouring rectangles with a hairline between
    them. Physically it is one opening and the model should carry one
    SubSurface.

    The union is the axis-aligned BOUNDING BOX of the group, so the result is a
    rectangle by construction -- never an L. That is also why the adjacency test
    is strict: two rectangles merge only when they are side by side (a gap in u
    under `gap`, while their z ranges genuinely overlap) or stacked (the mirror
    case). Diagonal neighbours are refused, because their bounding box would
    swallow the empty corner between them and invent glazing that is not there.

    Transitive and repeated to a fixed point: A-B and B-C means A-B-C.
    """
    if gap <= 0 or len(openings) < 2:
        return openings

    def adjacent(a, b):
        u_lo = max(a["u_min"], b["u_min"]); u_hi = min(a["u_max"], b["u_max"])
        z_lo = max(a["z_min"], b["z_min"]); z_hi = min(a["z_max"], b["z_max"])
        u_gap = max(0.0, max(a["u_min"], b["u_min"]) - min(a["u_max"], b["u_max"]))
        z_gap = max(0.0, max(a["z_min"], b["z_min"]) - min(a["z_max"], b["z_max"]))
        side_by_side = u_gap <= gap and (z_hi - z_lo) > 0
        stacked = z_gap <= gap and (u_hi - u_lo) > 0
        return side_by_side or stacked

    groups = defaultdict(list)
    for op in openings:
        groups[(op["box_id"], op["side"], op["class"])].append(op)

    out, n_merged = [], 0
    for key, ops in sorted(groups.items(), key=lambda kv: str(kv[0])):
        changed = True
        while changed:
            changed = False
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    if not adjacent(ops[i], ops[j]):
                        continue
                    a, b = ops[i], ops[j]
                    keep = a if a["n_points"] >= b["n_points"] else b
                    rec = dict(keep)
                    rec.update(
                        u_min=round(min(a["u_min"], b["u_min"]), 4),
                        u_max=round(max(a["u_max"], b["u_max"]), 4),
                        z_min=round(min(a["z_min"], b["z_min"]), 4),
                        z_max=round(max(a["z_max"], b["z_max"]), 4))
                    rec["width_m"] = round(rec["u_max"] - rec["u_min"], 4)
                    rec["height_m"] = round(rec["z_max"] - rec["z_min"], 4)
                    rec["n_points"] = a["n_points"] + b["n_points"]
                    rec["n_views"] = a.get("n_views", 1) + b.get("n_views", 1)
                    rec["merged_touching"] = (a.get("merged_touching", 1)
                                              + b.get("merged_touching", 1))
                    ops = [o for k, o in enumerate(ops) if k not in (i, j)] + [rec]
                    n_merged += 1
                    changed = True
                    break
                if changed:
                    break
        out.extend(ops)
    if n_merged:
        notes.append(f"touching merge: {n_merged} pair(s) fused within {gap:g} m "
                     f"-> {len(openings)} openings became {len(out)}")
    return out


def merge_corners(openings, rejected, dist):
    """Collapse the same physical opening fitted onto several faces at once.

    Kept by point count, not by how flat the fit is: at a corner every candidate
    face has points close to it, so off-plane distance does not separate them,
    while the face the opening is really in is the one most of the mask's
    returns landed on.
    """
    if dist <= 0 or len(openings) < 2:
        return openings
    aabbs = [rect_aabb(r) for r in openings]
    order = sorted(range(len(openings)), key=lambda i: -openings[i]["n_points"])
    keep, dropped = [], set()
    for i in order:
        if i in dropped:
            continue
        keep.append(i)
        for j in order:
            if j == i or j in dropped or j in keep:
                continue
            if openings[j]["class"] != openings[i]["class"]:
                continue
            if openings[j]["side"] == openings[i]["side"] and \
                    openings[j]["box_id"] == openings[i]["box_id"]:
                continue
            if aabb_distance(aabbs[j], aabbs[i]) <= dist:
                openings[j]["rejected"] = (
                    f"corner duplicate of the {openings[i]['class']} on box "
                    f"{openings[i]['box_id']} {openings[i]['side']} "
                    f"({openings[i]['n_points']} pts vs {openings[j]['n_points']})")
                rejected.append(openings[j])
                dropped.add(j)
    return [openings[i] for i in sorted(keep)]


def rect_corners(rec):
    """The rectangle's four 3-D corners, on its box face."""
    u0, u1, z0, z1, c = rec["u_min"], rec["u_max"], rec["z_min"], rec["z_max"], rec["plane_coord"]
    if rec["side"] in ("left", "right"):        # face at constant x, u runs along y
        return np.array([[c, u0, z0], [c, u1, z0], [c, u1, z1], [c, u0, z1]])
    return np.array([[u0, c, z0], [u1, c, z0], [u1, c, z1], [u0, c, z1]])


def show(boxes, openings, pts, cls, args):
    import pyvista as pv
    w, h = (int(v) for v in args.window_size.split(","))
    pl = pv.Plotter(off_screen=bool(args.screenshot), window_size=(w, h))
    pl.set_background("white")

    for b in boxes:
        cube = pv.Box(bounds=(b["x_min"], b["x_max"], b["y_min"], b["y_max"],
                              b["z_min"], b["z_max"]))
        pl.add_mesh(cube, color=BOX_COLOR, style="wireframe", line_width=2, opacity=0.55)

    if args.points_in_view and len(pts):
        for klass in sorted(set(cls)):
            m = cls == klass
            pl.add_mesh(pv.PolyData(pts[m]), color=CLASS_COLORS.get(klass, (120, 120, 120)),
                        point_size=2, opacity=0.25)

    drawn, n_synth = defaultdict(int), 0
    for rec in openings:
        c = rect_corners(rec)
        quad = pv.PolyData(c, faces=np.array([4, 0, 1, 2, 3]))
        colour = CLASS_COLORS.get(rec["class"], (120, 120, 120))
        if rec.get("synthetic"):
            # Hollow: an inferred slot must never look like a measured one.
            pl.add_mesh(quad, color=colour, style="wireframe", line_width=4)
            n_synth += 1
        else:
            pl.add_mesh(quad, color=colour, opacity=0.85, show_edges=True,
                        edge_color="black", line_width=2)
        drawn[rec["class"]] += 1

    # add_legend needs an actor per label; empty PolyData gives one that draws
    # nothing but still carries the colour into the legend box.
    for klass, n in sorted(drawn.items()):
        pl.add_mesh(pv.PolyData(np.zeros((1, 3))), color=CLASS_COLORS[klass], opacity=0.0,
                    label=f"{klass} ({n})")
    if n_synth:
        pl.add_mesh(pv.PolyData(np.zeros((1, 3))), color=(0, 0, 0), opacity=0.0,
                    label=f"{n_synth} inferred (hollow)")
    pl.add_mesh(pv.PolyData(np.zeros((1, 3))), color=BOX_COLOR, opacity=0.0,
                label=f"boxes ({len(boxes)})")
    pl.add_legend(bcolor="white", border=True, loc="upper right", size=(0.15, 0.11))
    pl.show_axes()
    pl.add_text(f"{Path(args.boxes).name} + {len(openings)} opening(s)",
                font_size=9, color="black")
    if args.screenshot:
        pl.show(screenshot=args.screenshot)
        print(f"Wrote {args.screenshot}")
    else:
        pl.show()


def main():
    args = parse_args()
    pts_path = Path(args.points)
    pts, cls, prov = load_points(pts_path)
    if not len(pts):
        sys.exit(f"{pts_path} has no points.")
    data = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
    boxes = data.get("boxes", [])
    if not boxes:
        sys.exit(f"no boxes in {args.boxes}.")
    print(f"{len(pts)} opening point(s), {len(boxes)} box(es)"
          + ("  [masks]" if args.masks else ""))

    if args.masks:
        if not args.session_dir:
            sys.exit("--masks needs --session-dir (sync_manifest.json and labels.npy).")
        openings, rejected, n_orphan, notes = fit_from_masks(pts, cls, prov, boxes, args)
    else:
        openings, rejected, n_orphan, notes = fit(pts, cls, boxes, args)
    print(f"{n_orphan} point(s) near no box face (>{args.max_face_dist:g} m), dropped")
    for n in notes:
        print(f"  {n}")
    kept = defaultdict(int)
    for r in openings:
        kept[r["class"]] += 1
    print("kept: " + (", ".join(f"{c}={n}" for c, n in sorted(kept.items())) or "nothing"))
    print(f"rejected: {len(rejected)} cluster(s)")
    for r in openings:
        print(f"  #{r['id']:<3} {r['class']:<7} box {r['box_id']} {r['side']:<7} "
              f"{r['width_m']:5.2f} x {r['height_m']:4.2f} m at {r['plane_coord']:7.2f}, "
              f"u {r['u_min']:6.2f}..{r['u_max']:6.2f}  z {r['z_min']:5.2f}..{r['z_max']:4.2f}  "
              + (f"  INFERRED, slot {r['regularized']['slot']}" if r.get("synthetic")
                 else f"{r['n_points']:6d} pts"
                      + (f", {r['n_views']} view(s)" if r.get("n_views") else "")
                      + (f", off-plane p50 {r['point_offset_p50']:.2f} m"
                         if r.get("point_offset_p50") is not None else "")))

    out = Path(args.out) if args.out else pts_path.parent / "openings.json"
    out.write_text(json.dumps({
        "source_points": str(pts_path),
        "source_boxes": str(args.boxes),
        "params": {k: getattr(args, k) for k in
                   ("max_face_dist", "span_margin", "cluster_cell", "min_points",
                    "min_width", "min_height", "max_width", "door_floor_tol",
                    "merge_corner_dist", "exterior_only", "regularize",
                    "regularize_fill", "regularize_tol", "regularize_size_tol")},
        "regularize_notes": notes,
        "openings": openings,
        "rejected": rejected,
    }, indent=1), encoding="utf-8")
    print(f"Wrote {len(openings)} opening(s) to {out}")

    if not args.no_show:
        show(boxes, openings, pts, cls, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
