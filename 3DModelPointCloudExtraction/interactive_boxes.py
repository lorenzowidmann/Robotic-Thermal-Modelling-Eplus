"""Interactive editor for fit_boxes.py's boxes.json: viewed from the top --
fix a box fit_boxes.py got wrong, add one it missed, delete a stray one --
then save an EDITED COPY of the json (the input file is never overwritten).

Only ONE box is "selected" (a live, draggable widget) at a time; every other
box is drawn as a plain static wireframe. This is deliberate, not just a UI
choice: a VTK box widget keeps observing every mouse/scroll event for as
long as it exists, so a boxes.json with a few dozen boxes (a corridor tiled
end to end, say) previously meant a few dozen *simultaneously* live widgets
-- which is what made zooming freeze the window. With only one live at a
time, zoom/pan/rotate stay responsive regardless of how many boxes there
are. Cycle the selection with 'n'; whichever box is selected is also the
one 'd'/Backspace deletes -- no more typing a box id into the terminal.

Every widget is a plain axis-aligned box (rotation disabled, unlike
PlaneFittingAttempt/interactive_boxes.py's free-rotating ROI-cropping tool
this file used to be a copy of): dragging a face/corner only ever changes
that box's x_min/x_max/y_min/y_max, so a box you edit is still guaranteed to
land on the same 90-deg-multiple grid every other box does. Height (z_min/
z_max) is intentionally NOT drag-editable -- VTK's box widget has no clean
way to lock dragging to just X/Y, so instead each box keeps the z-range it
was loaded with (or --z-min/--z-max / the first hall's floor_z/ceiling_z
from the json for a brand new box), and the widget's own Z is simply
ignored, same as PlaneFittingAttempt/interactive_boxes.py's --z-base/
--z-top used to do.

--boxes defaults to SavedBoxes/boxes.json (fit_boxes.py's own default
output) and --out to SavedBoxes/boxes_edited.json, both next to this
script and created if missing -- so the fit -> view -> edit chain needs no
path arguments at all.

Usage:
    python interactive_boxes.py [--boxes SavedBoxes/boxes.json]
        [--bag <rosbag2_folder> | --pcd <file.pcd>]  (optional, for context)
        [--topic /cloud_registered] [--store ROS2_HUMBLE]
        [--roi xmin,xmax,ymin,ymax,zmin,zmax] [--sor]
        [--declutter] [--single-hall] [--cluster-gap 0.30] [--min-cluster-points 200]
        [--voxel-display 0.05] [--max-display-points 400000] [--point-size 2]
        [--handle-size-m 0.15]
        [--z-min Z] [--z-max Z]  (height for boxes added with 'a', if the
                                   loaded json has no "halls" summary)
        [--out SavedBoxes/boxes_edited.json]

While the window is open:
    drag a face/corner handle     resize the SELECTED box (X/Y only)
    drag the middle               move the SELECTED box (X/Y only)
    'n' key (or Tab)              select the next box (cycles through all
                                   of them; the first box is selected
                                   automatically when the window opens)
    'a' key                       add a new box (default size, centered on
                                   the current camera focal point) and
                                   select it immediately
    'd' key (or Backspace)        delete the SELECTED box, then auto-select
                                   the next one
    'p' key                       print every box's current state without
                                   closing the window
    close the window               finalize: save every remaining box to
                                   --out, renumbered 0..N-1, with a
                                   same-geometry overlap check printed

The point cloud (--bag/--pcd) is optional and shown only as static context
(flat grey, never modified or re-saved) -- unlike
PlaneFittingAttempt/interactive_boxes.py which cropped and saved a .pcd per
box; this tool's output is boxes.json, nothing else. --max-display-points is
the other half of the zoom-freeze fix: a hard cap on how many context points
are ever rendered, applied after --voxel-display.

Venv: C:\\venvs\\planefit (same as fit_boxes.py; needs pyvista).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from fit_boxes import SAVED_BOXES_DIR, crop_roi, declutter_clusters, load_merged_cloud, \
    load_pcd_cloud, statistical_outlier_removal

PALETTE = ["red", "lime", "deepskyblue", "yellow", "magenta", "orange", "cyan", "black",
           "gold", "hotpink"]


def box_to_geometry(box_id, xmin, xmax, ymin, ymax, z0, z1, hall=None):
    width, depth, height = xmax - xmin, ymax - ymin, z1 - z0
    footprint = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    corners_3d = ([[x, y, z0] for x, y in footprint]
                   + [[x, y, z1] for x, y in footprint])
    out = {
        "id": box_id,
        "x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax, "z_min": z0, "z_max": z1,
        "width_m": float(width), "depth_m": float(depth), "height_m": float(height),
        "area_m2": float(width * depth), "volume_m3": float(width * depth * height),
        "corners_3d": corners_3d,
    }
    if hall is not None:
        out["hall"] = hall
    return out


def check_overlaps(boxes):
    """3D AABB overlap between saved boxes -- not fatal (manual edits can
    legitimately leave things imperfect), just printed so you notice before
    feeding this into anything downstream that assumes a clean tiling."""
    warnings = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ox = min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"])
            oy = min(a["y_max"], b["y_max"]) - max(a["y_min"], b["y_min"])
            oz = min(a["z_max"], b["z_max"]) - max(a["z_min"], b["z_min"])
            if ox > 1e-6 and oy > 1e-6 and oz > 1e-6:
                warnings.append((a["id"], b["id"], float(ox * oy)))
    return warnings


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=Path, default=SAVED_BOXES_DIR / "boxes.json",
                    help="boxes.json to load and edit (default: SavedBoxes/boxes.json next "
                         "to this script, i.e. fit_boxes.py's own default output)")
    ap.add_argument("--bag", type=Path, help="rosbag2 folder (raw /cloud_registered) -- "
                                              "optional, just for visual context")
    ap.add_argument("--pcd", type=Path, help="load context cloud from a .pcd instead of a bag")
    ap.add_argument("--topic", default="/cloud_registered")
    ap.add_argument("--store", default="ROS2_HUMBLE")
    ap.add_argument("--roi", help="xmin,xmax,ymin,ymax,zmin,zmax crop (metres)")
    ap.add_argument("--sor", action="store_true")
    ap.add_argument("--sor-k", type=int, default=16)
    ap.add_argument("--sor-std", type=float, default=1.5)
    ap.add_argument("--declutter", action="store_true",
                    help="match fit_boxes.py's declutter so the displayed cloud lines up "
                         "with what boxes.json was actually fit from")
    ap.add_argument("--single-hall", action="store_true")
    ap.add_argument("--cluster-gap", type=float, default=0.30)
    ap.add_argument("--min-cluster-points", type=int, default=200)
    ap.add_argument("--voxel-display", type=float, default=0.05,
                    help="downsample the context cloud for render speed (0 = off)")
    ap.add_argument("--max-display-points", type=int, default=400_000,
                    help="hard cap on displayed context points, applied after "
                         "--voxel-display (randomly subsampled further if still above "
                         "this) -- the other half of the zoom-freeze fix, see the top "
                         "of this file's docstring")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--handle-size-m", type=float, default=0.15,
                    help="widget handle size in METRES. VTK sizes a box widget's handles as "
                         "a fraction of the whole SCENE's bounding-box diagonal, not that "
                         "widget's own box -- so with many small boxes in one big scene "
                         "(e.g. a long corridor tiled into many boxes), a fixed fraction "
                         "makes every handle the same oversized sphere regardless of its "
                         "own box's size. This is converted to that fraction internally "
                         "(handle_size_m / scene diagonal), so it means the same physical "
                         "size no matter how big the scene is -- lower it if handles still "
                         "look too big, e.g. 0.05 for a small room")
    ap.add_argument("--z-min", type=float, default=None,
                    help="height base for a NEW box ('a') -- overrides the default (the "
                         "first hall's floor_z from --boxes)")
    ap.add_argument("--z-max", type=float, default=None,
                    help="height top for a NEW box ('a') -- overrides the default (the "
                         "first hall's ceiling_z from --boxes)")
    ap.add_argument("--out", type=Path, default=SAVED_BOXES_DIR / "boxes_edited.json",
                    help="where the edited copy is saved (default: SavedBoxes/"
                         "boxes_edited.json next to this script; the directory is created "
                         "if missing) -- --boxes itself is never overwritten, pass --out "
                         "equal to --boxes yourself if you do want to overwrite it")
    args = ap.parse_args()

    if args.bag and args.pcd:
        raise SystemExit("pass at most one of --bag or --pcd")

    meta = json.loads(args.boxes.read_text())
    boxes = meta.get("boxes", [])
    print(f"loaded {len(boxes)} box(es) from {args.boxes}")

    # Height for a brand-new box ('a'): fit_boxes.py now estimates floor_z/ceiling_z PER
    # HALL (its "halls" list), not one global value, since different halls can genuinely
    # be at different heights -- so there's no single "the" height to fall back to here.
    # Best available default: --z-min/--z-max if given, else the first hall's/first box's
    # height. If you're adding a box into a hall with a different height, follow up with
    # --z-min/--z-max explicitly.
    default_z = None
    if args.z_min is not None and args.z_max is not None:
        default_z = (args.z_min, args.z_max)
    elif meta.get("halls"):
        h0 = meta["halls"][0]
        default_z = (h0["floor_z"], h0["ceiling_z"])
    elif boxes:
        default_z = (boxes[0]["z_min"], boxes[0]["z_max"])

    xyz = None
    if args.pcd:
        xyz = load_pcd_cloud(args.pcd)
    elif args.bag:
        xyz = load_merged_cloud(args.bag, args.topic, args.store)
    if xyz is not None:
        if args.roi:
            roi = tuple(float(x) for x in args.roi.split(","))
            xyz = crop_roi(xyz, roi)
        if args.sor:
            xyz = statistical_outlier_removal(xyz, k=args.sor_k, std_ratio=args.sor_std)
        if args.declutter:
            xyz = declutter_clusters(xyz, gap=args.cluster_gap, min_points=args.min_cluster_points,
                                      single_hall=args.single_hall)
        if args.voxel_display > 0:
            keys = np.floor(xyz / args.voxel_display).astype(np.int64)
            _, idx = np.unique(keys, axis=0, return_index=True)
            xyz = xyz[idx]
            print(f"display downsample @ {args.voxel_display}m: {len(xyz)} points")
        if len(xyz) > args.max_display_points:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(xyz), size=args.max_display_points, replace=False)
            xyz = xyz[idx]
            print(f"display cap: further subsampled to {args.max_display_points} points "
                  f"(--max-display-points)")
        if default_z is None:
            default_z = (float(xyz[:, 2].min()), float(xyz[:, 2].max()))

    if default_z is None:
        raise SystemExit("no height available for new boxes -- pass --z-min/--z-max, or "
                          "load a --boxes.json that has floor_z/ceiling_z or at least one box")

    p = pv.Plotter(off_screen=False, window_size=(1600, 1000))
    p.enable_parallel_projection()  # avoid perspective flare on tall thin boxes

    if xyz is not None:
        # Flat grey, not colored by Z: this cloud is just spatial context for placing/
        # resizing boxes, and a turbo colormap competes for attention with the boxes'
        # own (much more important) PALETTE colors.
        cloud = pv.PolyData(xyz)
        p.add_mesh(cloud, color="#999999", point_size=args.point_size,
                   render_points_as_spheres=False)
    p.set_background("white")

    # VTK sizes a box widget's handles as a fraction of the whole SCENE's bounding-box
    # diagonal (see --handle-size-m's help), not that widget's own box -- compute that
    # diagonal from everything that will be on screen (boxes + context cloud) so
    # --handle-size-m means the same physical handle size regardless of scene scale.
    diag_pts = []
    if boxes:
        diag_pts.append(np.array([c for b in boxes for c in b["corners_3d"]]))
    if xyz is not None:
        diag_pts.append(xyz)
    if diag_pts:
        allc = np.vstack(diag_pts)
        scene_diag = float(np.linalg.norm(allc.max(axis=0) - allc.min(axis=0)))
    else:
        scene_diag = 0.0
    if scene_diag <= 1e-6:
        scene_diag = 10.0  # no boxes/cloud yet to measure -- arbitrary fallback, fine since
                            # the first 'a'-added box only needs a sane starting handle size
    handle_frac = args.handle_size_m / scene_diag
    print(f"scene diagonal {scene_diag:.2f} m -> handle fraction {handle_frac:.5f} "
          f"for {args.handle_size_m} m handles")

    # boxes_state[id]: committed geometry for a box that is NOT currently selected (dict
    # with xmin/xmax/ymin/ymax/z0/z1/color). The selected box's geometry instead lives in
    # `current[id]` (from its live widget's callback) until it's committed back here by
    # commit_selected() -- on deselect, delete, or save. `order` is the stable cycle order
    # 'n' steps through; only ONE widget (selected["widget"]) exists at a time (see
    # docstring for why: that's the zoom-freeze fix).
    # "hall" (the room a box belongs to, from fit_boxes.py) is carried along untouched by
    # drag/resize -- to_openstudio.py groups boxes into OpenStudio Spaces by this field, so
    # losing it here would silently turn every box back into its own single-box room.
    boxes_state = {}
    order = []
    for b in boxes:
        bid = b["id"]
        boxes_state[bid] = {"xmin": b["x_min"], "xmax": b["x_max"], "ymin": b["y_min"],
                             "ymax": b["y_max"], "z0": b["z_min"], "z1": b["z_max"],
                             "hall": b.get("hall", bid),
                             "color": PALETTE[bid % len(PALETTE)]}
        order.append(bid)
    static_actors = {}   # id -> (mesh_actor, label_actor), for every non-selected box
    current = {}          # id -> latest box_poly (pv.PolyData) from the live widget's callback
    selected = {"id": None, "widget": None, "label_actor": None}
    next_id_counter = [max((b["id"] for b in boxes), default=-1) + 1]

    def make_callback(box_id):
        def callback(box_poly):
            current[box_id] = box_poly.copy()
        return callback

    def draw_static(box_id):
        s = boxes_state[box_id]
        mesh = pv.Box(bounds=(s["xmin"], s["xmax"], s["ymin"], s["ymax"], s["z0"], s["z1"]))
        mesh_actor = p.add_mesh(mesh, color=s["color"], style="wireframe", line_width=3)
        label_actor = p.add_point_labels([[s["xmin"], s["ymin"], s["z0"]]], [str(box_id)],
                                          font_size=16, text_color=s["color"],
                                          shape_opacity=0.6, always_visible=True)
        static_actors[box_id] = (mesh_actor, label_actor)

    def undraw_static(box_id):
        mesh_actor, label_actor = static_actors.pop(box_id)
        p.remove_actor(mesh_actor)
        p.remove_actor(label_actor)

    def commit_selected():
        """Fold the selected box's live widget geometry back into boxes_state, tear the
        widget down, and redraw it as a static box so it stays VISIBLE in the scene --
        only a truly deleted box should ever disappear from view (delete_selected() does
        not call this). Safe to call with nothing selected."""
        bid = selected["id"]
        if bid is None:
            return
        box_poly = current.get(bid)
        if box_poly is not None:
            xmin, xmax, ymin, ymax, _, _ = box_poly.bounds
            boxes_state[bid]["xmin"], boxes_state[bid]["xmax"] = xmin, xmax
            boxes_state[bid]["ymin"], boxes_state[bid]["ymax"] = ymin, ymax
        selected["widget"].Off()
        p.remove_actor(selected["label_actor"])
        selected["id"] = None
        selected["widget"] = None
        selected["label_actor"] = None
        if bid in boxes_state:   # not deleted in the meantime -- stay visible
            draw_static(bid)

    def select_box(box_id):
        commit_selected()
        if box_id in static_actors:
            undraw_static(box_id)
        s = boxes_state[box_id]
        bounds = (s["xmin"], s["xmax"], s["ymin"], s["ymax"], s["z0"], s["z1"])
        widget = p.add_box_widget(callback=make_callback(box_id), bounds=bounds, factor=1.0,
                                   rotation_enabled=False, color=s["color"])
        widget.SetHandleSize(handle_frac)
        # add_box_widget fires the callback once immediately on creation, so current[box_id]
        # is already populated here with no extra nudge needed.
        label_actor = p.add_point_labels([bounds[0::2]], [f"{box_id} [selected]"],
                                          font_size=18, text_color=s["color"],
                                          shape_opacity=0.8, always_visible=True)
        selected["id"] = box_id
        selected["widget"] = widget
        selected["label_actor"] = label_actor
        p.render()
        print(f"selected box {box_id} -- drag to resize/move, 'n' next box, "
              f"'d'/Backspace delete it")

    def select_next():
        if not order:
            print("no boxes")
            return
        if selected["id"] is None:
            select_box(order[0])
            return
        idx = order.index(selected["id"])
        select_box(order[(idx + 1) % len(order)])

    def add_new_box():
        commit_selected()
        box_id = next_id_counter[0]
        next_id_counter[0] += 1
        fx, fy, _ = p.camera.focal_point
        half = 1.5
        boxes_state[box_id] = {"xmin": fx - half, "xmax": fx + half, "ymin": fy - half,
                                "ymax": fy + half, "z0": default_z[0], "z1": default_z[1],
                                "hall": box_id,  # a brand new box starts as its own room --
                                                 # 'p' to check ids, edit the saved json by
                                                 # hand to merge it into an existing one
                                "color": PALETTE[box_id % len(PALETTE)]}
        order.append(box_id)
        select_box(box_id)
        print(f"added box {box_id} (z locked to {default_z[0]:.2f}..{default_z[1]:.2f})")

    def delete_selected():
        bid = selected["id"]
        if bid is None:
            print("no box selected -- 'n' to select one first")
            return
        idx = order.index(bid)
        selected["widget"].Off()
        p.remove_actor(selected["label_actor"])
        selected["id"] = None
        selected["widget"] = None
        selected["label_actor"] = None
        boxes_state.pop(bid, None)
        current.pop(bid, None)
        order.remove(bid)
        print(f"deleted box {bid}")
        if order:
            # order.remove() shifted everything after the deleted box back by one, so
            # index `idx` now holds what used to be the NEXT box -- select that (wraps
            # to the first box if the deleted one was last).
            select_box(order[idx % len(order)])
        else:
            p.render()

    def print_current():
        print(f"\ncurrent state, {len(order)} box(es):")
        for bid in order:
            s = boxes_state[bid]
            if bid == selected["id"] and current.get(bid) is not None:
                xmin, xmax, ymin, ymax, _, _ = current[bid].bounds
            else:
                xmin, xmax, ymin, ymax = s["xmin"], s["xmax"], s["ymin"], s["ymax"]
            tag = " [selected]" if bid == selected["id"] else ""
            print(f"  {bid}{tag}: x={xmin:.2f}..{xmax:.2f} y={ymin:.2f}..{ymax:.2f} "
                  f"z={s['z0']:.2f}..{s['z1']:.2f} "
                  f"({xmax - xmin:.2f} x {ymax - ymin:.2f} x {s['z1'] - s['z0']:.2f} m)")

    for bid in order:
        draw_static(bid)
    select_next()  # selects order[0] if any boxes were loaded; no-op otherwise

    p.add_key_event("n", select_next)
    p.add_key_event("Tab", select_next)
    p.add_key_event("a", add_new_box)
    p.add_key_event("d", delete_selected)
    p.add_key_event("BackSpace", delete_selected)
    p.add_key_event("p", print_current)
    p.add_text("'n'/Tab select next -- drag handles to resize/move selected (X/Y only) -- "
               "'a' add box -- 'd'/Backspace delete selected -- 'p' print -- close to save",
               font_size=9, color="black")

    # Top-down: "edit boxes.json from the top" is a plan-view editing task, so start framed
    # that way. Still a real interactive camera -- orbit with the mouse if you want a 3D check.
    if boxes:
        allc = np.array([c for b in boxes for c in b["corners_3d"]])
        lo, hi = allc.min(axis=0), allc.max(axis=0)
        margin = (hi - lo) * 0.2
        bounds = (lo[0] - margin[0], hi[0] + margin[0], lo[1] - margin[1], hi[1] + margin[1],
                  lo[2], hi[2])
        p.reset_camera(bounds=bounds)
    p.camera_position = "xy"
    p.camera.up = (0.0, 1.0, 0.0)

    print(f"{len(order)} starting box(es), opening window -- 'n' to select, drag to "
          f"resize/move, 'a' to add, 'd'/Backspace to delete selected, 'p' to print, "
          f"close window to save")
    p.show()

    # Fold the last-selected box's edits back into boxes_state -- done by hand here (not
    # via commit_selected(), which also calls widget.Off()/p.remove_actor: the render
    # window is already gone at this point, so touching it further isn't safe).
    if selected["id"] is not None:
        bid = selected["id"]
        box_poly = current.get(bid)
        if box_poly is not None:
            xmin, xmax, ymin, ymax, _, _ = box_poly.bounds
            boxes_state[bid]["xmin"], boxes_state[bid]["xmax"] = xmin, xmax
            boxes_state[bid]["ymin"], boxes_state[bid]["ymax"] = ymin, ymax

    if not boxes_state:
        print("no boxes remain, nothing to save")
        return

    out_boxes = []
    for new_id, box_id in enumerate(order):
        s = boxes_state[box_id]
        xmin, xmax, ymin, ymax = s["xmin"], s["xmax"], s["ymin"], s["ymax"]
        if xmax <= xmin or ymax <= ymin:
            print(f"box {box_id}: degenerate footprint (zero/negative size), skipping")
            continue
        out_boxes.append(box_to_geometry(new_id, xmin, xmax, ymin, ymax, s["z0"], s["z1"],
                                          hall=s.get("hall")))

    overlaps = check_overlaps(out_boxes)
    if overlaps:
        print(f"\nWARNING: {len(overlaps)} overlapping box pair(s) after editing:")
        for a_id, b_id, area in overlaps:
            print(f"  box {a_id} and box {b_id} overlap by {area:.2f} m^2 (in plan)")

    meta = dict(meta)  # keep source/cell_size_m/yaw_deg/halls as loaded (note: "halls" is
                       # fit_boxes.py's ORIGINAL per-hall floor_z/ceiling_z summary -- it is
                       # NOT re-derived from your edits, so it can go stale if you moved a
                       # box into a different hall's area or changed its height via --z-min/
                       # --z-max on a re-add)
    meta["boxes"] = out_boxes
    meta["edited_from"] = str(args.boxes)
    meta["leftover_cells"] = None    # no longer meaningful after manual editing
    meta["leftover_area_m2"] = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {len(out_boxes)} box(es) to {args.out}")


if __name__ == "__main__":
    main()
