"""Merge several boxes.json "sessions" (e.g. one per scanned bag/run) into one final
boxes.json -- with a top-view window, like interactive_boxes.py's, that lets you
click-drag any box to slide its WHOLE SESSION into alignment with the others BEFORE the
merge is written (clicking one box of boxes_edited.json moves every box that came from
boxes_edited.json together, in lockstep -- not just that one box).

Unlike interactive_boxes.py, dragging here is translate-only: click any box from a
session and drag, and every box in that session shifts x_min/x_max/y_min/y_max by the
same dx/dy (so each box's own footprint size never changes, and their relative layout
within the session is preserved); z_min/z_max stay exactly as loaded, so a session can
never be dragged off its own floor. There is no resize. This is enforced by construction
(the drag handler only ever adds the same dx/dy to both min and max of every box in the
session, and never touches z), not by a size/height check after the fact.

The whole session can also be ROTATED, in 90 degree steps only ('r' = clockwise, 'R' =
counter-clockwise, applied to whichever session the last-clicked box belongs to -- same
box the 'd' delete uses). Every box in the session is rotated together about the
session's own current bounding-box center, so the family turns in place rather than
swinging around some other session's origin; each box's footprint stays axis-aligned
(width/height simply swap on an odd number of 90s) and z is untouched, same as a drag.
90-degree-only is a deliberate limit, not a missing feature: to_pcd.py has to recover
each session's total rotation from the merged file alone (see its own docstring), and
searching the 4 right angles for a match is exact and unambiguous -- an arbitrary angle
would need the rotation to be written out separately, which is what this tool has always
avoided (see "HOW THE PER-SESSION OFFSET IS RECOVERED" in to_pcd.py's docstring).

Every box keeps its origin ("session" = index into the file list below, "source_file",
"source_hall" = its hall id before merging, "source_id" = its id before renumbering) so
you can trace any box back to the session it came from after the merge. Boxes from
different sessions are colored differently (same PALETTE cycling as interactive_boxes.py,
by session index) so you can see at a glance which cluster is which while you drag them
into alignment. "hall" ids are offset per session (session_index * 1000) before merging
so two sessions' hall 0 don't collide -- to_openstudio.py's per-hall Space grouping still
works on the merged output, just with bigger hall numbers.

Architecture note: this does NOT use vtkBoxWidget per box like interactive_boxes.py does
(that file deliberately keeps only ONE widget alive at a time -- see its docstring -- since
a live box widget keeps observing every mouse/scroll event, and a few dozen of them at once
is what made zooming freeze). Here every box is a plain, static mesh (no widget), and a
single custom vtkInteractorStyleTrackballCamera subclass (DragStyle, below) owns exactly
one set of mouse observers for the whole scene: on left-click it cell-picks whichever box
you clicked (if any), then drags every box sharing that box's "session" by re-projecting
the mouse to the clicked box's own z-plane every move (exact ray/plane intersection, so
it works at any camera angle, not just top-down) and applying the resulting dx/dy to the
whole session; a click that misses every box falls through to the normal trackball
camera (rotate/pan/zoom all still work). So this scales fine to many boxes from many
sessions -- the cost is one mesh+wireframe+label per box, same as interactive_boxes.py's
static (non-selected) boxes, never a per-box widget.

Usage:
    python merge_boxes.py SESSION1.json SESSION2.json [SESSION3.json ...]
        [--out SavedBoxes/boxes_merged.json]

While the window is open:
    click + drag any box    move its WHOLE SESSION together (X/Y only -- size and Z are
                             locked, see above)
    click empty space + drag   normal camera orbit/pan/zoom (unchanged)
    'r' key                 rotate the last-clicked box's WHOLE SESSION 90 deg clockwise
                             about that session's own bounding-box center
    'R' key (shift+r)       same, counter-clockwise
    'd' key (or Backspace)  delete the last INDIVIDUAL box you clicked (drag or plain
                             click), then it's excluded from the merge -- the rest of its
                             session is untouched
    'p' key                 print every box's current state without closing the window
    close the window        finalize: write every remaining box to --out, renumbered
                             0..N-1, with a same-geometry overlap check printed (overlaps
                             across sessions are expected if they cover the same area --
                             just informational, not fatal)

Venv: C:\\venvs\\planefit (same as fit_boxes.py/interactive_boxes.py; needs pyvista+vtk).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

from fit_boxes import SAVED_BOXES_DIR
from interactive_boxes import PALETTE, box_to_geometry, check_overlaps

# Hall ids are offset per session by this much before merging, so e.g. session 0's hall 0
# and session 1's hall 0 don't collide in the merged output. Fine as long as no single
# session has >=1000 halls.
HALL_OFFSET_PER_SESSION = 1000


class DragStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Trackball camera style (rotate/pan/zoom all still work normally) that also lets you
    click-drag a whole SESSION at once. LeftButtonPress cell-picks the click; if it landed
    on one of the pickable box actors, the drag is handled entirely by the on_move/on_drop
    callbacks below -- translating every box belonging to that same session's X/Y in
    lockstep (the box actually under the cursor supplies the z-plane the drag ray is
    unprojected onto -- exact ray/plane intersection, correct at any camera angle -- but
    the resulting dx/dy is applied to the whole session, not just that one box) -- and the
    superclass's own OnLeftButtonDown/OnMouseMove/OnLeftButtonUp are never called for the
    duration of that drag, so the camera does not also rotate while you drag. A click that
    misses every box falls straight through to the superclass: normal orbit.
    """

    def __init__(self, renderer, actor_to_box, get_z, get_session, on_move, on_drop):
        self.renderer = renderer
        self.actor_to_box = actor_to_box
        self.get_z = get_z
        self.get_session = get_session
        self.on_move = on_move
        self.on_drop = on_drop
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.dragging_session = None
        self.last_id = None   # last individual box clicked (drag or not) -- what 'd' deletes
        self._grab = None     # last unprojected mouse world point during a drag
        self.AddObserver("LeftButtonPressEvent", self._press)
        self.AddObserver("MouseMoveEvent", self._move)
        self.AddObserver("LeftButtonReleaseEvent", self._release)

    def _unproject(self, x, y, z_plane):
        """Exact ray/plane intersection: unproject screen (x, y) to the world point where
        the camera ray through that pixel crosses the horizontal plane z = z_plane. Works
        for perspective or parallel projection, at any camera angle -- not just top-down."""
        self.renderer.SetDisplayPoint(x, y, 0.0)
        self.renderer.DisplayToWorld()
        x0, y0, z0, w0 = self.renderer.GetWorldPoint()
        x0, y0, z0 = x0 / w0, y0 / w0, z0 / w0
        self.renderer.SetDisplayPoint(x, y, 1.0)
        self.renderer.DisplayToWorld()
        x1, y1, z1, w1 = self.renderer.GetWorldPoint()
        x1, y1, z1 = x1 / w1, y1 / w1, z1 / w1
        dz = z1 - z0
        if abs(dz) < 1e-9:
            return None  # ray parallel to the plane -- shouldn't happen in practice
        t = (z_plane - z0) / dz
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)

    def _press(self, _obj, _event):
        x, y = self.GetInteractor().GetEventPosition()
        self.picker.Pick(x, y, 0, self.renderer)
        box_id = self.actor_to_box.get(self.picker.GetActor())
        if box_id is None:
            self.OnLeftButtonDown()  # missed every box -- normal camera rotate
            return
        wp = self._unproject(x, y, self.get_z(box_id))
        if wp is None:
            self.OnLeftButtonDown()
            return
        self.dragging_session = self.get_session(box_id)
        self.last_id = box_id
        self._grab = wp
        self._drag_z = self.get_z(box_id)  # fixed for the whole gesture, see _move

    def _move(self, _obj, _event):
        if self.dragging_session is None:
            self.OnMouseMove()  # normal orbit/pan
            return
        x, y = self.GetInteractor().GetEventPosition()
        wp = self._unproject(x, y, self._drag_z)
        if wp is None:
            return
        dx, dy = wp[0] - self._grab[0], wp[1] - self._grab[1]
        self._grab = wp
        self.on_move(self.dragging_session, dx, dy)

    def _release(self, _obj, _event):
        if self.dragging_session is None:
            self.OnLeftButtonUp()
            return
        self.on_drop(self.dragging_session)
        self.dragging_session = None


def load_session(path, session_idx):
    """Load one boxes.json, tagging every box/hall with where it came from and offsetting
    its hall ids so they don't collide with another session's."""
    meta = json.loads(path.read_text())
    offset = session_idx * HALL_OFFSET_PER_SESSION
    boxes = []
    for b in meta.get("boxes", []):
        nb = dict(b)
        orig_hall = nb.get("hall")
        nb["session"] = session_idx
        nb["source_file"] = str(path)
        nb["source_hall"] = orig_hall
        nb["source_id"] = nb.get("id")
        nb["hall"] = orig_hall + offset if orig_hall is not None else None
        boxes.append(nb)
    halls = []
    for h in meta.get("halls", []) or []:
        nh = dict(h)
        nh["session"] = session_idx
        nh["source_hall"] = nh.get("hall")
        if nh.get("hall") is not None:
            nh["hall"] = nh["hall"] + offset
        halls.append(nh)
    return meta, boxes, halls


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path,
                    help="two or more boxes.json files to merge, each one a separate "
                         "capture session (e.g. SavedBoxes/boxes.json from one bag and "
                         "SavedBoxes/boxes_edited.json from another)")
    ap.add_argument("--out", type=Path, default=SAVED_BOXES_DIR / "boxes_merged.json",
                    help="merged output json (default: SavedBoxes/boxes_merged.json next "
                         "to this script; created if missing)")
    args = ap.parse_args()

    all_boxes, all_halls, session_summaries = [], [], []
    for idx, path in enumerate(args.sessions):
        meta, boxes, halls = load_session(path, idx)
        print(f"session {idx}: {len(boxes)} box(es) from {path}")
        all_boxes.extend(boxes)
        all_halls.extend(halls)
        session_summaries.append({"index": idx, "file": str(path),
                                   "source": meta.get("source"), "n_boxes": len(boxes)})
    if not all_boxes:
        raise SystemExit("no boxes found in any session file")

    # boxes_state[id]: this tool's live/editable copy of each box (keyed by a freshly
    # assigned sequential id, NOT the original per-session id -- that's kept as
    # "source_id"). "hall" here is already offset (see load_session); "session"/
    # "source_file"/"source_hall"/"source_id" are carried straight through to the merged
    # output untouched by dragging, same as interactive_boxes.py carries "hall" through.
    boxes_state, order = {}, []
    for new_id, b in enumerate(all_boxes):
        boxes_state[new_id] = {
            "xmin": b["x_min"], "xmax": b["x_max"], "ymin": b["y_min"], "ymax": b["y_max"],
            "z0": b["z_min"], "z1": b["z_max"], "hall": b.get("hall"),
            "session": b["session"], "source_file": b["source_file"],
            "source_hall": b.get("source_hall"), "source_id": b.get("source_id"),
            "color": PALETTE[b["session"] % len(PALETTE)],
        }
        order.append(new_id)

    p = pv.Plotter(off_screen=False, window_size=(1600, 1000))
    p.enable_parallel_projection()
    p.set_background("white")

    meshes, box_actors, actor_to_box, label_actors, actor_offsets = {}, {}, {}, {}, {}

    def move_box_actor(bid, dx, dy):
        # Cheap translate: nudge the actors' own transform (SetPosition), never touch the
        # mesh's point data. During a drag this fires for every box in the session on
        # every mouse-move event, so this -- not rebuilding geometry via pv.Box() each
        # time, which was the original (laggy) approach -- is what keeps it smooth even
        # for a session with many boxes.
        off = actor_offsets[bid]
        off[0] += dx
        off[1] += dy
        acts = box_actors[bid]
        acts["solid"].SetPosition(off[0], off[1], 0.0)
        acts["wire"].SetPosition(off[0], off[1], 0.0)

    def refresh_label(bid):
        # remove+re-add is the only way to move a pyvista point label -- noticeably
        # heavier than move_box_actor's SetPosition, which is why on_session_move doesn't
        # call this per box per mouse-move (that was the original lag with a multi-box
        # session): it's only called once per box, in on_session_drop, when a drag ends.
        old = label_actors.pop(bid, None)
        if old is not None:
            p.remove_actor(old)
        s = boxes_state[bid]
        text = f"S{s['session']}:{bid}"
        label_actors[bid] = p.add_point_labels(
            [[s["xmin"], s["ymin"], s["z1"]]], [text], font_size=13, text_color=s["color"],
            shape_opacity=0.7, always_visible=True)

    def create_box_actor(bid):
        # (Re)builds bid's mesh + actors from its current boxes_state -- used both for the
        # initial draw and after a rotate, which (unlike a drag) changes the mesh's own
        # bounds (width/height swap on an odd number of 90s) so the cheap SetPosition
        # nudge move_box_actor uses does not apply; the actor has to be rebuilt instead.
        s = boxes_state[bid]
        mesh = pv.Box(bounds=(s["xmin"], s["xmax"], s["ymin"], s["ymax"], s["z0"], s["z1"]))
        meshes[bid] = mesh
        # solid, mostly-transparent fill: the pickable click target (a wireframe alone is
        # too thin to reliably click). wireframe on top of it for a crisp outline.
        solid_actor = p.add_mesh(mesh, color=s["color"], opacity=0.15, pickable=True)
        wire_actor = p.add_mesh(mesh, style="wireframe", line_width=2, color=s["color"],
                                 pickable=False)
        box_actors[bid] = {"solid": solid_actor, "wire": wire_actor}
        actor_to_box[solid_actor] = bid
        actor_offsets[bid] = [0.0, 0.0]
        refresh_label(bid)

    for bid in order:
        create_box_actor(bid)

    # Precomputed once: which box ids belong to each session, so a drag doesn't have to
    # scan every box in the merge on every single mouse-move event -- just its own session.
    session_members = {}
    for bid in order:
        session_members.setdefault(boxes_state[bid]["session"], []).append(bid)

    def get_box_z(bid):
        return boxes_state[bid]["z0"]

    def get_box_session(bid):
        return boxes_state[bid]["session"]

    def on_session_move(session, dx, dy):
        # every box that came from this session moves together, in lockstep -- each keeps
        # its own z and size, only x/y shift, same as a single box's move would. Labels are
        # intentionally NOT touched here (see refresh_label's cost note) -- they catch up
        # in on_session_drop once the drag ends.
        for bid in session_members.get(session, ()):
            s = boxes_state[bid]
            s["xmin"] += dx
            s["xmax"] += dx
            s["ymin"] += dy
            s["ymax"] += dy
            move_box_actor(bid, dx, dy)
        p.render()

    def on_session_drop(session):
        members = session_members.get(session, ())
        for bid in members:
            refresh_label(bid)  # snap labels to their final position now that dragging stopped
        p.render()
        print(f"session {session}: moved {len(members)} box(es)")

    style = DragStyle(p.renderer, actor_to_box, get_box_z, get_box_session,
                       on_session_move, on_session_drop)
    # Must go through pyvista's own iren.style SETTER, not interactor.SetInteractorStyle()
    # directly -- p.show() calls self.iren.update_style() right before the interactive
    # loop starts, which re-applies whatever pyvista's OWN _style_class bookkeeping still
    # points at. Setting the raw interactor bypasses that bookkeeping, so update_style()
    # silently reverts to the default trackball style right as the window opens (looked
    # like dragging did nothing but rotate). Going through iren.style keeps pyvista's
    # cached style in sync with ours, so update_style() re-applies DragStyle, not the
    # default.
    p.iren.style = style

    def rotate_point_cw(x, y, cx, cy, clockwise):
        # 90 deg rotation about (cx, cy): clockwise maps (0,1) (north) -> (1,0) (east), i.e.
        # (dx, dy) -> (dy, -dx); counter-clockwise is the inverse, (dx, dy) -> (-dy, dx).
        dx, dy = x - cx, y - cy
        if clockwise:
            return cx + dy, cy - dx
        return cx - dy, cy + dx

    def rotate_session(clockwise):
        bid = style.last_id
        if bid is None or bid not in boxes_state:
            print("click a box first (drag or just click it), then 'r'/'R' to rotate its "
                  "whole session 90 deg")
            return
        session = boxes_state[bid]["session"]
        members = session_members.get(session, [])
        if not members:
            return
        xs = [boxes_state[m]["xmin"] for m in members] + \
             [boxes_state[m]["xmax"] for m in members]
        ys = [boxes_state[m]["ymin"] for m in members] + \
             [boxes_state[m]["ymax"] for m in members]
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        for m in members:
            s = boxes_state[m]
            x0, y0 = rotate_point_cw(s["xmin"], s["ymin"], cx, cy, clockwise)
            x1, y1 = rotate_point_cw(s["xmax"], s["ymax"], cx, cy, clockwise)
            s["xmin"], s["xmax"] = min(x0, x1), max(x0, x1)
            s["ymin"], s["ymax"] = min(y0, y1), max(y0, y1)
            acts = box_actors.pop(m)
            p.remove_actor(acts["solid"])
            p.remove_actor(acts["wire"])
            del actor_to_box[acts["solid"]]
            create_box_actor(m)
        p.render()
        direction = "clockwise" if clockwise else "counter-clockwise"
        print(f"session {session}: rotated {len(members)} box(es) 90 deg {direction} "
              f"about ({cx:.2f}, {cy:.2f})")

    def delete_last():
        bid = style.last_id
        if bid is None or bid not in boxes_state:
            print("click a box first (drag or just click it), then 'd' to delete it")
            return
        acts = box_actors.pop(bid)
        p.remove_actor(acts["solid"])
        p.remove_actor(acts["wire"])
        lbl = label_actors.pop(bid, None)
        if lbl is not None:
            p.remove_actor(lbl)
        del actor_to_box[acts["solid"]]
        session_members[boxes_state[bid]["session"]].remove(bid)
        actor_offsets.pop(bid, None)
        del boxes_state[bid]
        order.remove(bid)
        style.last_id = None
        print(f"deleted box {bid}")
        p.render()

    def print_all():
        print(f"\ncurrent state, {len(order)} box(es):")
        for bid in order:
            s = boxes_state[bid]
            print(f"  {bid} [session {s['session']}, orig id {s['source_id']} from "
                  f"{s['source_file']}]: x={s['xmin']:.2f}..{s['xmax']:.2f} "
                  f"y={s['ymin']:.2f}..{s['ymax']:.2f} z={s['z0']:.2f}..{s['z1']:.2f}")

    p.add_key_event("d", delete_last)
    p.add_key_event("BackSpace", delete_last)
    p.add_key_event("p", print_all)
    p.add_key_event("r", lambda: rotate_session(True))
    p.add_key_event("R", lambda: rotate_session(False))
    p.add_text("click+drag any box to move its WHOLE SESSION together (X/Y only -- size "
               "& floor locked) -- 'r'/'R' rotate last-clicked box's session 90 deg "
               "CW/CCW -- click empty space to orbit/pan/zoom as usual -- "
               "'d'/Backspace delete last-clicked box -- 'p' print -- close to save",
               font_size=9, color="black")

    # Top-down start, framed over every session's boxes combined -- still a real
    # interactive camera (orbit with the mouse over empty space), the drag math doesn't
    # depend on staying top-down (see DragStyle's docstring).
    allc = np.array([[s["xmin"], s["ymin"], s["z0"]] for s in boxes_state.values()]
                     + [[s["xmax"], s["ymax"], s["z1"]] for s in boxes_state.values()])
    lo, hi = allc.min(axis=0), allc.max(axis=0)
    margin = (hi - lo) * 0.2
    bounds = (lo[0] - margin[0], hi[0] + margin[0], lo[1] - margin[1], hi[1] + margin[1],
              lo[2], hi[2])
    p.reset_camera(bounds=bounds)
    p.camera_position = "xy"
    p.camera.up = (0.0, 1.0, 0.0)

    print(f"{len(order)} box(es) from {len(args.sessions)} session(s) -- drag to align, "
          f"'r'/'R' to rotate last-clicked box's session 90 deg CW/CCW, "
          f"'d'/Backspace to delete last-clicked, 'p' to print, close window to save")
    p.show()

    if not boxes_state:
        print("no boxes remain, nothing to save")
        return

    out_boxes = []
    for new_id, bid in enumerate(order):
        s = boxes_state[bid]
        geo = box_to_geometry(new_id, s["xmin"], s["xmax"], s["ymin"], s["ymax"],
                               s["z0"], s["z1"], hall=s["hall"])
        geo["session"] = s["session"]
        geo["source_file"] = s["source_file"]
        geo["source_hall"] = s["source_hall"]
        geo["source_id"] = s["source_id"]
        out_boxes.append(geo)

    overlaps = check_overlaps(out_boxes)
    if overlaps:
        print(f"\nNOTE: {len(overlaps)} overlapping box pair(s) after merge (expected if "
              f"sessions cover the same area -- not fatal):")
        for a_id, b_id, area in overlaps:
            print(f"  box {a_id} and box {b_id} overlap by {area:.2f} m^2 (in plan)")

    merged = {
        "source": f"merged from {len(args.sessions)} session(s): "
                  + ", ".join(str(s) for s in args.sessions),
        "sessions": session_summaries,
        "cell_size_m": None,   # per-session values may differ -- no single value applies
        "yaw_deg": None,
        "halls": all_halls,    # each session's ORIGINAL halls summary, hall-id-offset --
                                # not re-derived from your drags, same staleness caveat as
                                # interactive_boxes.py's meta["halls"]
        "boxes": out_boxes,
        "leftover_cells": None,
        "leftover_area_m2": None,
        "merged_from": [str(s) for s in args.sessions],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, indent=2))
    print(f"\nwrote {len(out_boxes)} box(es) to {args.out}")


if __name__ == "__main__":
    main()
