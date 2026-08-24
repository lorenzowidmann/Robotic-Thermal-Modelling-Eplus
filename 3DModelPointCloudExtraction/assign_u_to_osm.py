"""Assign measured per-voxel U-values onto an OpenStudio model's surfaces,
splitting each surface into rectangular tiles wherever the measured U varies.

Runs AFTER openings exist: it consumes an .osm that already has its windows and
doors as SubSurfaces (i.e. to_openstudio.py --openings output), and it keeps
them attached.

What it does, per exterior surface
----------------------------------
1. Lays a grid over the surface, tile pitch --tile-size.
2. SNAPS the grid lines onto every opening edge, so no window or door ever
   straddles a tile boundary -- each opening is then exactly a whole number of
   cells.
3. Around each opening, reserves a HOST rectangle: the opening's cells expanded
   by one cell ring. The opening is re-parented to that host. This is required
   because OpenStudio needs a SubSurface to lie strictly inside one parent
   Surface -- a host exactly equal to the window would have zero opaque area
   and be rejected. Overlapping hosts are merged.
4. Every remaining cell becomes its own Surface, carrying the U measured for
   that patch of wall. Hosts get a U too (from their own opaque area).
5. The original Surface is removed; hosts + tiles retile it exactly, so the
   space stays enclosed and the model remains simulatable.

Tiles whose U differs end up as separate Surfaces with separate Constructions,
which is the only way OpenStudio can carry more than one U across one wall.

U -> Construction
-----------------
The measured U is an AIR-TO-AIR transmittance: it already contains both surface
films (it was derived as hsi*(Tint-Tsurf)/(Tint-Text)). EnergyPlus adds the
films itself from the surface's exposure, so the construction must carry only
the layer resistance in between:

    R_layer = 1/U - 1/hsi - 1/he

This has a hard consequence worth knowing: with hsi=7.7 and he=25,
1/hsi + 1/he = 0.170 m2K/W, so no air-to-air U above 1/0.170 = 5.89 W/m2K can
be represented by ANY construction -- the films alone already conduct that
much. Measured U above that ceiling is physically impossible for an opaque
element, and this script does not silently clamp it: those tiles are reported,
named ...__UNPHYSICAL, and given --min-r so the model still loads. Treat them
as a flag that the measurement, not the model, needs revisiting.

Which voxels count
------------------
--u-column (default u_value_corrected_w_m2k) is read from --voxels; rows with
an empty value are skipped -- that is how voxel_solar_ns.py marks surfaces it
declared unmeasurable (e.g. the floor, whose far side is not outdoor air).
--require-note / --skip-note filter further by correction_note if only the
physically-validated groups are wanted.

Tiles with no voxels: --fill-neighbors
--------------------------------------
The thermal camera did not see every patch of every wall, so plenty of tiles
end up with zero contributing voxels and, by default, no construction at all.
--fill-neighbors gives those tiles the MEAN U of the tiles they physically
touch which do have a value, spreading outwards one ring at a time (a tile
with no measured neighbour waits for a ring in which one of its neighbours has
been filled, and so on). Only edge-sharing neighbours count -- corner-only
contact does not -- and only within the SAME parent surface: a wall with no
measurement anywhere on it stays empty rather than inheriting a number from
the wall around the corner. Each ring's donors are frozen at the previous
ring, so the result does not depend on iteration order. --fill-max-rings caps
how far a value may travel (0 = unlimited, the default).

This is INTERPOLATION, not measurement, so it is labelled as such everywhere
and never silently mixed into the measured population:
  * the Surface is renamed  ..._t12_3__NFILL2  (the number is the ring, i.e.
    how many tiles away the nearest real measurement was),
  * it gets its own Construction, named  U_8.50_Wm2K__NEIGHBOR_FILLED , which
    is never shared with a measured tile of the same U,
  * OpenStudio AdditionalProperties are set on the Surface: u_source =
    neighbor_fill, fill_ring, fill_from_n (how many neighbours were averaged),
  * the --report CSV gains a u_source / fill_ring / fill_from_n column trio,
    with u_source = measured | neighbor_fill | none.
Without --fill-neighbors nothing above happens and every tile is either
measured or bare, exactly as before.

Coordinate frames
-----------------
The .osm is in LOCAL coordinates: to_openstudio.py translates all geometry by
-origin. --osm-origin must therefore be added back to reach the voxel/SLAM
frame. For session 9 that origin is the min corner of boxes_s9_edited.json,
i.e. 4.11,-0.80,-0.33.

Only axis-aligned surfaces are handled (this model is built from axis-aligned
boxes); anything else is reported and left untouched.

Usage:
    python assign_u_to_osm.py \\
        --osm OpenStudioModel/session9_openings.osm \\
        --voxels <run>/thermal_voxels_u_solair.csv \\
        --osm-origin 4.11,-0.80,-0.33 \\
        --tile-size 0.5 --out OpenStudioModel/session9_u.osm \\
        [--fill-neighbors [--fill-max-rings 3]]

Venv: C:\\venvs\\planefit (openstudio 3.11 + numpy).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import openstudio

TOL = 1e-6


def load_model(path):
    m = openstudio.osversion.VersionTranslator().loadModel(openstudio.toPath(str(path)))
    if not m.is_initialized():
        raise SystemExit(f"could not load {path}")
    return m.get()


def axis_of(vertices):
    """(constant axis index, its value) for an axis-aligned planar surface, else None."""
    V = np.array([[p.x(), p.y(), p.z()] for p in vertices])
    for ax in (0, 1, 2):
        if np.ptp(V[:, ax]) < 1e-4:
            return ax, float(V[:, ax].mean())
    return None, None


def inplane_axes(const_ax):
    """The two in-plane axis indices (u, v). v is z for walls, y for horizontal."""
    return {0: (1, 2), 1: (0, 2), 2: (0, 1)}[const_ax]


def rect_of(vertices, const_ax):
    ua, va = inplane_axes(const_ax)
    V = np.array([[p.x(), p.y(), p.z()] for p in vertices])
    return V[:, ua].min(), V[:, ua].max(), V[:, va].min(), V[:, va].max()


def make_points(u0, u1, v0, v1, const_ax, const_val, want_normal):
    """Point3dVector for a tile, wound so its outward normal matches want_normal."""
    ua, va = inplane_axes(const_ax)
    quad = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
    pts = []
    for u, v in quad:
        c = [0.0, 0.0, 0.0]
        c[const_ax] = const_val
        c[ua], c[va] = u, v
        pts.append(c)
    P = np.array(pts)
    n = np.cross(P[1] - P[0], P[2] - P[0])
    if np.dot(n, want_normal) < 0:
        pts = pts[::-1]
    vec = openstudio.Point3dVector()
    for c in pts:
        vec.append(openstudio.Point3d(float(c[0]), float(c[1]), float(c[2])))
    return vec


def snapped_lines(lo, hi, pitch, extra):
    """Grid lines from lo to hi at `pitch`, with `extra` coordinates forced in."""
    n = max(1, int(round((hi - lo) / pitch)))
    lines = list(np.linspace(lo, hi, n + 1))
    lines += [e for e in extra if lo + TOL < e < hi - TOL]
    lines = sorted(lines)
    out = [lines[0]]
    for x in lines[1:]:
        if x - out[-1] > 1e-4:
            out.append(x)
    out[-1] = hi
    return out


def merge_rects(rects):
    """Iteratively union overlapping index-rects (i0,i1,j0,j1) into bounding boxes."""
    rects = list(rects)
    changed = True
    while changed:
        changed = False
        for a in range(len(rects)):
            for b in range(a + 1, len(rects)):
                i0, i1, j0, j1 = rects[a]
                k0, k1, l0, l1 = rects[b]
                if i0 < k1 and k0 < i1 and j0 < l1 and l0 < j1:
                    rects[a] = (min(i0, k0), max(i1, k1), min(j0, l0), max(j1, l1))
                    rects.pop(b)
                    changed = True
                    break
            if changed:
                break
    return rects


def edge_adjacency(rects, tol=1e-4):
    """Index adjacency for coplanar (u0,u1,v0,v1) rects that share an EDGE.

    Two rects are neighbours when they abut along one axis (one's max meets the
    other's min) and genuinely overlap along the other. The overlap must exceed
    `tol`, so rects meeting at a single corner are NOT neighbours -- 4- rather
    than 8-connectivity, which is what "touching tile" means for a wall.
    Handles the hosts too: they are just bigger rects, no index bookkeeping.
    """
    adj = [set() for _ in rects]
    for a in range(len(rects)):
        a0, a1, b0, b1 = rects[a]
        for b in range(a + 1, len(rects)):
            c0, c1, d0, d1 = rects[b]
            du = min(a1, c1) - max(a0, c0)            # overlap along u
            dv = min(b1, d1) - max(b0, d0)            # overlap along v
            vert = abs(a1 - c0) < tol or abs(c1 - a0) < tol      # abut in u
            horz = abs(b1 - d0) < tol or abs(d1 - b0) < tol      # abut in v
            if (vert and dv > tol) or (horz and du > tol):
                adj[a].add(b)
                adj[b].add(a)
    return adj


def fill_from_neighbors(res, adj, max_rings):
    """Give every u-less entry of `res` the mean U of its touching neighbours.

    Grows outwards one ring at a time. A ring's donors are exactly the entries
    settled in a STRICTLY earlier ring (measured ones are ring 0), so every
    tile filled in ring k averages only values that were already final when
    the ring started -- the outcome is independent of iteration order.
    Stops when a ring fills nothing, or at `max_rings` (0 = unlimited).
    """
    ring = 0
    while True:
        ring += 1
        if max_rings and ring > max_rings:
            break
        pending = {}
        for k, r in enumerate(res):
            if r["u"] is not None:
                continue
            vals = [res[m]["u"] for m in adj[k]
                    if res[m]["u"] is not None and res[m]["ring"] < ring]
            if vals:
                pending[k] = (float(np.mean(vals)), len(vals))
        if not pending:
            break
        for k, (v, n_from) in pending.items():
            res[k].update(u=v, src="neighbor_fill", ring=ring, from_n=n_from)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--osm", type=Path, required=True, help="input .osm, openings already defined")
    ap.add_argument("--voxels", type=Path, required=True, help="voxel CSV with x,y,z and a U column")
    ap.add_argument("--out", type=Path, required=True, help="output .osm")
    ap.add_argument("--osm-origin", default="0,0,0", metavar="X,Y,Z",
                    help="added to .osm coords to reach the voxel frame (session 9: "
                         "4.11,-0.80,-0.33)")
    ap.add_argument("--u-column", default="u_value_corrected_w_m2k")
    ap.add_argument("--fallback-u-column", default="",
                    help="used only where --u-column is empty; default '' (disabled). Enabling "
                         "it does NOT resurrect voxels marked non_misurabile_* -- an empty U "
                         "there is a deliberate 'no valid U exists', not a missing value.")
    ap.add_argument("--require-note", default=None, metavar="LIST",
                    help="comma-separated correction_note values to keep (default: keep all)")
    ap.add_argument("--skip-note", default="", metavar="LIST",
                    help="comma-separated correction_note values to drop")
    ap.add_argument("--tile-size", type=float, default=0.5, metavar="M",
                    help="nominal tile pitch in metres (default 0.5)")
    ap.add_argument("--max-dist", type=float, default=0.25, metavar="M",
                    help="max distance from the surface plane for a voxel to belong to it "
                         "(default 0.25)")
    ap.add_argument("--pad", type=float, default=0.10, metavar="M",
                    help="in-plane tolerance when collecting a tile's voxels (default 0.10)")
    ap.add_argument("--min-voxels", type=int, default=1,
                    help="tiles with fewer contributing voxels get no construction (default 1)")
    ap.add_argument("--u-bin", type=float, default=0.25, metavar="W_M2K",
                    help="round U to this bin so tiles share Constructions (default 0.25)")
    ap.add_argument("--fill-neighbors", action="store_true",
                    help="give tiles with no voxels the mean U of the tiles they touch, "
                         "spreading outwards ring by ring; filled tiles are renamed __NFILL<ring>, "
                         "get their own __NEIGHBOR_FILLED constructions and u_source "
                         "AdditionalProperties. Interpolated, not measured -- off by default")
    ap.add_argument("--fill-max-rings", type=int, default=0, metavar="N",
                    help="with --fill-neighbors, how far a value may travel from a real "
                         "measurement (0 = unlimited, the default)")
    ap.add_argument("--hsi", type=float, default=7.7, help="internal film used to strip Rsi (default 7.7)")
    ap.add_argument("--he", type=float, default=25.0, help="external film used to strip Rse (default 25)")
    ap.add_argument("--min-r", type=float, default=0.001, metavar="M2K_W",
                    help="layer resistance given to tiles whose U exceeds the film-only ceiling "
                         "(default 0.001)")
    ap.add_argument("--surface-types", default="Wall,RoofCeiling",
                    help="surface types to tile (default Wall,RoofCeiling)")
    ap.add_argument("--boundary", default="Outdoors",
                    help="only tile surfaces with this outside boundary condition (default Outdoors)")
    ap.add_argument("--report", type=Path, default=None, help="optional per-tile CSV report")
    args = ap.parse_args()

    origin = np.array([float(v) for v in args.osm_origin.split(",")])
    want_types = {s.strip() for s in args.surface_types.split(",") if s.strip()}
    keep_notes = ({s.strip() for s in args.require_note.split(",")}
                  if args.require_note else None)
    drop_notes = {s.strip() for s in args.skip_note.split(",") if s.strip()}

    # --- voxels ---
    rows = list(csv.DictReader(open(args.voxels, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {args.voxels}")
    if args.u_column not in rows[0]:
        raise SystemExit(f"{args.voxels} has no {args.u_column} column")
    P, U = [], []
    n_empty = n_note = 0
    fb = args.fallback_u_column if args.fallback_u_column and args.fallback_u_column in rows[0] else None
    for r in rows:
        note = r.get("correction_note", "")
        if (keep_notes is not None and note not in keep_notes) or note in drop_notes:
            n_note += 1
            continue
        raw = r[args.u_column]
        # A voxel_solar_ns.py --unmeasurable-plane-id row is blank on purpose:
        # its far side is not the outdoor air, so no U exists at all. Falling
        # back to the plain column there would reinstate exactly the number
        # that run decided was meaningless.
        if raw == "" and fb and not note.startswith("non_misurabile"):
            raw = r.get(fb, "")
        if raw == "":
            n_empty += 1
            continue
        P.append([float(r["x"]), float(r["y"]), float(r["z"])])
        U.append(float(raw))
    if not P:
        raise SystemExit("no usable voxels after filtering")
    P, U = np.array(P), np.array(U)
    print(f"{len(rows)} voxel row(s) -> {len(P)} usable "
          f"({n_empty} empty U, {n_note} filtered by note)")
    print(f"  U: min {U.min():.2f} max {U.max():.2f} mean {U.mean():.2f} W/m2K")

    u_ceiling = 1.0 / (1.0 / args.hsi + 1.0 / args.he)
    print(f"  film-only ceiling with hsi={args.hsi}, he={args.he}: "
          f"U cannot exceed {u_ceiling:.2f} W/m2K in any construction")

    model = load_model(args.osm)
    constructions = {}

    def construction_for(u, filled=False):
        # `filled` is part of the key on purpose: a neighbour-filled tile must
        # never end up sharing a Construction with a measured tile of the same
        # U, or the model would lose the only record of which is which.
        key = round(u / args.u_bin) * args.u_bin
        if (key, filled) in constructions:
            return constructions[(key, filled)], key
        r_layer = 1.0 / key - 1.0 / args.hsi - 1.0 / args.he
        unphysical = r_layer <= args.min_r
        if unphysical:
            r_layer = args.min_r
        tag = ("__NEIGHBOR_FILLED" if filled else "") + ("__UNPHYSICAL" if unphysical else "")
        mat = openstudio.model.MasslessOpaqueMaterial(model)
        mat.setRoughness("MediumRough")
        mat.setThermalResistance(float(r_layer))
        mat.setName(("neighborfill_R" if filled else "measured_R") + f"{r_layer:.4f}"
                    + ("__UNPHYSICAL" if unphysical else ""))
        layers = openstudio.model.MaterialVector()
        layers.append(mat)
        con = openstudio.model.Construction(model)
        con.setLayers(layers)
        con.setName(f"U_{key:.2f}_Wm2K" + tag)
        constructions[(key, filled)] = (con, unphysical)
        return constructions[(key, filled)], key

    report = []
    n_src = n_tile = n_host = n_noassign = n_unphys = n_filled = 0
    fill_rings = defaultdict(int)
    skipped = defaultdict(int)

    for surf in list(model.getSurfaces()):
        if surf.surfaceType() not in want_types:
            skipped[f"type={surf.surfaceType()}"] += 1
            continue
        if surf.outsideBoundaryCondition() != args.boundary:
            skipped[f"boundary={surf.outsideBoundaryCondition()}"] += 1
            continue
        const_ax, const_val = axis_of(surf.vertices())
        if const_ax is None:
            skipped["not axis-aligned"] += 1
            continue
        space = surf.space()
        if not space.is_initialized():
            skipped["no space"] += 1
            continue
        space = space.get()
        n_src += 1

        ua, va = inplane_axes(const_ax)
        u0, u1, v0, v1 = rect_of(surf.vertices(), const_ax)
        onrm = surf.outwardNormal()
        want_normal = np.array([onrm.x(), onrm.y(), onrm.z()])
        base = surf.name().get()

        subs = list(surf.subSurfaces())
        sub_rects = [rect_of(s.vertices(), const_ax) for s in subs]

        ulines = snapped_lines(u0, u1, args.tile_size,
                               [e for r in sub_rects for e in (r[0], r[1])])
        vlines = snapped_lines(v0, v1, args.tile_size,
                               [e for r in sub_rects for e in (r[2], r[3])])
        nu, nv = len(ulines) - 1, len(vlines) - 1

        def cell_span(a0, a1, lines):
            i0 = int(np.argmin([abs(x - a0) for x in lines]))
            i1 = int(np.argmin([abs(x - a1) for x in lines]))
            return min(i0, i1), max(i0, i1)

        # host rects (cell-index space), expanded one ring so the opening fits strictly inside
        hosts = []
        for (a0, a1, b0, b1) in sub_rects:
            i0, i1 = cell_span(a0, a1, ulines)
            j0, j1 = cell_span(b0, b1, vlines)
            hosts.append((max(0, i0 - 1), min(nu, i1 + 1), max(0, j0 - 1), min(nv, j1 + 1)))
        hosts = merge_rects(hosts)

        consumed = np.zeros((nu, nv), bool)
        for (i0, i1, j0, j1) in hosts:
            consumed[i0:i1, j0:j1] = True

        def u_for(a0, a1, b0, b1):
            """median U of voxels on this surface within the (u,v) rect."""
            lo = np.array([0.0, 0.0, 0.0]); hi = np.array([0.0, 0.0, 0.0])
            lo[const_ax], hi[const_ax] = const_val - args.max_dist, const_val + args.max_dist
            lo[ua], hi[ua] = a0 - args.pad, a1 + args.pad
            lo[va], hi[va] = b0 - args.pad, b1 + args.pad
            local = P - origin                       # voxel points in .osm local frame
            m = np.all((local >= lo) & (local <= hi), axis=1)
            return (float(np.median(U[m])), int(m.sum())) if m.any() else (None, 0)

        pieces = []                                   # (name, u0,u1,v0,v1, is_host, host_idx)
        for h, (i0, i1, j0, j1) in enumerate(hosts):
            pieces.append((f"{base}__host{h}", ulines[i0], ulines[i1],
                           vlines[j0], vlines[j1], True, h))
        for i in range(nu):
            for j in range(nv):
                if consumed[i, j]:
                    continue
                pieces.append((f"{base}__t{i}_{j}", ulines[i], ulines[i + 1],
                               vlines[j], vlines[j + 1], False, None))

        # phase 1: what the voxels actually measured, for every piece. A piece
        # below --min-voxels counts as unmeasured, so it can be filled and
        # cannot itself act as a donor.
        res = []
        for (name, a0, a1, b0, b1, is_host, hidx) in pieces:
            uval, cnt = u_for(a0, a1, b0, b1)
            if uval is not None and cnt < args.min_voxels:
                uval = None
            res.append({"u": uval, "n": cnt, "ring": 0, "from_n": 0,
                        "src": "measured" if uval is not None else "none"})

        # phase 2: spread measured values into the gaps (opt-in)
        if args.fill_neighbors:
            fill_from_neighbors(res, edge_adjacency([p[1:5] for p in pieces]),
                                args.fill_max_rings)

        # phase 3: build the surfaces
        new_surfaces = {}
        for (name, a0, a1, b0, b1, is_host, hidx), r in zip(pieces, res):
            filled = r["src"] == "neighbor_fill"
            if filled:
                name = f"{name}__NFILL{r['ring']}"
            pts = make_points(a0, a1, b0, b1, const_ax, const_val, want_normal)
            s = openstudio.model.Surface(pts, model)
            s.setSpace(space)
            s.setSurfaceType(surf.surfaceType())
            s.setOutsideBoundaryCondition(surf.outsideBoundaryCondition())
            s.setSunExposure(surf.sunExposure())
            s.setWindExposure(surf.windExposure())
            s.setName(name)
            uval, cnt = r["u"], r["n"]
            unphys = False
            if uval is not None:
                (con, unphys), key = construction_for(uval, filled)
                s.setConstruction(con)
                if unphys:
                    n_unphys += 1
                if filled:
                    n_filled += 1
                    fill_rings[r["ring"]] += 1
            else:
                n_noassign += 1
            # machine-readable provenance, so a reader of the .osm never has to
            # infer "measured or interpolated?" from the name alone
            props = s.additionalProperties()
            props.setFeature("u_source", r["src"])
            if uval is not None:
                props.setFeature("u_w_m2k", float(round(uval, 4)))
            if filled:
                props.setFeature("fill_ring", int(r["ring"]))
                props.setFeature("fill_from_n", int(r["from_n"]))
            if is_host:
                new_surfaces[hidx] = s
                n_host += 1
            else:
                n_tile += 1
            report.append({
                "surface": name, "parent": base, "kind": "host" if is_host else "tile",
                "u_w_m2k": "" if uval is None else round(uval, 4),
                "u_source": r["src"],
                "fill_ring": r["ring"] if filled else "",
                "fill_from_n": r["from_n"] if filled else "",
                "n_voxels": cnt, "unphysical": int(bool(unphys)),
                "area_m2": round((a1 - a0) * (b1 - b0), 4),
            })

        # re-parent the openings onto their host, then drop the original surface
        for s_sub, r in zip(subs, sub_rects):
            for h, (i0, i1, j0, j1) in enumerate(hosts):
                if (ulines[i0] - TOL <= r[0] and r[1] <= ulines[i1] + TOL and
                        vlines[j0] - TOL <= r[2] and r[3] <= vlines[j1] + TOL):
                    s_sub.setSurface(new_surfaces[h])
                    break
            else:
                print(f"  WARNING: opening {s_sub.name().get()} found no host on {base} "
                      f"-- left on the original surface, which is NOT removed")
                break
        else:
            surf.remove()

    print(f"\n{n_src} source surface(s) -> {n_host} host + {n_tile} tile surface(s)")
    for k, v in sorted(skipped.items()):
        print(f"  skipped {v} surface(s): {k}")
    print(f"{len(constructions)} construction(s) created (U binned to {args.u_bin} W/m2K)")
    if args.fill_neighbors:
        print(f"  {n_filled} piece(s) NEIGHBOUR-FILLED (interpolated, not measured): "
              + ", ".join(f"ring {k}: {v}" for k, v in sorted(fill_rings.items())))
        print(f"    named __NFILL<ring>, construction __NEIGHBOR_FILLED, "
              f"AdditionalProperties u_source=neighbor_fill")
    if n_noassign:
        print(f"  {n_noassign} piece(s) got NO construction (no voxels within "
              f"--max-dist {args.max_dist} / --min-voxels {args.min_voxels}"
              + (", and no measured piece touching them" if args.fill_neighbors else "") + ")")
    if n_unphys:
        print(f"  WARNING: {n_unphys} piece(s) have U above the {u_ceiling:.2f} W/m2K "
              f"film-only ceiling -- named __UNPHYSICAL, given R={args.min_r}. "
              f"These are not representable as an opaque construction.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(openstudio.toPath(str(args.out)), True)
    print(f"wrote {args.out}")

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
            w.writeheader()
            w.writerows(report)
        print(f"wrote {args.report} ({len(report)} piece(s))")


if __name__ == "__main__":
    main()
