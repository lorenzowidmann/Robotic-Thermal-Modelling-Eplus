"""Render an OpenStudio .osm's actual geometry, straight from the model.

show_boxes.py draws boxes.json; fit_openings.py draws its own rectangles. This
draws what is really IN the .osm -- every Surface and SubSurface, with the
vertices OpenStudio itself reports -- so what you see is what EnergyPlus will
read, not an upstream intermediate that is merely supposed to match it.

Surfaces are coloured by type (Wall / Floor / RoofCeiling) and sub-surfaces by
sub-surface type (Door red, every glazing class blue), the same convention
WindowsDoorsDetection uses end to end.

Vertices come back in the SPACE's coordinate system, so each surface is pushed
through its space's transformation before being drawn -- skipping that puts
every space at the origin on top of the others.

Usage:
    C:\\venvs\\planefit\\Scripts\\python.exe show_osm.py ^
        OpenStudioModel\\session9_consensus.osm --screenshot model.png

    ... --no-roof          :: drop the ceilings, to see inside
    ... --wireframe        :: surfaces as edges only

Venv: C:\\venvs\\planefit (openstudio + pyvista).
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import openstudio
import pyvista as pv

SURFACE_COLORS = {
    "Wall": (196, 178, 128),
    "Floor": (110, 110, 118),
    "RoofCeiling": (150, 70, 60),
}
SUBSURFACE_COLORS = {"Door": (220, 40, 40)}
GLAZING_COLOR = (40, 110, 230)      # every window-ish sub-surface type


def parse_args():
    p = argparse.ArgumentParser(description="Render an .osm's surfaces and sub-surfaces")
    p.add_argument("osm", help="Path to the .osm")
    p.add_argument("--screenshot", default=None, metavar="PNG")
    p.add_argument("--no-roof", action="store_true",
                   help="Hide RoofCeiling surfaces, so the interior is visible.")
    p.add_argument("--wireframe", action="store_true",
                   help="Draw surfaces as edges only; sub-surfaces stay solid.")
    p.add_argument("--opacity", type=float, default=0.55, metavar="A")
    p.add_argument("--window-size", default="1600,1000", metavar="W,H")
    p.add_argument("--azimuth", type=float, default=None, metavar="DEG",
                   help="Camera azimuth for the screenshot; default is an isometric view.")
    return p.parse_args()


def poly(vertices):
    """openstudio Point3dVector -> a pyvista polygon."""
    pts = np.array([[v.x(), v.y(), v.z()] for v in vertices], dtype=float)
    if len(pts) < 3:
        return None
    faces = np.hstack([[len(pts)], np.arange(len(pts))])
    return pv.PolyData(pts, faces=faces)


def main():
    args = parse_args()
    path = Path(args.osm)
    model = openstudio.osversion.VersionTranslator().loadModel(
        openstudio.path(str(path)))
    if not model.is_initialized():
        raise SystemExit(f"{path} did not load as an OpenStudio model.")
    model = model.get()

    w, h = (int(v) for v in args.window_size.split(","))
    pl = pv.Plotter(off_screen=bool(args.screenshot), window_size=(w, h))
    pl.set_background("white")

    n_surf, n_sub = Counter(), Counter()
    for surface in model.getSurfaces():
        stype = surface.surfaceType()
        if args.no_roof and stype == "RoofCeiling":
            continue
        space = surface.space()
        # Surface vertices are in space coordinates; without the space's own
        # transformation every space would be drawn stacked at the origin.
        tf = space.get().transformation() if space.is_initialized() \
            else openstudio.Transformation()
        mesh = poly(tf * surface.vertices())
        if mesh is None:
            continue
        colour = SURFACE_COLORS.get(stype, (160, 160, 160))
        if args.wireframe:
            pl.add_mesh(mesh, color=colour, style="wireframe", line_width=2)
        else:
            pl.add_mesh(mesh, color=colour, opacity=args.opacity,
                        show_edges=True, edge_color=(70, 70, 70), line_width=1)
        n_surf[stype] += 1

        for sub in surface.subSurfaces():
            sub_mesh = poly(tf * sub.vertices())
            if sub_mesh is None:
                continue
            st = sub.subSurfaceType()
            colour = SUBSURFACE_COLORS.get(st, GLAZING_COLOR)
            # Lifted a millimetre along the wall normal: a sub-surface is exactly
            # coplanar with its parent, and coplanar faces z-fight into a
            # speckled mess at any distance.
            nrm = np.array(sub_mesh.face_normals[0], dtype=float)
            sub_mesh = sub_mesh.translate(nrm * 0.001, inplace=False)
            pl.add_mesh(sub_mesh, color=colour, opacity=0.95, show_edges=True,
                        edge_color="black", line_width=2)
            n_sub[st] += 1

    print(f"{path.name}: {len(model.getSpaces())} space(s)")
    print("  surfaces:     " + ", ".join(f"{k}={v}" for k, v in sorted(n_surf.items())))
    print("  sub-surfaces: " + (", ".join(f"{k}={v}" for k, v in sorted(n_sub.items()))
                                or "none"))
    area = sum(s.grossArea() for s in model.getSubSurfaces())
    wall = sum(s.grossArea() for s in model.getSurfaces() if s.surfaceType() == "Wall")
    if wall:
        print(f"  opening area {area:.1f} m2 of {wall:.1f} m2 wall -> WWR {100*area/wall:.1f}%")

    for st, n in sorted(n_sub.items()):
        pl.add_mesh(pv.PolyData(np.zeros((1, 3))), opacity=0.0,
                    color=SUBSURFACE_COLORS.get(st, GLAZING_COLOR), label=f"{st} ({n})")
    if n_sub:
        pl.add_legend(bcolor="white", border=True, loc="upper right", size=(0.15, 0.10))
    pl.show_axes()
    pl.add_text(f"{path.name}", font_size=9, color="black")
    if args.azimuth is not None:
        pl.camera.azimuth = args.azimuth

    if args.screenshot:
        pl.show(screenshot=args.screenshot)
        print(f"Wrote {args.screenshot}")
    else:
        pl.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
