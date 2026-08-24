"""Solar correction for the NORTH and SOUTH corridor walls, per voxel,
via the SOL-AIR temperature.

Takes voxel_u_value.py's output and recomputes U with the solar gain placed
where the sun physically is -- on the OUTSIDE face -- instead of adding it to
the interior surface being imaged.

Why not voxel_u_value.py's additive form
----------------------------------------
voxel_u_value.py --irradiance uses

    U = [hsi * (Tint - Tsurf) + alpha * I] / (Tint - Text)

which is right for an opaque wall whose Tsurf is measured on the SAME face
the sun hits (an exterior-side measurement). In this dataset Tsurf is the
INTERIOR surface: the sun never lands on it (bar the transmitted sunpatch,
out of scope -- see below). Adding alpha*I there credits the interior face
with an absorption that happens outside, and it shows: at transmittance 1.0
that form drove 330/562 north and 86/90 south voxels to a NEGATIVE U, and
only a transmittance around 0.35 pulled them back into a sane range -- a
value that implies solar-control/selective glazing, which this building
(wooden frames, column radiators, early-1900s corridor) does not have. A
physical parameter forced to an implausible value to make the arithmetic
work is a sign the model is misapplied, not a parameter to tune.

Sol-air form used here
----------------------
    T_sol_air = Text + (alpha * I) / he
    U         = hsi * (Tint - Tsurf) / (Tint - T_sol_air)

  alpha  solar absorptance, per voxel, from the input CSV (never invented)
  I      POA irradiance on THAT wall's own orientation, from sun_incidence.py
  he     exterior surface heat transfer coefficient, W/m2K (--he, default 25,
         the standard summer value in UNI EN ISO 6946)

The solar term now divides by he (~25) instead of needing a fudge factor, so
there is NO free parameter left to tune: he is normative, alpha comes from
emissivity_table.csv, I from pvlib. Measured effect on a south voxel
(alpha=0.65, I=169.7): T_sol_air = 36.9 + 4.4 = 41.3 C, U 7.35 -> 4.5 W/m2K,
positive and plausible, and moving in the right direction (part of the heat
was solar, not conductive).

This also removes the need for the glass/opaque special case the additive
form required: alpha already separates them (0.10 for glass, 0.65 for
painted metal), so the same formula applies to every voxel of an exterior
wall and the physics self-regulates.

Full form (--hce/--delta-r/--fsh)
---------------------------------
    T_sol_air = Text + (alpha * fsh * I) / hce - (eps * dR) / hce
    U         = hsi * (Tint - Tsurf) / (Tint - T_sol_air)

--he (default 25) is UNI EN ISO 6946's COMBINED convective+radiative
exterior coefficient (hc~20 + hr~5) -- it already has an average long-wave
loss baked in. The full form separates them: --hce is convective-only
(default 20), and --delta-r/--eps subtract the long-wave exchange with the
sky explicitly instead of leaving it folded into he. Passing --hce switches
the solar term's denominator from 25 to 20 (a ~25% LARGER solar_rise for
the same alpha*I) and requires setting --delta-r yourself for the eps*dR
term to do anything (default 0, see below) -- --he alone keeps the original
simple form.

dR is the net long-wave exchange with the sky (surface radiates to sky,
receives less back since the sky is colder than ambient air). ASHRAE's
tabulated default is dR~63-93 W/m2 for a HORIZONTAL surface (a roof, full
sky view) -- but for a VERTICAL wall the standard textbook assumption is
dR=0: the wall sees roughly half sky, half ground, and the ground radiates
close to ambient air temperature, so the two nearly cancel. All three walls
here (north, south, head) are vertical, so --delta-r defaults to 0 and the
eps*dR term contributes nothing unless you override it with a measured/
estimated sky temperature for this specific capture instant.

fsh (--fsh, default 1.0) is the self-shading fraction: 1.0 means the whole
alpha*I lands unobstructed. This dataset has NO shading geometry computed
(the recessed window reveals and pillars could partially self-shade some
wall voxels at this sun angle) -- 1.0 is "shading not modeled", not "no
shading exists". Needs its own reveal-depth-vs-sun-angle analysis to set
per voxel; left uniform here.

Glazing is NOT sol-air (--glass-materials)
-------------------------------------------
Sol-air, in EITHER form above, is a conduction model: it collapses exterior
convection+radiation into one equivalent driving temperature for
hsi*(Tint-Tsurf) through an OPAQUE mass. Glass doesn't work that way -- most
of the incident beam TRANSMITTS through it (governed by SHGC, a completely
different mechanism), not conducts through it. Feeding glass's alpha=0.10
into sol-air (an earlier version of this script did) uses the right formula
on the wrong material: alpha=0.10 correctly describes what the pane itself
absorbs, but sol-air then treats that absorbed sliver as if it drove
conduction to an interior face the way solar-heated masonry does, which
isn't the physical mechanism for a window at all.
--glass-materials (default "glass") lists materials skipped by the solar
term entirely: those voxels get the PLAIN formula (I=0 in T_sol_air, i.e.
T_sol_air=Text) and correction_note flags them as unmodeled, not silently
corrected with a model that does not apply. A real glazing treatment needs
an SHGC-based transmission model -- out of scope here, same as the
transmitted sunpatch below.

Out of scope, and still a declared limit of the method: the sunpatch
TRANSMITTED through the glazing onto interior surfaces (floor, opposite
wall). Sol-air models a wall's own exterior absorption, not redistributed
transmitted beam -- that needs raytracing regardless of which materials
sol-air is applied to.

Which voxels
------------
Geometric, by distance to the plane in planes.json (not the sign of y), same
criterion as voxel_u_value.py --correction-mode plane. Walls face 180 deg
apart (north ~13, south ~193) and see very different irradiance, so each
gets its own POA -- which is the whole reason this script exists, since
voxel_u_value.py's --plane-id is a scalar.

fit_planes.py does not orient its normals consistently -- plane 1's points
INTO the room, so sun_incidence.py would read its bearing 180 deg wrong and
hand the north wall the south wall's azimuth. Every plane is re-oriented to
point away from the scene interior before being handed to sun_incidence.py,
via a temporary planes.json. sun_incidence.py itself is not modified and
stays the only place the pvlib pipeline lives.

Usage:
    py voxel_solar_ns.py --in <session>/voxel_map/thermal_voxels_u.csv \\
        --tint 29 --text 36.9 --hsi 8

Needs only numpy; the subprocess needs pvlib, so --python defaults to
C:\\venvs\\planefit (sun_incidence.py's venv) when it exists.
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PLANES = HERE.parent / "OpenStudioModel" / "planes.json"
DEFAULT_SUN = HERE.parent / "OpenStudioModel" / "sun_incidence.py"
DEFAULT_GHI = HERE.parent / "OpenStudioModel" / "legnaro_ghi.csv"
PLANEFIT_PY = Path(r"C:\venvs\planefit\Scripts\python.exe")


def outward_normal(plane, interior):
    """(unit normal, d) re-oriented to point away from `interior`."""
    n = np.asarray(plane["normal"], dtype=float)
    d = float(plane["d"])
    k = np.linalg.norm(n)
    n, d = n / k, d / k
    centre = np.asarray(plane["center_3d"], dtype=float)
    if float(n @ (centre - interior)) < 0.0:
        n, d = -n, -d
    return n, d


def run_sun_incidence(args, planes_path, plane_id, surface_tilt=90.0):
    """Call sun_incidence.py for one plane at one instant, return its row.

    Asks for a 2-step window and keeps the first row: sun_incidence.py
    derives its integration step from times[1] and raises IndexError on a
    single timestep.

    surface_tilt: 90 (default) for a vertical wall, 0 for a horizontal
    ceiling/roof facing up (surface_azimuth is irrelevant at tilt=0)."""
    end = (datetime.fromisoformat(args.instant) + timedelta(minutes=1)).isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / f"plane{plane_id}_instant.csv"
        cmd = [str(args.python), str(args.sun_incidence),
               "--planes", str(planes_path), "--plane-id", str(plane_id),
               "--surface-tilt", str(surface_tilt),
               "--north-offset-deg", str(args.north_offset_deg),
               "--lat", str(args.lat), "--lon", str(args.lon), "--tz", args.tz,
               "--altitude", str(args.altitude),
               "--start", args.instant, "--end", end, "--freq", "1min",
               "--model", args.model, "--iam-model", args.iam_model,
               "--out", str(out_csv)]
        if args.irradiance_csv:
            cmd += ["--irradiance-csv", str(args.irradiance_csv)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"sun_incidence.py failed for plane {plane_id}:\n"
                             f"{proc.stdout}\n{proc.stderr}")
        for line in proc.stdout.splitlines():
            if line.startswith("plane ") or line.startswith("irradiance from"):
                print(f"  [sun_incidence] {line}")
        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"sun_incidence.py returned no timestep for plane {plane_id}")
    return {k: (v if k == "timestamp" else float(v)) for k, v in rows[0].items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="voxel_u_value.py's thermal_voxels_u.csv (needs x,y,z,t_mean_c,"
                         "solar_absorptance,u_value_w_m2k)")
    ap.add_argument("--tint", type=float, required=True, help="indoor air temperature, deg C")
    ap.add_argument("--text", type=float, required=True, help="outdoor air temperature, deg C")
    ap.add_argument("--hsi", type=float, default=None,
                    help="internal surface heat transfer coefficient, W/m2K, uniform for every "
                         "voxel. Must match the voxel_u_value.py run that produced --in -- the "
                         "uncorrected voxels are recomputed and cross-checked against its "
                         "u_value_w_m2k column. Omit if --in has its own 'hsi_used' column "
                         "(per-surface-type ISO 6946 values, e.g. wall 7.7/floor 5.9/ceiling "
                         "10.0) -- that column is used per-voxel instead, and is more correct "
                         "than a single --hsi wherever floor/ceiling voxels get a solar term "
                         "too. Required if --in has no 'hsi_used' column.")
    ap.add_argument("--he", type=float, default=None, metavar="W_M2K",
                    help="COMBINED (convective+radiative) exterior coefficient, simple sol-air "
                         "form: T_sol_air = Text + alpha*I/he (default 25, UNI EN ISO 6946 "
                         "summer value). Mutually exclusive with --hce/--delta-r (the full form) "
                         "-- if neither is given, --he=25 is used.")
    ap.add_argument("--hce", type=float, default=None, metavar="W_M2K",
                    help="CONVECTIVE-only exterior coefficient, full sol-air form: T_sol_air = "
                         "Text + alpha*fsh*I/hce - eps*dR/hce (default when used: 20). Passing "
                         "this switches to the full form -- see module docstring for why this "
                         "isn't just --he with a different number.")
    ap.add_argument("--delta-r", type=float, default=0.0, metavar="W_M2",
                    help="net long-wave exchange with the sky, full form only (default 0, the "
                         "standard assumption for a VERTICAL wall -- see module docstring; "
                         "ASHRAE's horizontal/roof default is 63-93, do not reuse that here "
                         "without a reason).")
    ap.add_argument("--eps", type=float, default=None, metavar="0-1",
                    help="long-wave emissivity for the eps*dR term, full form only (default: "
                         "each voxel's own material emissivity if the input CSV has an "
                         "'emissivity' column, else 0.90 as a flat fallback)")
    ap.add_argument("--fsh", type=float, default=1.0, metavar="0-1",
                    help="self-shading fraction, full form only (default 1.0 = unobstructed -- "
                         "shading geometry is NOT modeled here, see module docstring)")
    ap.add_argument("--glass-materials", default="glass", metavar="LIST",
                    help="comma-separated materials sol-air is NOT applied to (default 'glass') "
                         "-- these get the plain no-solar formula and a distinct correction_note "
                         "instead, since sol-air is a conduction model and does not describe "
                         "transmission through glazing. See module docstring. Empty string "
                         "applies sol-air to every material (not recommended).")
    # --- geometry ----------------------------------------------------------
    ap.add_argument("--planes", type=Path, default=DEFAULT_PLANES,
                    help=f"fit_planes.py planes.json, SAME SESSION as --in (default {DEFAULT_PLANES})")
    ap.add_argument("--north-plane-id", type=int, default=1,
                    help="north wall, azimuth ~13 deg (default 1)")
    ap.add_argument("--south-plane-id", type=int, default=4,
                    help="south wall, azimuth ~193 deg (default 4)")
    ap.add_argument("--done-plane-id", type=int, default=3,
                    help="head wall already corrected in its own run -- copied through "
                         "untouched (default 3). -1 disables. See --redo-done-plane.")
    ap.add_argument("--redo-done-plane", action="store_true",
                    help="also recompute --done-plane-id in sol-air form instead of copying it "
                         "through. Off by default (the standing instruction is to leave it "
                         "alone), but note that copying it through leaves ONE column holding two "
                         "different physics: the additive alpha*I form for that plane and the "
                         "sol-air form everywhere else.")
    ap.add_argument("--plane-threshold", type=float, default=0.15, metavar="M",
                    help="max distance (m) from a plane for a voxel to belong to it "
                         "(default 0.15, same as voxel_u_value.py)")
    ap.add_argument("--room-bbox", action=argparse.BooleanOptionalAction, default=True,
                    help="drop voxels outside the x/y footprint of --floor-plane-id's floor "
                         "plane (expanded by --plane-threshold) before any plane mask is "
                         "computed -- removes through-glass LiDAR returns that the "
                         "infinite-plane distance test would otherwise count as 'on' a wall. "
                         "Default on.")
    ap.add_argument("--floor-plane-id", type=int, default=0,
                    help="id of the floor plane whose corners_3d define the room footprint "
                         "(default 0)")
    ap.add_argument("--unmeasurable-plane-id", type=int, action="append", default=[],
                    metavar="ID",
                    help="plane whose voxels CANNOT yield a valid U with this method: its "
                         "far-side boundary is not the outdoor air at --text, so Tsurf is not "
                         "bounded by [Tint, Text] and the formula has no steady-state meaning "
                         "there. Repeatable. Their u_value_corrected_w_m2k is written EMPTY and "
                         "correction_note is set to non_misurabile_<id>, rather than emitting a "
                         "number that looks measured but isn't. They are also left out of the "
                         "summary statistics. Typical use: the floor (id 0), whose underside is "
                         "the room below at roughly Tint, not outdoor air -- see README.")
    ap.add_argument("--ceiling-plane-id", type=int, default=-1,
                    help="id of the ceiling/roof plane to sol-air correct as a horizontal "
                         "(surface_tilt=0) surface -- open sky view, same as a flat roof. "
                         "-1 disables (default): ceiling voxels then get no solar term, same "
                         "as floor/other. Always solved for in sol-air form (no 'already "
                         "corrected, copy through' mode like --done-plane-id, since there is no "
                         "prior run to copy from).")
    # --- sun position / irradiance ----------------------------------------
    ap.add_argument("--instant", default="2026-07-30T18:13:00",
                    help="local representative capture instant (default 2026-07-30T18:13:00). "
                         "The real window is 18:12:23.9-18:14:30.2 and the solar zenith moves "
                         "<0.2 deg across it, so one instant stands in for the session.")
    ap.add_argument("--north-offset-deg", type=float, default=193.0,
                    help="true compass bearing of the SLAM +Y axis (default 193, Session 9)")
    ap.add_argument("--lat", type=float, default=45.405, help="site latitude, deg (default 45.405)")
    ap.add_argument("--lon", type=float, default=11.875, help="site longitude, deg (default 11.875)")
    ap.add_argument("--tz", default="Europe/Rome", help="IANA timezone (default Europe/Rome)")
    ap.add_argument("--altitude", type=float, default=0.0, help="site elevation, m (default 0)")
    ap.add_argument("--model", default="perez",
                    choices=["isotropic", "klucher", "haydavies", "reindl", "king", "perez"],
                    help="sky-diffuse transposition model (default perez)")
    ap.add_argument("--iam-model", default="physical",
                    choices=["physical", "ashrae", "martin_ruiz", "none"],
                    help="glass incidence-angle modifier for the beam component (default physical)")
    ap.add_argument("--irradiance-csv", type=Path, default=DEFAULT_GHI,
                    help=f"measured timestamp,ghi CSV for sun_incidence.py (default {DEFAULT_GHI})")
    # --- escape hatches ----------------------------------------------------
    ap.add_argument("--north-poa", type=float, default=None, metavar="W_M2",
                    help="use this poa_effective for the north wall instead of running "
                         "sun_incidence.py")
    ap.add_argument("--south-poa", type=float, default=None, metavar="W_M2",
                    help="use this poa_effective for the south wall instead of running "
                         "sun_incidence.py")
    ap.add_argument("--sun-incidence", type=Path, default=DEFAULT_SUN,
                    help=f"path to sun_incidence.py (default {DEFAULT_SUN})")
    ap.add_argument("--python", type=Path, default=None,
                    help="interpreter for the sun_incidence.py subprocess -- it needs pvlib "
                         f"(default {PLANEFIT_PY} if it exists, else this interpreter)")
    ap.add_argument("--out", type=Path, default=None, help="default: <in>_solair.csv next to the input")
    ap.add_argument("--out-ply", type=Path, default=None, help="default: <in>_solair.ply next to the input")
    args = ap.parse_args()

    if abs(args.tint - args.text) < 1e-9:
        raise SystemExit("--tint and --text are equal -- U is undefined (division by zero)")
    if args.he is not None and args.hce is not None:
        raise SystemExit("--he and --hce are mutually exclusive -- pick the simple form (--he) "
                         "or the full form (--hce/--delta-r/--fsh), see module docstring")
    if args.hce is None:
        # simple form: T_sol_air = Text + alpha*I/he
        h_denom, use_full_form = (args.he if args.he is not None else 25.0), False
    else:
        # full form: T_sol_air = Text + alpha*fsh*I/hce - eps*dR/hce
        h_denom, use_full_form = args.hce, True
    if h_denom <= 0:
        raise SystemExit(f"the exterior coefficient must be positive, got {h_denom}")
    if not (0.0 <= args.fsh <= 1.0):
        raise SystemExit(f"--fsh must be in [0,1], got {args.fsh}")
    glass_materials = {m.strip() for m in args.glass_materials.split(",") if m.strip()}
    if args.python is None:
        args.python = PLANEFIT_PY if PLANEFIT_PY.exists() else Path(sys.executable)

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")
    for col in ("x", "y", "z", "t_mean_c", "solar_absorptance", "u_value_w_m2k"):
        if col not in rows[0]:
            raise SystemExit(f"{args.inp} has no {col} column -- run voxel_u_value.py first")
    has_hsi_col = "hsi_used" in rows[0]
    if not has_hsi_col and args.hsi is None:
        raise SystemExit(f"--hsi is required: {args.inp} has no 'hsi_used' column to read it from")
    if has_hsi_col and args.hsi is not None:
        print(f"NOTE: {args.inp.name} has its own hsi_used column (per-surface-type) -- using "
              f"that per-voxel, ignoring --hsi {args.hsi}")

    plane_data = json.loads(args.planes.read_text())
    by_id = {p["id"]: p for p in plane_data["planes"]}

    if args.room_bbox:
        if args.floor_plane_id not in by_id:
            raise SystemExit(f"no plane with id {args.floor_plane_id} in {args.planes}")
        corners = np.array(by_id[args.floor_plane_id]["corners_3d"])
        (rx0, ry0), (rx1, ry1) = corners[:, :2].min(0), corners[:, :2].max(0)
        rx0, ry0 = rx0 - args.plane_threshold, ry0 - args.plane_threshold
        rx1, ry1 = rx1 + args.plane_threshold, ry1 + args.plane_threshold
        n_all = len(rows)
        rows = [r for r in rows
                if rx0 <= float(r["x"]) <= rx1 and ry0 <= float(r["y"]) <= ry1]
        print(f"--room-bbox: floor plane {args.floor_plane_id} footprint x[{rx0:.2f} {rx1:.2f}] "
              f"y[{ry0:.2f} {ry1:.2f}] -- dropped {n_all - len(rows)}/{n_all} voxel(s) outside it")
        if not rows:
            raise SystemExit("--room-bbox left no voxels -- check --planes/--floor-plane-id")

    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    tsurf = np.array([float(r["t_mean_c"]) for r in rows])
    alpha = np.array([float(r["solar_absorptance"]) if r["solar_absorptance"] != "" else 0.0
                      for r in rows])
    u_in = np.array([float(r["u_value_w_m2k"]) for r in rows])
    hsi = (np.array([float(r["hsi_used"]) for r in rows]) if has_hsi_col
          else np.full(len(rows), args.hsi))
    material = np.array([r.get("material", "") for r in rows])
    is_glass = np.isin(material, list(glass_materials)) if glass_materials else np.zeros(len(rows), bool)
    if args.eps is not None:
        eps = np.full(len(rows), args.eps)
    elif "emissivity" in rows[0]:
        eps = np.array([float(r["emissivity"]) if r["emissivity"] != "" else 0.90 for r in rows])
    else:
        eps = np.full(len(rows), 0.90)   # flat fallback, no per-voxel emissivity column available
    # Present when the input already had a solar term applied (voxel_u_value.py
    # --irradiance). Used only to know which rows are a clean plain-formula
    # baseline for the cross-check below.
    st_in = np.array([float(r.get("solar_term_w_m2") or 0.0) for r in rows]) \
        if "solar_term_w_m2" in rows[0] else np.zeros(len(rows))

    # --- plane membership --------------------------------------------------
    interior = np.mean([p["center_3d"] for p in plane_data["planes"]], axis=0)

    def mask_for(pid):
        if pid not in by_id:
            raise SystemExit(f"no plane with id {pid} in {args.planes}")
        n, d = outward_normal(by_id[pid], interior)
        return np.abs(xyz @ n + d) < args.plane_threshold

    on_done = mask_for(args.done_plane_id) if args.done_plane_id >= 0 else np.zeros(len(rows), bool)
    if args.redo_done_plane:
        on_done_keep = np.zeros(len(rows), bool)
        on_done_solair = on_done & ~is_glass
    else:
        on_done_keep = on_done
        on_done_solair = np.zeros(len(rows), bool)
    # Priority chain done > north > south > ceiling, made mutually exclusive
    # by construction (each excludes every higher-priority mask already
    # assigned) so the later irr[]/note[] writes below can't disagree about
    # which wall a corner voxel belongs to -- ~on_done_keep alone used to let
    # on_north/on_south/on_ceiling double-claim a voxel already on_done_solair
    # (--redo-done-plane path), so irr[] (last-write-wins there) and note[]
    # (different last-write-wins order) picked DIFFERENT walls for the same
    # voxel: a real corner-voxel mismatch, not just a display glitch.
    # Unmeasurable planes are claimed FIRST, ahead of every solar role: a voxel
    # on one of them gets no U at all, so it must not also be counted as a
    # corrected wall voxel at a corner.
    on_unmeas = np.zeros(len(rows), bool)
    for pid in args.unmeasurable_plane_id:
        on_unmeas |= mask_for(pid)
    on_done &= ~on_unmeas
    on_done_keep &= ~on_unmeas
    on_done_solair &= ~on_unmeas

    on_north = mask_for(args.north_plane_id) & ~on_unmeas & ~on_done & ~is_glass
    on_south = mask_for(args.south_plane_id) & ~on_unmeas & ~on_done & ~on_north & ~is_glass
    on_ceiling = (mask_for(args.ceiling_plane_id) & ~on_unmeas & ~on_done & ~on_north
                  & ~on_south & ~is_glass
                 if args.ceiling_plane_id >= 0 else np.zeros(len(rows), bool))
    # Glass on an exterior wall gets neither sol-air (see module docstring)
    # nor a plain U it can be judged by without saying so -- flagged
    # separately so it is visibly "not modeled", not silently wrong.
    on_wall_glass = (mask_for(args.north_plane_id) | mask_for(args.south_plane_id) |
                     (mask_for(args.done_plane_id) if args.done_plane_id >= 0 else False) |
                     (mask_for(args.ceiling_plane_id) if args.ceiling_plane_id >= 0 else False)) & \
                    is_glass & ~on_unmeas & ~on_done_keep

    print(f"{len(rows)} voxel(s) from {args.inp.name}, plane threshold {args.plane_threshold} m")
    for pid in args.unmeasurable_plane_id:
        print(f"  plane {pid} (UNMEASURABLE): {int(mask_for(pid).sum())} -- no U written, "
              f"far-side boundary is not --text")
    print(f"  plane {args.done_plane_id} (head wall):  {int(on_done.sum())} "
          f"({'recomputed in sol-air' if args.redo_done_plane else 'copied through untouched'})")
    print(f"  plane {args.north_plane_id} (north):      {int(on_north.sum())}")
    print(f"  plane {args.south_plane_id} (south):      {int(on_south.sum())}")
    if args.ceiling_plane_id >= 0:
        print(f"  plane {args.ceiling_plane_id} (ceiling, tilt=0):  {int(on_ceiling.sum())}")
    print(f"  glass on an exterior wall (not modeled, {sorted(glass_materials)}): "
          f"{int(on_wall_glass.sum())}")
    print(f"  no solar term (floor/ceiling/other):  "
          f"{int((~(on_unmeas | on_done | on_north | on_south | on_ceiling | on_wall_glass)).sum())}")

    # --- POA per wall ------------------------------------------------------
    poa = {}
    need_sun = (args.north_poa is None and on_north.any()) or \
               (args.south_poa is None and (on_south.any() or on_done_solair.any())) or \
               on_ceiling.any()
    if need_sun:
        oriented = json.loads(args.planes.read_text())
        for p in oriented["planes"]:
            n, d = outward_normal(p, interior)
            p["normal"], p["d"] = [float(c) for c in n], float(d)
        with tempfile.TemporaryDirectory() as tmp:
            planes_out = Path(tmp) / "planes_outward.json"
            planes_out.write_text(json.dumps(oriented))
            print(f"sun_incidence.py at {args.instant} ({args.tz}), model={args.model}, "
                  f"iam={args.iam_model}:")
            for label, pid, override in (("north", args.north_plane_id, args.north_poa),
                                         ("south", args.south_plane_id, args.south_poa)):
                poa[label] = ({"poa_effective_w_m2": override} if override is not None
                              else run_sun_incidence(args, planes_out, pid))
            if on_done_solair.any():
                poa["done"] = run_sun_incidence(args, planes_out, args.done_plane_id)
            if on_ceiling.any():
                poa["ceiling"] = run_sun_incidence(args, planes_out, args.ceiling_plane_id,
                                                   surface_tilt=0.0)
    for label, override in (("north", args.north_poa), ("south", args.south_poa)):
        if label not in poa:
            poa[label] = {"poa_effective_w_m2": override if override is not None else 0.0}

    for label in [k for k in ("north", "south", "done", "ceiling") if k in poa]:
        r = poa[label]
        if "aoi_deg" not in r:
            print(f"{label}: poa_effective={r['poa_effective_w_m2']:.1f} W/m2 (given)")
            continue
        if r["solar_zenith_deg"] > 87.0:
            print(f"WARNING: solar zenith {r['solar_zenith_deg']:.1f} deg -- the Perez "
                  f"transposition is numerically unstable this close to the horizon")
        print(f"{label}: aoi={r['aoi_deg']:.1f} deg  poa_direct={r['poa_direct_w_m2']:.1f}  "
              f"poa_diffuse={r['poa_diffuse_w_m2']:.1f}  iam={r['iam']:.3f}  "
              f"-> poa_effective={r['poa_effective_w_m2']:.1f} W/m2")
    if on_north.any() and poa["north"].get("poa_direct_w_m2", 0.0) > 0.0:
        print(f"WARNING: the north wall got a non-zero beam component "
              f"({poa['north']['poa_direct_w_m2']:.1f} W/m2) -- at this instant the sun should "
              f"be behind it (aoi > 90 deg). Check --north-offset-deg and --instant.")

    # --- sol-air U ---------------------------------------------------------
    # Simple:  T_sol_air = Text + alpha*I/he
    # Full:    T_sol_air = Text + alpha*fsh*I/hce - eps*dR/hce   (--hce given)
    # Per voxel (alpha, eps vary; I is per wall). Voxels on no exterior wall,
    # or glass on one (see on_wall_glass above), keep I=0, i.e. T_sol_air =
    # Text and the plain formula, unchanged.
    irr = np.zeros(len(rows))
    irr[on_north] = poa["north"]["poa_effective_w_m2"]
    irr[on_south] = poa["south"]["poa_effective_w_m2"]
    if on_done_solair.any():
        irr[on_done_solair] = poa["done"]["poa_effective_w_m2"]
    if on_ceiling.any():
        irr[on_ceiling] = poa["ceiling"]["poa_effective_w_m2"]

    corrected = on_north | on_south | on_done_solair | on_ceiling
    if use_full_form:
        # eps*dR is gated by `corrected` too: it's the wall's own long-wave
        # loss, meaningless (and, with a nonzero --delta-r, wrongly nonzero)
        # for voxels that were never given a solar gain term in the first
        # place -- floor/ceiling, glass, the untouched head wall.
        solar_rise = np.where(corrected, (alpha * args.fsh * irr - eps * args.delta_r) / h_denom, 0.0)
    else:
        solar_rise = alpha * irr / h_denom          # K added to Text; irr already 0 off-wall
    t_sol_air = args.text + solar_rise
    denom = args.tint - t_sol_air
    conduction = hsi * (args.tint - tsurf)

    # Guard: T_sol_air crossing Tint makes the denominator cross zero and U
    # explode. Physically it means the surface's driving temperature equals
    # the indoor air -- no gradient, U is undefined there, not infinite.
    degenerate = np.abs(denom) < 1e-6
    u_corr = np.where(degenerate, np.nan, conduction / np.where(degenerate, 1.0, denom))
    if degenerate.any():
        print(f"WARNING: {int(degenerate.sum())} voxel(s) have T_sol_air within 1e-6 of Tint "
              f"-- U undefined there, written as empty")

    # plane 3 keeps whatever its own run produced, unless --redo-done-plane
    u_corr[on_done_keep] = u_in[on_done_keep]
    solar_rise[on_done_keep] = np.nan

    note = np.full(len(rows), "", dtype=object)
    note[on_done_keep] = "gia_corretto_id3"
    note[on_done_solair] = "testa_solair"
    note[on_north] = "nord_solair"
    note[on_south] = "sud_solair"
    note[on_ceiling] = "soffitto_solair"
    note[on_wall_glass] = "vetro_non_modellato"   # sol-air skipped, see module docstring
    # Unmeasurable last: it overrides every other label, and blanks the U so no
    # number is emitted that could be mistaken for a measurement.
    for pid in args.unmeasurable_plane_id:
        m = mask_for(pid)
        note[m] = f"non_misurabile_{pid}"
        u_corr[m] = np.nan
        solar_rise[m] = np.nan

    # Cross-check: where no solar term is applied by EITHER run, the recomputed
    # plain U must reproduce the input column. A mismatch means --tint/--text/
    # --hsi differ from the voxel_u_value.py run that made --in.
    untouched = ~(on_unmeas | on_done | on_north | on_south | on_ceiling) & (st_in == 0)
    if untouched.any():
        drift = np.nanmax(np.abs(u_corr[untouched] - u_in[untouched]))
        if drift > 1e-3:
            print(f"WARNING: recomputing the uncorrected voxels drifts up to {drift:.4f} W/m2K "
                  f"from the input's u_value_w_m2k -- --tint/--text/--hsi do not match the "
                  f"voxel_u_value.py run that produced {args.inp.name}")
        else:
            print(f"baseline check: uncorrected voxels reproduce u_value_w_m2k to {drift:.2e} W/m2K")
    if on_done_keep.any() and (st_in[on_done_keep] > 0).any():
        print(f"NOTE: {int((st_in[on_done_keep] > 0).sum())} plane-{args.done_plane_id} voxel(s) "
              f"carry the OLD additive alpha*I correction from their own run. They are copied "
              f"through as instructed, so u_value_corrected_w_m2k mixes two formulations. "
              f"--redo-done-plane recomputes them in sol-air for a consistent column.")

    for r, ui, sr, ta, nt in zip(rows, u_corr, solar_rise, t_sol_air, note):
        r["t_sol_air_c"] = "" if np.isnan(sr) else round(float(ta), 3)
        r["solar_rise_k"] = "" if np.isnan(sr) else round(float(sr), 3)
        r["u_value_corrected_w_m2k"] = "" if np.isnan(ui) else round(float(ui), 4)
        r["correction_note"] = nt

    out = args.out or args.inp.with_name(args.inp.stem + "_solair.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    form_desc = (f"full form, hce={h_denom} W/m2K, delta_r={args.delta_r} W/m2, fsh={args.fsh}"
                if use_full_form else f"simple form, he={h_denom} W/m2K")
    print(f"\nsol-air applied to {int(corrected.sum())}/{len(rows)} voxel(s) ({form_desc})")
    if on_wall_glass.any():
        print(f"  {int(on_wall_glass.sum())} glass voxel(s) on an exterior wall skipped "
              f"(correction_note=vetro_non_modellato) -- plain formula, not sol-air")
    if corrected.any():
        sr = solar_rise[corrected]
        print(f"  T_sol_air rise over Text: min {np.nanmin(sr):.2f}, max {np.nanmax(sr):.2f}, "
              f"mean {np.nanmean(sr):.2f} K")
        for label, m in (("nord_solair", on_north), ("sud_solair", on_south),
                         ("testa_solair", on_done_solair), ("soffitto_solair", on_ceiling)):
            if not m.any():
                continue
            ok = np.isfinite(u_corr[m])
            sane = ok & (u_corr[m] >= 0) & (u_corr[m] <= 5)
            print(f"  {label}: U {np.nanmean(u_in[m]):.3f} -> {np.nanmean(u_corr[m]):.3f} W/m2K "
                  f"(mean), {int(sane.sum())}/{int(m.sum())} in [0,5]")
    fin = np.isfinite(u_corr)
    if on_unmeas.any():
        print(f"  {int(on_unmeas.sum())} voxel(s) on unmeasurable plane(s) "
              f"{args.unmeasurable_plane_id}: U left EMPTY, excluded from the stats below")
    print(f"U corrected: min {np.nanmin(u_corr):.3f}, max {np.nanmax(u_corr):.3f}, "
          f"mean {np.nanmean(u_corr):.3f}, median {np.nanmedian(u_corr):.3f} W/m2K "
          f"({int(fin.sum())}/{len(rows)} defined)")
    sane = fin & (u_corr >= 0) & (u_corr <= 5)
    print(f"  in [0,5] W/m2K: {int(sane.sum())}/{int(fin.sum())} of the measurable voxels "
          f"({int(sane.sum())/max(1,int(fin.sum())):.1%})")
    print(f"wrote {out}")

    out_ply = args.out_ply or args.inp.with_name(args.inp.stem + "_solair.ply")
    uplot = np.where(fin, u_corr, np.nanmedian(u_corr))
    lo_u, hi_u = np.percentile(uplot, [5, 95])
    norm = np.clip((uplot - lo_u) / max(1e-9, hi_u - lo_u), 0, 1)
    rgb = (np.stack([norm, np.zeros_like(norm), 1.0 - norm], 1) * 255).astype(int)
    with open(out_ply, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(xyz, rgb):
            f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")
    print(f"wrote {out_ply} (blue=low U, red=high U, 5-95th percentile)")


if __name__ == "__main__":
    main()
