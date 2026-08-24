"""Per-voxel U-value from voxel_consensus.py's corrected-temperature voxel map.

Two formulas, selected by whether --irradiance is given.

Without --irradiance -- plain steady-state quantitative-IR-thermography
formula (ISO 9869 style), no solar term:

    U = hsi * (Tint - Tsurf) / (Tint - Text)

With --irradiance -- full formula, adding back the radiative gain a
sun-exposed surface picks up (this is what actually explains a Tsurf hotter
than both Tint and Text, which the plain formula can't represent):

    U = [hsi * (Tint - Tsurf) + alpha * I] / (Tint - Text)

  Tsurf  = corrected interior surface temperature, per voxel (t_mean_c column
           of thermal_voxels.csv -- already emissivity + reflected-temperature
           corrected by correct_session.py, then voxel-averaged by
           voxel_consensus.py --stage thermal). Real measured data, not
           modeled -- this script does no thermal correction of its own.
  Tint   = indoor air temperature (deg C), constant for the session
  Text   = outdoor air temperature (deg C), constant for the session
  hsi    = internal surface heat transfer coefficient (W/m2K), constant
  alpha  = solar_absorptance, per voxel (column added by voxel_consensus.py
           --stage thermal from emissivity_table.csv's consensus-material
           lookup -- rerun that first if your thermal_voxels.csv predates it)
  I      = incident solar irradiance on the surface, W/m2, constant for the
           run -- pass the effective (IAM-corrected) POA irradiance from
           sun_incidence.py for the actual capture instant. One scalar
           applied uniformly to every voxel regardless of that voxel's own
           wall orientation -- sun_incidence.py was only run for one plane
           (the glazed wall), not per-orientation for every wall/floor/
           ceiling in the scene, so this is a simplification, not a claim
           that every surface saw that exact irradiance.

--correction-mode selects WHERE the solar term gets applied (only matters
when --irradiance is given):
  plane (default)  geometric: I is applied only to voxels within
                    --plane-threshold of --plane-id's plane in --planes.
                    A spatial criterion, independent of the raw-Tsurf flags
                    below. Requires --planes and --plane-id.
  flag              I is applied only to voxels already flagged
                    solar_suspected (out-of-range Tsurf AND material in
                    --solar-materials -- see below). Needs only --irradiance
                    and the input CSV's solar_absorptance/material columns --
                    --planes/--plane-id/--plane-threshold are not used.
                    WARNING: not an independent check -- the same Tsurf
                    anomaly used to flag solar_suspected is also what decides
                    where the solar explanation for that anomaly is applied.
                    State this clearly if used in the thesis.

Sign convention is direction-agnostic: works the same whether Tint > Text
(heating season) or Text > Tint (cooling season, e.g. this dataset).

A voxel's Tsurf outside [min(Tint,Text), max(Tint,Text)] is physically
implausible for a *pure-conduction, no-solar* passive surface. This flag is
independent of which formula is used -- it just describes the raw Tsurf, so
it's still printed with --irradiance even though the solar term may now
fully explain it:
  solar_suspected  Tsurf out of range AND material is in --solar-materials
                    (glass, painted_metal by default): high confidence this
                    is solar/radiative gain.
  solar_possible   Tsurf out of range AND material is in --maybe-solar-
                    materials (paint, rubber by default) -- could also be
                    shaded, no per-voxel sun-angle check here.
  implausible      Tsurf out of range on any other material -- no known
                    physical explanation.
Nothing is dropped: the U value is still written for all three.

--transmittance (default 1.0) attenuates I before alpha*I is applied: use this
when --irradiance is the exterior incident irradiance on a GLAZED surface but
Tsurf is measured on its interior side (or on objects behind it) -- most of
that exterior I is reflected/transmitted by the glass, not absorbed at the
face being imaged, so feeding it in unattenuated overstates the solar term.
Leave at 1.0 for an opaque wall, where sun and Tsurf are on the same surface.

Usage:
    python voxel_u_value.py --in thermal_voxels.csv --tint 29 --text 36.9 --hsi 8
        [--irradiance 613.6 --correction-mode {plane,flag} --transmittance 0.85]
        [--planes planes.json --plane-id 3 --plane-threshold 0.15]  # plane mode
        [--out thermal_voxels_u.csv] [--out-ply thermal_voxels_u.ply]
        [--solar-materials glass,painted_metal] [--maybe-solar-materials paint,rubber]

No LiDAR/rosbags/pandas needed -- just numpy. Any venv with numpy works.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="thermal_voxels.csv from voxel_consensus.py --stage thermal")
    ap.add_argument("--tint", type=float, required=True, help="indoor air temperature, deg C")
    ap.add_argument("--text", type=float, required=True, help="outdoor air temperature, deg C")
    ap.add_argument("--hsi", type=float, default=None,
                    help="internal surface heat transfer coefficient, W/m2K, uniform for every "
                         "voxel. Omit if --in has its own 'hsi_used' column (per-surface-type "
                         "ISO 6946 values -- see add_surface_hsi.py); that column is then used "
                         "per-voxel instead. Required if --in has no 'hsi_used' column.")
    ap.add_argument("--solar-materials", default="painted_metal,glass", metavar="LIST",
                    help="Comma-separated consensus-material names that are sun-absorptive "
                         "regardless of orientation (default: painted_metal,glass). Out-of-range "
                         "Tsurf on these gets flag_reason=solar_suspected (high confidence). "
                         "Empty string disables the distinction.")
    ap.add_argument("--maybe-solar-materials", default="paint,rubber", metavar="LIST",
                    help="Comma-separated consensus-material names that are ONLY sun-absorptive "
                         "when actually sun-exposed -- could be shaded instead (default: "
                         "paint,rubber). Out-of-range Tsurf on these gets "
                         "flag_reason=solar_possible (lower confidence than solar_suspected, "
                         "since there's no per-voxel shading check). Empty string disables it.")
    ap.add_argument("--irradiance", type=float, default=None, metavar="W_M2",
                    help="incident solar irradiance, W/m2 (e.g. sun_incidence.py's "
                         "poa_effective_w_m2 at the capture instant). If given, uses the full "
                         "formula U = [hsi*(Tint-Tsurf) + alpha*I] / (Tint-Text), alpha from "
                         "the input CSV's solar_absorptance column (must be present -- rerun "
                         "voxel_consensus.py --stage thermal if it's missing). If omitted, uses "
                         "the plain no-solar formula (previous behaviour). Where it gets applied "
                         "is controlled by --correction-mode.")
    ap.add_argument("--correction-mode", choices=["plane", "flag"], default="plane",
                    help="how to decide which voxels receive the solar term (default: plane). "
                         "plane: geometric, --planes/--plane-id required (see those). "
                         "flag: only voxels already flagged solar_suspected from raw Tsurf + "
                         "material -- see module docstring for the caveat this implies. Only "
                         "matters when --irradiance is given.")
    ap.add_argument("--transmittance", type=float, default=1.0, metavar="0-1",
                    help="fraction of --irradiance that actually reaches the surface being "
                         "measured (default 1.0, i.e. no attenuation -- correct for an OPAQUE "
                         "exterior wall, where the sun hits the same surface Tsurf is measured "
                         "on). For a GLAZED surface, --irradiance is the exterior incident I on "
                         "the outward face of the glass, but Tsurf is measured on the interior "
                         "side -- most of that I is reflected or transmitted through the glass, "
                         "not absorbed at the face you're imaging (see emissivity_table.csv's "
                         "glass row). Set this below 1.0 (e.g. 0.85 for typical clear single "
                         "glazing's solar transmittance) to attenuate I before alpha*I is applied, "
                         "instead of feeding it the full exterior value. Applied uniformly to "
                         "every voxel that receives the solar term, same as I and alpha.")
    ap.add_argument("--planes", type=Path, default=None,
                    help="fit_planes.py's planes.json, same SLAM world frame as --in's x,y,z. "
                         "Required with --irradiance and --correction-mode plane: I is only "
                         "valid on the specific surface sun_incidence.py computed it for (one "
                         "wall's own orientation), so it gets applied ONLY to voxels within "
                         "--plane-threshold of --plane-id's plane -- every other voxel keeps the "
                         "plain (I=0) formula. Applying one wall's irradiance to the whole scene "
                         "(floor, ceiling, other walls) wildly overcorrects them -- measured: "
                         "mean U drops to -41 W/m2K, only 5%% of voxels land in a sane [0,5] "
                         "range, if you skip this filter. Unused with --correction-mode flag.")
    ap.add_argument("--plane-id", type=int, default=None,
                    help="id in --planes that --irradiance was computed for (the glazed "
                         "surface). Required with --correction-mode plane, unused with flag.")
    ap.add_argument("--plane-threshold", type=float, default=0.15, metavar="M",
                    help="max distance (m) from --plane-id's plane for a voxel to receive the "
                         "solar term (default 0.15, matching fit_planes.py/show_planes.py's "
                         "typical label-assignment tolerance). Unused with --correction-mode flag.")
    ap.add_argument("--out", type=Path, default=None, help="default: <in>_u.csv next to the input")
    ap.add_argument("--out-ply", type=Path, default=None, help="default: <in>_u.ply next to the input")
    args = ap.parse_args()

    if abs(args.tint - args.text) < 1e-9:
        raise SystemExit("--tint and --text are equal -- U is undefined (division by zero)")

    with open(args.inp, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")
    has_hsi_col = "hsi_used" in rows[0]
    if not has_hsi_col and args.hsi is None:
        raise SystemExit(f"--hsi is required: {args.inp} has no 'hsi_used' column to read it from")
    if has_hsi_col and args.hsi is not None:
        print(f"NOTE: {args.inp.name} has its own hsi_used column (per-surface-type) -- using "
              f"that per-voxel, ignoring --hsi {args.hsi}")

    tsurf = np.array([float(r["t_mean_c"]) for r in rows])
    hsi = (np.array([float(r["hsi_used"]) for r in rows]) if has_hsi_col
          else np.full(len(rows), args.hsi))
    conduction_term = hsi * (args.tint - tsurf)

    # Flags from raw Tsurf + material, computed up front (independent of the
    # correction method) -- --correction-mode flag needs solar_suspected as
    # an INPUT to the solar-term calc below, not just an output label.
    lo, hi = min(args.tint, args.text), max(args.tint, args.text)
    implausible = (tsurf < lo) | (tsurf > hi)

    solar_materials = {m.strip() for m in args.solar_materials.split(",") if m.strip()}
    maybe_solar_materials = {m.strip() for m in args.maybe_solar_materials.split(",") if m.strip()}
    material = [r.get("material", "") for r in rows]
    solar_suspected = implausible & np.array([m in solar_materials for m in material])
    solar_possible = implausible & ~solar_suspected & np.array(
        [m in maybe_solar_materials for m in material])

    if args.irradiance is not None:
        if "solar_absorptance" not in rows[0]:
            raise SystemExit(
                f"{args.inp} has no solar_absorptance column -- rerun "
                f"voxel_consensus.py --stage thermal (it now writes this column) before "
                f"using --irradiance")
        if not (0.0 <= args.transmittance <= 1.0):
            raise SystemExit(f"--transmittance must be in [0,1], got {args.transmittance}")

        effective_irradiance = args.irradiance * args.transmittance
        i_desc = (f"I={args.irradiance} W/m2" if args.transmittance == 1.0 else
                  f"I={args.irradiance} W/m2 x transmittance={args.transmittance} = "
                  f"{effective_irradiance:.1f} W/m2 effective")

        if args.correction_mode == "plane":
            if args.planes is None or args.plane_id is None:
                raise SystemExit("--irradiance --correction-mode plane requires --planes and "
                                 "--plane-id (see --help -- applying it to every voxel "
                                 "unfiltered wildly overcorrects)")
            plane_data = json.loads(args.planes.read_text())
            plane = next((p for p in plane_data["planes"] if p["id"] == args.plane_id), None)
            if plane is None:
                raise SystemExit(f"no plane with id {args.plane_id} in {args.planes}")
            n = np.array(plane["normal"])
            xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
            dist_to_plane = np.abs(xyz @ n + plane["d"]) / np.linalg.norm(n)
            on_surface = dist_to_plane < args.plane_threshold
            print(f"plane {args.plane_id} ({plane.get('orientation', '?')}): "
                  f"{int(on_surface.sum())}/{len(rows)} voxel(s) within {args.plane_threshold}m "
                  f"-- only these get the {i_desc} solar term")
        else:  # flag
            on_surface = solar_suspected
            print(f"flag-based correction: {int(on_surface.sum())}/{len(rows)} voxel(s) flagged "
                  f"solar_suspected get the {i_desc} solar term")
            print("WARNING: flag-based correction -- the same Tsurf anomaly used to flag "
                  "solar_suspected is used to decide where to apply the solar explanation for "
                  "that anomaly -- this is not an independent geometric prediction, state this "
                  "clearly if used in the thesis.")

        alpha = np.array([float(r["solar_absorptance"]) if r["solar_absorptance"] != "" else 0.0
                          for r in rows])
        solar_term = np.where(on_surface, alpha * effective_irradiance, 0.0)
        u = (conduction_term + solar_term) / (args.tint - args.text)
    else:
        alpha = None
        solar_term = None
        u = conduction_term / (args.tint - args.text)

    for i, (r, ui, bad, solar, maybe) in enumerate(
            zip(rows, u, implausible, solar_suspected, solar_possible)):
        r["u_value_w_m2k"] = round(float(ui), 4)
        r["plausible"] = "0" if bad else "1"
        r["flag_reason"] = ("solar_suspected" if solar else
                            "solar_possible" if maybe else
                            "implausible" if bad else "")
        if solar_term is not None:
            r["solar_term_w_m2"] = round(float(solar_term[i]), 2)

    out = args.out or args.inp.with_name(args.inp.stem + "_u.csv")
    fieldnames = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    n_bad = int(implausible.sum())
    n_solar = int(solar_suspected.sum())
    n_unexplained = n_bad - n_solar
    hsi_desc = "per-voxel (hsi_used column)" if has_hsi_col else f"{args.hsi} W/m2K"
    print(f"{len(rows)} voxel(s), hsi={hsi_desc}, Tint={args.tint}C, Text={args.text}C"
          + (f", {i_desc} (full formula)" if args.irradiance is not None
             else " (no-solar formula)"))
    print(f"U: min {u.min():.3f}, max {u.max():.3f}, mean {u.mean():.3f}, "
          f"median {np.median(u):.3f} W/m2K")
    if args.irradiance is not None:
        # Physically sane opaque-envelope U is roughly 0.1-5 W/m2K -- outside
        # that, either alpha or I is a bad fit for that voxel's real exposure
        # (--irradiance is one scalar for the whole scene, not per-orientation).
        sane = (u >= 0.0) & (u <= 5.0)
        print(f"solar term (alpha*I): min {solar_term.min():.1f}, max {solar_term.max():.1f}, "
              f"mean {solar_term.mean():.1f} W/m2")
        print(f"{int(sane.sum())}/{len(rows)} voxel(s) ({100*sane.mean():.1f}%) now land in a "
              f"physically sane U range [0, 5] W/m2K")
        if n_bad:
            still_bad_after = ((u < 0) | (u > 5)) & implausible
            print(f"of the {n_bad} voxel(s) flagged from raw Tsurf (solar_suspected/possible/"
                  f"implausible): {int(still_bad_after.sum())} still give an implausible U even "
                  f"after the solar correction")
    if n_bad:
        print(f"WARNING: {n_bad}/{len(rows)} voxel(s) ({100 * n_bad / len(rows):.1f}%) have a "
              f"corrected surface temperature outside [{lo:.1f}, {hi:.1f}] C (between Tint/Text) "
              f"-- flagged plausible=0 but still written to the U column")
        if solar_materials:
            print(f"  {n_solar} solar_suspected (material in {sorted(solar_materials)}): "
                  f"plausibly real solar/radiative gain, {'now corrected for via --irradiance' if args.irradiance is not None else 'this hsi-only formula cannot represent'}")
            print(f"  {n_unexplained} implausible (other materials): no known physical "
                  f"explanation, worth checking correction/measurement")
    print(f"wrote {out}")

    out_ply = args.out_ply or args.inp.with_name(args.inp.stem + "_u.ply")
    xs = np.array([float(r["x"]) for r in rows])
    ys = np.array([float(r["y"]) for r in rows])
    zs = np.array([float(r["z"]) for r in rows])
    lo_u, hi_u = np.percentile(u, [5, 95])
    norm = np.clip((u - lo_u) / max(1e-9, hi_u - lo_u), 0, 1)
    rgb = (np.stack([norm, np.zeros_like(norm), 1.0 - norm], 1) * 255).astype(int)
    with open(out_ply, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for x, y, z, c in zip(xs, ys, zs, rgb):
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {c[0]} {c[1]} {c[2]}\n")
    print(f"wrote {out_ply} (blue=low U, red=high U, 5-95th percentile)")


if __name__ == "__main__":
    main()
