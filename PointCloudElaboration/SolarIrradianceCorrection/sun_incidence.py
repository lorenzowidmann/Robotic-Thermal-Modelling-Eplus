"""Sun position + corrected incidence irradiance on a vertical glazed
surface, using pvlib.

Pulls the wall's outward horizontal direction from fit_planes.py's
planes.json (surface_tilt is always 90 -- vertical) and turns it into a true
compass azimuth via a manual --north-offset-deg, since the LiDAR/SLAM world
frame has no georeferencing of its own (no known relationship to true north).

Azimuth convention (matches pvlib: compass bearing, clockwise from north,
0=N/90=E/180=S/270=W):

    local_bearing_deg = atan2(normal_x, normal_y) mod 360
        -- i.e. the plane's local +Y axis is treated as a *local reference
        north* (bearing 0) and +X as local reference east (bearing 90).
    surface_azimuth_deg = (local_bearing_deg + north_offset_deg) mod 360
        -- north_offset_deg is the true compass bearing OF the SLAM frame's
        local +Y axis. You need to know this from something external to the
        scan itself (e.g. a compass reading taken during capture, or lining
        up the corridor against a satellite/floor-plan view). If you don't
        know it, --north-offset-deg 0 just assumes local +Y already IS true
        north -- fine for a self-consistent "what if the corridor faced
        north" study, wrong for real solar-gain numbers.

Pipeline (all pvlib):
    1. Location(lat, lon, tz, altitude).get_solarposition(times)
    2. Location.get_clearsky(times, model="ineichen")  -- GHI/DNI/DHI, no
       measured weather data required. Use --irradiance-csv to override with
       real data instead: give ghi,dni,dhi columns directly, or just a ghi
       column and it's split into DNI+DHI via the Erbs decomposition model
       (needs only GHI + solar position -- no pressure/dew-point inputs).
       This is the normal path if all you have is a global irradiance
       reading (pyranometer, weather station, PVGIS, ...): GHI alone can't
       be transposed onto a vertical surface directly, because a tilted/
       vertical plane sees the beam and diffuse components differently
       (beam scales with cos(AOI), diffuse does not) -- they have to be
       separated first.
    3. irradiance.aoi(90, surface_azimuth, solar_zenith, solar_azimuth)
    4. irradiance.get_total_irradiance(...) -- POA global/direct/diffuse via
       a transposition model (--model, default isotropic).
    5. iam.physical(aoi) (or --iam-model) -- glass reflection loss, applied
       to the POA BEAM component only (poa_direct * iam); diffuse is left
       uncorrected, the standard simplification (diffuse arrives from a
       spread of angles, a single-AOI IAM does not apply to it).
    6. poa_effective = poa_direct * iam + poa_diffuse
       -- this is the "corrected incidence irradiance" actually available to
       drive solar gain / the solar_absorptance term in voxel_u_value.py.

Usage:
    # list wall planes and their local bearing, to pick --plane-id
    python sun_incidence.py --planes planes.json --list

    python sun_incidence.py --planes planes.json --plane-id 1 \\
        --north-offset-deg 173 --lat 45.4642 --lon 9.1900 --tz Europe/Rome \\
        --date 2026-07-30 --out plane1_irradiance.csv --plot plane1_irradiance.png

Venv: C:\\venvs\\planefit (pip install pvlib -- pulls in pandas).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib


def wall_planes(data):
    return [p for p in data["planes"] if p.get("orientation") == "wall" and not p.get("synthetic")]


def local_bearing_deg(normal):
    nx, ny = normal[0], normal[1]
    return float(np.degrees(np.arctan2(nx, ny)) % 360.0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planes", type=Path, default=Path("planes.json"))
    ap.add_argument("--plane-id", type=int, default=None,
                    help="id from planes.json to use as the glazed surface (see --list)")
    ap.add_argument("--surface-tilt", type=float, default=90.0,
                    help="surface tilt, deg from horizontal (default 90 = vertical wall; use 0 "
                         "for a horizontal ceiling/roof facing up -- surface_azimuth is "
                         "irrelevant at tilt=0, the AOI formula drops the azimuth term)")
    ap.add_argument("--list", action="store_true",
                    help="print wall planes (id, normal, local bearing) and exit")
    ap.add_argument("--solve-north-offset", metavar="PLANE_ID:TRUE_BEARING_DEG",
                    help="e.g. --solve-north-offset 3:92 if you know plane 3's real compass "
                         "bearing is ~92 deg (from a floor plan/satellite map/compass reading) "
                         "-- prints the --north-offset-deg to use for every other plane, and exits")
    ap.add_argument("--north-offset-deg", type=float, default=None,
                    help="true compass bearing of the SLAM frame's local +Y axis (required "
                         "unless --list)")
    ap.add_argument("--lat", type=float, default=None, help="site latitude, deg (required unless --list)")
    ap.add_argument("--lon", type=float, default=None, help="site longitude, deg (required unless --list)")
    ap.add_argument("--tz", default=None, help="IANA timezone, e.g. Europe/Rome (required unless --list)")
    ap.add_argument("--altitude", type=float, default=0.0, help="site elevation, m (default 0)")
    ap.add_argument("--date", default=None,
                    help="single day, YYYY-MM-DD (local tz). Mutually exclusive with --start/--end.")
    ap.add_argument("--start", default=None, help="local ISO datetime, e.g. 2026-07-30T06:00")
    ap.add_argument("--end", default=None, help="local ISO datetime, e.g. 2026-07-30T20:00")
    ap.add_argument("--freq", default="5min", help="pandas frequency string (default 5min)")
    ap.add_argument("--model", default="isotropic",
                    choices=["isotropic", "klucher", "haydavies", "reindl", "king", "perez"],
                    help="sky-diffuse transposition model (default isotropic)")
    ap.add_argument("--iam-model", default="physical", choices=["physical", "ashrae", "martin_ruiz", "none"],
                    help="glass incidence-angle-modifier model for the beam component "
                         "(default physical: n=1.526, K=4, L=0.002 -- clear glass)")
    ap.add_argument("--irradiance-csv", type=Path, default=None,
                    help="override the clear-sky model: CSV with columns timestamp,ghi,dni,dhi "
                         "(timestamp parseable by pandas, localized to --tz if naive)")
    ap.add_argument("--out", type=Path, default=None, help="default: plane<id>_irradiance.csv")
    ap.add_argument("--plot", type=Path, default=None, help="optional PNG plot of GHI/POA/effective")
    args = ap.parse_args()

    data = json.loads(args.planes.read_text())
    walls = wall_planes(data)

    if args.solve_north_offset:
        pid_str, bearing_str = args.solve_north_offset.split(":")
        pid, known_bearing = int(pid_str), float(bearing_str)
        plane = next((p for p in walls if p["id"] == pid), None)
        if plane is None:
            raise SystemExit(f"no wall plane with id {pid} (see --list)")
        lb = local_bearing_deg(plane["normal"])
        offset = (known_bearing - lb) % 360.0
        print(f"plane {pid}: local_bearing={lb:.1f} deg, known true bearing={known_bearing:.1f} deg")
        print(f"--north-offset-deg {offset:.1f}")
        print("\nSame offset applies to every plane in this planes.json (they share one SLAM frame):")
        for p in walls:
            print(f"  plane {p['id']}: local_bearing={local_bearing_deg(p['normal']):.1f} deg "
                  f"-> true bearing={(local_bearing_deg(p['normal']) + offset) % 360.0:.1f} deg")
        return

    if args.list or args.plane_id is None:
        if not walls:
            raise SystemExit(f"no wall planes in {args.planes}")
        print(f"{'id':>3}  {'normal':<24}  {'local_bearing_deg':>18}  area_m2")
        for p in walls:
            print(f"{p['id']:>3}  {str([round(c, 3) for c in p['normal']]):<24}  "
                  f"{local_bearing_deg(p['normal']):>18.1f}  {p['area_m2']:.1f}")
        if args.plane_id is None:
            return

    missing = [name for name, v in [
        ("--north-offset-deg", args.north_offset_deg), ("--lat", args.lat),
        ("--lon", args.lon), ("--tz", args.tz)] if v is None]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")

    plane = next((p for p in data["planes"] if p["id"] == args.plane_id), None)
    if plane is None:
        raise SystemExit(f"no plane with id {args.plane_id} in {args.planes}")
    if plane.get("orientation") != "wall" and args.surface_tilt == 90.0:
        print(f"WARNING: plane {args.plane_id} orientation is "
              f"'{plane.get('orientation')}', not 'wall' -- proceeding anyway, "
              f"treating it as vertical (tilt=90). Pass --surface-tilt 0 for a "
              f"horizontal ceiling/floor instead.")

    bearing = local_bearing_deg(plane["normal"])
    surface_azimuth = (bearing + args.north_offset_deg) % 360.0
    print(f"plane {args.plane_id}: normal={[round(c, 3) for c in plane['normal']]}  "
          f"local_bearing={bearing:.1f} deg  north_offset={args.north_offset_deg:.1f} deg  "
          f"-> surface_azimuth={surface_azimuth:.1f} deg (0=N,90=E,180=S,270=W)")

    if args.date:
        times = pd.date_range(f"{args.date} 00:00", f"{args.date} 23:59", freq=args.freq, tz=args.tz)
    elif args.start and args.end:
        times = pd.date_range(args.start, args.end, freq=args.freq, tz=args.tz)
    else:
        raise SystemExit("need --date, or both --start and --end")

    site = pvlib.location.Location(args.lat, args.lon, tz=args.tz, altitude=args.altitude)
    solpos = site.get_solarposition(times)

    if args.irradiance_csv:
        irr = pd.read_csv(args.irradiance_csv, parse_dates=["timestamp"], index_col="timestamp")
        if irr.index.tz is None:
            irr = irr.tz_localize(args.tz)
        # Linear time-interpolation onto `times`, not nearest-with-tolerance:
        # source data is commonly hourly (e.g. an ARPAV station export) while
        # `times` can be much finer (--freq), so a tight tolerance silently
        # drops every row and irradiance goes to 0. Interpolating also avoids
        # a stair-step artifact in the transposition that nearest-match gives.
        combined_index = irr.index.union(times).sort_values()
        irr = irr.reindex(combined_index).interpolate(method="time").reindex(times)
        if irr.isna().any().any():
            print(f"WARNING: {int(irr.isna().any(axis=1).sum())} requested timestamp(s) fall "
                  f"outside {args.irradiance_csv}'s coverage -- left as NaN")
        if "dni" in irr.columns and "dhi" in irr.columns:
            ghi, dni, dhi = irr["ghi"], irr["dni"], irr["dhi"]
            print(f"irradiance from {args.irradiance_csv} ({len(irr)} row(s) matched, "
                  f"ghi/dni/dhi columns given directly)")
        else:
            # Real pyranometer/weather-station data is almost always GHI only.
            # Erbs splits it into DNI+DHI from the clearness index kt = GHI /
            # extraterrestrial horizontal irradiance -- no extra inputs needed
            # beyond GHI and solar position, unlike DISC (needs pressure) or
            # DIRINT (needs dew point).
            ghi = irr["ghi"].fillna(0.0).clip(lower=0.0)
            decomp = pvlib.irradiance.erbs(ghi, solpos["apparent_zenith"], times)
            dni, dhi = decomp["dni"], decomp["dhi"]
            # Erbs (like any GHI-only decomposition) divides by cos(zenith) to
            # back out DNI -- as the sun nears the horizon that denominator
            # goes to 0 and DNI blows up to physically impossible values (seen
            # here: 1241 W/m2 beam from a 79 W/m2 GHI reading at zenith 87 deg,
            # i.e. more direct beam than the sun delivers at noon). Physical
            # fix: DNI can't exceed the clear-sky DNI for that same sun
            # position -- clouds/atmosphere only ever attenuate beam relative
            # to clear-sky, they don't enhance it. Clipping to the Ineichen
            # clear-sky DNI (which already tapers smoothly with airmass near
            # the horizon) fixes the spike without an artificial hard cutoff
            # that would show up as a discontinuity in the plot.
            clearsky_dni = site.get_clearsky(times, model="ineichen")["dni"]
            n_clipped = int((dni > clearsky_dni).sum())
            dni = dni.clip(upper=clearsky_dni)
            print(f"irradiance from {args.irradiance_csv} ({len(irr)} row(s) matched, "
                  f"GHI-only -- DNI/DHI split via Erbs decomposition, "
                  f"{n_clipped} timestep(s) beam-clipped to clear-sky DNI)")
    else:
        clearsky = site.get_clearsky(times, model="ineichen")
        ghi, dni, dhi = clearsky["ghi"], clearsky["dni"], clearsky["dhi"]
        print("irradiance from pvlib clear-sky (Ineichen) model -- no measured weather data")

    aoi = pvlib.irradiance.aoi(args.surface_tilt, surface_azimuth, solpos["apparent_zenith"], solpos["azimuth"])
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=args.surface_tilt, surface_azimuth=surface_azimuth,
        solar_zenith=solpos["apparent_zenith"], solar_azimuth=solpos["azimuth"],
        dni=dni, ghi=ghi, dhi=dhi, dni_extra=dni_extra, model=args.model)

    if args.iam_model == "none":
        iam = pd.Series(1.0, index=times)
    elif args.iam_model == "physical":
        iam = pvlib.iam.physical(aoi)
    elif args.iam_model == "ashrae":
        iam = pvlib.iam.ashrae(aoi)
    else:
        iam = pvlib.iam.martin_ruiz(aoi)
    iam = iam.clip(lower=0.0, upper=1.0).fillna(0.0)

    poa_direct = total["poa_direct"].clip(lower=0.0)
    poa_diffuse = total["poa_diffuse"].clip(lower=0.0)
    poa_effective = poa_direct * iam + poa_diffuse

    out = pd.DataFrame({
        "solar_zenith_deg": solpos["apparent_zenith"],
        "solar_azimuth_deg": solpos["azimuth"],
        "aoi_deg": aoi,
        "ghi_w_m2": ghi,
        "dni_w_m2": dni,
        "dhi_w_m2": dhi,
        "poa_global_w_m2": total["poa_global"].clip(lower=0.0),
        "poa_direct_w_m2": poa_direct,
        "poa_diffuse_w_m2": poa_diffuse,
        "iam": iam,
        "poa_effective_w_m2": poa_effective,
    })

    out_path = args.out or Path(f"plane{args.plane_id}_irradiance.csv")
    out.to_csv(out_path, index_label="timestamp")

    daylight = out[out["poa_effective_w_m2"] > 0]
    step_h = (times[1] - times[0]).total_seconds() / 3600.0
    daily_insolation_wh = out["poa_effective_w_m2"].sum() * step_h
    print(f"{len(out)} timestep(s), surface_tilt={args.surface_tilt:.0f} "
          f"surface_azimuth={surface_azimuth:.1f}, "
          f"iam_model={args.iam_model}, transposition_model={args.model}")
    if len(daylight):
        peak_t = out["poa_effective_w_m2"].idxmax()
        print(f"peak effective POA: {out['poa_effective_w_m2'].max():.1f} W/m2 at {peak_t}")
    print(f"daily insolation on this surface: {daily_insolation_wh:.1f} Wh/m2 "
          f"({daily_insolation_wh / 1000:.2f} kWh/m2)")
    print(f"wrote {out_path}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(out.index, out["ghi_w_m2"], label="GHI (horizontal)", alpha=0.5)
        ax.plot(out.index, out["poa_global_w_m2"], label="POA global (on wall)", alpha=0.7)
        ax.plot(out.index, out["poa_effective_w_m2"], label="POA effective (IAM-corrected)", lw=2)
        ax.set_ylabel("W/m2")
        ax.set_title(f"Plane {args.plane_id} -- surface_azimuth={surface_azimuth:.0f} deg")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
