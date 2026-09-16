"""Estimate the thermal camera's sensor offset from the indoor air temperature,
without needing a thermocouple.

The FLIR reads several degrees hotter than reality (measured +4.5..+5.4 C on the
AcquisitionGroundTruth sessions, drifting upward as the camera warms up). That
offset is a sensor property: it is the same on every material, so no emissivity
or atmosphere model removes it. It has to be measured against something whose
temperature is known.

A thermal camera cannot see air -- air is transparent in the LWIR band, only
surfaces are imaged. So the reference has to be a surface known to sit at air
temperature. An *interior* wall is that surface: the same air is on both sides,
no heat flows through it, and it settles at air temperature. Windows (outdoor
air on the far side), radiators, floors (solar gain) and doors do not qualify
and are excluded by ADE class.

Not every wall pixel is at air temperature either -- some are warmed by a nearby
radiator or by sun. The *coldest* wall pixels are the ones actually in
equilibrium with the air, so the estimator takes a low percentile of the pooled
wall pixels rather than their mean:

    offset = percentile(wall pixels, P) - T_air

P defaults to 10, chosen by sweeping P against thermocouple ground truth on
A1/A9/A10: it was the value whose residual was both near zero and most
consistent across sessions (spread 0.34 C, no fitted constant needed). Lower
percentiles gave a tighter spread but needed an additive constant; higher ones
drifted.

CAVEAT: that tuning rests on three sessions in one building on one day, so 0.34 C
is an indication, not a validated uncertainty. Re-check with --thermocouple
whenever a probe reading is available.

Usage:
    py estimate_offset.py --session-dir ...\\ZED\\20260911_094055\\fullrate --air-temp 21
    py estimate_offset.py --session-dir ... --air-temp 21 --thermocouple 22.19
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_PROXY = "wall,ceiling,column"
REPORT_PERCENTILES = (0.5, 1, 2, 5, 10, 15, 20)


def parse_args():
    p = argparse.ArgumentParser(
        description="Estimate the camera's sensor offset from the indoor air temperature")
    p.add_argument("--session-dir", required=True, metavar="DIR",
                    help="ZED session folder with sync_manifest.json and emissivity_map/")
    p.add_argument("--air-temp", type=float, required=True,
                    help="Measured indoor air temperature in deg C")
    p.add_argument("--material-map-dir", default=None, metavar="DIR",
                    help="Where the per-frame segments.json live (default: "
                         "<session-dir>/material_map_consensus)")
    p.add_argument("--corrected-name", default="corrected_temperature.npy",
                    help="Corrected map to read inside each emissivity_map/<frame>/")
    p.add_argument("--percentile", type=float, default=10.0, metavar="P",
                    help="Percentile of the pooled proxy pixels taken as the camera's "
                         "reading of an air-temperature surface (default 10)")
    p.add_argument("--proxy-classes", default=DEFAULT_PROXY, metavar="LIST",
                    help=f"Comma-separated ADE classes used as the air-temperature "
                         f"proxy (default: {DEFAULT_PROXY})")
    p.add_argument("--thermocouple", type=float, default=None, metavar="C",
                    help="Thermocouple reading, if available: reports the residual of "
                         "the air-based estimate against it (validation only)")
    p.add_argument("--every-n", type=int, default=1, metavar="N")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--out-name", default="offset_report.json",
                    help="Report filename written in <session-dir> ('-' to skip)")
    return p.parse_args()


def collect_proxy_pixels(triplets, emis_dir, material_dir, corrected_name, proxy_classes):
    """Pool the corrected temperatures of every proxy-class segment in the session."""
    by_class = {}
    n_frames = 0

    for tr in triplets:
        stem = Path(tr["flir"]["file"]).stem
        frame_dir = emis_dir / stem
        corrected_path = frame_dir / corrected_name
        segment_path = frame_dir / "segment_id.npy"
        seg_json = material_dir / stem / "segments.json"

        if not (corrected_path.exists() and segment_path.exists() and seg_json.exists()):
            continue

        corrected = np.load(corrected_path)
        segment_id = np.load(segment_path)
        if corrected.shape != segment_id.shape:
            print(f"skip {stem}: shape mismatch corrected{corrected.shape} "
                  f"segment_id{segment_id.shape}", file=sys.stderr)
            continue

        used = False
        for seg in json.loads(seg_json.read_text(encoding="utf-8"))["segments"]:
            ade = seg.get("ade")
            if ade not in proxy_classes:
                continue
            values = corrected[(segment_id == seg["id"]) & np.isfinite(corrected)]
            if values.size:
                by_class.setdefault(ade, []).append(values)
                used = True
        n_frames += used

    return {k: np.concatenate(v) for k, v in by_class.items()}, n_frames


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    emis_dir = session_dir / "emissivity_map"
    material_dir = (Path(args.material_map_dir) if args.material_map_dir
                    else session_dir / "material_map_consensus")
    proxy_classes = {c.strip() for c in args.proxy_classes.split(",") if c.strip()}

    manifest = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    print(f"Session {session_dir.name}, air {args.air_temp:.1f} C")
    print(f"Proxy ADE classes: {', '.join(sorted(proxy_classes))}\n")

    by_class, n_frames = collect_proxy_pixels(
        triplets, emis_dir, material_dir, args.corrected_name, proxy_classes)

    if not by_class:
        print(f"No proxy-class pixels found under {emis_dir}.\n"
              f"Check that {args.corrected_name} exists (run correct_session.py first) "
              f"and that segments.json in {material_dir} carries an 'ade' field.",
              file=sys.stderr)
        return 1

    pooled = np.concatenate(list(by_class.values()))

    print(f"{n_frames} frame(s), {pooled.size} proxy pixel(s)")
    for ade in sorted(by_class, key=lambda k: -by_class[k].size):
        v = by_class[ade]
        print(f"   {ade:12s} {v.size:9d} px   p{args.percentile:g} "
              f"{np.percentile(v, args.percentile):6.2f}   median {np.median(v):6.2f}")

    reference = float(np.percentile(pooled, args.percentile))
    offset = reference - args.air_temp

    print(f"\nSensitivity to the percentile choice:")
    for p in REPORT_PERCENTILES:
        mark = "  <--" if abs(p - args.percentile) < 1e-9 else ""
        print(f"   p{p:<4g} -> offset {np.percentile(pooled, p) - args.air_temp:+6.2f} C{mark}")

    print(f"\np{args.percentile:g} of proxy pixels = {reference:.2f} C, air = {args.air_temp:.1f} C")
    print(f"ESTIMATED SENSOR OFFSET  {offset:+.2f} C     ->  T_real = T_corrected {-offset:+.2f}")

    residual = None
    if args.thermocouple is not None:
        # The probe measures one surface, so this only validates the estimate if the
        # scene is near-isothermal apart from the classes already excluded.
        median_all = float(np.median(pooled))
        truth = median_all - args.thermocouple
        residual = offset - truth
        print(f"\nValidation against the thermocouple ({args.thermocouple:.2f} C):")
        print(f"   offset from thermocouple {truth:+.2f} C  (median proxy {median_all:.2f})")
        print(f"   residual of the air-based estimate {residual:+.2f} C")

    if args.out_name != "-":
        (session_dir / args.out_name).write_text(json.dumps({
            "schema": "sensor_offset/v1",
            "generated_by": "estimate_offset.py",
            "air_temp_c": args.air_temp,
            "proxy_classes": sorted(proxy_classes),
            "percentile": args.percentile,
            "corrected_name": args.corrected_name,
            "n_frames": n_frames,
            "n_pixels": int(pooled.size),
            "reference_c": round(reference, 3),
            "offset_c": round(offset, 3),
            "offset_by_percentile_c": {
                f"p{p:g}": round(float(np.percentile(pooled, p)) - args.air_temp, 3)
                for p in REPORT_PERCENTILES},
            "pixels_by_class": {k: int(v.size) for k, v in by_class.items()},
            "thermocouple_c": args.thermocouple,
            "residual_c": round(residual, 3) if residual is not None else None,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {session_dir / args.out_name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
