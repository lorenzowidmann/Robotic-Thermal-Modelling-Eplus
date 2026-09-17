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

--interactive replaces the automatic ADE-class pooling with a manual pick: step
through frames, drag a rectangle over a patch you trust is a plain interior
wall (no radiator glow, no window edge, no visible superpixel-boundary
artifact), and only that patch feeds the percentile. Useful when the automatic
classes are too sparse (few wall pixels in the session) or visibly include a
warmed patch the ADE segmentation missed.

--apply writes a debiased corrected-temperature map per frame (does not touch
correct_session.py's own output): <debiased-name> = corrected_temperature - offset.

Usage:
    py estimate_offset.py --session-dir ...\\ZED\\20260911_094055\\fullrate --air-temp 21
    py estimate_offset.py --session-dir ... --air-temp 21 --thermocouple 22.19
    py estimate_offset.py --session-dir ... --air-temp 21 --interactive
    py estimate_offset.py --session-dir ... --air-temp 21 --apply
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
    p.add_argument("--interactive", action="store_true",
                    help="Pick the reference patch by hand instead of the automatic "
                         "ADE-class pooling: step through frames (left/right or a/d), "
                         "drag a rectangle over a plain interior-wall patch, Enter to "
                         "confirm, Esc to cancel.")
    p.add_argument("--apply", action="store_true",
                    help="Write a debiased corrected-temperature map for every frame "
                         "in the session: <debiased-name> = <corrected-name> - offset. "
                         "The original correct_session.py output is left untouched.")
    p.add_argument("--debiased-name", default="corrected_temperature_debiased.npy",
                    help="Filename written per frame when --apply is set")
    return p.parse_args()


def pick_frame_and_roi(triplets, emis_dir, corrected_name):
    """Steppable viewer over corrected_temperature.npy with a draggable
    rectangle: pick one frame and one region to use as the air-temperature
    reference. Left/Right (or a/d) step frames, drag selects, Enter confirms,
    Esc cancels. Returns (stem, (x0, y0, x1, y1)) in pixel coords, or None."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    usable = [Path(tr["flir"]["file"]).stem for tr in triplets
              if (emis_dir / Path(tr["flir"]["file"]).stem / corrected_name).exists()]
    if not usable:
        print(f"No frame has {corrected_name} under {emis_dir} -- run "
              f"correct_session.py first", file=sys.stderr)
        return None

    state = {"i": 0, "box": None, "result": None}
    fig, ax = plt.subplots()

    def on_select(eclick, erelease):
        x0, y0, x1, y1 = eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        state["box"] = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    selector = RectangleSelector(ax, on_select, useblit=True, button=[1],
                                  interactive=True, minspanx=3, minspany=3)

    def show():
        data = np.load(emis_dir / usable[state["i"]] / corrected_name)
        ax.clear()
        ax.imshow(data, cmap="inferno")
        ax.set_title(
            f"{usable[state['i']]}   frame {state['i'] + 1}/{len(usable)}\n"
            "<- / -> step frame   drag = select wall patch   Enter=confirm   Esc=cancel")
        selector.set_active(True)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["i"] = min(state["i"] + 1, len(usable) - 1)
            state["box"] = None
            show()
        elif event.key in ("left", "a"):
            state["i"] = max(state["i"] - 1, 0)
            state["box"] = None
            show()
        elif event.key in ("enter", "return"):
            if state["box"] is None:
                print("Drag a rectangle over the wall patch first.", file=sys.stderr)
            else:
                state["result"] = (usable[state["i"]], state["box"])
                plt.close(fig)
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    return state["result"]


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
    all_triplets = manifest["triplets"]
    triplets = all_triplets[::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    print(f"Session {session_dir.name}, air {args.air_temp:.1f} C")

    roi_meta = None
    if args.interactive:
        picked = pick_frame_and_roi(triplets, emis_dir, args.corrected_name)
        if picked is None:
            print("No ROI selected -- aborting.", file=sys.stderr)
            return 1
        stem, (x0, y0, x1, y1) = picked
        data = np.load(emis_dir / stem / args.corrected_name)
        h, w = data.shape
        xi0, xi1 = max(0, int(round(x0))), min(w, int(round(x1)) + 1)
        yi0, yi1 = max(0, int(round(y0))), min(h, int(round(y1)) + 1)
        roi = data[yi0:yi1, xi0:xi1]
        pooled = roi[np.isfinite(roi)]
        if pooled.size == 0:
            print("Selected ROI has no valid (finite) pixels.", file=sys.stderr)
            return 1
        n_frames = 1
        by_class = {"manual_roi": pooled}
        roi_meta = {"frame": stem, "px_x0": xi0, "px_y0": yi0, "px_x1": xi1, "px_y1": yi1}
        print(f"Manual ROI: {stem} [x {xi0}:{xi1}, y {yi0}:{yi1}], {pooled.size} px\n")
    else:
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
            "method": "manual_roi" if args.interactive else "ade_class_pooling",
            "roi": roi_meta,
            "air_temp_c": args.air_temp,
            "proxy_classes": None if args.interactive else sorted(proxy_classes),
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
            "applied_to": args.debiased_name if args.apply else None,
        }, indent=2), encoding="utf-8")
        print(f"\nWrote {session_dir / args.out_name}")

    if args.apply:
        n_written = 0
        for tr in all_triplets:
            stem = Path(tr["flir"]["file"]).stem
            src = emis_dir / stem / args.corrected_name
            if not src.exists():
                continue
            corrected = np.load(src)
            debiased = corrected - offset
            np.save(emis_dir / stem / args.debiased_name, debiased.astype(np.float32))
            n_written += 1
        print(f"\n--apply: wrote {args.debiased_name} for {n_written} frame(s) "
              f"(= {args.corrected_name} {-offset:+.2f} C, every frame in the session, "
              f"not just the ones used to estimate the offset)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
