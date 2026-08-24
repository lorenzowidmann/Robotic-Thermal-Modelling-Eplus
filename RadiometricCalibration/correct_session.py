"""Batch radiometric correction over a whole synced session, with a
physical-plausibility check that can reject a material call and retry.

Per frame it needs three things already produced upstream:
    apparent temperature  -- the rot180 FLIR .npy (deg C)
    distance.npy          -- EmissivityCalculation/project_to_flir.py
    segment_id.npy        -- ditto, links each FLIR pixel to its ZED superpixel
plus that frame's segments.json (classify_session.py) for the per-segment
material candidates.

Why the retry exists
--------------------
The correction divides by emissivity, so a wrong low-e call is catastrophic:
e=0.07 turns a 37 degC apparent reading into ~158 degC. CLIP picks those
classes with 15-30% confidence among near-ties, so the call is often a coin
flip. classify_session.py's gate already refuses weak low-e calls up front;
this is the second net, and it uses physics rather than confidence: compute
the corrected temperature a candidate implies, and if the result is not a
temperature this scene can produce, drop that candidate and try the next one
from the segment's ranking. A segment where no candidate produces a plausible
temperature is written as NaN, never as a made-up number.

Usage:
    py correct_session.py --session-dir ...\\ZED\\20260730_161223\\fullrate
        --flir-dir ...\\Flir\\session9_only_rot180
        --humidity 50 --air-temp 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from radiometric import correct_temperature, transmittance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EmissivityCalculation"))
from emissivity import EmissivityTable  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch radiometric correction with plausibility-driven material retry")
    p.add_argument("--session-dir", required=True, metavar="DIR",
                    help="ZED session folder with sync_manifest.json, material_map/ and emissivity_map/")
    p.add_argument("--flir-dir", required=True, metavar="DIR",
                    help="Folder with the rot180 FLIR .npy apparent-temperature frames")
    p.add_argument("--humidity", type=float, required=True,
                    help="Relative humidity in percent (assumed value is fine: "
                         "RH 20-80%% moves the result by <0.4 K at these distances)")
    p.add_argument("--air-temp", type=float, required=True, help="Air temperature in deg C")
    p.add_argument("--reflected-temp", type=float, default=None,
                    help="Reflected apparent temperature in deg C (default: air temperature)")
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    p.add_argument("--material-map-dir", default=None, metavar="DIR",
                    help="Where the per-frame segments.json live (default: "
                         "<session-dir>/material_map). Point this at the output of "
                         "EmissivityCalculation/voxel_consensus.py to correct with "
                         "multi-view consensus materials instead of per-frame ones.")

    p.add_argument("--t-min", type=float, default=-20.0, metavar="C",
                    help="Lowest physically plausible surface temperature (default -20).")
    p.add_argument("--t-max", type=float, default=80.0, metavar="C",
                    help="Highest physically plausible surface temperature (default 80, "
                         "above a working radiator but far below the ~158 C an e=0.07 misfire gives).")
    p.add_argument("--min-plausible-frac", type=float, default=0.99, metavar="F",
                    help="Fraction of a segment's pixels that must fall inside "
                         "[t-min, t-max] for a candidate to be accepted (default 0.99).")
    p.add_argument("--min-emissivity", type=float, default=0.5, metavar="E",
                    help="Candidates below this emissivity are not even tried "
                         "(same threshold as classify_session.py's gate; default 0.5).")
    p.add_argument("--allow-low-emissivity", action="store_true",
                    help="Also try candidates below --min-emissivity (last resort, off by default).")

    p.add_argument("--every-n", type=int, default=1, metavar="N")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--out-name", default="corrected_temperature.npy",
                    help="Filename written inside each emissivity_map/<frame>/ folder.")
    return p.parse_args()


def candidate_list(seg, table, min_emissivity, allow_low):
    """Materials to try for one segment, best first.

    The accepted call comes first, then the rest of its stored ranking. Low-e
    classes are held back to the end (and skipped entirely unless explicitly
    allowed): they are both rare in an indoor scene and the ones that blow the
    correction up, so they should never win by being tried first.
    """
    ordered, seen = [], set()

    def push(material):
        if material in seen:
            return
        seen.add(material)
        try:
            eps = table.lookup(material).emissivity
        except KeyError:
            return          # stale table vs. stale segments.json, skip quietly
        ordered.append((material, eps))

    push(seg["top_material"])
    for material, _conf in seg.get("top_k", []):
        push(material)

    high = [c for c in ordered if c[1] >= min_emissivity]
    low = [c for c in ordered if c[1] < min_emissivity]
    return high + low if allow_low else high


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    flir_dir = Path(args.flir_dir)
    material_dir = Path(args.material_map_dir) if args.material_map_dir else session_dir / "material_map"
    emis_dir = session_dir / "emissivity_map"

    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    reflected_temp = args.reflected_temp if args.reflected_temp is not None else args.air_temp

    manifest = json.loads((session_dir / "sync_manifest.json").read_text(encoding="utf-8"))
    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]

    print(f"Correcting up to {len(triplets)} frame(s)")
    print(f"RH {args.humidity:.0f}%, air {args.air_temp:.1f} C, reflected {reflected_temp:.1f} C")
    print(f"Plausible range [{args.t_min:.0f}, {args.t_max:.0f}] C, "
          f">= {args.min_plausible_frac * 100:.0f}% of a segment's pixels must fit\n")

    n_done = 0
    tot_retried = tot_failed = tot_segments = 0

    for tr in triplets:
        stem = Path(tr["flir"]["file"]).stem
        frame_emis = emis_dir / stem
        seg_json = material_dir / stem / "segments.json"
        thermal_path = flir_dir / f"{stem.replace('_R', '')}.npy"

        missing = [str(p) for p in (frame_emis / "distance.npy", frame_emis / "segment_id.npy",
                                     seg_json, thermal_path) if not p.exists()]
        if missing:
            print(f"skip {stem}: missing {', '.join(Path(m).name for m in missing)}", file=sys.stderr)
            continue

        apparent = np.load(thermal_path).astype(np.float64)
        distance = np.load(frame_emis / "distance.npy").astype(np.float64)
        segment_id = np.load(frame_emis / "segment_id.npy")
        seg_data = json.loads(seg_json.read_text(encoding="utf-8"))
        seg_by_id = {s["id"]: s for s in seg_data["segments"]}

        if apparent.shape != distance.shape or apparent.shape != segment_id.shape:
            print(f"skip {stem}: shape mismatch thermal{apparent.shape} "
                  f"distance{distance.shape} segment_id{segment_id.shape}", file=sys.stderr)
            continue

        tau = transmittance(distance, args.humidity, args.air_temp)

        corrected = np.full(apparent.shape, np.nan, dtype=np.float64)
        chosen_eps = np.full(apparent.shape, np.nan, dtype=np.float64)
        decisions = []
        n_retried = n_failed = 0

        for sid in np.unique(segment_id):
            sid = int(sid)
            mask = segment_id == sid
            seg = seg_by_id.get(sid)
            if seg is None:
                n_failed += 1
                decisions.append({"segment_id": sid, "outcome": "no_segment_record",
                                  "n_pixels": int(mask.sum())})
                continue

            candidates = candidate_list(seg, table, args.min_emissivity, args.allow_low_emissivity)
            accepted = None
            tried = []
            for material, eps in candidates:
                t_seg = correct_temperature(apparent[mask], eps, tau[mask], reflected_temp, args.air_temp)
                frac_ok = float(np.mean((t_seg >= args.t_min) & (t_seg <= args.t_max)))
                tried.append({"material": material, "emissivity": eps,
                              "frac_in_range": round(frac_ok, 4),
                              "t_median": round(float(np.median(t_seg)), 2)})
                if frac_ok >= args.min_plausible_frac:
                    accepted = (material, eps, t_seg)
                    break

            if accepted is None:
                n_failed += 1
                decisions.append({"segment_id": sid, "outcome": "no_plausible_candidate",
                                  "n_pixels": int(mask.sum()), "tried": tried})
                continue        # pixels stay NaN -- never invent a number

            material, eps, t_seg = accepted
            corrected[mask] = t_seg
            chosen_eps[mask] = eps
            if material != seg["top_material"]:
                n_retried += 1
                decisions.append({"segment_id": sid, "outcome": "retried",
                                  "n_pixels": int(mask.sum()),
                                  "from": seg["top_material"], "to": material, "tried": tried})

        np.save(frame_emis / args.out_name, corrected.astype(np.float32))
        np.save(frame_emis / "emissivity_used.npy", chosen_eps.astype(np.float32))

        n_seg = len(np.unique(segment_id))
        valid = np.isfinite(corrected)
        (frame_emis / "correction_report.json").write_text(json.dumps({
            "schema": "radiometric_correction/v1",
            "generated_by": "correct_session.py",
            "source_flir_frame": tr["flir"]["file"],
            "conditions": {"humidity_pct": args.humidity, "air_temp_c": args.air_temp,
                            "reflected_temp_c": reflected_temp,
                            "humidity_source": "assumed (not measured during acquisition)"},
            "plausible_range_c": [args.t_min, args.t_max],
            "min_plausible_frac": args.min_plausible_frac,
            "n_segments": n_seg,
            "n_segments_retried": n_retried,
            "n_segments_failed": n_failed,
            "n_pixels_valid": int(valid.sum()),
            "n_pixels_nan": int((~valid).sum()),
            "corrected_c": {
                "min": round(float(np.nanmin(corrected)), 2) if valid.any() else None,
                "max": round(float(np.nanmax(corrected)), 2) if valid.any() else None,
                "mean": round(float(np.nanmean(corrected)), 2) if valid.any() else None,
            },
            "decisions": decisions,
        }, indent=2), encoding="utf-8")

        rng = (f"{np.nanmin(corrected):.1f}..{np.nanmax(corrected):.1f} C" if valid.any() else "all NaN")
        print(f"{stem}: {n_seg} segments, {n_retried} retried, {n_failed} unresolved -> {rng}")

        n_done += 1
        tot_retried += n_retried
        tot_failed += n_failed
        tot_segments += n_seg

    if tot_segments:
        print(f"\n{n_done} frame(s). Segments retried {tot_retried}/{tot_segments} "
              f"({100.0 * tot_retried / tot_segments:.1f}%), "
              f"unresolved {tot_failed}/{tot_segments} ({100.0 * tot_failed / tot_segments:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
