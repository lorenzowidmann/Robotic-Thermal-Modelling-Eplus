"""Material + emissivity per region, for every frame of a recorded ZED
session (driven by SensorFusion/sync_manifest.py's sync_manifest.json).

This is the "smallest possible zone" version of the fixed 3x3 grid the removed
main.py used: each frame is segmented into regions that respect real visual
edges instead, and every region is classified independently with the same CLIP
zero-shot classifier + emissivity_table.csv lookup already used elsewhere in
this module.

Segmentation defaults to SAM (Segment Anything), whose masks follow actual
objects: on a corridor frame the floor comes out as ONE region and the
radiators, pillars and window bays are separated, at 99% pixel coverage. SLIC
remains available with --segmenter slic and is ~70x faster, but it is a
PARTITION -- it emits ~n_segments cells whatever the content, so a uniform
floor is split into dozens of cells that then get classified independently
and disagree with each other.

Only the part of the ZED frame the FLIR can actually see is segmented and
classified (--no-crop-to-flir-fov for the whole frame). project_to_flir.py
keeps solely the LiDAR points valid in BOTH cameras, so a region outside the
FLIR frustum can never reach a FLIR pixel -- on this rig that frustum is ~16%
of the ZED frame, so most of the CLIP calls used to be discarded downstream.
There is no direct FLIR<->ZED calibration, but both extrinsics live in the
LiDAR frame and compose; see Calibration/projection.py::flir_fov_bbox_in_zed.

Output per frame, under <out-dir>/<flir_frame_stem>/:
    labels.npy    -- int32 HxW superpixel-id raster, ZED pixel grid, always
                      full-frame; -1 outside the crop when cropping is on, so
                      project_to_flir.py needs no change either way
    segments.json -- schema "material_map/v1": per-segment id, centroid,
                      area, top material, confidence, emissivity, top_k
                      (centroids in full-frame ZED pixels, cropped or not)
    overlay.png   -- optional (--overlay), superpixel boundaries + labels

Defaults are the best-quality configuration measured on this rig: SAM masks,
CLIP ViT-H/14, and the geometric zone prior on. That prior matters more than it
looks -- unconstrained on a full frame, ViT-H proposed a bare metal on 27% of
regions and one survived the confidence gate at 0.79, which would have turned a
37 degC reading into ~156 degC on that region. With the prior on, the gate
never fired at all.

Usage:
    py classify_session.py --session-dir ...\\ZED\\20260730_161223\\fullrate --limit 3 --overlay
    py classify_session.py --session-dir ...\\fullrate --every-n 5

    # fast, lower quality (the pre-2026-08 behaviour):
    py classify_session.py --session-dir ...\\fullrate --segmenter slic \\
        --clip-model openai/clip-vit-large-patch14 \\
        --no-zone-constraint --no-crop-to-flir-fov
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from emissivity import EmissivityTable, MaterialClassifier
from emissivity.segmentation import superpixel_segments, sam_segments, segment_boxes
from emissivity.zones import restrict_ranking, zone_of

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Calibration"))
from rig_calibration import load_rig_calibration
from projection import flir_fov_bbox_in_zed


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_zed_frames_dir(session_dir: Path) -> Path:
    """Same convention as SensorFusion/sync_manifest.py::load_zed_frames --
    frames live in metadata.json's recording.frames_dir (default "frames")."""
    meta_path = session_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return session_dir / (meta.get("recording", {}).get("frames_dir") or "frames")


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-superpixel material/emissivity classification for a synced ZED session")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder holding metadata.json + frames/ + sync_manifest.json "
        "(the output of SensorFusion/sync_manifest.py).",
    )
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output root (default: <session-dir>/material_map).",
    )
    # --- segmenter ---------------------------------------------------------
    p.add_argument("--segmenter", choices=("sam", "slic"), default="sam",
                    help="sam (default): Segment Anything masks follow real objects -- the "
                         "floor comes out as one region -- at ~27 s/frame on CPU. "
                         "slic: ~0.4 s/frame, but it partitions the image into ~n_segments "
                         "cells regardless of content, so uniform surfaces get shredded.")
    p.add_argument("--n-segments", type=int, default=100, metavar="N",
                    help="Target SLIC superpixel count per frame (default 100). slic only.")
    p.add_argument("--compactness", type=float, default=10.0,
                    help="SLIC compactness -- higher favors regular shapes over edge fidelity (default 10.0).")
    p.add_argument("--sigma", type=float, default=0.0,
                    help="SLIC pre-blur, segmentation decision only (default 0.0 = historical "
                         "behaviour). 2-3 gives cleaner boundaries at no cost. slic only.")
    p.add_argument("--sam-model", default="facebook/sam-vit-base", metavar="REPO")
    p.add_argument("--sam-grid", type=int, default=16, metavar="N",
                    help="SAM point-prompt grid, N x N (default 16). Lower is faster.")
    p.add_argument("--sam-min-area", type=int, default=1500, metavar="PX",
                    help="Drop SAM masks smaller than this (default 1500).")
    p.add_argument("--sam-nms-iou", type=float, default=0.7, metavar="IOU",
                    help="Drop SAM masks overlapping a better one above this IoU (default 0.7).")
    # --- geometric prior ---------------------------------------------------
    p.add_argument("--zone-constraint", action=argparse.BooleanOptionalAction, default=True,
                    help="Restrict CLIP's candidate materials by region geometry "
                         "(floor/ceiling/vertical). On by default: it is free -- it reuses the "
                         "same forward pass -- and it makes a bare-metal call on an ordinary "
                         "surface impossible, which is the only misclassification that changes "
                         "the corrected temperature by more than ~1 degC. --no-zone-constraint "
                         "to disable. See emissivity/zones.py.")
    # --- crop to the FLIR field of view ----------------------------------
    p.add_argument("--crop-to-flir-fov", action=argparse.BooleanOptionalAction, default=True,
                    help="Segment/classify only the part of the ZED frame the FLIR can see "
                         "(~16%% of it on this rig; 26.5%% with the default margin). On by "
                         "default: project_to_flir.py keeps only points valid in BOTH cameras, "
                         "so everything outside is discarded downstream anyway, and cropping "
                         "cuts a 107-frame session from ~3.9 h to ~2.3 h. "
                         "--no-crop-to-flir-fov to segment the whole frame.")
    p.add_argument("--fov-margin-px", type=int, default=45, metavar="PX",
                    help="Pad the FLIR-FOV crop by this many ZED pixels (default 45). Covers "
                         "the error of composing the two LiDAR<->camera extrinsics.")
    p.add_argument("--calibration", default=None, metavar="YAML",
                    help="Rig calibration (default: ../Calibration/rig_calibration.yaml). "
                         "Read whenever --crop-to-flir-fov is on, which is the default.")
    p.add_argument("--top-k", type=int, default=3, metavar="N",
                    help="How many (material, confidence) candidates to keep per segment (default 3).")
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    # --- low-emissivity gate ---------------------------------------------
    # A low-e class is catastrophic downstream: the radiometric correction
    # divides by e, so e=0.07 amplifies the correction ~14x (a 37 degC
    # apparent reading becomes ~158 degC). Bare polished metal is also
    # essentially absent indoors. So a low-e class is only accepted on
    # strong evidence; otherwise the best non-low-e candidate is used.
    p.add_argument("--low-emissivity-max", type=float, default=0.5, metavar="E",
                    help="Classes with emissivity below this are gated (default 0.5).")
    p.add_argument("--low-emissivity-min-conf", type=float, default=0.50, metavar="P",
                    help="Min top-1 confidence to accept a low-emissivity class (default 0.50).")
    p.add_argument("--low-emissivity-min-margin", type=float, default=0.15, metavar="P",
                    help="Min confidence margin over the runner-up to accept a low-emissivity class (default 0.15).")
    p.add_argument("--no-gating", action="store_true",
                    help="Disable the low-emissivity gate (keep CLIP's raw top-1).")
    p.add_argument(
        "--clip-model", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="HF CLIP model. Default ViT-H/14 (LAION): on SAM masks it labels every radiator "
             "painted_metal at 0.85-1.00 where ViT-L/14 said plaster/glass/brick at 0.25-0.61, "
             "and its confidences are calibrated enough to be worth weighting by. Costs ~2.0 s "
             "per region vs 0.8 s, and ~3.9 GB on first download. Use "
             "openai/clip-vit-large-patch14 for the faster, weaker previous default.",
    )
    p.add_argument("--every-n", type=int, default=1, metavar="N",
                    help="Process every Nth triplet from sync_manifest.json (default 1 = all).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                    help="Stop after N processed frames (for a quick test run).")
    p.add_argument("--overlay", action="store_true",
                    help="Also save a superpixel-boundary + material-label PNG per frame.")
    return p.parse_args()


def apply_low_emissivity_gate(ranked, table, eps_max, min_conf, min_margin):
    """Decide a segment's material from the full ranking, refusing to hand out
    a low-emissivity class on weak evidence.

    `ranked` is the complete [(material, confidence), ...] list, best first.

    Returns (material, confidence, gate_info). gate_info is None when the gate
    did not fire, otherwise a dict recording what was overridden and why --
    the override is always auditable in segments.json, never silent.
    """
    top_material, top_conf = ranked[0]
    if table.lookup(top_material).emissivity >= eps_max:
        return top_material, top_conf, None

    runner_up_conf = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_conf - runner_up_conf
    if top_conf >= min_conf and margin >= min_margin:
        return top_material, top_conf, None    # strong enough, accept as-is

    # Weak low-e call: fall back to the best candidate above the threshold.
    for material, conf in ranked[1:]:
        if table.lookup(material).emissivity >= eps_max:
            return material, conf, {
                "overrode": top_material,
                "overrode_confidence": top_conf,
                "overrode_emissivity": table.lookup(top_material).emissivity,
                "margin": margin,
                "reason": ("confidence below threshold" if top_conf < min_conf
                            else "margin over runner-up below threshold"),
            }
    # Every class in the table is low-e (only possible with a custom table).
    return top_material, top_conf, {
        "overrode": None,
        "reason": "no candidate above the emissivity threshold; kept top-1",
    }


def draw_overlay(image: np.ndarray, labels: np.ndarray, segments: list[dict]) -> np.ndarray:
    import cv2
    from skimage.segmentation import mark_boundaries

    bounded = (mark_boundaries(image, labels, color=(0, 1, 0)) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(bounded, cv2.COLOR_RGB2BGR)
    for seg in segments:
        cx, cy = seg["centroid_px"]
        label = f"{seg['top_material'][:10]} e={seg['emissivity']:.2f}"
        cv2.putText(bgr, label, (int(cx) - 25, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 0), 1)
    return bgr


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)

    manifest_path = session_dir / "sync_manifest.json"
    if not manifest_path.exists():
        print(f"No sync_manifest.json in {session_dir} -- run SensorFusion/sync_manifest.py first.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_dir = load_zed_frames_dir(session_dir)

    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "material_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    classifier = MaterialClassifier(table, model_name=args.clip_model)
    # Used by the zone prior to drop catastrophic candidates where the zone
    # itself carries no categorical restriction.
    eps_of = {m: table.lookup(m).emissivity for m in table.materials}

    # Loaded once: SAM's weights are ~375 MB and loading them per frame would
    # dominate the runtime.
    sam_model = None
    if args.segmenter == "sam":
        from transformers import SamModel
        print(f"Loading SAM ({args.sam_model}) ...")
        sam_model = SamModel.from_pretrained(args.sam_model).eval()

    # Resolved on the first frame, once the ZED capture size is known.
    crop_box = None
    cal = None
    if args.crop_to_flir_fov:
        cal_path = args.calibration or (
            Path(__file__).resolve().parent.parent / "Calibration" / "rig_calibration.yaml")
        cal = load_rig_calibration(cal_path)

    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]
    seg_note = (f"SAM {args.sam_grid}x{args.sam_grid} prompts" if args.segmenter == "sam"
                else f"SLIC {args.n_segments} segments/frame target")
    print(f"Processing {len(triplets)} frame(s) from {manifest_path.name} ({seg_note})")
    if args.zone_constraint:
        print("Zone constraint: ON (candidates restricted by region geometry)")
    if args.no_gating:
        print("Low-emissivity gate: DISABLED (raw CLIP top-1)")
    else:
        print(f"Low-emissivity gate: e<{args.low_emissivity_max} needs "
              f"conf>={args.low_emissivity_min_conf:.2f} and margin>={args.low_emissivity_min_margin:.2f}")

    total_gated = 0
    total_zoned = 0
    total_segments = 0

    for i, triplet in enumerate(triplets):
        t0 = time.time()
        n_gated = 0
        n_zoned = 0
        zed_file = triplet["zed"]["file"]
        flir_file = triplet["flir"]["file"]
        flir_stem = Path(flir_file).stem

        image = np.asarray(Image.open(frames_dir / zed_file).convert("RGB"))

        # The crop is a fixed rig property, so it is computed once, on the
        # first frame (which is where the ZED capture size becomes known).
        if args.crop_to_flir_fov and crop_box is None:
            zh, zw = image.shape[:2]
            crop_box = flir_fov_bbox_in_zed(cal, zw, zh, margin_px=args.fov_margin_px)
            cx0, cy0, cx1, cy1 = crop_box
            pct = 100.0 * (cx1 - cx0) * (cy1 - cy0) / (zw * zh)
            print(f"FLIR FOV in ZED: x[{cx0} {cx1}] y[{cy0} {cy1}]  "
                  f"{cx1 - cx0}x{cy1 - cy0} px = {pct:.1f}% of the frame "
                  f"(margin {args.fov_margin_px} px)")

        if crop_box is None:
            view, off_x, off_y = image, 0, 0
        else:
            cx0, cy0, cx1, cy1 = crop_box
            view, off_x, off_y = image[cy0:cy1, cx0:cx1], cx0, cy0

        if args.segmenter == "sam":
            labels = sam_segments(view, model_name=args.sam_model, grid=args.sam_grid,
                                  min_area=args.sam_min_area, nms_iou=args.sam_nms_iou,
                                  model=sam_model)
        else:
            labels = superpixel_segments(view, n_segments=args.n_segments,
                                         compactness=args.compactness, sigma=args.sigma)
        boxes = segment_boxes(labels)
        vh, vw = view.shape[:2]
        zones = [zone_of(seg, vh, vw) for seg in boxes]

        crops = [view[y0:y1, x0:x1] for seg in boxes for (x0, y0, x1, y1) in [seg["bbox"]]]

        # From here on everything is reported in full-frame ZED pixels, so the
        # output is identical in meaning whether or not the crop was applied.
        if crop_box is not None:
            for seg in boxes:
                bx0, by0, bx1, by1 = seg["bbox"]
                seg["bbox"] = (bx0 + off_x, by0 + off_y, bx1 + off_x, by1 + off_y)
                seg["centroid_px"] = [seg["centroid_px"][0] + off_x,
                                      seg["centroid_px"][1] + off_y]
            labels_full = np.full(image.shape[:2], -1, dtype=np.int32)
            labels_full[cy0:cy1, cx0:cx1] = labels
            labels = labels_full
        # Rank against EVERY class, not just top-k: the gate needs a fallback
        # candidate, and the top-3 can legitimately be all-metal (a grey
        # reflective blob ranks steel/aluminium/copper 1-2-3), leaving nothing
        # above the emissivity threshold to fall back to.
        results = classifier.classify_batch(crops, top_k=len(table.materials))

        segments = []
        for seg, zone, ranked in zip(boxes, zones, results):
            # Geometric prior first: the gate below then works on candidates
            # that are already possible for this kind of surface.
            if args.zone_constraint:
                restricted = restrict_ranking(ranked, zone, eps_of=eps_of,
                                              min_eps=args.low_emissivity_max)
                if restricted is not ranked:
                    n_zoned += ranked[0][0] != restricted[0][0]
                    ranked = restricted
            if args.no_gating:
                material, confidence = ranked[0]
                gate_info = None
            else:
                material, confidence, gate_info = apply_low_emissivity_gate(
                    ranked, table,
                    args.low_emissivity_max,
                    args.low_emissivity_min_conf,
                    args.low_emissivity_min_margin,
                )
            rec = table.lookup(material)
            record = {
                "id": seg["id"],
                "centroid_px": seg["centroid_px"],
                "area_px": seg["area_px"],
                "top_material": material,
                "confidence": confidence,
                "emissivity": rec.emissivity,
                "solar_absorptance": rec.solar_absorptance,
                "top_k": [(m, c) for m, c in ranked[:args.top_k]],
            }
            if args.zone_constraint:
                record["zone"] = zone
            if gate_info is not None:
                record["gated"] = gate_info
                n_gated += 1
            segments.append(record)

        frame_dir = out_dir / flir_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "labels.npy", labels.astype(np.int32))
        (frame_dir / "segments.json").write_text(json.dumps({
            "schema": "material_map/v1",
            "generated_by": "classify_session.py",
            "generated_utc": utc_now_iso(),
            "source_zed_frame": zed_file,
            "source_flir_frame": flir_file,
            "n_segments": len(segments),
            # null when the whole frame was used; otherwise the FLIR-FOV crop
            # the superpixels were computed in (full-frame ZED pixels).
            "flir_fov_crop": None if crop_box is None else {
                "x0": crop_box[0], "y0": crop_box[1],
                "x1": crop_box[2], "y1": crop_box[3],
                "margin_px": args.fov_margin_px,
            },
            "gate": {
                "enabled": not args.no_gating,
                "low_emissivity_max": args.low_emissivity_max,
                "min_confidence": args.low_emissivity_min_conf,
                "min_margin": args.low_emissivity_min_margin,
                "n_gated": n_gated,
            },
            "segments": segments,
        }, indent=2), encoding="utf-8")

        if args.overlay:
            import cv2
            cv2.imwrite(str(frame_dir / "overlay.png"), draw_overlay(image, labels, segments))

        total_gated += n_gated
        total_zoned += n_zoned
        total_segments += len(segments)
        dt = time.time() - t0
        gated_note = f", {n_gated} gated" if n_gated else ""
        zoned_note = f", {n_zoned} zone-corrected" if n_zoned else ""
        print(f"[{i + 1}/{len(triplets)}] {flir_stem}: {len(segments)} segments in {dt:.1f}s "
              f"({dt / max(1, len(segments)) * 1000:.0f} ms/segment){gated_note}{zoned_note}")

    if total_segments:
        print(f"\nLow-emissivity gate fired on {total_gated}/{total_segments} segments "
              f"({100.0 * total_gated / total_segments:.1f}%)")
        if args.zone_constraint:
            print(f"Zone constraint changed the material on {total_zoned}/{total_segments} "
                  f"segments ({100.0 * total_zoned / total_segments:.1f}%)")
    print(f"Done. Output in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
