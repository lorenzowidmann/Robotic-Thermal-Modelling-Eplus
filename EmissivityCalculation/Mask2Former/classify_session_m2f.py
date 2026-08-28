"""Material + emissivity per region, with Mask2Former as the segmenter and the
ADE20K class as the material prior.

Same job as ../classify_session.py, same output, same downstream consumers
(../project_to_flir.py, ../voxel_consensus.py). Two things change, and only two.

1. The segmenter
----------------
SAM-everything (or SLIC) is replaced by one Mask2Former forward pass on the
ADE20K-150 taxonomy -- the same switch WindowsDoorsDetection made for door and
window detection, for the same reasons: SAM decoded its masks at 256x256 and
resized them to 1920x1080 with INTER_NEAREST, so boundaries arrived as ~7 px
staircases and the gap-fill invented the rest; and it cost ~27 s/frame against
4.7-5.2 s here. See m2f_materials/segmentation_m2f.py.

2. Where the prior comes from
-----------------------------
../emissivity/zones.py::zone_of decides which materials CLIP is allowed to pick
from the region's BBOX SHAPE -- wide and low means floor, tall means vertical.
That prior is doing real work (unconstrained on a full frame, ViT-H proposed a
bare metal on 27% of regions, and one survived the gate at 0.79, which would
have turned a 37 degC reading into ~156 degC), but its source is a guess, and
the guess has a documented failure mode: a wide region high in the frame is
called `ceiling` whatever it is.

Mask2Former already knows what the region IS, so the prior is read off the ADE
class instead: ade_material_prior.csv says a `column` is not made of glass, a
`ceiling` is not made of glass, a `windowpane` is not made of rubber. An ADE
label with no row keeps the old `any` behaviour -- no categorical list, only the
emissivity floor that keeps bare metal out.

Everything else is deliberately untouched: CLIP still chooses the material from
../emissivity_table.csv's prompts, the low-emissivity gate is the same function
with the same defaults, the FLIR-FOV crop is the same, and segments.json is
still schema "material_map/v1" with only additive keys (`ade`, `ade_confidence`,
`crop`, `prior`). `zone` is still written -- ADE-derived now -- because
../voxel_consensus.py re-applies the geometric prior on it when pooling votes
across frames.

What CLIP is shown
------------------
Still the region's bounding box, as in ../classify_session.py. An ADE region is
object-shaped, so its bbox is often mostly other objects -- 54% foreign pixels
on an L-shaped wall region -- and two alternatives that fix that are
implemented (--crop-mode masked, --crop-mode texture). Measured on this rig
they are both WORSE: ../emissivity_table.csv's prompts name objects ("a painted
metal radiator", "a window pane"), so deleting the object leaves CLIP less to
go on, not more. The numbers are in m2f_materials/crops.py.

Output per frame, under <out-dir>/<flir_frame_stem>/:
    labels.npy    -- int32 HxW region-id raster, ZED pixel grid, always
                     full-frame; -1 outside the crop and wherever no region
                     reached --min-area, so ../project_to_flir.py needs no
                     change either way
    segments.json -- schema "material_map/v1"
    overlay.png   -- optional (--overlay): region contours labelled
                     <ade> -> <material> e=<eps>

Venv: C:\\venvs\\emissivity (torch, torchvision, transformers, opencv, pandas).

Usage:
    py classify_session_m2f.py --session-dir ...\\ZED\\20260730_161223\\fullrate --limit 3 --overlay
    py classify_session_m2f.py --session-dir ...\\fullrate --every-n 5

    # the prior off, to see what it was doing:
    py classify_session_m2f.py --session-dir ...\\fullrate --limit 3 --no-category-prior
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

# --- the parent module ------------------------------------------------------
# ../emissivity/ is imported unchanged: the emissivity table, the CLIP
# classifier and the gate are shared with ../classify_session.py so there is
# one emissivity_table.csv, which ../voxel_consensus.py also reads.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from emissivity import EmissivityTable, MaterialClassifier          # noqa: E402
from m2f_materials import category_prior as prior_mod               # noqa: E402
from m2f_materials import crops as crop_mod                         # noqa: E402
from m2f_materials import segmentation_m2f as m2f                   # noqa: E402


def _find_root(start: Path) -> Path:
    """Walk up until the directory holding SensorFusionLoader/ is found.

    Copied from WindowsDoorsDetection/classify_openings.py, and it matters here:
    ../classify_session.py hardcodes `../Calibration`, which does not exist in
    this repo -- the calibration loader lives in SensorFusionLoader/ -- so its
    --crop-to-flir-fov path is broken as checked in. Searching also survives the
    next time this module is moved, which a fixed count of .parent hops does not.
    """
    for d in [start, *start.parents]:
        if (d / "SensorFusionLoader").is_dir():
            return d
    raise RuntimeError(
        f"SensorFusionLoader/ not found in any parent of {start} -- it holds "
        "rig_calibration.py/.yaml and projection.py, which --crop-to-flir-fov needs.")


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
        description="Per-region material/emissivity classification for a synced ZED "
                    "session, segmented by Mask2Former and primed by its ADE class")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder holding metadata.json + frames/ + sync_manifest.json "
             "(the output of SensorFusion/sync_manifest.py).",
    )
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Output root (default: <session-dir>/material_map_m2f). Deliberately not "
             "material_map/, so a run here never overwrites ../classify_session.py's "
             "output and the two can be diffed.",
    )
    # --- segmenter ---------------------------------------------------------
    p.add_argument("--model", default=m2f.DEFAULT_MODEL, metavar="REPO",
                   help="Mask2Former semantic checkpoint on the ADE20K-150 taxonomy "
                        f"(default {m2f.DEFAULT_MODEL}). A panoptic or COCO checkpoint "
                        "will load but has different labels and will fail the ade lookup "
                        "in ade_material_prior.csv.")
    p.add_argument("--min-area", type=int, default=m2f.MIN_COMPONENT_AREA, metavar="PX",
                   help="Connected components below this never become a region: they stay "
                        f"at -1 in labels.npy and are never classified (default "
                        f"{m2f.MIN_COMPONENT_AREA}, the number SAM used as --sam-min-area, "
                        "kept so region counts stay comparable across the switch).")
    p.add_argument("--crop-mode", choices=("bbox", "masked", "texture"), default="bbox",
                   help="What CLIP is shown per region. bbox (default): the region's "
                        "bounding box, exactly what ../classify_session.py does -- measured "
                        "best of the three on this rig, see m2f_materials/crops.py for the "
                        "numbers and why. masked: bbox crop with the outside of the mask "
                        "flattened to the region's mean colour. texture: a 224x224 swatch "
                        "tiled from patches cut strictly inside the mask, so an L-shaped "
                        "wall region is not judged on the door inside its bounding box -- "
                        "sound in principle, but it deletes the object-level cue "
                        "../emissivity_table.csv's prompts are written against.")
    # --- the ADE category prior -------------------------------------------
    p.add_argument("--category-prior", action=argparse.BooleanOptionalAction, default=True,
                   help="Restrict CLIP's candidate materials by the region's ADE class "
                        "(default on). This is the ../emissivity/zones.py prior with a "
                        "semantic source instead of a bbox-shape one: a column cannot be "
                        "glass, a windowpane cannot be rubber. Free -- it reuses the same "
                        "forward pass, since restricting a softmax and renormalising does "
                        "not change the ordering. --no-category-prior to disable.")
    p.add_argument("--prior-table", default=None, metavar="CSV",
                   help="Alternative ade_material_prior.csv (group,ade,zone,materials,notes).")
    # --- crop to the FLIR field of view ------------------------------------
    p.add_argument("--crop-to-flir-fov", action=argparse.BooleanOptionalAction, default=True,
                   help="Segment/classify only the part of the ZED frame the FLIR can see "
                        "(~16%% of it on this rig; 26.5%% with the default margin). On by "
                        "default: ../project_to_flir.py keeps only points valid in BOTH "
                        "cameras, so everything outside is discarded downstream anyway. "
                        "--no-crop-to-flir-fov to segment the whole frame.")
    p.add_argument("--fov-margin-px", type=int, default=45, metavar="PX",
                   help="Pad the FLIR-FOV crop by this many ZED pixels (default 45). Covers "
                        "the error of composing the two LiDAR<->camera extrinsics.")
    p.add_argument("--calibration", default=None, metavar="YAML",
                   help="Rig calibration (default: SensorFusionLoader/rig_calibration.yaml). "
                        "Read whenever --crop-to-flir-fov is on, which is the default.")
    p.add_argument("--top-k", type=int, default=3, metavar="N",
                   help="How many (material, confidence) candidates to keep per segment (default 3).")
    p.add_argument("--table", default=None, help="Path to a custom emissivity CSV")
    # --- low-emissivity gate ------------------------------------------------
    # Unchanged from ../classify_session.py, defaults included. A low-e class is
    # catastrophic downstream: the radiometric correction divides by e, so
    # e=0.07 amplifies it ~14x (37 degC apparent becomes ~158 degC). The ADE
    # prior above now makes most of those impossible by construction, but the
    # gate stays: it is the net for unmapped ADE labels.
    p.add_argument("--low-emissivity-max", type=float, default=0.5, metavar="E",
                   help="Classes with emissivity below this are gated (default 0.5).")
    p.add_argument("--low-emissivity-min-conf", type=float, default=0.50, metavar="P",
                   help="Min top-1 confidence to accept a low-emissivity class (default 0.50).")
    p.add_argument("--low-emissivity-min-margin", type=float, default=0.15, metavar="P",
                   help="Min confidence margin over the runner-up to accept a low-emissivity "
                        "class (default 0.15).")
    p.add_argument("--no-gating", action="store_true",
                   help="Disable the low-emissivity gate (keep CLIP's raw top-1).")
    p.add_argument(
        "--clip-model", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        help="HF CLIP model. Default ViT-H/14 (LAION): on real masks it labels every "
             "radiator painted_metal at 0.85-1.00 where ViT-L/14 said plaster/glass/brick "
             "at 0.25-0.61. Costs ~2.0 s per region vs 0.8 s, and ~3.9 GB on first "
             "download. Use openai/clip-vit-large-patch14 for the faster, weaker one.",
    )
    p.add_argument("--every-n", type=int, default=1, metavar="N",
                   help="Process every Nth triplet from sync_manifest.json (default 1 = all).")
    p.add_argument("--limit", type=int, default=None, metavar="N",
                   help="Stop after N processed frames (for a quick test run).")
    p.add_argument("--overlay", action="store_true",
                   help="Also save a region-contour + material-label PNG per frame.")
    p.add_argument("--check-only", action="store_true",
                   help="Load the checkpoint and ade_material_prior.csv, validate the two "
                        "against each other, print the mapping and exit. No frames are "
                        "read and CLIP is never loaded -- the cheap way to check a table "
                        "edit.")
    return p.parse_args()


def apply_low_emissivity_gate(ranked, table, eps_max, min_conf, min_margin):
    """Decide a segment's material from the full ranking, refusing to hand out a
    low-emissivity class on weak evidence.

    Copied unchanged from ../classify_session.py -- the gate is not what this
    module is changing, and keeping it byte-identical is what makes a diff
    between the two scripts' outputs attributable to the segmenter and the
    prior.

    `ranked` is the complete [(material, confidence), ...] list, best first.

    Returns (material, confidence, gate_info). gate_info is None when the gate
    did not fire, otherwise a dict recording what was overridden and why -- the
    override is always auditable in segments.json, never silent.
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
    """Region CONTOURS, not bounding boxes, labelled `<ade> -> <material>`.

    Contours because a box misrepresents what was classified: a thin reveal
    strip beside a window bay looks contained in the bay on boxes and separate
    on masks, and it is the mask that CLIP was shown. Same reasoning, and the
    same cv2.findContours approach, as
    WindowsDoorsDetection/classify_openings.py::draw_overlay.

    The ADE name is printed next to the material because that is the pair that
    has to be argued with: `column -> concrete` is right, `column -> glass`
    would mean the prior table is wrong, and neither is visible from the
    material alone.
    """
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).copy()
    for seg in segments:
        mask = (labels == int(seg["id"])).astype(np.uint8)
        if not mask.any():
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Colour by whether anything overrode CLIP: green ordinary, orange the
        # prior stepped in, red the low-e gate fired.
        color = (0, 255, 0)
        if seg.get("prior"):
            color = (0, 165, 255)
        if seg.get("gated"):
            color = (0, 0, 255)
        cv2.drawContours(bgr, contours, -1, color, 2)
        cx, cy = seg["centroid_px"]
        text = f"{seg['ade']} -> {seg['top_material']} e={seg['emissivity']:.2f}"
        cv2.putText(bgr, text, (int(cx) - 60, int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)
    return bgr


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)

    # --- the two tables, validated against each other before anything heavy --
    table = EmissivityTable(args.table) if args.table else EmissivityTable()
    prior = prior_mod.AdeMaterialPrior(
        table, args.prior_table or prior_mod.DEFAULT_PRIOR_TABLE)
    # Used for the unmapped-ADE fallback, where the only restriction left is
    # the emissivity floor.
    eps_of = {m: table.lookup(m).emissivity for m in table.materials}

    print(f"Loading {args.model} ...")
    t_load = time.time()
    proc, model = m2f.load_model(args.model)
    names = set(m2f.ade_name_map(model).values())
    missing = sorted(set(prior.ade_labels) - names)
    if missing:
        print(f"{prior.path.name} maps ADE labels this checkpoint does not have: {missing}. "
              "Those rows can never fire, which silently widens the prior -- fix the "
              "spelling or the checkpoint.", file=sys.stderr)
        return 1
    print(f"  loaded in {time.time() - t_load:.1f}s; {len(prior.groups)} prior group(s) "
          f"over {len(prior.ade_labels)} ADE label(s), "
          f"{len(names - set(prior.ade_labels))} label(s) unmapped "
          f"(-> emissivity floor only)")
    for rec in prior.groups:
        print(f"    {rec.group:<12} [{rec.zone:^8}] {', '.join(rec.materials)}")
    if args.check_only:
        print("--check-only: tables agree with the checkpoint. Nothing else run.")
        return 0

    manifest_path = session_dir / "sync_manifest.json"
    if not manifest_path.exists():
        print(f"No sync_manifest.json in {session_dir} -- run SensorFusion/sync_manifest.py "
              "first.", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames_dir = load_zed_frames_dir(session_dir)

    out_dir = Path(args.out_dir) if args.out_dir else session_dir / "material_map_m2f"
    out_dir.mkdir(parents=True, exist_ok=True)

    classifier = MaterialClassifier(table, model_name=args.clip_model)

    # Resolved on the first frame, once the ZED capture size is known.
    crop_box = None
    cal = None
    flir_fov_bbox_in_zed = None
    if args.crop_to_flir_fov:
        root = _find_root(_HERE)
        sys.path.insert(0, str(root / "SensorFusionLoader"))
        from rig_calibration import load_rig_calibration
        from projection import flir_fov_bbox_in_zed
        cal = load_rig_calibration(
            args.calibration or (root / "SensorFusionLoader" / "rig_calibration.yaml"))

    triplets = manifest["triplets"][::args.every_n]
    if args.limit:
        triplets = triplets[:args.limit]
    print(f"Processing {len(triplets)} frame(s) from {manifest_path.name} "
          f"(Mask2Former, min area {args.min_area} px, --crop-mode {args.crop_mode})")
    if args.category_prior:
        print(f"Category prior: ON ({prior.path.name}; candidates restricted by ADE class)")
    else:
        print("Category prior: OFF (CLIP scores every material on every region)")
    if args.no_gating:
        print("Low-emissivity gate: DISABLED (raw CLIP top-1)")
    else:
        print(f"Low-emissivity gate: e<{args.low_emissivity_max} needs "
              f"conf>={args.low_emissivity_min_conf:.2f} and "
              f"margin>={args.low_emissivity_min_margin:.2f}")

    total_gated = 0
    total_primed = 0
    total_segments = 0
    total_ade = Counter()

    for i, triplet in enumerate(triplets):
        t0 = time.time()
        n_gated = 0
        n_primed = 0
        zed_file = triplet["zed"]["file"]
        flir_file = triplet["flir"]["file"]
        flir_stem = Path(flir_file).stem

        image = np.asarray(Image.open(frames_dir / zed_file).convert("RGB"))

        # The crop is a fixed rig property, so it is computed once, on the first
        # frame (which is where the ZED capture size becomes known).
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

        labels, regions, masks = m2f.m2f_regions(view, proc, model, min_area=args.min_area)
        for r in regions:
            total_ade[r["ade"]] += 1

        # What CLIP is shown, built from the mask rather than the bbox -- see
        # m2f_materials/crops.py for why the default is a texture swatch.
        crops, crop_notes = [], []
        for reg, mask in zip(regions, masks):
            crop, note = crop_mod.region_crop(view, mask, reg["bbox"], args.crop_mode,
                                              seed=reg["id"])
            crops.append(crop)
            crop_notes.append(note)

        # From here on everything is reported in full-frame ZED pixels, so the
        # output is identical in meaning whether or not the crop was applied.
        if crop_box is not None:
            for reg in regions:
                bx0, by0, bx1, by1 = reg["bbox"]
                reg["bbox"] = (bx0 + off_x, by0 + off_y, bx1 + off_x, by1 + off_y)
                reg["centroid_px"] = [reg["centroid_px"][0] + off_x,
                                      reg["centroid_px"][1] + off_y]
            labels_full = np.full(image.shape[:2], -1, dtype=np.int32)
            labels_full[cy0:cy1, cx0:cx1] = labels
            labels = labels_full

        # Rank against EVERY class, not just top-k: the gate needs a fallback
        # candidate, and the top-3 can legitimately be all-metal (a grey
        # reflective blob ranks steel/aluminium/copper 1-2-3), leaving nothing
        # above the emissivity threshold to fall back to.
        results = (classifier.classify_batch(crops, top_k=len(table.materials))
                   if crops else [])

        segments = []
        for reg, note, ranked in zip(regions, crop_notes, results):
            ade = reg["ade"]
            allowed = prior.candidates(ade) if args.category_prior else None
            # Category prior first: the gate below then works on candidates that
            # are already possible for this kind of object.
            if args.category_prior:
                restricted = prior_mod.restrict_to_candidates(
                    ranked, allowed, eps_of=eps_of, min_eps=args.low_emissivity_max)
                prior_info = None
                if restricted is not ranked and ranked[0][0] != restricted[0][0]:
                    prior_info = {
                        "group": prior.group_of(ade),
                        "overrode": ranked[0][0],
                        "overrode_confidence": ranked[0][1],
                        "reason": (f"{ade} cannot be {ranked[0][0]}"
                                   if allowed else
                                   f"{ade} is unmapped; emissivity floor "
                                   f"{args.low_emissivity_max}"),
                    }
                    n_primed += 1
                ranked = restricted
            else:
                prior_info = None

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
                "id": reg["id"],
                "centroid_px": reg["centroid_px"],
                "area_px": reg["area_px"],
                "top_material": material,
                "confidence": confidence,
                "emissivity": rec.emissivity,
                "solar_absorptance": rec.solar_absorptance,
                "top_k": [(m, c) for m, c in ranked[:args.top_k]],
                # Additive to material_map/v1: what the segmenter said, how
                # sure it was, and what CLIP was actually shown.
                "ade": ade,
                "ade_confidence": reg["ade_confidence"],
                "crop": note,
                # ADE-derived, and still written under this name because
                # ../voxel_consensus.py re-applies the geometric prior on it
                # when it pools votes across frames.
                "zone": prior.zone(ade),
            }
            if args.category_prior:
                record["prior_group"] = prior.group_of(ade)
                if allowed:
                    record["allowed_materials"] = list(allowed)
                if prior_info is not None:
                    record["prior"] = prior_info
            if gate_info is not None:
                record["gated"] = gate_info
                n_gated += 1
            segments.append(record)

        frame_dir = out_dir / flir_stem
        frame_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "labels.npy", labels.astype(np.int32))
        (frame_dir / "segments.json").write_text(json.dumps({
            "schema": "material_map/v1",
            "generated_by": "classify_session_m2f.py",
            "generated_utc": utc_now_iso(),
            "source_zed_frame": zed_file,
            "source_flir_frame": flir_file,
            "n_segments": len(segments),
            "segmenter": {
                "kind": "mask2former",
                "model": args.model,
                "min_area_px": args.min_area,
                "crop_mode": args.crop_mode,
            },
            # null when the whole frame was used; otherwise the FLIR-FOV crop
            # the regions were computed in (full-frame ZED pixels).
            "flir_fov_crop": None if crop_box is None else {
                "x0": crop_box[0], "y0": crop_box[1],
                "x1": crop_box[2], "y1": crop_box[3],
                "margin_px": args.fov_margin_px,
            },
            # Replaces the old zone_constraint block. `zone` per segment is
            # still written for ../voxel_consensus.py, but it is now read off
            # the ADE class rather than guessed from bbox shape.
            "category_prior": {
                "enabled": bool(args.category_prior),
                "table": str(prior.path),
                "n_corrected": n_primed,
                "note": "candidates restricted by ADE class; supersedes zones.zone_of",
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
        total_primed += n_primed
        total_segments += len(segments)
        dt = time.time() - t0
        gated_note = f", {n_gated} gated" if n_gated else ""
        primed_note = f", {n_primed} prior-corrected" if n_primed else ""
        print(f"[{i + 1}/{len(triplets)}] {flir_stem}: {len(segments)} regions in {dt:.1f}s "
              f"({dt / max(1, len(segments)) * 1000:.0f} ms/region){gated_note}{primed_note}")

    if total_segments:
        print(f"\nLow-emissivity gate fired on {total_gated}/{total_segments} segments "
              f"({100.0 * total_gated / total_segments:.1f}%)")
        if args.category_prior:
            print(f"Category prior changed the material on {total_primed}/{total_segments} "
                  f"segments ({100.0 * total_primed / total_segments:.1f}%)")
        print("regions by ADE class: " +
              ", ".join(f"{a}={n}" for a, n in total_ade.most_common(15)))
    print(f"Done. Output in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
