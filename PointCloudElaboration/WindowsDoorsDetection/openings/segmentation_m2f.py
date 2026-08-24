"""Mask2Former ADE20K segmentation + classification in one forward pass.

Replaces the SAM-everything + CLIP-zero-shot pair that stage 1 used to run.
That pair had two structural problems, both visible on session 9 overlays:

  * SAM's masks were decoded at 256x256 and resized to 1920x1080 with
    INTER_NEAREST, so every boundary arrived as a ~7 px staircase, and
    `_fill_gaps` then painted the unlabelled remainder with the nearest id --
    inventing boundaries where SAM had abstained.
  * CLIP had to answer "door / window / other" from a crop, with `other`
    carrying one prompt for walls, floors, ceilings, radiators, pillars,
    people and clutter. It was the least well-posed of the three.

Mask2Former answers both questions at once, from a taxonomy (ADE20K-150) that
already contains `windowpane` and `door` as first-class entries, and that also
gives us `floor`, `ceiling` and `wall` for free -- which is what lets
zone_prior's bbox-shape heuristic be retired (see `zone_from_ade`).

Measured on session 9, 7 frames, CPU: 4.7-5.2 s/frame against SAM's ~27 s.

Per-pixel confidence
--------------------
`post_process_semantic_segmentation` returns an argmax and nothing else, but
stage 2 gates votes on `--min-vote-confidence 0.5`, so a real number is
needed. Mask2Former's semantic map is

    seg[c] = sum_q softmax(class_logits[q])[c] * sigmoid(mask_logits[q])

and the confidence reported here is the winner's share of that sum,
`seg.max(c) / seg.sum(c)`. It is a well-defined per-pixel quantity in [0, 1]
and it is NOT a calibrated probability -- treat it as a ranking score.

The einsum is done at FULL resolution, after upsampling the query masks,
because doing it at the decoder's native ~1/4 resolution and upsampling the
argmax afterwards would reintroduce exactly the staircase this module exists
to remove. It is row-chunked to keep the (150, H, W) intermediate off the
heap: at 1920x1080 that tensor alone is 1.24 GB.
"""

import numpy as np

DEFAULT_MODEL = "facebook/mask2former-swin-large-ade-semantic"

# ADE20K names that stand in for the zone prior's geometric guesses. Matched
# on the first comma-separated synonym of id2label, lowercased, so this
# survives the checkpoint spelling its labels "floor;flooring".
FLOOR_ADE_NAMES = {"floor", "rug", "path", "road", "earth", "sand"}
CEILING_ADE_NAMES = {"ceiling", "sky"}

# Components smaller than this never become a segment id: they stay at -1 in
# the raster and therefore never vote in stage 2. Same number SAM used as
# --sam-min-area, kept so the two are comparable frame to frame.
MIN_COMPONENT_AREA = 1500


def _first_name(label: str) -> str:
    return label.split(",")[0].strip().lower()


def ade_name_map(model) -> dict[int, str]:
    """{ade id -> first synonym}, read off the checkpoint rather than hardcoded.

    The 150-class list is stable across the ADE checkpoints, but reading it
    from config means a mismatched checkpoint fails loudly at the lookup
    instead of silently classifying the wrong index.
    """
    return {int(i): _first_name(name) for i, name in model.config.id2label.items()}


def zone_from_ade(ade_name: str, bbox) -> str:
    """The zone prior, taken from semantics instead of from bbox shape.

    zone_prior.py's README section "The ceiling rule eats high windows"
    documents the failure this replaces: zone_of()'s `cy < 0.30 and
    bw > 0.8 * bh` was tuned on FLIR-FOV-cropped frames, and on the uncropped
    frames this stage runs, a wide window high in the frame satisfies it and
    is forced to `other` -- undetectable by construction.

    Here a region is `ceiling` because Mask2Former called it ceiling, not
    because of where its box sits, so a clerestory stays a windowpane. The
    vertical/any split is kept only because segments.json's consumers read the
    key; nothing downstream restricts on it any more.
    """
    if ade_name in FLOOR_ADE_NAMES:
        return "floor"
    if ade_name in CEILING_ADE_NAMES:
        return "ceiling"
    x0, y0, x1, y1 = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    return "vertical" if bh > 1.3 * bw else "any"


def zone_for_merged(rec: dict, image_height: int, image_width: int) -> str:
    """The `zone_fn` stage 1B calls when it builds a MERGED region's record.

    A merged region is by construction a door or a window -- geometry.py only
    merges those two classes -- so it can never be floor or ceiling, and the
    only thing left to decide is the vertical/any shape split. Signature
    matches the zone_of() it replaces so apply_geometry_filter does not care
    which it was handed.
    """
    return zone_from_ade("", rec["bbox"])


def load_model(model_name: str = DEFAULT_MODEL):
    """(processor, model). Loaded once by the caller and reused across frames --
    the swin-large weights are ~850 MB."""
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).eval()
    return proc, model


def semantic_with_confidence(image: np.ndarray, proc, model,
                             row_chunk: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """RGB HxWx3 uint8 -> (ade_ids int32 HxW, confidence float32 HxW).

    See the module docstring for what `confidence` means and why the einsum
    runs at full resolution in row chunks.
    """
    import torch

    h, w = image.shape[:2]
    inputs = proc(images=image, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)

    # (Q, C) -- the no-object column is dropped, as post_process does.
    class_probs = out.class_queries_logits.softmax(-1)[0, :, :-1]
    masks = torch.nn.functional.interpolate(
        out.masks_queries_logits, size=(h, w), mode="bilinear", align_corners=False)[0]
    masks = masks.sigmoid()                         # (Q, H, W)

    sem = np.empty((h, w), np.int32)
    conf = np.empty((h, w), np.float32)
    with torch.no_grad():
        for r0 in range(0, h, row_chunk):
            r1 = min(h, r0 + row_chunk)
            seg = torch.einsum("qc,qhw->chw", class_probs, masks[:, r0:r1])
            best, idx = seg.max(0)
            total = seg.sum(0).clamp_min(1e-6)
            sem[r0:r1] = idx.numpy().astype(np.int32)
            conf[r0:r1] = (best / total).numpy().astype(np.float32)
    return sem, conf


def m2f_segments(image: np.ndarray, proc, model, ade_to_class: dict[str, str],
                 other_class: str, min_area: int = MIN_COMPONENT_AREA):
    """One frame -> (labels, segments, sem, conf).

    `labels` is an int32 HxW raster, one id per connected component of one ADE
    class, -1 where the component was below `min_area`. Unlike SAM's path there
    is no gap fill: a pixel that did not make it into a component simply does
    not vote, which is the honest behaviour and is what stage 2 already assumes
    for sid < 0.

    `segments` carries the same keys stage 1B and stage 2 read -- id, bbox,
    centroid_px, area_px, top_class, confidence, top_k, zone -- plus `ade` (the
    ADE class name the region came from), which is the audit trail for why a
    region got the class it did.

    Components are split per ADE class, NOT merged across classes: a mullion
    labelled `wall` between two `windowpane` bays stays its own region, and it
    is stage 1B's dilation that decides whether the bays fuse.
    """
    import cv2

    names = ade_name_map(model)
    sem, conf = semantic_with_confidence(image, proc, model)

    labels = np.full(sem.shape, -1, np.int32)
    segments = []
    next_id = 0
    for ade_id in np.unique(sem):
        ade_id = int(ade_id)
        ade_name = names.get(ade_id, f"ade_{ade_id}")
        cls = ade_to_class.get(ade_name, other_class)
        mask = (sem == ade_id).astype(np.uint8)
        n, comp, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            region = comp == k
            ys, xs = np.nonzero(region)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            c = float(conf[region].mean())
            labels[region] = next_id
            segments.append({
                "id": next_id,
                "bbox": bbox,
                "centroid_px": [float(xs.mean()), float(ys.mean())],
                "area_px": area,
                "top_class": cls,
                "confidence": c,
                "top_k": [(cls, c)],
                "zone": zone_from_ade(ade_name, bbox),
                "ade": ade_name,
            })
            next_id += 1
    return labels, segments, sem, conf
