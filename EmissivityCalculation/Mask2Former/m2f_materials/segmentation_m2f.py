"""Mask2Former ADE20K segmentation, as the material pipeline's segmenter.

Adapted from PointCloudElaboration/WindowsDoorsDetection/openings/
segmentation_m2f.py, which introduced it for door/window detection. Everything
about the forward pass, the per-pixel confidence and the connected-component
split is kept verbatim from there; read that module's header for the full
argument. The short version:

  * ../emissivity/segmentation.py::sam_segments decodes masks at 256x256 and
    resizes to 1920x1080 with INTER_NEAREST, so every boundary arrives as a
    ~7 px staircase, and `_fill_gaps` then paints the unlabelled remainder with
    the nearest id -- inventing boundaries where SAM abstained. Regions here
    come out of a semantic argmax computed at FULL resolution instead.
  * ~4.7-5.2 s/frame on CPU against SAM's ~27 s, measured on session 9.
  * The ADE class of every region is carried through as `ade`, which is what
    category_prior.py turns into a material restriction. SAM produced no such
    label, which is why the prior used to have to be guessed from bbox shape.

What this module does NOT do, unlike the openings version
---------------------------------------------------------
It does not assign a class. There, ADE `door` IS the answer. Here the ADE class
is only the prior -- the material still comes from CLIP, because ADE says what
an object is, not what it is made of, and emissivity depends on the latter
(a `door` is wood or painted metal or glass, at eps 0.90/0.94/0.92).

Per-pixel confidence
--------------------
`post_process_semantic_segmentation` returns an argmax and nothing else, so the
confidence reported here is derived directly from the semantic map

    seg[c] = sum_q softmax(class_logits[q])[c] * sigmoid(mask_logits[q])

as the winner's share of the total, `seg.max(c) / seg.sum(c)`. Well defined in
[0, 1], NOT a calibrated probability -- a ranking score. It is reported as
`ade_confidence` and is deliberately kept separate from the CLIP `confidence`
that decides the material; the low-emissivity gate reads only the latter.

The einsum runs at FULL resolution, after upsampling the query masks, because
doing it at the decoder's native ~1/4 resolution and upsampling the argmax
afterwards would reintroduce exactly the staircase this module exists to
remove. It is row-chunked to keep the (150, H, W) intermediate off the heap: at
1920x1080 that tensor alone is 1.24 GB.
"""

import numpy as np

DEFAULT_MODEL = "facebook/mask2former-swin-large-ade-semantic"

# Components smaller than this never become a region: they stay at -1 in the
# raster, are never classified, and never reach project_to_flir.py. Same number
# ../classify_session.py used as --sam-min-area, kept so a frame's region count
# is comparable across the switch.
MIN_COMPONENT_AREA = 1500


def _first_name(label: str) -> str:
    return label.split(",")[0].strip().lower()


def ade_name_map(model) -> dict[int, str]:
    """{ade id -> first synonym}, read off the checkpoint rather than hardcoded.

    The 150-class list is stable across the ADE checkpoints, but reading it from
    config means a mismatched checkpoint fails loudly at the lookup instead of
    silently applying the wrong row of ade_material_prior.csv.
    """
    return {int(i): _first_name(name) for i, name in model.config.id2label.items()}


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

    See the module docstring for what `confidence` means and why the einsum runs
    at full resolution in row chunks.
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


def m2f_regions(image: np.ndarray, proc, model, min_area: int = MIN_COMPONENT_AREA):
    """One frame -> (labels, regions, masks).

    `labels` is an int32 HxW raster, one id per connected component of one ADE
    class, -1 where the component was below `min_area`. There is no gap fill: a
    pixel that did not make it into a region simply does not vote, which is the
    honest behaviour and is what ../project_to_flir.py and ../voxel_consensus.py
    already assume for sid < 0.

    `regions` carries the keys ../classify_session.py's loop reads -- id, bbox,
    centroid_px, area_px -- plus `ade` (the ADE class name the region came from)
    and `ade_confidence`. It carries NO material: that is CLIP's answer, added by
    the driver.

    `masks` is the boolean HxW mask per region, in the same order, because
    crops.py builds its CLIP input from the mask rather than from the bbox --
    an ADE region is object-shaped, so its bounding box is routinely mostly
    other objects.

    Components are split per ADE class, NOT merged across classes: a mullion
    labelled `wall` between two `windowpane` bays stays its own region and gets
    its own material, which is the correct behaviour here (the frame really is a
    different material from the glass).
    """
    import cv2

    names = ade_name_map(model)
    sem, conf = semantic_with_confidence(image, proc, model)

    labels = np.full(sem.shape, -1, np.int32)
    regions: list[dict] = []
    masks: list[np.ndarray] = []
    next_id = 0
    for ade_id in np.unique(sem):
        ade_id = int(ade_id)
        ade_name = names.get(ade_id, f"ade_{ade_id}")
        mask = (sem == ade_id).astype(np.uint8)
        n, comp, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            region = comp == k
            ys, xs = np.nonzero(region)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            labels[region] = next_id
            regions.append({
                "id": next_id,
                "bbox": bbox,
                "centroid_px": [float(xs.mean()), float(ys.mean())],
                "area_px": area,
                "ade": ade_name,
                "ade_confidence": float(conf[region].mean()),
            })
            masks.append(region)
            next_id += 1
    return labels, regions, masks
