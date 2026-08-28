"""What CLIP is actually shown for one region.

Three modes. The default is `bbox` -- the region's bounding box, exactly what
../classify_session.py hands CLIP today -- and the reason is measured, not
assumed, because the assumption went the other way.

The argument for NOT using the bbox
-----------------------------------
Mask2Former regions are semantic and object-shaped, so a bounding box is often
mostly other objects: the `wall` region of a corridor frame is an L wrapping
around a door and two window bays. On a synthetic L of the same shape, 54% of
the pixels in the bbox crop lie outside the region; the texture swatch below is
0%. A thin window reveal has the opposite problem -- a 12 px strip upsampled to
224 is a smear.

Why the bbox wins anyway
------------------------
Measured on session 20260730_161223, frame 20250906_233144, 12 regions, all
three modes, same seed and same prior:

    ade         bbox                  masked                texture
    windowpane  glass 1.00            glass 0.98            plastic 0.52
    windowpane  glass 0.91            plastic 0.86          plastic 0.95
    windowpane  glass 0.94            glass 0.98            painted_metal 0.72
    radiator    painted_metal 1.00    painted_metal 1.00    painted_metal 0.58
    radiator    painted_metal 1.00    painted_metal 1.00    ceramic 0.55
    radiator    painted_metal 0.97    painted_metal 0.81    ceramic 0.96
    floor       rubber 0.79           concrete 0.43         ceramic 0.75

The bbox column is right where the other two are wrong, and confident where
they are not: glass on all three window bays, painted_metal on all three
radiator sections, and rubber on the corridor floor -- which is what that floor
actually is. The category prior also had to correct only 1 of 12 regions on
bbox against 5 on masked and 6 on texture, i.e. CLIP's raw answer already
agreed with the prior far more often.

The reason is in ../emissivity_table.csv: its prompts are OBJECT-level, not
texture-level -- "a photo of a painted metal radiator or painted metal panel
with glossy enamel paint", "a photo of a smooth glass surface or window pane".
A swatch of pure surface deletes exactly the cue those prompts are written
against. The plausible-sounding argument that Mask2Former has already supplied
the object identity, so CLIP only needs the texture, is wrong about how CLIP is
being asked the question here.

Kept, for the cases where that reasoning may not hold -- a region whose bbox is
genuinely dominated by something else, or a table rewritten with texture-level
prompts:

  masked  -- bbox crop with everything outside the mask replaced by the
             region's own mean colour. Mean, not black: a black surround drags
             CLIP toward asphalt and rubber, the two darkest classes in the
             table.
  texture -- a 224x224 swatch tiled from patches cut strictly inside the mask,
             so CLIP sees a solid field of that surface and nothing else.
             Patches are cut at NATIVE resolution and tiled, not resized to
             fill, so the apparent scale of the grain is preserved -- brick
             reads as brick because the courses are the size CLIP saw them in
             training.
"""

import numpy as np

# CLIP's input side for every model in use here (ViT-L/14 and ViT-H/14 are both
# 224). The swatch is built at exactly this size so the processor's resize is a
# no-op and the patches reach the model at the scale they were cut.
CANVAS_PX = 224
# Starting patch side. Halved until patches fit strictly inside the mask;
# below MIN_PATCH_PX a swatch stops being a texture and starts being noise, so
# the region falls back to `masked` instead.
PATCH_PX = 64
MIN_PATCH_PX = 8


def bbox_crop(image: np.ndarray, bbox) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return image[y0:y1, x0:x1]


def masked_crop(image: np.ndarray, mask: np.ndarray, bbox) -> np.ndarray:
    """Bbox crop with the outside of the mask flattened to the region's mean colour."""
    x0, y0, x1, y1 = bbox
    sub = image[y0:y1, x0:x1].copy()
    m = mask[y0:y1, x0:x1]
    if m.all() or not m.any():
        return sub
    sub[~m] = sub[m].mean(axis=0).astype(sub.dtype)
    return sub


def _inside_patch_origins(mask_sub: np.ndarray, p: int) -> np.ndarray:
    """Top-left corners of every p x p window lying STRICTLY inside the mask.

    Via a summed-area table, so this is O(area) per patch size rather than
    O(area * p^2): a window is fully inside iff its mask sum is exactly p*p.
    Strictly inside matters -- a patch that clipped the region's edge would
    carry a stripe of the neighbouring object into the swatch, and tiling it 16
    times would turn one bad edge into a texture.

    Returns an (N, 2) int array of (y, x); empty if no window of this size fits.
    """
    import cv2

    h, w = mask_sub.shape
    if h < p or w < p:
        return np.empty((0, 2), np.int64)
    integral = cv2.integral(mask_sub.astype(np.uint8))          # (h+1, w+1), int32
    total = (integral[p:, p:] - integral[:-p, p:]
             - integral[p:, :-p] + integral[:-p, :-p])
    ys, xs = np.nonzero(total == p * p)
    return np.stack([ys, xs], axis=1)


def texture_swatch(image: np.ndarray, mask: np.ndarray, bbox,
                   canvas: int = CANVAS_PX, patch: int = PATCH_PX,
                   min_patch: int = MIN_PATCH_PX, seed: int = 0):
    """Tile patches cut from inside `mask` into a `canvas` x `canvas` RGB swatch.

    Returns (swatch, patch_px) or (None, 0) when the region is too thin to yield
    even a min_patch x min_patch window -- the caller falls back to masked_crop.

    `seed` is the region id, so the same frame reprocessed gives the same swatch
    and a diff between two runs means a real change, not a reshuffle.
    """
    x0, y0, x1, y1 = bbox
    sub = image[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1]

    p = patch
    origins = np.empty((0, 2), np.int64)
    while p >= min_patch:
        origins = _inside_patch_origins(m, p)
        if len(origins):
            break
        p //= 2
    if not len(origins):
        return None, 0

    rng = np.random.default_rng(seed)
    grid = (canvas + p - 1) // p                      # tiles per side, rounded up
    picks = rng.choice(len(origins), size=grid * grid, replace=len(origins) < grid * grid)

    big = np.empty((grid * p, grid * p, 3), image.dtype)
    for t, idx in enumerate(picks):
        i, j = divmod(t, grid)
        py, px = origins[idx]
        tile = sub[py:py + p, px:px + p]
        # Mirror alternate rows/columns so the tiling seams do not read as a
        # hard edge -- CLIP is perfectly capable of latching onto a periodic
        # grid of seams instead of onto the material.
        if j % 2:
            tile = tile[:, ::-1]
        if i % 2:
            tile = tile[::-1, :]
        big[i * p:(i + 1) * p, j * p:(j + 1) * p] = tile
    return big[:canvas, :canvas], p


def region_crop(image: np.ndarray, mask: np.ndarray, bbox, mode: str, seed: int = 0):
    """Dispatch on --crop-mode. Returns (crop, note) where `note` records what
    was actually built, so segments.json can say so per region rather than only
    naming the mode that was requested."""
    if mode == "bbox":
        return bbox_crop(image, bbox), "bbox"
    if mode == "masked":
        return masked_crop(image, mask, bbox), "masked"
    if mode == "texture":
        swatch, p = texture_swatch(image, mask, bbox, seed=seed)
        if swatch is not None:
            return swatch, f"texture:{p}px"
        # Too thin for even an 8x8 window inside the mask.
        return masked_crop(image, mask, bbox), "masked:fallback"
    raise ValueError(f"Unknown crop mode {mode!r}")
