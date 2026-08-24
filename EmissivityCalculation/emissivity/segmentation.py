"""Segmentation into zones for material classification: the "smallest possible
zone" when true per-pixel labeling isn't feasible (CLIP classifies whole crops,
not individual pixels).

Two segmenters:

  superpixel_segments()  -- SLIC. Cheap (0.4 s) and always available, but it is
      a PARTITION: it produces ~n_segments cells regardless of content, so a
      uniform floor is split into dozens of cells no matter how the parameters
      are set. `sigma` (pre-blur) is worth setting to 2-3; it costs nothing and
      stops boundaries chasing sensor noise.

  sam_segments()         -- Segment Anything, automatic mask generation from a
      grid of point prompts. ~27 s/frame on CPU, but the masks follow real
      objects: on a corridor frame the floor comes out as ONE mask and the
      radiators/pillars/window bays are separated, at 99% pixel coverage.

Both return an int label raster; -1 means "no zone" (SAM leaves a few percent
uncovered before gap-filling, and classify_session.py's FLIR-FOV crop marks
everything outside the crop this way).
"""

import numpy as np
from skimage.segmentation import slic

SAM_MODEL = "facebook/sam-vit-base"
# SAM's own preprocessing constants (it does NOT use the CLIP/ImageNet pair).
_SAM_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
_SAM_STD = np.array([58.395, 57.12, 57.375], np.float32)


def superpixel_segments(image: np.ndarray, n_segments: int = 100,
                        compactness: float = 10.0, sigma: float = 0.0) -> np.ndarray:
    """RGB HxWx3 uint8 image -> int label raster (HxW), one id per superpixel.

    sigma: Gaussian pre-blur applied for the segmentation decision only -- the
    labels are used on the original image, so no texture is lost downstream.
    Default 0.0 keeps the historical behaviour; 2-3 gives visibly cleaner
    boundaries at the same segment count.
    """
    return slic(image, n_segments=n_segments, compactness=compactness, sigma=sigma,
                start_label=0, channel_axis=-1)


def _sam_preprocess(image: np.ndarray, size: int = 1024):
    """SAM's transform: longest side to `size`, normalise, pad to square.
    Done here rather than via SamProcessor, which pulls in torchvision."""
    import cv2

    h, w = image.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    padded = np.zeros((size, size, 3), np.float32)
    padded[:nh, :nw] = (r - _SAM_MEAN) / _SAM_STD
    return padded.transpose(2, 0, 1)[None], nh, nw


def _fill_gaps(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Give every pixel inside `valid` a label, by nearest labelled neighbour.
    SAM's masks leave ~1% of pixels uncovered; without this they would be
    dropped as unknown ids by project_to_flir.py."""
    from scipy import ndimage        # a scikit-image dependency, always present

    holes = valid & (labels < 0)
    if not holes.any() or (labels >= 0).sum() == 0:
        return labels
    _d, idx = ndimage.distance_transform_edt(labels < 0, return_indices=True)
    out = labels.copy()
    out[holes] = labels[idx[0][holes], idx[1][holes]]
    return out


def sam_segments(image: np.ndarray, model_name: str = SAM_MODEL, grid: int = 16,
                 batch: int = 32, min_area: int = 1500, nms_iou: float = 0.7,
                 fill_gaps: bool = True, model=None) -> np.ndarray:
    """RGB HxWx3 uint8 image -> int label raster (HxW), one id per SAM mask.

    A `grid` x `grid` lattice of point prompts is pushed through the mask
    decoder (cheap) after a single image-encoder pass (the expensive part), the
    best mask per prompt is kept, tiny masks are dropped, and duplicates are
    removed by NMS on IoU. Masks are painted largest-first so small objects
    stay visible on top of the surface they sit on.

    `model` lets a caller reuse an already-loaded SamModel across frames --
    loading it per frame would dominate the runtime.
    """
    import cv2
    import torch
    from transformers import SamModel

    if model is None:
        model = SamModel.from_pretrained(model_name).eval()

    h, w = image.shape[:2]
    pixel_values, nh, nw = _sam_preprocess(image)
    with torch.no_grad():
        emb = model.get_image_embeddings(torch.from_numpy(pixel_values))

    ys = (np.arange(grid) + 0.5) / grid * nh
    xs = (np.arange(grid) + 0.5) / grid * nw
    pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)

    masks, scores = [], []
    for i in range(0, len(pts), batch):
        chunk = torch.from_numpy(pts[i:i + batch].astype(np.float32)).reshape(1, -1, 1, 2)
        with torch.no_grad():
            out = model(image_embeddings=emb, input_points=chunk, multimask_output=True)
        low_res = out.pred_masks[0]
        iou = out.iou_scores[0].numpy()
        up = torch.nn.functional.interpolate(low_res, size=(1024, 1024), mode="bilinear",
                                             align_corners=False).numpy()
        for j in range(up.shape[0]):
            k = int(iou[j].argmax())
            m = up[j, k, :nh, :nw] > 0
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if m.sum() >= min_area:
                masks.append(m)
                scores.append(float(iou[j, k]))

    keep = []
    for idx in np.argsort(scores)[::-1]:
        m = masks[idx]
        if all((m & masks[k]).sum() / max(1, (m | masks[k]).sum()) < nms_iou for k in keep):
            keep.append(idx)

    labels = np.full((h, w), -1, np.int32)
    for new_id, idx in enumerate(sorted(keep, key=lambda i: -masks[i].sum())):
        labels[masks[idx]] = new_id
    if fill_gaps:
        labels = _fill_gaps(labels, np.ones((h, w), bool))
    return labels


def segment_boxes(labels: np.ndarray) -> list[dict]:
    """Per-segment id, bounding box (x0,y0,x1,y1), centroid, and area -- used to
    crop each superpixel for classification without needing to mask/warp the
    image itself. (Same shape of output as the grid_boxes() of the removed
    main.py, which tiled a fixed NxN grid instead of segmenting.)

    Negative ids mean "no zone" and are skipped.
    """
    segments = []
    for seg_id in np.unique(labels):
        if seg_id < 0:
            continue
        ys, xs = np.nonzero(labels == seg_id)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        segments.append({
            "id": int(seg_id),
            "bbox": (x0, y0, x1, y1),
            "centroid_px": [float(xs.mean()), float(ys.mean())],
            "area_px": int(len(xs)),
        })
    return segments
