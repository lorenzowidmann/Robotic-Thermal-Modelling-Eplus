"""Mask2Former segmentation + ADE-category material prior for classify_session_m2f.py.

The material table, the CLIP classifier and the low-emissivity gate are NOT
here: they are the parent module's (../../emissivity/), imported unchanged, so
emissivity_table.csv stays the one source of truth that ../../voxel_consensus.py
also reads.
"""

from .category_prior import (
    DEFAULT_PRIOR_TABLE,
    AdeMaterialPrior,
    PriorRecord,
    restrict_to_candidates,
)
from .crops import CANVAS_PX, PATCH_PX, bbox_crop, masked_crop, region_crop, texture_swatch
from .segmentation_m2f import (
    DEFAULT_MODEL,
    MIN_COMPONENT_AREA,
    ade_name_map,
    load_model,
    m2f_regions,
    semantic_with_confidence,
)

__all__ = [
    "AdeMaterialPrior",
    "PriorRecord",
    "DEFAULT_PRIOR_TABLE",
    "restrict_to_candidates",
    "CANVAS_PX",
    "PATCH_PX",
    "bbox_crop",
    "masked_crop",
    "texture_swatch",
    "region_crop",
    "DEFAULT_MODEL",
    "MIN_COMPONENT_AREA",
    "ade_name_map",
    "load_model",
    "semantic_with_confidence",
    "m2f_regions",
]
