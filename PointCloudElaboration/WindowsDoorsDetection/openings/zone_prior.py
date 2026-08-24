"""The geometric prior, translated from materials to openings.

The geometry itself is NOT re-derived here: zone_of() is imported unchanged
from EmissivityCalculation/emissivity/zones.py, so a region is called
floor/ceiling/vertical/any by exactly the same bbox rules the material
pipeline uses. Only the per-zone candidate list is ours, and it is trivial:

    floor   -> other        a floor patch is not a door or a window
    ceiling -> other        a ceiling patch is not a door or a window
    vertical-> no restriction (this is where doors and windows live)
    any     -> no restriction

That is the same free win zones.py describes: it reuses the same CLIP forward
pass, and restricting a softmax to a subset and renormalising does not change
the ordering, so it is exactly equivalent to having scored only the allowed
classes.

KNOWN FAILURE MODE -- read before trusting a window count
---------------------------------------------------------
zone_of()'s ceiling rule is `cy < 0.30 and bw > 0.8 * bh`. The 0.8 was tuned
on FLIR-FOV-CROPPED frames, where the crop keeps ~39% of the width but ~68%
of the height, so a real ceiling patch loses most of the "wide" shape it has
in a full frame. This module runs UNCROPPED (doors and windows can be
anywhere in the ZED frame), so that threshold is being used outside the
regime it was measured in: a wide window high in the frame -- a clerestory,
the top of a tall window bay, a skylight -- satisfies cy<0.30 and bw>0.8*bh
and is forced to "other" here, i.e. made undetectable by construction.

zones.py's own comment reports that true window segments with cy<0.30 topped
out at bw/bh = 0.768 on this rig, a clean split from the ceiling cases at
0.769 -- but that measurement is on cropped frames and does not transfer.
n_zone_forced is printed per zone by classify_openings.py precisely so the
size of this loss is visible; if ceiling-forced segments are a large fraction,
retune the rule or run with --no-zone-constraint and compare.
"""

try:                                    # imported as part of the package
    from .table import OTHER_CLASS
except ImportError:                     # imported as a bare module by path, by
    from table import OTHER_CLASS       # the torch-free consensus stage

# Mirrors emissivity/zones.py::ZONE_CANDIDATES. None means "no categorical
# restriction". There is no emissivity-style min_eps fallback for "any" here:
# no opening class is catastrophic downstream the way a bare-metal call is,
# so an unconstrained zone is simply unconstrained.
OPENING_ZONE_CANDIDATES = {
    "floor": [OTHER_CLASS],
    "ceiling": [OTHER_CLASS],
    "vertical": None,
    "any": None,
}


def restrict_opening_ranking(ranked, zone):
    """Drop the classes the zone forbids, renormalising the rest.

    `ranked` is the classifier's full [(class, prob), ...], best first. Same
    contract and same maths as emissivity/zones.py::restrict_ranking -- it is
    reimplemented rather than reused because restrict_ranking reads
    ZONE_CANDIDATES from its own module globals and cannot be handed a
    different table.

    Falls back to the unrestricted ranking if the filter would leave nothing,
    so a custom opening_table.csv can never produce an empty result.
    """
    allowed = OPENING_ZONE_CANDIDATES.get(zone)
    if not allowed:
        return ranked
    keep = [(c, p) for c, p in ranked if c in allowed]
    if not keep:
        return ranked
    total = sum(p for _c, p in keep) or 1.0
    return [(c, p / total) for c, p in keep]
