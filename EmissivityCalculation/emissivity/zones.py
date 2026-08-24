"""Geometric priors on a region: restrict which materials CLIP is allowed to
pick, based on where the region sits in the frame and what shape it has.

Why this exists
---------------
The correction divides by emissivity, so a bare-metal call is catastrophic
(e=0.07 turns a 37 degC apparent reading into ~156 degC), while confusing any
two ordinary indoor materials is worth under 1 degC -- every non-metal in
emissivity_table.csv sits between 0.90 and 0.95.

Measured on one corridor frame with CLIP ViT-H/14: unconstrained, it produced
steel_polished twice, steel_oxidized once and copper_oxidized once out of 27
regions. Constraining candidates by zone removed all four. This makes the
catastrophic error structurally impossible rather than relying on
classify_session.py's confidence gate or correct_session.py's plausibility
retry to catch it afterwards -- those remain as further nets.

The assumption
--------------
`vertical` deliberately excludes bare metals: an indoor wall/pillar/window
reveal is not polished steel. A genuinely exposed metal duct would be forced
to painted_metal (e=0.94 instead of 0.07). Given the table above that is the
right trade, but it IS an assumption -- pass zone_constraint=False to disable.

A better version of this would come from LiDAR surface normals (a horizontal
plane below the sensor is a floor) rather than from bbox shape. The cloud is
already projected into the ZED frame by project_to_flir.py, but that runs
after this step, so the geometric proxy below is what is available here.
"""

ZONE_CANDIDATES = {
    "floor": ["rubber", "ceramic", "concrete", "wood", "asphalt", "plastic"],
    "ceiling": ["plaster", "paint", "concrete", "wood"],
    "vertical": ["plaster", "paint", "brick", "concrete", "glass", "wood",
                 "painted_metal", "plastic", "ceramic", "fabric"],
    # No categorical prior for a region whose shape says nothing -- but "no
    # prior" must not mean "anything goes". A region that matched none of the
    # above previously got the full 20-class list, and ViT-H duly returned
    # steel_polished at 0.53 with a 0.23 margin, which sails through the
    # confidence gate: e=0.07, a ~150 degC region. So `any` still drops the
    # classes whose emissivity would blow the correction up; see
    # restrict_ranking's min_eps.
    "any": None,
}


def zone_of(segment: dict, image_height: int, image_width: int) -> str:
    """Classify a region's geometry into a zone name.

    `segment` is one entry from segmentation.segment_boxes(): it carries bbox,
    centroid_px and area_px, all in the coordinate frame of the image that was
    segmented (the FLIR-FOV crop when cropping is on).
    """
    x0, y0, x1, y1 = segment["bbox"]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    cy = segment["centroid_px"][1] / max(1, image_height)
    fill = segment["area_px"] / float(bw * bh)
    touches_bottom = y1 >= image_height - 3

    if cy > 0.55 and bw > 1.6 * bh and touches_bottom and fill > 0.5:
        return "floor"
    # 0.8, not 1.6: with --crop-to-flir-fov (the default), the crop keeps
    # ~39% of the frame's width but ~68% of its height (session 9/6 rig:
    # 751x733 px out of 1920x1080), so a real ceiling patch loses most of
    # the "wide" shape a full, uncropped frame would give it -- 1.6 was
    # tuned for the uncropped case and let ~110 real ceiling segments (this
    # session alone) fall through to "any" misclassified as glass, since
    # "any"'s emissivity gate only blocks bare metal (eps<0.5), not glass
    # (eps=0.92). Measured on this rig's actual segments: true window
    # ("vertical") segments with cy<0.30 top out at bw/bh=0.768; the missed
    # ceiling candidates start at 0.769 -- a clean, zero-overlap split.
    if cy < 0.30 and bw > 0.8 * bh:
        return "ceiling"
    if bh > 1.3 * bw:
        return "vertical"
    return "any"


def restrict_ranking(ranked, zone, eps_of=None, min_eps=0.5):
    """Drop candidates the zone forbids, renormalising the remaining scores.

    `ranked` is CLIP's full [(material, prob), ...] best-first. Restricting a
    softmax to a subset and renormalising does not change the ordering, so this
    is exactly equivalent to having scored only the allowed classes -- no extra
    forward pass, no extra cost.

    For a zone with no categorical list ("any"), candidates below `min_eps` are
    still dropped when `eps_of` is supplied: no indoor surface this pipeline
    sees is bare polished metal, and letting one through costs ~120 degC while
    every ordinary confusion costs under 1 degC.

    Falls back to the unrestricted ranking if the filter would leave nothing,
    so a table with unusual material names can never produce an empty result.
    """
    allowed = ZONE_CANDIDATES.get(zone)
    if allowed:
        keep = [(m, p) for m, p in ranked if m in allowed]
    elif eps_of is not None:
        keep = [(m, p) for m, p in ranked if eps_of.get(m, 1.0) >= min_eps]
    else:
        return ranked
    if not keep:
        return ranked
    total = sum(p for _m, p in keep) or 1.0
    return [(m, p / total) for m, p in keep]
