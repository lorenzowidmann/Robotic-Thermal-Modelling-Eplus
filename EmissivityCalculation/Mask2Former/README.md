# Mask2Former material + emissivity

`classify_session_m2f.py` does the same job as `../classify_session.py` — a material and
an emissivity for every region of every frame of a synced ZED session — with two things
changed and nothing else.

1. **The segmenter.** SAM-everything (or SLIC) is replaced by one Mask2Former forward
   pass on the ADE20K-150 taxonomy, the same switch
   `PointCloudElaboration/WindowsDoorsDetection` made for door/window detection.
2. **Where CLIP's prior comes from.** `../emissivity/zones.py::zone_of` guesses what a
   region is from its **bbox shape** — wide and low means floor, tall means vertical — and
   restricts CLIP's candidate materials accordingly. Mask2Former already *knows* what the
   region is, so the restriction is read off the ADE class instead:
   `ade_material_prior.csv` says a `column` is not made of glass, a `ceiling` is not made
   of glass, a `windowpane` is not made of rubber.

Everything after that is the parent module's, unchanged and imported rather than copied:
the CLIP classifier, `../emissivity_table.csv`, the low-emissivity gate, the FLIR-FOV
crop, and the `material_map/v1` output schema. There is one emissivity table, and
`../voxel_consensus.py` reads the same one.

```
py classify_session_m2f.py --session-dir ...\ZED\20260730_161223\fullrate --limit 3 --overlay
py classify_session_m2f.py --session-dir ...\fullrate --every-n 5
py classify_session_m2f.py --session-dir . --check-only          # validate a table edit
```

Venv: `C:\venvs\emissivity`. Output goes to `<session-dir>/material_map_m2f/` — *not*
`material_map/`, so a run here never overwrites `../classify_session.py`'s output and the
two can be diffed.

## What the prior is for

The radiometric correction divides by emissivity, so a bare-metal call is catastrophic
(ε=0.07 turns a 37 °C apparent reading into ~156 °C) while confusing any two ordinary
indoor materials costs under 1 °C — every non-metal in `../emissivity_table.csv` sits
between 0.90 and 0.95. The prior's job is to make the first kind of error *structurally
impossible* rather than to catch it afterwards.

Measured on session `20260730_161223`, frames `20250906_233144_R` and `..._45_R`, 24
regions, defaults. `--no-category-prior` against the default run:

| ADE class | prior off | prior on |
|---|---|---|
| painting | **steel_oxidized** 0.18 (ε=0.79) | plastic 0.50 (ε=0.94) |
| painting | **steel_oxidized** 0.45 (ε=0.79) | fabric 0.40 (ε=0.90) |
| wall | **glass** 0.30 | paint 0.90 |
| door | **rubber** 0.34 | glass 0.46 |

Note the first two. `steel_oxidized` at ε=0.79 is *above* the low-emissivity gate's 0.5
threshold, so the gate cannot see it — the gate fired on 0 of 24 segments in both runs.
The prior catches a class of error the gate structurally cannot.

The prior also raises confidence everywhere it does not change the answer (glass on a
window bay goes 0.43 → 0.94, painted_metal on a radiator 0.56 → 0.97), because
restricting a softmax to a subset and renormalising is exactly equivalent to having
scored only the allowed classes — no extra forward pass, no extra cost.

With the prior on, the two frames come out: every window bay `glass`, every radiator
`painted_metal`, the corridor floor `rubber`, walls and ceiling `paint`. No region
anywhere is a bare metal.

## `ade_material_prior.csv`

The taxonomy lives in the file, so a change is a CSV edit and not a code edit — same
convention as `WindowsDoorsDetection/opening_table.csv`.

| column | meaning |
|---|---|
| `group` | free-text row name, echoed into `segments.json` as `prior_group` |
| `ade` | semicolon-separated ADE20K-150 labels, matched on the **first** comma-separated synonym, lowercased — so the checkpoint's `floor;flooring` matches `floor` here |
| `zone` | `floor` / `ceiling` / `vertical` / `any` — *not* used to pick materials, see below |
| `materials` | semicolon-separated names from `../emissivity_table.csv` |
| `notes` | why this row allows what it allows |

19 groups cover 119 of the 150 ADE classes. The remaining 31 (water, sea, car, boat, …)
are **unmapped by omission** and fall back to the emissivity floor only — the same
fallback `ZONE_CANDIDATES["any"]` has, for the same reason: "no prior" must not mean
"anything goes".

No group lists a bare metal anywhere.

### Editing it

`--check-only` loads the checkpoint and the CSV, validates them against each other,
prints the mapping and exits without touching CLIP or any frames. Five things fail
loudly:

- a material that is not in `../emissivity_table.csv` (a typo would otherwise silently
  drop out of the filter and *widen* the prior it was meant to narrow);
- an ADE label the loaded checkpoint does not have — same reason;
- one ADE label claimed by two groups (the prior applied would depend on row order);
- an unknown `zone` name;
- an empty `materials` list (a row that allows nothing falls back to the *unrestricted*
  ranking, i.e. the opposite of what it says — delete the row instead).

### The `zone` column and why it exists

`../voxel_consensus.py:270` re-applies the **geometric** prior when it pools votes across
frames: `restrict_ranking(ranked, seg.get("zone", "any"))`. So `segments.json` still
carries a `zone` per segment — now read off the ADE class rather than guessed from bbox
shape.

For the two stages not to contradict each other, a row with `zone != "any"` must list a
**subset** of `emissivity.zones.ZONE_CANDIDATES[zone]`. That is checked at load time, not
left as a comment. A row whose materials do not fit any geometric zone — a `rug` is
fabric, and `ZONE_CANDIDATES["floor"]` has no fabric — declares `zone = any` and stage 2
leaves it alone.

Verified on the two-frame run: stage 2's zone re-application strips 0 of 24 segments.

## What CLIP is shown: `--crop-mode`

Default `bbox` — the region's bounding box, exactly what `../classify_session.py` does.
This is measured, and the measurement went against the expectation.

Mask2Former regions are semantic and object-shaped, so a bounding box is often mostly
*other objects*: on a synthetic L of the same shape as a corridor `wall` region, 54% of
the bbox crop's pixels lie outside the region. Two modes fix that — `masked` (outside the
mask flattened to the region's mean colour) and `texture` (a 224×224 swatch tiled from
patches cut strictly inside the mask, 0% foreign pixels). Both are **worse**:

| ADE class | `bbox` | `masked` | `texture` |
|---|---|---|---|
| windowpane | **glass 1.00** | glass 0.98 | plastic 0.52 |
| windowpane | **glass 0.91** | plastic 0.86 | plastic 0.95 |
| windowpane | **glass 0.94** | glass 0.98 | painted_metal 0.72 |
| radiator | **painted_metal 1.00** | painted_metal 1.00 | painted_metal 0.58 |
| radiator | **painted_metal 1.00** | painted_metal 1.00 | ceramic 0.55 |
| radiator | **painted_metal 0.97** | painted_metal 0.81 | ceramic 0.96 |
| floor | **rubber 0.79** | concrete 0.43 | ceramic 0.75 |

The category prior also had to correct only 1 of 12 regions on `bbox`, against 5 on
`masked` and 6 on `texture` — CLIP's raw answer already agreed with the prior far more
often.

The reason is in `../emissivity_table.csv`: its prompts are **object-level**, not
texture-level — *"a photo of a painted metal radiator or painted metal panel with glossy
enamel paint"*, *"a photo of a smooth glass surface or window pane"*. A swatch of pure
surface deletes exactly the cue those prompts are written against. The plausible argument
that Mask2Former has already supplied the object identity, so CLIP only needs the
texture, is wrong about how CLIP is being asked the question here.

`masked` and `texture` are kept for the case where that stops holding — a region whose
bbox really is dominated by something else, or a table rewritten with texture-level
prompts.

## Output

`<out-dir>/<flir_frame_stem>/`:

- `labels.npy` — int32 H×W region-id raster on the full ZED pixel grid; −1 outside the
  FLIR-FOV crop and wherever no connected component reached `--min-area`. No gap fill: a
  pixel that did not make it into a region simply does not vote, which is what
  `../project_to_flir.py` and `../voxel_consensus.py` already assume for `sid < 0`.
- `segments.json` — schema `material_map/v1`, unchanged for both consumers. Additive keys
  only: `ade`, `ade_confidence`, `crop`, `prior_group`, `allowed_materials`, and a
  `prior` block (mirroring the existing `gated` block) recording what the prior overrode
  and why. At document level, `segmenter` and `category_prior` replace the SAM/SLIC and
  `zone_constraint` blocks; `gate` is unchanged.
- `overlay.png` (`--overlay`) — region **contours**, not boxes, labelled
  `<ade> -> <material> e=<ε>`. Green ordinary, orange where the category prior stepped
  in, red where the low-emissivity gate fired. Contours because a box misrepresents what
  was classified: a thin reveal beside a window bay looks contained in it on boxes and
  separate on masks.

`ade_confidence` is Mask2Former's per-pixel `seg.max(c) / seg.sum(c)` averaged over the
region — a ranking score in [0, 1], **not** a calibrated probability. It is deliberately
separate from the CLIP `confidence` that decides the material; the gate reads only the
latter.

## Cost

Roughly 5 s/frame for Mask2Former plus ~2 s/region for CLIP ViT-H/14 on CPU — about 30 s
for a 12-region frame. SAM alone was ~27 s/frame for the segmentation. Both checkpoints
are ~850 MB and ~3.9 GB respectively and are loaded once for the whole run.

## Layout

```
Mask2Former/
  classify_session_m2f.py     the driver
  ade_material_prior.csv      ADE class -> allowed materials
  m2f_materials/
    segmentation_m2f.py       Mask2Former forward pass -> regions + ade + confidence
    category_prior.py         the CSV, its validation, and the ranking restriction
    crops.py                  the three --crop-mode implementations
  requirements.txt
```

`../emissivity/` (table, CLIP classifier) and `../../SensorFusionLoader/` (rig
calibration, for the FLIR-FOV crop) are imported, not copied. The calibration path is
found by searching upward for `SensorFusionLoader/`, copied from
`WindowsDoorsDetection/classify_openings.py::_find_root` — note that
`../classify_session.py` hardcodes `../Calibration`, which does not exist in this repo, so
its `--crop-to-flir-fov` is broken as checked in and this script's is not.
