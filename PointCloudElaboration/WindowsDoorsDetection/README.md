# WindowsDoorsDetection — doors and windows from a ZED session

Classifies every region of a recorded ZED session as **door / window / other**,
then pools multi-view votes per 3-D voxel so a physical opening ends up with
whichever class the majority of the views that actually saw it agree on.

Stage 1 is one **Mask2Former** forward pass on the **ADE20K-150** taxonomy,
which already contains `windowpane` and `door`. It replaced SAM-everything +
CLIP-zero-shot; see "Why Mask2Former replaced SAM + CLIP" below for the
measurements that motivated the swap.

Downstream of this, and deliberately **out of scope** here: fitting planes and
polygons to the consensus voxels, and the OpenStudio `.osm` export. That step
consumes `door_window_voxels.csv`.

## Layout

```
WindowsDoorsDetection/
  classify_openings.py         stage 1 — Mask2Former + stage 1B geometry
  opening_voxel_consensus.py   stage 2 — multi-view voxel vote + 3-D map
  view_openings.py             viewer — the stage 2 voxels, over the scene cloud
  paint_openings.py            viewer — stage 1 only, no vote: points by segment
  fit_openings.py              openings.json — rectangles on box faces, for .osm
  opening_table.csv            the taxonomy: class, ade, prompt, notes
  overlay_lidar.py             QA view: a projected LiDAR point against the mask it landed in
  openings/
    segmentation_m2f.py        Mask2Former -> regions, confidence, zones
    lidar_metrics.py           metric size of a masked region, from the bag
    geometry.py                stage 1B: merge + plausibility rules
    table.py                   OpeningTable (stdlib csv, no pandas)
    zone_prior.py              stage 2's --respect-zones (default on): floor/ceiling
                                segments can't be door or window, torch-free by
                                design so the rosbags-venv consensus stage can use it
```

## Running it

```powershell
# stage 1 — CLIP-free now, but still the torch venv. --bag turns on the
# metric door check and additionally needs rosbags.
& C:\venvs\emissivity\Scripts\python.exe classify_openings.py `
    --session-dir ...\ZED\20260730_161223\fullrate `
    --bag ...\rosbag2_2026_07_30-18_12_20 --limit 5 --overlay

# stage 2 — rosbags venv, unchanged
& C:\venvs\sensorfusion\Scripts\python.exe opening_voxel_consensus.py `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20

# viewer — planefit venv, the only one with pyvista. Voxels alone, or with
# --bag/--session-dir the scene cloud rebuilt behind them.
& C:\venvs\planefit\Scripts\python.exe view_openings.py `
    --consensus-dir ...\fullrate\opening_map_consensus `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20 `
    --every-n 10 --scene-max-range 15

# --paint-cloud draws no cubes: it colours the scene points themselves by the
# voxel each falls in, so the openings show at the cloud's resolution, not the
# vote's 0.20 m.
& C:\venvs\planefit\Scripts\python.exe view_openings.py `
    --consensus-dir ...\fullrate\opening_map_consensus `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20 `
    --paint-cloud

# paint_openings.py — stage 1 only, no vote, every gate off. Answers "is the
# consensus what is removing my openings?" --depth-band is the occlusion proxy
# stage 2 does not have; see "The door is not where the door is" below.
& C:\venvs\planefit\Scripts\python.exe paint_openings.py `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20 `
    --opening-map-dir ...\fullrate\opening_map `
    --depth-band 1.5 --dedup 0.05 --out-ply raw_openings.ply

# fit_openings.py — labelled points + boxes -> rectangles on the box faces,
# and the model view: boxes wireframed, openings drawn as coloured planes.
& C:\venvs\planefit\Scripts\python.exe fit_openings.py `
    --points openings_pts.csv `
    --boxes ...\3DModelPointCloudExtraction\SavedBoxes\boxes_edited2.json `
    --out openings.json `
    --regularize-fill --regularize-size-tol 2.5

# the wall finds the openings, the ZED says what is in each one. Needs the
# non-opening returns as well -- they carry the bay rhythm.
& C:\venvs\planefit\Scripts\python.exe fit_openings.py `
    --points pts_prov.csv --boxes ...\SavedBoxes\boxes_s9_edited.json `
    --masks --session-dir ...\fullrate `
    --opening-map-dir ...\fullrate\opening_map_m2f_full `
    --image-only-windows --scene-points scene_pts.csv `
    --merge-touch-gap 0.02 --bays-from-wall `
    --out openings_bays.json --no-show

# and into the model (from 3DModelPointCloudExtraction/)
& C:\venvs\planefit\Scripts\python.exe to_openstudio.py `
    --boxes SavedBoxes\boxes_edited2.json --openings ...\openings.json `
    --name Session9 --out OpenStudioModel\session9_openings.osm
```

These are **PowerShell** blocks — the backtick is its line continuation, `#` its
comment, and `&` the call operator that runs an interpreter by path. The
backtick must be the **last character on the line**: one trailing space after it
and PowerShell parses the next line on its own and fails on `--`. Paths with
spaces need quoting, `& "C:\Program Files\...\python.exe"`.

In **cmd.exe** the same commands take `^` for continuation and `::` for comments,
and no `&`.

Outputs:

| path | what |
|---|---|
| `<session>/opening_map/<stem>/labels.npy` | int32 HxW region-id raster, full ZED grid, **-1 where no component reached `--min-area`** |
| `<session>/opening_map/<stem>/segments.json` | schema `opening_map/v1` — id, bbox, centroid_px, area_px, top_class, confidence, top_k, zone, `ade` |
| `<session>/opening_map/<stem>/overlay.png` | `--overlay` only; kept openings in colour, **rejected candidates in yellow with the rule that killed them** |
| `<session>/opening_map_consensus/<stem>/segments.json` | same, with the consensus class substituted and a `consensus` block per segment |
| `<session>/opening_map_consensus/door_window_voxels.csv` | **the deliverable**: one row per opening voxel, within `--max-range` |
| `<session>/opening_map_consensus/door_window_voxels.ply` | same, coloured door=red / window=blue |

`door_window_voxels.csv` columns: `x,y,z,opening_class,agreement,n_observations,n_votes,w_door,w_other,w_window`.
Every class's pooled weight is written, not just the winner's, so the polygon
step can apply its own threshold without re-running the vote. `n_observations`
is the number of distinct frames that saw the voxel; `n_votes` counts LiDAR
points and therefore grows with how densely the scan happened to hit the
surface — it is a sampling statistic, not a view count.

## Why Mask2Former replaced SAM + CLIP

Measured on session 9, same frames, same CPU:

| | SAM vit-base + CLIP | Mask2Former swin-large ADE |
|---|---|---|
| runtime / frame | ~27 s | **4.7–7.2 s** |
| boundary resolution | masks decoded at 256×256, `INTER_NEAREST` to 1920×1080 → ~7 px staircase | decoder logits upsampled bilinear at full res |
| unlabelled pixels | `_fill_gaps` painted them with the nearest id, **inventing** outlines across the floor and radiators | left at -1, they simply do not vote |
| `other` | one CLIP prompt covering walls, floors, ceilings, radiators, pillars, people, clutter | the other 148 ADE classes, each keeping its own region |
| corridor-end door | **missed** — CLIP called it `other` | found, conf 0.90 |

Two structural wins came free with the taxonomy:

* **The floor is a real mask.** Stage 1B's floor-contact test used to run
  against `zone_of()`'s bbox guess; it now runs against ADE `floor`.
* **The ceiling rule that ate high windows is gone.** `zone_prior.py` documents
  `zone_of()` calling a wide, high window `ceiling` — `cy < 0.30 and
  bw > 0.8 * bh`, a threshold tuned on FLIR-FOV-cropped frames and used here on
  uncropped ones — and forcing it to `other`, i.e. undetectable by
  construction. Zones now come from the ADE class
  (`segmentation_m2f.zone_from_ade`), so a clerestory stays a windowpane.
  `zone_prior.py` still runs, though: `restrict_opening_ranking` is stage 2's
  `--respect-zones` gate (default **on**), applied to the pooled voxel vote so a
  voxel straddling the wall/floor junction can't hand a floor segment a door
  class. What retired is only the classification-time use, the ceiling rule
  above. `classifier.py`, the CLIP zero-shot path, really is unused now and has
  been deleted.

Per-pixel **confidence** is derived rather than taken from the model, because
`post_process_semantic_segmentation` returns an argmax and stage 2 gates on
`--min-vote-confidence 0.5`. It is the winner's share of Mask2Former's semantic
map, `seg.max(c) / seg.sum(c)`. Well-defined, in [0, 1], and **not calibrated** —
treat it as a ranking score. Observed: 0.90 on the corridor door, 0.61–0.84 on
the glazed bays, 0.43–0.45 on the persistent false positive.

## Stage 1B — merge + geometric plausibility

Runs inside `classify_openings.py`, after segmentation, before the write.
`--no-geometry-filter` turns it off and writes the raw per-ADE-region result.

1. **Merge** touching same-class regions by connected components, with a 2 px
   dilation to bridge the hairline seam a mullion leaves between two fragments
   of the same window. The dilation decides connectivity only — a merge never
   absorbs another class's pixels. `merged_from` records what each detection
   absorbed.
2. **Window rule** — OFF by default (`--window-filter` to enable). Windows are
   merged and kept; only doors are filtered. The rule rejected a real bay at
   h_ratio 0.577 against a 0.60 threshold, and unlike a door a window has no
   metric check to arbitrate, because glazing returns no LiDAR.
3. **Door rules**, in order: metric veto → bay-edge veto → the three size rules
   → floor contact.

Rejected detections keep their pixels and carry a `rejected` block naming the
rule. Nothing is silent, and `segments.json` is normalised against the raster at
the end, so every id in `labels.npy` has exactly one record.

### Everything here is mask shape, never bbox

The bbox is derived and reported; it never decides, draws or exports. The
overlay draws `cv2.findContours` outlines of the merged masks. This is not
cosmetic — it is what made the original QA read correct.

### The LiDAR now arbitrates door size, and only door size

`--bag` measures every door candidate in metres before any size rule may
discard it (`openings/lidar_metrics.py`). Three outcomes:

* **ok** — rescues a candidate a size rule wanted to kill. Recorded as
  `geometry.rescued_by_lidar` and counted under `rescued` in the report.
* **bad** — rejects a candidate the size rules would have kept, rule
  `door_metric_dims`. Fires *before* the pixel rules: a real measurement beats
  every proxy below it.
* **unknown** — no measurement. The pixel rules decide alone. This is not a
  soft `bad`, and conflating the two would make every distant or glazed
  candidate fail for lack of evidence.

Rescuable rules are exactly the depth-dependent ones —
`door_below_min_area`, `door_below_min_width`, `door_taller_than_glass_wall`.
Floor contact and bay-edge adjacency are **never** rescued: they are statements
about where a region sits, which no metric measurement answers.

Size is `extent_px * median_depth / focal` — the mask's full pixel extent, with
only the scale borrowed from the LiDAR. Measuring the 3-D extent of the returns
instead would under-report systematically: LiDAR covers a door leaf densely but
stops short of the top edge and the reveal.

An abstention always says **why**, `few_points` or `multi_depth`, and the
distinction is the actionable part: `few_points` means the surface returned
nothing, `multi_depth` means the mask is not one surface and the fix is
upstream in what got merged into it.

Windows are untouched by any of this. Glazing returns nothing — see the
measurement in the next section — so a metric window gate would fail hardest on
the class it is meant to validate.

### Three things the measurements overturned (SAM era, still true)

| assumption | what the data said |
|---|---|
| `touches_floor` = `y1 >= H-3` | **0 of 19** detections pass. The floor is segmented as one region reaching y=H, so a door's bottom edge is at the wall/floor junction mid-frame. Replaced by mask-adjacency to the floor region, tolerance 120 px (measured gaps: 1/31/40/48/101/114 standing vs 370/462 floating). |
| containment veto, >50% overlap | **Never fires** — real door↔window mask overlap is 0.1–6.3%. Replaced by one-sided edge adjacency. |
| glass wall at h/H ≥ 0.75–0.85 | The **actual** glass wall measures 0.68; 0.75 rejects everything. Lowered to 0.60. Thin evidence — and see "Known problems" below, where 0.60 is now rejecting a real bay at 0.577. |

## The cloud both stages read: `--cloud-source`

Default **`raw`**. Both stages call `lidar_metrics.load_clouds`, so they cannot
silently read different clouds — stage 1's door measurements and stage 2's
votes have to agree about what the sensor saw.

`registered` (FAST-LIO's `/cloud_registered`) is kept for comparison and is
measurably broken for this purpose. `raw` rebuilds world clouds from
`/livox/lidar` + `/Odometry` through `../LivoxLidarOdometryLoader`. The poses
are FAST-LIO's either way — this changes which *points* are carried, not where
the rig thought it was.

Session 9, 5 frames, everything else identical:

| | `registered` | `raw` |
|---|---|---|
| points / scan | 6 349–6 465 | **79 765–84 196** |
| footprint in frame | u 666..1309 (8% of area) | u 0..1920, v 275..898 |
| votes | 16 699 | **320 286** |
| voxels | 513 | 1 326 |
| **opening voxels** | **0** | **82** (window 82, door 0) |
| agreement | — | median 1.00, 100% ≥ 0.5 |
| observations/voxel | — | median 4 |

The 82 voxels land as two strips down opposite sides of the corridor —
y ∈ [0.5, 1.7] and y ∈ [−1.3, −1.1], x 0.9–4.9 m — which is the two glazed
walls. Two things to read off them before trusting the polygon fit:

* **z spans only −0.1 to 1.3 m**, concentrated at 0.1–0.5. The HAP's 25°
  vertical against the ZED's 54° means the laser reaches roughly the middle of
  the frame height, so a floor-to-ceiling bay is captured over its lower ~1.4 m
  and no more. Not fixable by re-running FAST-LIO; it is the sensor.
* **Side A is 1.2 m thick, side B is one voxel thick.** Side B (21 of 24
  voxels at y = −1.1) is a clean plane. Side A is smeared because ADE
  `windowpane` swallows the radiators standing in front of the glass, so their
  surfaces vote `window` too.

`door = 0` at the default range: the only door found is the corridor-end
opening at 34.4 m, and `--max-range 8` drops its votes. At `--max-range 40` it
yields 474 door voxels — but `n_observations` falls from a median of 4 to 2 and
474 voxels is far more than a 2.5 × 2.7 m opening should fill, so those are
smeared along the ray, not a measurement. The honest fix is to use frames where
the rover is closer to that door, not a wider gate.

## Range limit: `--max-range` (stage 2, default 8 m)

A surface classified from 20 m away is a few pixels of an oblique, blurred
region. The gate lives at the **vote**, not in stage 1B, and that placement is
the whole point:

* it uses **exact per-point depth**, so no densification is involved. Filling
  the sparse LiDAR by nearest neighbour and trimming per pixel was measured and
  does not work: the fill hands the far glass the depth of the near mullions.
* there is **no fail-open case**. At the vote, a point that does not exist
  simply does not vote, which is already the correct behaviour.

**This changes the 3-D product only.** Stage 1's `overlay.png` still draws the
full-length regions. Judge the result on `door_window_voxels.csv`/`.ply`.

## The hand-off to OpenStudio

An OpenStudio opening is a **SubSurface**, and it is not free-floating: it must
be coplanar with, and strictly inside, a parent `Surface`. So the deliverable
the `.osm` needs is not a polygon fit to the opening points — it is *which wall,
and which rectangle of that wall*. That is a far weaker demand on the LiDAR than
fitting glass which barely returns anything, and it is why this order works
where a direct polygon fit does not:

```
paint_openings.py --out-csv   x,y,z,opening_class          (stage 1, no vote)
fit_boxes.py                  boxes.json                   (the rooms)
fit_openings.py               openings.json                (rectangles ON faces)
to_openstudio.py --openings   .osm with Door / FixedWindow SubSurfaces
```

**Feed it stage 1's points, not the consensus.** `fit_openings.py` takes any
`x,y,z,opening_class` csv, and `door_window_voxels.csv` has those columns, so
stage 2's output can be dropped in — but measured on session 9 it removes far
more than it should:

| | stage 1 points | consensus voxels |
|---|---|---|
| windows | 8 | 7 |
| doors | 1 | **0** |
| opening area | 44.7 m² | 10.8 m² |
| WWR | 28.9% | **6.6%** |

The consensus *does* separate the bays cleanly — `--max-range 8` discards the
distant grazing patches that otherwise tile one wall into a single ribbon — but
the same gate deletes the corridor-end door at 34.4 m outright, and 56% of the
995 voxels land more than 0.60 m from any box face and are dropped. 6.6% does
not describe a corridor whose long wall is a run of tall windows. Use the
consensus for cross-checking, not as the input.

`fit_openings.py` assigns each point to the nearest vertical box face it lies
within (`--max-face-dist`, default 0.60 m — the returns come off the reveal and
the mullions, not the glass, so this cannot be tight), clusters the points in
that face's own (u, z) plane, and takes each cluster's extent as the rectangle.
The rectangle sits on the **box face**, never on the mean of the points.

### Where an opening is allowed to be

Size alone is not enough — a rectangle has to be somewhere an opening can exist.
Four rules, each of which fired on session 9:

* **The door band**, `MIN/MAX_DOOR_W_M` and `MIN/MAX_DOOR_H_M` **imported from
  `openings/lidar_metrics.py`**, not restated: a candidate stage 1B's metric
  check accepted must not be rejected here on different numbers.
* **A door reaches the floor** (`--door-floor-tol`, 0.30 m). This is what
  removes the fragments a door mask leaves on the side walls beside the real
  opening — they float.
* **A door does not reach the ceiling** (`--door-head-clearance`, 0.12 m). The
  mask runs to the top of the opening, so the rectangle lands exactly on the
  box's own `z_max` and the model gets a full-height hole where the building has
  a lintel. Session 9's door measured 2.23 m — the box's entire interior height
  — and clamps to **2.11 m**. **Clamped, not rejected**: the door is real, only
  its top edge is wrong. Windows are exempt; a clerestory legitimately sits
  high. Re-applied after `--regularize`, because a median head can land back on
  the ceiling even when every input was clamped off it.
* **`--merge-corner-dist`**, measured **box-to-box, not centre-to-centre**. An
  opening at the end of a corridor is in reach of three faces at once — the end
  wall and both side walls. The corridor-end door's two fits nearly touch at the
  corner (0.45 m apart) while their *centres* are 1.57 m apart, so a
  centre-distance test at 1.5 m kept a second, impossible door on the side wall.
* **`--exterior-only`**: an opening on a face another box sits against can never
  be built — `to_openstudio.py` puts no Surface there, the two spaces are
  already open to each other. Tested at the **rectangle's own centre**, not over
  the whole face: `build_wall_segments` subtracts only the covered *part* of a
  face and still builds walls either side of it. Rejecting on the whole face
  discarded 11 of 13 real openings along this corridor, where the parallel boxes
  cover each other's long sides over part of their length.

The glazing rejected on box 0's covered faces is not lost, incidentally — the
same bays are fitted again on the *outer* boxes' faces, which is where the real
exterior wall is.

### `--regularize`: a repeated bay is evidence, a missed one is a gap

A facade bay repeats; the LiDAR's view of each copy does not. The same window
seen from four angles and three distances comes out as rectangles of different
widths at slightly wrong spacing, and one bay the rover never got a clean look
at comes out as nothing at all.

`--regularize` fits, per wall face and class, a lattice `u = offset + k·pitch`
and snaps every opening on that face to one median size at its own slot.
`--regularize-fill` additionally *creates* an opening at each empty slot — those
are **inferred, not measured**: `n_points: 0` and `"synthetic": true` in
`openings.json`, and the viewer draws them hollow so they can never be mistaken
for a measurement. Both are **off by default**; the run prints every change.

Three guards, and on session 9 two of them fire:

* at least `--regularize-min-count` (3) openings on the face — two is a
  coincidence;
* widths spread (p95−p05) under `--regularize-size-tol`. **At the default 0.60
  this refuses session 9's long wall**, spread 1.94 m, because the clustering
  merged two adjacent bays into one 3.94 m rectangle. `--regularize-size-tol
  2.5` accepts it; the median width is robust to the merged one either way;
* lattice RMS under `--regularize-tol` (0.35 m).

**The pitch is searched, not summarised.** Every observed gap and that gap over
2, 3 and 4 is tried, best RMS wins, ties to the coarser lattice. A summary
statistic cannot do this: three bays at 5, 9 and 17 m have gaps {4, 8}, whose
median is 6 — a pitch fitting none of the three. The true pitch, 4, is only the
*smaller* gap. Verified on exactly that case: slot 2 is recovered at 12.00–14.00
against a true centre of 13.

Session 9, `--regularize-fill --regularize-size-tol 2.5`:

```
box 1 bottom window: 5 opening(s) on a 3.61 m pitch, RMS 0.17 m -> 2.59 x 1.69 m each
```

Five bays down the long glazed wall, one size, one pitch, no empty slots to
fill.

`--regularize-extend` continues a lattice to both ends of the face instead of
only between the openings found. On session 9 it adds **nothing**, and the
reason is the useful part:

### The window pattern stops at x = 21.11 because the box does

The long glazed wall looks like one wall and is modelled as two faces:

| | face | x span |
|---|---|---|
| box 1 `bottom` | y = −1.20 | 3.11 … **21.11** |
| box 0 `bottom` | y = −0.80 | 3.01 … 34.51 |

A lattice belongs to one face and cannot cross onto another — a different face
is a different plane, here 0.40 m away. The five regularised bays already fill
box 1's face exactly (slots 0–4 of a face that holds 0–4), so there is nothing
for `--regularize-extend` to add. **The limit is the box model, not the
openings.**

Tested by extending box 1 to x_max 34.51 and re-running: two more window
clusters immediately appear at x 22.0 and 25.3 — the points were always there,
they were simply landing on box 0's face — and with the size guard opened to
3.5 m the whole wall regularises:

```
box 1 bottom window: 7 opening(s) on a 3.40 m pitch, RMS 0.34 m -> 1.99 x 1.48 m each
box 1 bottom window: filled 2 empty slot(s) -- INFERRED, not measured
```

Nine bays from x 4.6 to 33.8, the last two inferred where the scan never got a
clean look. So if that corridor really is one uniform width, the fix is in
`interactive_boxes.py` — stretch box 1 the full length — and then re-run this.
It is not a change to make here: whether the 0.40 m jog at x = 21.11 is a real
recess or a tiling artefact is a question about the cloud, not about openings.

Measured, session 9, `boxes_edited2.json` + 36 frames of `opening_map_m2f_full`:

| | |
|---|---|
| labelled points in | 59 177 (door 20 835, window 38 342, deduped at 3 cm) |
| near no box face | 4 896 dropped |
| rectangles kept | **1 door, 10 window** (32 clusters rejected: 24 too small, 7 on a covered face, 1 corner duplicate) |
| into the `.osm` | 11 SubSurfaces — 1 Door, 10 FixedWindow, **30.2 m² total** |
| placed nowhere | none |

**Read the sills before trusting the numbers.** Every fitted window starts at
z ≈ −0.1, i.e. on the floor, because ADE `windowpane` swallows the radiators
standing under the glass (see "Known problems"). The heights are real, the
**sill heights are not** — they are the top of the radiator, not the bottom of
the glazing. For an energy model that matters: it inflates the glazed area and
puts it at the wrong height. Nothing here fixes it; excluding the radiators has
to happen in stage 1.

## `--bays-from-wall`: the wall finds the openings, the ZED says what is in them

The division of labour that `--masks` starts — LiDAR for the plane, camera for
the outline — taken to its end. Neither the returns inside an opening nor a
mask's own extent set the rectangle any more:

| question | answered by | why |
|---|---|---|
| where is there a hole in this wall | the wall's **piers**, in plan view | opaque, dense, and the only thing on the facade the laser measures honestly |
| how wide is it | the same piers, one width for the whole wall | the bay is one repeated element; its copies are one measurement each, their common width is nine |
| how high does it go | the same plan view at **every height** | a stack of plan views is a section |
| is it a window or a door | the ZED | the LiDAR cannot tell glass from a gap, and a door from a doorway |

**Pitch, from every height band at once.** The rhythm belongs to the structure,
so it shows in every plan view that cuts the wall. Session 9, 21 bands of 0.20 m
from the floor to 1.87 m:

| | bands reading 3.20 m | stacked peak | single `--pitch-z` band |
|---|---|---|---|
| south wall | 19 / 21 | **3.180 m** | 3.184 m |
| north wall | 19 / 21 | **3.175 m** | 3.168 m |

The same answer — but it no longer depends on `--pitch-z` having been chosen
well. One band *can* be wrong: the north wall reads 1.60 m at z 0.07–0.27,
picking up the skirting and the radiator feet. One bad band in twenty cannot
move the stacked autocorrelation. `--no-pitch-bands` restores the single band.

### Why the bays measured a third of their width

A window bay is **not empty in plan view.** Its mullions, its frame and the
reveal each return one dense bin right inside the opening. Session 9, south
wall, the 0.10 m profile across four consecutive bays (pier level 105, cut 37):

```
slot 1   72  52  21   8   1   3   1   2   3   5  12 [90]  5   2   8   5   3   1 [135] 47 ...
slot 2   57  88  38   8   7   3   1   1   1   4   6 [94] 38   0   0   2   7   1 [38] 182 ...
slot 3   98  69  23   8   2   1   1   1   2   2   3 [120] 11  1   1   0   5   2 [115] 161 ...
```

Two spikes, at the same offset in every bay, 0.7 m apart, each **one bin wide**.
The old edge walk started at the bay centre and stopped at the first bin over
the cut — the mullion, 0.30 m away — and called that the window's edge.

`--bay-min-pier` (0.35 m) is the fix, and it is a statement about the building:
**a pier is a metre of wall, a mullion is one bin.** The occupancy mask is
opened by that width before the edges are read, so runs shorter than a pier
disappear and the piers keep their exact extent.

| | old | new |
|---|---|---|
| south wall bay width | 1.50 m | **2.33 m** (per bay 2.18 … 2.43) |
| north wall bay width | 1.55 m | **2.13 m** (per bay 1.61 … 2.50) |
| windows confirmed | 6 + 4 | **7 + 5** |

Two more things came with it. The threshold is now the **local** pier level (a
rolling 90th percentile two pitches wide), because the rover passed close to one
end of the corridor and far from the other and the same pier returns 60 points
at one end and 700 at the other. And each edge is interpolated to where the
profile crosses the cut, instead of being quantised to a 0.10 m bin.

### Each bay sits at its own trough, not on the lattice

The lattice decides **how many** bays there are and they all share **one
width**. Their centres are measured. Session 9 says why — the phase offset per
bay, south wall:

```
slot  0 1 2 3 4   -0.34 -0.35 -0.31 -0.31 -0.31
slot  5 6 7 8     +0.08 +0.12 +0.08 +0.04
```

Two rhythms in phase with each other but not with a single offset, breaking at
x ≈ 21 m — the corridor is not one build. On the north wall the drift runs
+0.47 … +0.94 m. A strict lattice puts five of nine bays a third of a metre off
the opening they describe; `--bay-snap-phase` forces it back on if wanted.

### Sill and head, per bay, and the door test

The vertical scan is per bay now, not one pair of numbers for the whole wall:
in each height band the bay's own occupancy is compared with the two piers
flanking it *in that band*, and the opening is the run of z where that contrast
holds. Two bays on session 9 come out reaching the floor —

```
box 0 bottom slot 0   z -0.13..1.77   reaches the floor
box 1 top    slot 3   z -0.13..1.77   reaches the floor
```

— against z 0.57…0.87 upward for every other bay. That is the wall's own
evidence for a doorway rather than glazing over a spandrel, and it is written to
`openings.json` as `bay.reaches_floor`, with every contrast run in `bay.z_runs`.

The class is still the camera's call, not the wall's: `--bay-doors` (on) lets a
`door` mask claim a bay, and the class is whichever class's masks cover most of
it. A door claiming a bay keeps its **own** width (`--bay-door-width mask`) —
session 9's bays are 2.3 m and its doors under 1.1 m, the door stands *in* the
structural opening — and its sill goes to the floor (`--bay-door-to-floor`),
because a door reaches the floor by definition and the scan cannot see the
bottom of the opening past whatever is standing in it.

`--bay-height-from-mask` still overrides the sill with the masks'. Note what
that trades: the masks reach the floor and so does the radiator they swallow, so
the sill is wrong in the way "Read the sills before trusting the numbers" above
describes; the wall's own sill is the honest one, but it is the top of the
radiator wherever a radiator stands under the glass.

### One size for the repeated element: `--bay-uniform-size`

A bay is one window built many times, so its copies are many measurements of
**one size**, not many sizes. The width already leaves the wall as a single
number. The sill and head do not — each bay's vertical scan is stopped somewhere
different by a radiator, a curtain, or the rover's own line of sight — so on
session 9's south wall seven identical windows came out anywhere from 0.90 m to
1.90 m tall. `--bay-uniform-size` (**on**) gives the height the same standing as
the width.

The average is over the copies that **agree**: everything within
`--bay-size-nstd` (1.0) standard deviations of the mean. The spread of the
copies is itself the estimate of how far a copy may sit from the truth — inside
it they are one window measured repeatedly, outside it they are a measurement of
something else. Sill and head are trimmed **independently**, since a bay can be
typical at one edge and an outlier at the other:

```
box 0 bottom: 7 window bay(s) share one size 2.31 x 1.07 m at z 0.70..1.77
              (mean over the bays within 1 sd; sill from 6/7, head from 5/7)
    slot  0 z -0.13..1.77 -- sill outside the spread, not averaged in
    slot  4 z  0.67..1.87 -- head outside the spread, not averaged in
```

Slot 0 is the doorway that runs to the floor: it is excluded from the **average**
without being excluded from the **output**, which is the whole point of trimming
rather than filtering. Every exclusion is printed, and each opening keeps what it
measured on its own in `bay.z_before_uniform`.

Doors are exempt — a door does not repeat, and it already keeps its own width and
its own sill. `--no-bay-uniform-size` restores the per-bay heights.

## Why the fitted windows are short: the LiDAR never sees their top

The segmentation is not the weak link. Measured on 5 frames of session 9, per
`windowpane` segment, comparing the rows the **mask** covers against the rows
that carry a **LiDAR return**:

| frame | mask rows | LiDAR rows | covered | gap at top |
|---|---|---|---|---|
| `..233144_R` id14 | 0 … 685 | 292 … 682 | 56.9% | **292 px** |
| `..233159_R` id12 | 0 … 785 | 339 … 785 | 56.8% | **339 px** |
| `..233214_R` id11 | 0 … 750 | 315 … 750 | 58.0% | **315 px** |
| `..233244_R` id13 | 0 … 790 | 281 … 788 | 64.2% | **281 px** |

Median coverage over 11 segments: **56.8%**. The gap at the **bottom** is 0–3 px
in every single case.

So the laser reaches the bottom edge of every window exactly, and stops roughly
halfway up. It is not sampling noise — the LiDAR's rows never start above ~270
of 1080 in any frame, which is the HAP's 25° vertical against the ZED's 54°.
Two segments in these 5 frames (`mask_v 0..298`, `mask_v 0..241`) sit entirely
above the band and get **zero** returns: a clerestory is invisible by
construction.

**This is why every fitted window comes out ~1.5 m tall in a 2.23 m room and
sits low on the wall.** The rectangle is bounded by the sensor's vertical reach,
not by the opening. Nothing in stage 1, the vote, the box model or the drift
correction touches it.

### `--masks`: the fix

**The LiDAR supplies the plane, the camera supplies the outline.** The points
answer only the two questions they answer well — *which* box face this segment
sits on, and how far away — and the rectangle then comes from the mask itself,
back-projected onto that face:

* each mask pixel is a ray, `dir = R · (R_lc⁻¹ · K⁻¹[u,v,1])`, with the camera
  centre from the same pose chain `project_lidar_to_camera` uses forwards;
* the face is `x = const` or `y = const`, so the intersection is one division;
* only the mask **boundary** is projected — per row the leftmost and rightmost
  pixel, per column the top and bottom — which bounds the region exactly at a
  fraction of the pixels.

**Instance identity comes free.** One rectangle per `(frame, segment)` is one
window in one view, so nothing has to be re-separated by clustering afterwards —
which is what turned a run of bays into a single 15.6 m rectangle.

Two guards, both counted in the run's output:

* `--mask-min-cos` (0.20): a ray nearly parallel to its own wall slides the
  intersection metres for one pixel of noise. 7 838 boundary pixels refused on
  session 9.
* `--mask-max-depth` (25 m): 4 796 refused.
* `--mask-min-points` (20): below that, nothing says *which* wall the mask is
  on, and projecting onto a guess is worse than not projecting. 7 segments.

Views of one opening are merged by **containment — intersection over the
smaller rectangle — not IoU**. A view that caught only part of a bay is largely
*inside* the fuller view, so its IoU is low while its containment is near 1; a
first pass at 0.30 IoU left three overlapping rectangles where there is one bay.
Each edge of the merged rectangle is the **median** across views, never the
union, which would grow with every extra view and every bad one.

Measured on session 9, same 59 177 points and same boxes, point extent vs mask:

| | point extent | `--masks` |
|---|---|---|
| windows | 10 | **16** |
| widest window | **3.94 m** | **2.35 m** |
| height min / median / max | 1.01 / 1.40 / 1.75 m | 1.02 / 1.46 / **2.08** m |
| total opening area | 32.9 m² | 35.4 m² |
| into the `.osm` | — | 17 SubSurfaces, WWR 19.9% |

The bay separation is the clear win: no rectangle spans more than one opening
any more. **The height gain is real but smaller than the 57% coverage figure
suggests** — the rectangle is still clipped to the box's own height (2.23 m
here), and many `(frame, segment)` pairs are partial views that never saw the
whole opening either. The tallest windows now reach 2.08 m against a 1.75 m
ceiling before.

Off by default. `--masks` needs `--session-dir`, and a csv from
`paint_openings.py`, which writes the `frame` and `segment_id` of every point
for exactly this purpose.

## The door is not where the door is

Measured with `paint_openings.py`, session 9, `opening_map_m2f_full`, within a
**single** frame — so this is not drift and not the vote:

| frame | door segment | its points, depth p5 / p50 / p95 | world x span |
|---|---|---|---|
| `..233144_R` | id18, conf 0.90, 77×84 px | 25.9 / **34.4** / 34.4 m | 24.1 … 34.5 |
| `..233214_R` | id13, conf 0.93 | 18.8 / **24.1** / 24.2 m | 25.9 … 34.5 |
| `..233314_R` | id19, conf 0.81 | 24.6 / **30.5** / 30.6 m | 26.5 … 34.4 |

The door plane is at x ≈ 34.4. A tenth of the points labelled `door` are 8–10 m
in front of it, which is what draws a red tube down the corridor instead of a
door.

**Cause: the projection has no visibility test.** A point is labelled by
whatever mask its pixel lands in. The corridor-end door sits at the vanishing
point, so rays through its ~80×80 px are nearly parallel to the corridor and
every grazing hit on the floor and side walls falls inside it. Nothing is
mis-registered; the mask genuinely covers those pixels.

`--depth-band 1.5` removes it — the segment's *median* depth is already the
door — but it is a proxy. The real fix is a per-pixel depth buffer, keeping only
the nearest return per pixel and then requiring the mask's points to be one
surface.

**The same effect is why window points do not describe windows.** In
`..233144_R`, `windowpane` id14 is 170 147 px spanning v 0…686, and its 5 833
LiDAR points sit at 1.6–2.6 m: they are the reveal, the mullions and the
radiators standing in front of the glass, over the middle band of the frame the
HAP's 25° vertical can reach. Fitting a polygon to *those* points fits the
clutter, not the opening. For the OpenStudio export the workable order is the
other way round: fit the **wall** plane from the surrounding returns, then
back-project the image mask onto that plane. The mask has the opening's shape at
full camera resolution; the LiDAR only has to supply the plane.

## Known problems, honestly

Measured on session 9 frames 1–5 with `--bag`, after the switch.

* **~~Stage 2 starves~~ — FIXED, see "The cloud both stages read" below.**
  `/cloud_registered` is a heavily cropped version of what the sensor
  recorded. Measured on session 9, first triplet:

  | | raw `/livox/lidar` | FAST-LIO `/cloud_registered` |
  |---|---|---|
  | points / message | 83 096 | **6 465** (−92%) |
  | azimuth | −59.5° … +60.8° | **−17.2° … +17.4°** |
  | elevation | −13.4° … +13.2° | −4.3° … +8.7° |
  | min range | **1.02 m** | **4.05 m** |

  Projected into the ZED frame that is a patch spanning u ∈ [666, 1309],
  v ∈ [369, 628] — about **8% of the frame area**, and *every one* of the 6465
  points already lands inside the image, so the camera FOV clips nothing. Both
  window bays fall entirely outside it: **0 returns even inside their bounding
  boxes**, not just outside their masks.

  The 4.05 m floor looks like FAST-LIO's `preprocess/blind`, and the ±17°
  azimuth cone like `mapping/fov_degree`; the sensor is a Livox HAP (120° × 25°),
  not a Mid-360. Re-running FAST-LIO with those relaxed, or registering
  `/livox/lidar` against `/Odometry` directly, is the fix. It is not a
  Mask2Former problem and not a glass problem.

  **`../LivoxLidarOdometryLoader/` rebuilds the cloud from the raw topic** and
  recovers it without re-running anything. Measured, frame `20250906_233144_R`:

  | | in frame | footprint | window id=14 | window id=15 |
  |---|---|---|---|---|
  | `/cloud_registered` | 6 465 | u 666..1309 | **0** | **0** |
  | rebuilt raw | 77 894 | u 0..1920 | **5 833** | **5 897** |

  **This disproves a claim this README made for a long time.** "Glazing returns
  no LiDAR — 11 of 19 opening detections got ZERO points, including the
  315×735 px glass wall at conf 0.98–0.99" was measured on the cropped cloud,
  which cannot tell glass apart from out-of-footprint. On the rebuilt cloud
  those same bays return **thousands of points at 1.6–2.5 m**.

  That premise is the stated reason the window rules are pixel-only, the reason
  the metric gate was put at stage 2 rather than stage 1B, and the reason
  `--max-range` gates at the vote. All three need re-deriving on a full cloud.
  Nothing in this module is *wrong* as code; its justifications are.
* **ADE `windowpane` is the glazed bay, occluders included.** A radiator
  standing in front of glass is inside the window mask. Arguably right for a
  polygon fit of the frame; not what the class name says.
* **Windows have no filter at all now.** `--window-filter` is off, so a
  Mask2Former `windowpane` false positive has nothing standing between it and
  the vote. Nothing in these 5 frames exercises that, which means it is
  untested rather than safe.
* **`--door-h-m` / `--door-w-m` are wide enough to admit a passage** (3.00 ×
  2.80 m), which is deliberate — the corridor-end opening is 2.77 × 2.55 m and
  is real — but it means the band no longer discriminates a door from a
  garage-sized hole. The lower bound is doing the work.

### Fixed, and what the fix was worth

Three thresholds were measured wrong on the first pass. All were guesses that
sat just inside the data.

| was | is | why |
|---|---|---|
| `MAX_DEPTH_RATIO 1.35` | **1.50** | 4 of 6 door candidates abstained `multi_depth` at 1.36/1.40/1.41/1.42 — the check was switching itself off on almost every real candidate, and the same object flipped `unknown`↔`bad` between frames on scan jitter. Now `unknown=0`. |
| `MAX_DOOR_W_M 2.20` | **2.80** | The corridor-end opening measures 2.36–2.55 m wide, consistently, and is a real passage. 2.20 rejected it as `door_metric_dims`. `MAX_DOOR_H_M` 2.80 → 3.00 for the same reason. |
| window rule ON | **`--window-filter` off** | It rejected a real bay at h_ratio 0.577 against a 0.60 threshold whose own comment warned it separated by 0.08. |

Effect on session 9 frames 1–5, doors only:

| | before | after |
|---|---|---|
| metric verdicts | ok=0, bad=2, unknown=4 | **ok=3, bad=3, unknown=0** |
| corridor-end door (2.67–2.74 × 2.36–2.52 m) | rejected / abstained | **kept, 3/3 frames, conf 0.87–0.90** |
| radiator-recess FP (0.37–0.82 × 0.40–0.43 m) | kept, 3/3 frames | **rejected, 3/3 frames** |
| windows kept | 11 of 15 | **15 of 15** |

The false positive is the case the metric check was built for: it is *stable*
across frames, so multi-view voting would never have removed it, and it is
0.4 m wide, so one look through the LiDAR settles it.

## Dependency note: `SensorFusionLoader`, not `Calibration`

Both scripts import the calibration loader from `Thesis-final-wt2/SensorFusionLoader/`
(found by searching upwards, see `_find_root`), which holds `rig_calibration.py`,
`rig_calibration.yaml` and `projection.py`.

`EmissivityCalculation`'s own scripts (`classify_session.py`,
`voxel_consensus.py`, `project_to_flir.py`) still hardcode
`Path(__file__).resolve().parent.parent / "Calibration"`, a directory that does
not exist in this repo — **they are broken here as-is.** That is a pre-existing
gap this module does not fix.

It does have to work *around* it, in two places now:
`opening_voxel_consensus.py` and `classify_openings.py::_load_lidar_stack` both
reuse `project_to_flir.nearest_clouds_for_targets` for the one-pass bag read,
and importing that module executes its broken `sys.path.insert`. It loads
cleanly only because `SensorFusionLoader` is imported *first*, putting
`rig_calibration` and `projection` into `sys.modules` before `project_to_flir`
asks for them. **Do not reorder those imports.**
