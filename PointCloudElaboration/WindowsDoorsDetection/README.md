# WindowsDoorsDetection — doors and windows from a ZED session

Classifies every region of a recorded ZED session as **door / window / other**,
then pools multi-view votes per 3-D voxel so a physical opening ends up with
whichever class the majority of the views that actually saw it agree on.

Stage 1 is one **Mask2Former** forward pass on the **ADE20K-150** taxonomy,
which already contains `windowpane` and `door`. Stage 1B (inside the same
script) merges touching same-class regions and rejects implausible ones, with
an optional LiDAR metric check that measures every door candidate in metres
before a size rule may discard it. Stage 2 pools per-frame calls into a 3-D
voxel vote.

Downstream, and **out of scope here**: fitting rectangles to the consensus
voxels (`fit_openings.py`, in this folder) and the OpenStudio `.osm` export
(`to_openstudio.py`, in `3DModelPointCloudExtraction/`).

## Layout

```
WindowsDoorsDetection/
  classify_openings.py         stage 1 — Mask2Former + stage 1B geometry
  opening_voxel_consensus.py   stage 2 — multi-view voxel vote + 3-D map
  fit_openings.py              labelled points/masks -> rectangles on box faces
  view_openings.py             viewer — stage 2 voxels, over the scene cloud
  paint_openings.py            viewer/exporter — stage 1 only, no vote
  overlay_lidar.py             QA: projected LiDAR points drawn on the mask overlay
  opening_table.csv            taxonomy: class, ade, prompt (dead), notes
  openings/
    segmentation_m2f.py        Mask2Former -> regions, confidence, zones
    lidar_metrics.py           metric size of a masked region, from the bag
    geometry.py                stage 1B: merge + plausibility rules
    table.py                   OpeningTable (stdlib csv, no pandas)
    zone_prior.py              geometric prior; consensus's --respect-zones
```

## Pipeline

```powershell
# stage 1 — torch venv. --bag adds the metric door check and needs rosbags too.
& C:\venvs\emissivity\Scripts\python.exe classify_openings.py `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20 `
    --limit 5 --overlay

# stage 2 — rosbags venv
& C:\venvs\sensorfusion\Scripts\python.exe opening_voxel_consensus.py `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20

# viewer — planefit venv (only one with pyvista)
& C:\venvs\planefit\Scripts\python.exe view_openings.py `
    --consensus-dir ...\fullrate\opening_map_consensus `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20 `
    --every-n 10 --scene-max-range 15

# stage 1 -> points -> rectangles -> .osm (the hand-off to OpenStudio, from
# PointCloudElaboration/WindowsDoorsDetection/ and .../3DModelPointCloudExtraction/)
& C:\venvs\planefit\Scripts\python.exe paint_openings.py `
    --session-dir ...\fullrate --bag ...\rosbag2_2026_07_30-18_12_20 `
    --opening-map-dir ...\fullrate\opening_map `
    --out-csv openings_pts.csv --out-scene-csv scene_pts.csv --no-show

& C:\venvs\planefit\Scripts\python.exe fit_openings.py `
    --points openings_pts.csv --boxes ...\SavedBoxes\boxes_edited2.json `
    --out openings.json --no-show

& C:\venvs\planefit\Scripts\python.exe to_openstudio.py `
    --boxes ...\SavedBoxes\boxes_edited2.json --openings openings.json `
    --name Session9 --out session9_openings.osm
```

These are **PowerShell** blocks — backtick is the line continuation, `#` the
comment, `&` the call operator by path. The backtick must be the **last
character on the line**. In **cmd.exe** the same commands use `^` and `::`,
and no `&`.

## classify_openings.py — stage 1

One Mask2Former forward pass (`--model`, default
`facebook/mask2former-swin-large-ade-semantic`) segments and classifies in
one step. Components below `--min-area` (default 1500 px) never become a
region and stay `-1` in the raster.

**Stage 1B** (inside this script, `--geometry-filter`, default on) then:
1. merges touching same-class regions (`--merge-dilate-px`, default 2),
2. rejects windows standing on the floor but too short to be a glazed wall
   (`--window-filter`, **off by default** — it rejected a real bay at h_ratio
   0.577 against the 0.60 threshold),
3. rejects doors by floor contact, bay-edge adjacency, and size.

Everything here runs on the pixel **mask**, never the bounding box — the
bbox is reported but never decides, draws, or exports.

**`--bag`** turns on the LiDAR metric check for doors only (windows are
untouched — glazing returns no LiDAR). Every door candidate is measured in
metres (`openings/lidar_metrics.py`) before a size rule may discard it:
- metric door-sized -> **rescues** a candidate a pixel rule wanted to kill
- metric not door-sized -> **rejects** one a pixel rule would have kept
  (rule `door_metric_dims`, fires before the pixel rules)
- too few points or multi-depth mask -> **abstains**, pixel rules decide alone

Door size band: `--door-h-m` (default 1.60–3.00 m), `--door-w-m` (default
0.50–2.80 m), `--metric-min-points` (12), `--metric-max-depth-ratio` (1.50).
`--overlay` writes `overlay.png` with kept openings outlined and every
**rejected** candidate in yellow, labelled with the rule that killed it.

## opening_voxel_consensus.py — stage 2

Pools every LiDAR point's world position + the segment its pixel landed in
into a `--voxel` grid (default 0.20 m), and takes each voxel's majority
class. `other` competes on equal terms, so a wall voxel resolves to `other`
and a door voxel has to beat it — one confident bad view is not enough to
invent an opening.

Key flags: `--min-vote-confidence` (0.5, drop weak per-frame calls before
pooling), `--max-range` (8.0 m, ignore LiDAR points farther than this — a
surface classified from 20 m away is a blurred handful of pixels),
`--depth-power` (0.0, no distance weighting — measured worse than
distance-weighted), `--respect-zones` (on — re-applies the floor/ceiling
prior to the pooled vote so a voxel straddling the wall/floor junction can't
hand a floor segment a door class).

## `--cloud-source`: raw vs registered

Both stage 1 (`--bag`) and stage 2 read LiDAR through the same
`lidar_metrics.load_clouds`, default **`raw`**. `raw` rebuilds world clouds
from `/livox/lidar` + `/Odometry` via `../LivoxLidarOdometryLoader`.
`registered` reads FAST-LIO's `/cloud_registered`, which is measurably
cropped (a ~35° cone from ~4 m on session 9) and misses whole window bays
entirely — kept only for comparison, never use it to produce anything.

## Output

| path | what |
|---|---|
| `<session>/opening_map/<stem>/labels.npy` | int32 HxW region-id raster, `-1` where no component reached `--min-area` |
| `<session>/opening_map/<stem>/segments.json` | schema `opening_map/v1` — id, bbox, centroid, area, top_class, confidence, top_k, zone, `ade`; stage 1B's merge/reject/rescue history |
| `<session>/opening_map/<stem>/overlay.png` | `--overlay` only |
| `<session>/opening_map_consensus/<stem>/segments.json` | same schema, consensus class substituted, `consensus` block per segment |
| `<session>/opening_map_consensus/door_window_voxels.csv` / `.ply` | the stage 2 deliverable: one row per opening voxel within `--max-range` |

`door_window_voxels.csv` columns: `x,y,z,opening_class,agreement,n_observations,n_votes,w_door,w_other,w_window`.
Every class's pooled weight is written, not just the winner's. `n_observations`
is distinct frames that saw the voxel (the real view count); `n_votes` counts
LiDAR points and grows with scan density, so it's a sampling statistic, not a
view count.

## fit_openings.py — points/masks -> wall-mounted rectangles

OpenStudio needs a `SubSurface` coplanar with, and inside, a parent `Surface`
— not a free-floating polygon. So this fits a **rectangle on a box face**,
not a polygon to the points. Input: `--points` (x,y,z,opening_class — from
`paint_openings.py --out-csv` or stage 2's `door_window_voxels.csv`) and
`--boxes` (`fit_boxes.py`'s `boxes.json`).

Default path: assign each point to the nearest vertical box face within
`--max-face-dist` (0.60 m), cluster on that face by `--cluster-cell` (0.30 m),
take each cluster's extent as the rectangle, filter by `--min-width`/
`--min-height`/`--min-points` plus a door plausibility band imported from
`openings/lidar_metrics.py` (so it can't drift from stage 1B's own numbers).
`--door-floor-tol` (0.30 m) rejects a door that doesn't reach the floor;
`--door-head-clearance` (0.12 m) clamps (not rejects) a door head that would
otherwise land on the box ceiling. `--exterior-only` (on) drops openings on
faces another box covers, since no wall gets built there.

Point extent under-measures window height, because the LiDAR reaches only
about the bottom half of a window bay (see "Known problems"). Alternate
modes, all off by default:
- **`--masks`** — fit each opening from its Mask2Former mask projected onto
  the wall plane instead of from point extent (needs `--session-dir` and a
  csv with `frame`/`segment_id`, i.e. `paint_openings.py --out-csv`).
- **`--image-only-windows`** / **`--void-evidence`** — confirm/size windows
  from the camera or from a hole in the wall's LiDAR coverage, since a
  well-glazed bay returns almost no points to measure at all. Needs
  `--scene-points` (`paint_openings.py --out-scene-csv`).
- **`--regularize`** / **`--regularize-fill`** / **`--regularize-extend`** —
  fit a repeated-bay lattice per wall face and snap/fill openings to it.
  Inferred (filled) slots carry `"synthetic": true`, `n_points: 0`, and draw
  hollow in the viewer.
- **`--bays-from-wall`** — find openings from the wall's own pier/trough
  rhythm in plan view (needs `--scene-points`); the camera only decides
  window vs. door vs. nothing-seen per bay.

See `--help` for the full flag list — each group above has its own tunables
(dozens of them), all documented inline.

Output: `openings.json` (default `<points dir>/openings.json`) — one record
per rectangle with `class`, `box_id`, `side`, `u_min/u_max`, `z_min/z_max`,
`width_m`/`height_m`, `n_points`, `evidence` (`points`/`masks`/`void`/`image`),
plus `rejected` clusters and a `regularize_notes` log of every change made.
`--screenshot` / default interactive window draws boxes wireframed and
openings as coloured planes; `--no-show` skips the viewer.

## paint_openings.py / view_openings.py / overlay_lidar.py — viewers & QA

- **`paint_openings.py`** — stage 1 only, no vote, every gate off by default:
  colours LiDAR points by the mask each projects into. `--min-confidence`,
  `--max-range` mirror stage 2's gates (0 = off here); `--depth-band` is an
  occlusion proxy stage 2 doesn't have (drops points far from a segment's
  median depth — see "The door is not where the door is"). `--out-csv` /
  `--out-scene-csv` write the two csvs `fit_openings.py`'s `--masks`/
  `--void-evidence` need.
- **`view_openings.py`** — stage 2's consensus voxels as coloured cubes
  (door red / window blue), optionally over a rebuilt scene cloud
  (`--bag`+`--session-dir`, or `--scene-ply`). `--paint-cloud` colours the
  scene points themselves instead of drawing cubes, at the cloud's own
  resolution rather than the vote's 0.20 m.
- **`overlay_lidar.py`** — draws projected LiDAR points back onto stage 1's
  `overlay.png`, colour-coded by the class each point was given, so a
  projection/labelling bug is visible at the pixel it happened at rather
  than in a 3-D view.

## Known problems

- **A point extent under-measures window height.** The HAP's 25° vertical
  FOV against the ZED's 54° means the LiDAR reaches roughly the bottom half
  of a window bay and no more (measured: median 56.8% mask coverage, gap
  always at the top). Fitted windows from point extent alone come out short
  and low on the wall. `--masks` in `fit_openings.py` fixes this by fitting
  the camera mask's outline instead.
- **The door-in-mid-air artifact.** `paint_openings.py`'s projection has no
  visibility test — a point is labelled by whichever mask its pixel lands
  in, so a grazing hit on a near floor/wall in line with a far door gets
  labelled `door` too. `--depth-band` is a proxy fix (drop points far from
  the segment's median depth); the real fix would be a per-pixel depth
  buffer.
- **`--window-filter` is off by default**, so a Mask2Former `windowpane`
  false positive has nothing standing between it and the vote.
- **ADE `windowpane` includes occluders.** A radiator standing in front of
  glass is inside the window mask, which is what makes every fitted
  window's sill measure to the floor even though the actual glazing starts
  higher.

## Dependency note: `SensorFusionLoader`, not `Calibration`

Every script here imports the calibration loader from
`Thesis-final-wt2/SensorFusionLoader/` (`rig_calibration.py`,
`rig_calibration.yaml`, `projection.py`), found by searching upward from the
script's own path (`_find_root`), not by counting `.parent` hops.

`EmissivityCalculation`'s own scripts still hardcode a `Calibration/`
directory that doesn't exist in this repo and are broken here as-is — a
pre-existing gap this module doesn't fix but has to work around: importing
`project_to_flir` (used for `--cloud-source registered`) triggers that
broken path insert, which is only harmless because `SensorFusionLoader` is
imported **first**, putting `rig_calibration`/`projection` into
`sys.modules` before `project_to_flir` asks for them. Do not reorder those
imports if you touch `_load_lidar_stack` / the equivalent bootstrap in each
script.
