# OcTreeVoxel — full-building leveling + octree voxelization + viewer

Self-contained (no cross-import from OpenStudioModel or OcTree -- the files
it reuses are copied in, see "Provenance" below). Three-step pipeline:

1. **`fit_closed_planes.py`** -- RANSAC-fits a single watertight box (floor,
   ceiling, 4 walls) from the raw LiDAR point cloud. Writes
   `OcTreeVoxel_out/planes.json`.
2. **`aligned_octree.py`** -- derives a 3-axis "building frame" rotation from
   that closed box, applies it (+ a translation putting the floor at z=0) as
   a rigid transform to the *entire* raw cloud, then octree-voxelizes the
   now axis-aligned result. Writes `voxels.npz`, `transform.json`, and
   `planes_aligned.json` (the closed box re-derived *in* the aligned frame),
   all into `OcTreeVoxel_out/`.
3. **`view_voxels.py`** -- pyvista viewer for `OcTreeVoxel_out/voxels.npz`: cubes colored by
   point count (no semantic classes here), with the closed box from
   `planes_aligned.json` overlaid as a sanity check (should meet the voxel
   walls at an exact 90 degrees), and an optional raw-points overlay.
   Interactive window by default, with live keyboard controls (see below),
   or `--screenshot out.png` / `--orbit-gif out.gif` for a headless render.

## Why

Generalizes the leveling step in De Pazzi, Chiodini, Pertile (Sensors 2022),
["3D Radiometric Mapping by Means of LiDAR SLAM and Thermal Camera Data
Fusion"](https://doi.org/10.3390/s22134794) -- that paper levels against one
ground plane vs. gravity (Sec. 4.2, Eq. 14), which only corrects a tilt. This
corridor sits at a real *yaw* in the SLAM (`camera_init`) frame, not just a
tilt, so one plane isn't enough: the full closed set of walls/floor/ceiling
is used instead to define all 3 axes. The octree step itself is the paper's
actual method, unchanged: one root bin encompassing all points, recursively
subdivided into 8 occupied children per level.

No thermal/temperature averaging per voxel yet -- geometry/alignment only.

Optional 4th script, joining in a *different* pipeline's output rather than
part of the pipeline above: **`view_voxels_thermal.py`** -- same
`voxels.npz` viewer as `view_voxels.py`, but colored by temperature (real
degrees C) instead of point count, joined against
`EmissivityCalculation/voxel_consensus.py --stage thermal`'s
`thermal_voxels.csv`. See "`view_voxels_thermal.py`" below.

## Usage

```powershell
C:\venvs\planefit\Scripts\python.exe fit_closed_planes.py --bag <rosbag2_folder>
C:\venvs\planefit\Scripts\python.exe aligned_octree.py --voxel-size 0.15
C:\venvs\planefit\Scripts\python.exe view_voxels.py
```

Every generated file lands in `OcTreeVoxel_out\` next to the scripts (created
if missing), and every script reads its inputs from there by default, so the
three steps chain with no path flags. The whole folder is `.gitignore`d --
these are build products of one specific bag + `--voxel-size`, not committed
data. Override individually with `--out` (step 1),
`--planes`/`--voxels-out`/`--transform-out`/`--planes-aligned-out` (step 2),
`--voxels`/`--transform`/`--planes-aligned` (step 3).

`--bag` defaults to the reference bag:
`C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_filtered`
(single merged `/cloud_registered` message, ~1,066,093 points, fields
x,y,z float32, frame_id=camera_init).

`aligned_octree.py` re-reads the same bag itself (via the `bag`/`topic`/
`store` fields recorded in `OcTreeVoxel_out/planes.json`, overridable) -- the planes are used
*only* to compute the alignment transform, every point in the bag is still
loaded and transformed, none are clipped/dropped by plane membership.

`aligned_octree.py --voxel-size 0.15` voxelizes at an arbitrary metric size
(plain uniform grid, `octree.voxelize`). Omit it and use `--depth N` instead
for the power-of-two octree lattice (`build_octree` / `voxelize_octree`,
edge = root_extent / 2**depth) -- the paper's literal recursive-subdivision
method; an arbitrary size like 0.15 m generally isn't reachable that way for
any integer depth.

See each script's module docstring (`--help`) for the full flag list.

### `view_voxels.py` live keys (interactive window only)

| Key | Effect |
|---|---|
| `]` | raise the min-count filter by `--min-count-step` (default 1) |
| `[` | lower it |
| `0` | reset min-count to 1 (show every voxel) |
| `d` | toggle declutter (drop voxels not in the largest 26-connected component -- on by default; raising min-count can strand a voxel that used to touch a since-hidden neighbour) |

Not bound to `+`/`-`/arrow keys: pyvista's own defaults already use those for
camera zoom and point size, so reusing them would fire both at once.

## `view_voxels_thermal.py`

Same `voxels.npz` cube-glyph viewer as `view_voxels.py` (same `]`/`[`/`0`/`d`
live keys, same closed-box overlay, same `--screenshot`/`--orbit-gif`
headless options), but color is mean temperature (deg C, real colorbar) from
a **different pipeline's** output, joined in for display only -- nothing
here is written back into `OcTreeVoxel_out/`.

```powershell
C:\venvs\planefit\Scripts\python.exe view_voxels_thermal.py --thermal-csv <path/to/thermal_voxels.csv>
```

Inputs:
- `OcTreeVoxel_out/voxels.npz` + `transform.json` (this folder's own build
  products, defaulted same as `view_voxels.py`).
- `--thermal-csv` (**required, no default**) -- `thermal_voxels.csv` from
  `EmissivityCalculation/voxel_consensus.py --stage thermal`
  (`x,y,z,t_mean_c,t_std_c,n_obs,material,solar_absorptance`; only
  `x,y,z,t_mean_c` are used here). Unlike this folder's own outputs, that
  file lives elsewhere on disk (e.g.
  `...\SLAM\ZED\<session>\fullrate\voxel_map*\thermal_voxels.csv`), so
  there's deliberately no hardcoded default path.

**Frame-join caveat** -- the two grids don't share a frame *or* a lattice:
- `voxels.npz` is in the aligned (building-frame) coordinates
  `aligned_octree.py` produced (0.15 m voxels in the reference bag).
- `thermal_voxels.csv`'s `x,y,z` are in the **raw, pre-alignment** SLAM
  (`camera_init`) frame -- same session, same underlying LiDAR, but
  `aligned_octree.py` never touches this file, so it is never
  rotated/translated anywhere upstream. There is no pre-aligned copy of the
  thermal data sitting on disk anywhere; alignment happens fresh, in memory,
  every time this script runs.

  `view_voxels_thermal.py` transforms the CSV's raw-frame `x,y,z` into the
  aligned frame itself, on load, using `transform.json`'s *forward*
  convention -- `aligned = raw @ rotation.T + translation` (same direction
  `aligned_octree.py` applied to the whole LiDAR cloud, same direction
  `view_voxels.load_aligned_points()` uses for its `--points` overlay) --
  **not** `rotation_inv`/`translation_inv`, which map the other way
  (aligned back to raw).
- The two grids are also different sizes (0.15 m vs. 0.20 m in the
  reference data) and don't share a lattice origin, so a voxel center in one
  essentially never lands on a voxel center in the other even once both are
  in the same frame. Matching is therefore a radius search
  (`scipy.spatial.cKDTree.query_ball_point`, `--match-radius`, default
  0.20 m -- one thermal-voxel edge length), not an index lookup: each
  `voxels.npz` cell gets the mean `t_mean_c` of every thermal-CSV voxel
  within that radius (0, one, or several).

Before any rendering, the script prints: both point sets' bounding boxes in
the aligned frame, whether they actually overlap, the match count/rate at
`--match-radius`, and a sensitivity sweep at 0.5x/1x/1.5x/2x/3x that radius.
On the reference bag + `20260730_161223/fullrate/voxel_map_m2f/thermal_voxels.csv`
this comes out ~99.5% matched at the 0.20 m default. A non-overlapping bbox
or a very low match rate (<10%, see `LOW_MATCH_RATE_WARN`) prints a loud
warning instead of silently rendering a mostly-grey/empty scene -- that
signals a wrong transform direction or a mismatched bag/thermal-csv pairing,
not a radius that merely needs tuning. `--diagnostics-only` prints all of
this and exits without building any visualization.

Unmatched voxels (no thermal-CSV voxel within `--match-radius`) render in a
distinct neutral color (`--unmatched-color`, default `lightgray`) rather
than being hidden, so it's visually clear which cells are measured vs.
simply absent from the thermal data. Color defaults to `plasma` (sequential,
perceptually uniform -- not MATLAB `ViewThermalCSV.m`'s `jet`), fixed
`--clim` optional as in that script's `tempCLimits`.

Self-contained like the rest of this folder: only `x,y,z,t_mean_c` parsing
logic is copied in (plain `csv.DictReader`, no pandas); nothing is imported
from `EmissivityCalculation/voxel_consensus.py`.

## Output

All of these go to `OcTreeVoxel_out/`.

- `planes.json` -- `normal`, `d`,
  `orientation` (`wall` / `floor_ceiling`), `tilt_deg`, `centroid_3d`,
  `corners_3d`, etc. Closed box only (`close_geometry` + `cap_open_faces`
  always on), no free-standing fragments. Still in the raw SLAM frame.
- `voxels.npz` -- `centers` (M,3), `counts` (M,), `voxel_size` (scalar),
  `origin` (3,), `depth` (scalar, or `-1` if built with `--voxel-size`):
  occupied voxel centers + point counts of the aligned cloud, same fields
  `octree/voxelizer.py`'s `VoxelGrid` produces.
- `transform.json` -- `rotation` (3x3), `translation` (3,) and their inverse
  (`rotation_inv`, `translation_inv`), so voxel coordinates can be mapped
  back to the original SLAM/map (`camera_init`) frame later. Row-vector
  convention: `aligned = points @ rotation.T + translation`.
- `planes_aligned.json` -- the closed box re-derived *in* the aligned frame
  (`aligned_octree.align_and_reclose_planes`). Not just `planes.json`'s
  corners rotated by `transform.json`: `planes.json`'s box was axis-snapped
  against the *original* (pre-alignment) frame, which doesn't exactly match
  the more precise rotation `aligned_octree.py` derives from the true
  measured wall/floor normals -- naively rotating it leaves a small residual
  tilt (a couple degrees). This file re-closes the box after rotating, so
  it's exactly axis-aligned, consistent with the (always axis-aligned by
  construction) voxel grid. This is what `view_voxels.py` overlays.

## Provenance (self-contained copies, adapted where noted)

- `fit_closed_planes.py`'s RANSAC plane-fitting logic (`load_merged_cloud`,
  `segment_planes`, `dedupe_planes`, `close_geometry`, `tilt_from_structure_deg`,
  and their helpers) started as a copy of `Thesis/OpenStudioModel/fit_planes.py`,
  trimmed to what this pipeline actually uses (dropped the unused ROI/SOR/
  declutter CLI options) and folded into this one file rather than kept as a
  separate imported module, since nothing else in this pipeline needs those
  functions except through `fit_closed_planes.py`. `aligned_octree.py`
  imports `canonical_normal_offset`, `close_geometry`, `load_merged_cloud`
  from it; `view_voxels.py` imports `load_merged_cloud`.
- `octree/octree.py` -- copied verbatim from
  `PointCloudElaboration/OcTree/octree/octree.py` (numpy-only, no adaptation
  needed).
- `octree/voxelizer.py` -- copied from the same module, with the
  `classes.py`-dependent `MAX_CLASS_ID` import inlined as a local constant
  (`= 0`): `classes.py` (TUM-FACADE semantic ids) was intentionally not
  copied here since this pipeline doesn't classify points. The `VoxelGrid`
  fields used (`centers`, `counts`, `voxel_size`, `origin`) are unaffected;
  the unused `labels` field is always 0.
- `octree/smoothing.py`, `viewer.py`, `classes.py`, `las_loader.py`,
  `rosbag_loader.py`, `openstudio_adapter.py` are **not** copied -- not
  needed for geometry/alignment-only voxelization.
