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
