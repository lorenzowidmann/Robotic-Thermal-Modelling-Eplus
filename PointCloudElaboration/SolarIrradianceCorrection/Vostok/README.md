# SolarShadowVoxel — VOSTOK shadow mask per surface voxel

> **Status: not wired into the U-value pipeline.** Nothing outside this folder
> reads `shadow_mask.json`. The live chain is `../parse_arpav.py` ->
> `../voxel_u_value.py` -> `../voxel_solar_ns.py` (which calls
> `../sun_incidence.py`) ->
> `../../../3DModelPointCloudExtraction/assign_u_to_osm.py`, and no step in it
> consults a shadow mask. This folder was separated from those live scripts to
> make that boundary explicit; it is kept, not deleted, because the mask is
> exactly the missing input for a per-voxel shadow-aware irradiance term.
> The one run on disk used the placeholder `--north-bearing-deg 193`, so its
> output is a mechanics check, not a trustworthy result -- see "Verified".

Self-contained (no cross-import -- the pieces it reuses are copied in, see
"Provenance" below). Two parts:

- **VOSTOK** ([3dgeo-heidelberg/vostok](https://github.com/3dgeo-heidelberg/vostok),
  cloned to `../../../../vostok`, i.e. `Thesis-final-wt2/../vostok` =
  `Desktop/Measurment_v2/ClaudeCode/vostok`, sibling to `Easy3D`) -- built
  from source, see "Building VOSTOK" below. Does the actual raycast/octree
  shadow computation; this folder doesn't reimplement any of that.
- **`solar_shadow_voxel.py`** -- prepares VOSTOK's inputs from
  `OcTreeVoxel`'s output, runs `vostok.exe`, and parses the result back
  into a per-voxel shadow mask.

## Why

Gets a proper shadow-aware sunlit/occluded mask per surface voxel for
Session 9 at its actual capture time, to later combine with the existing
pvlib/Erbs/Perez irradiance pipeline (`../sun_incidence.py`,
`parse_arpav.py`) instead of trusting VOSTOK's own built-in clear-sky
irradiance model (a cruder Linke-turbidity-fixed-to-3 approximation than
what's already validated there). **This script stops at the shadow mask --
it does not multiply by irradiance magnitude.** That combination (irradiance
x mask -> sol-air correction) is a separate follow-up, once this output
exists and has been sanity-checked.

## Prerequisites

Two things must exist before this script can run; neither is committed.

### 1. `../../OcTreeVoxel`'s output (the point cloud, aligned + voxelized)

This script does not fit planes or voxelize anything itself -- it consumes
`voxels.npz`, `transform.json` and `planes_aligned.json` from
`../../OcTreeVoxel/OcTreeVoxel_out/`. They are build products of that pipeline, not
committed files, so a fresh clone has none of them. Run its two-step
pipeline first:

```powershell
cd ..\..\OcTreeVoxel
C:\venvs\planefit\Scripts\python.exe fit_closed_planes.py
C:\venvs\planefit\Scripts\python.exe aligned_octree.py --voxel-size 0.15
```

`--bag` defaults to the reference cloud
(`C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_filtered`,
~1,066,093 points); pass `--bag <rosbag2_folder>` to both scripts to use a
different one. `solar_shadow_voxel.py` then re-reads that same bag by itself
-- it takes the `bag`/`topic`/`store` fields straight out of
`transform.json`, so it always uses whatever cloud `OcTreeVoxel` was
actually run on, with no flag to keep in sync (`--bag`/`--topic`/`--store`
override it if needed).

Both steps write to `../../OcTreeVoxel/OcTreeVoxel_out/`, which is where this
script looks by default; override with `--voxels` / `--transform` /
`--planes-aligned` if they live elsewhere.

### 2. `vostok.exe`

Built from source once -- see "Building VOSTOK" below. Expected at
`..\..\..\..\vostok\build\vostok.exe`; override with `--vostok-exe`.

## Python environment

```powershell
py -3.12 -m venv C:\venvs\planefit
C:\venvs\planefit\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` here is only `numpy` + `rosbags` -- this script's whole
job is geometry prep plus reading the bag; the raycasting is `vostok.exe`'s
(C++, not a Python dependency). The venv is deliberately shared with
`../../OcTreeVoxel`, whose own `requirements.txt` is a superset of this one (it
adds `open3d`/`opencv-python`/`pyvista` for its RANSAC and viewer steps), so
installing that one covers both and the prerequisite step above runs from
the same interpreter.

## Building VOSTOK

No C++ toolchain existed on this machine going in. Installed via winget
(confirmed via VOSTOK's own `CMakeLists.txt`/README that it needs g++, not
MSVC -- the CMakeLists uses GCC-only flags like `-std=c++0x -m64` that
`cl.exe` doesn't understand):

```powershell
winget install --id Kitware.CMake -e
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e
```

Then, from a shell with both on PATH (a fresh terminal picks this up
automatically once winget's PATH changes take effect; refresh manually in
an existing session with
`$env:PATH = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")`):

```powershell
cd C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\vostok
mkdir build; cd build
cmake -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release `
  "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" `
  "-DCMAKE_EXE_LINKER_FLAGS=-static -static-libgcc -static-libstdc++" `
  "-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs" ..
cmake --build . -j 4
```

Three things needed working around, none of them VOSTOK source bugs:

1. **`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`** -- CMake 4.x dropped support for
   `cmake_minimum_required(VERSION 3.0.2)` (what VOSTOK's `CMakeLists.txt`
   declares). This flag tells CMake 4.x "configure it anyway, using policies
   from 3.5 onward" without editing the vendored `CMakeLists.txt`.
2. **`-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs`** -- `main.cpp` uses
   `std::experimental::filesystem` (the pre-C++17 TS API), which even on a
   modern GCC still needs an explicit link against `libstdc++fs` (unlike
   the mainlined `<filesystem>`, folded into libstdc++ itself since GCC 9).
   `CMakeLists.txt` doesn't link it. Using `CMAKE_CXX_STANDARD_LIBRARIES`
   specifically (not `CMAKE_EXE_LINKER_FLAGS`) matters: GNU `ld` needs a
   static library listed *after* the object files that reference its
   symbols, and `CMAKE_CXX_STANDARD_LIBRARIES` is what CMake appends at the
   very end of the link line.
3. **`-static -static-libgcc -static-libstdc++`** -- not required to build,
   but without it the resulting `vostok.exe` depends on `libgomp-1.dll`
   (OpenMP runtime) being on `PATH` at runtime, including whenever this
   folder's script later invokes it via `subprocess`. Static linking makes
   it a single self-contained binary with no MinGW DLL dependencies at all
   -- confirmed by running it with a stripped-down `PATH`.

Verify: `vostok\build\vostok.exe` with no arguments should print
`ERROR: Please specify a .sol project file.` (its usage message).

## Usage

```powershell
C:\venvs\planefit\Scripts\python.exe solar_shadow_voxel.py --north-bearing-deg <measured value>
```

Everything a run generates goes to `solar_shadow_voxel_out\` next to this
script (created if missing) -- VOSTOK's inputs, VOSTOK's own output, and the
final `shadow_mask.json`. Nothing is written into the module folder itself,
and the whole directory is `.gitignore`d. Redirect with `--workdir` (VOSTOK's
inputs + `shadow_clouds/`) and `--out` (the mask JSON) if wanted.

**`--north-bearing-deg` is required, no default -- this is the open
"resolve south wall / spatial mask" item.** It's the true compass bearing
(degrees from North, clockwise) of the *aligned* building frame's +X axis
(`aligned_octree.py`'s "forward", not +Y/"right"). This is deliberately
**not** the same number as `../voxel_solar_ns.py`'s
`--north-offset-deg` (default 193, documented there as "bearing of the SLAM
+Y axis"): that's the bearing of the *raw, pre-alignment* SLAM frame, and
`aligned_octree.py`'s `compute_building_frame` applies its own extra
rotation (a measured ~2.6 deg yaw correction, see `OcTreeVoxel/README.md`)
on top of it, so the aligned frame's own axis bearing is close to, but not
exactly, that number. Reusing 193 (or 193-90) directly here without
re-deriving it for the *aligned* frame specifically would silently be wrong
by roughly that residual rotation -- hence this script fails loudly instead
of guessing.

All other flags have defaults (site lat/lon, capture time, timezone, etc.
-- see `--help`). See each function's docstring for the pipeline's math
(`north_alignment_rotation`, `assign_surface_voxels`).

### Verified

The full pipeline (transform loading, north-rotation math, surface-voxel
face matching, VOSTOK invocation, shadow-cloud parsing, row-order matching)
was re-run end-to-end in this repo, from a clean state: `fit_closed_planes.py`
-> `aligned_octree.py --voxel-size 0.15` in `../../OcTreeVoxel` (6-plane closed
box, 1,066,093 points, 5,056 occupied voxels) -> `solar_shadow_voxel.py
--north-bearing-deg 193`. That bearing is a placeholder, explicitly *not* the
correct aligned-frame value (see above) -- this was a mechanics-only check,
the numeric result is not meant to be trusted.

Result was physically plausible: 2,980 / 5,056 voxels matched to a surface,
VOSTOK built an 8-deep octree over the 9.99 x 33.63 x 2.58 m cloud and ran
in ~16 s, the closest available shadow file was 2 minutes from the 18:12
target (`2026_211_18-14_shadow.txt` -- VOSTOK only evaluates
sunrise-to-sunset minutes, 05:59-20:38 that day, not midnight-aligned, so an
exact match isn't guaranteed even at a 5-minute step), the position sanity
check matched to 0.0005 m, and the per-face illumination pattern made sense:

| face | lit / total | note |
|---|---|---|
| 0 | 10 / 874 (1.1%) | floor -- interior, mostly occluded |
| 1 | 39 / 165 (23.6%) | wall |
| 2 | 129 / 459 (28.1%) | wall |
| 3 | 226 / 516 (43.8%) | wall |
| 4 | 946 / 946 (100%) | ceiling/roof -- open sky, no surrounding buildings are modeled as shadow points |
| 5 | 20 / 20 (100%) | end cap, also sky-facing |

1,370 / 2,980 lit overall (46.0%); the wall spread (24-44%) varies by
orientation as expected for a single afternoon sun position.

None of that run's output files are committed -- they used the placeholder
bearing and would be misleading if mistaken for a real result (and
`solar_shadow_voxel_out/` is `.gitignore`d anyway). Rerun with the real
`--north-bearing-deg` to get a trustworthy one.

## Output

`solar_shadow_voxel_out/shadow_mask.json` (or `--out`): `voxel_index` (row into `voxels.npz`'s
`centers`/`counts`), `center_aligned` (aligned frame, NOT north-rotated --
consistent with the rest of the `OcTreeVoxel`/`TemperatureToVoxel`
pipeline), `face_id` (0-5, `planes_aligned.json` index), `normal_aligned`
(outward-pointing, aligned frame), `illuminated` (bool). Plus
`shadow_file`/`capture_datetime`/`north_bearing_deg` for provenance.

Also written to `--workdir` (default: `solar_shadow_voxel_out/`): `shadow_points.xyz`,
`query_points.xyz`, `run.sol` (VOSTOK's inputs), and `shadow_clouds/`
(VOSTOK's own output -- one `<year>_<day>_<hour>-<minute>_shadow.txt` per
evaluated minute step across the whole day, sunrise to sunset; only the one
closest to the requested capture time is parsed, the rest are left in place
in case another time of day is wanted later), plus
`run_irradiation_ignored.xyz` (VOSTOK's cumulative-irradiation column, which
this pipeline discards -- see "Why" above) and the `.vostokmeta` sidecars
VOSTOK caches next to each input cloud. None of these are meant to be
committed -- `.gitignore` excludes the whole `solar_shadow_voxel_out/`
directory, and they're regenerated by every run anyway.

## Provenance (self-contained copies, adapted where noted)

- `load_merged_cloud`/`read_pointcloud2` in `solar_shadow_voxel.py` --
  copied from `OcTreeVoxel/fit_closed_planes.py` (same convention: raw,
  unaligned bag points in the SLAM/map frame).
- Everything else (`north_alignment_rotation`, `compose_transform`,
  `face_bounds`, `outward_normal`, `assign_surface_voxels`, the `.sol`
  writer, the VOSTOK runner/parser) is new, written directly against
  VOSTOK's own source (`IrradianceCalc.cpp`, `ProjectConfig.cpp`,
  `main.cpp`, `solpos00.h`) rather than trusting its README prose alone --
  see the module docstring for exactly which source lines back which
  design decision (the +Y=North/+X=East/Z=up convention, the SOLPOS
  timezone sign, the shadow-file row-order guarantee, the `.sol` line
  format).
