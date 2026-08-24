# SolarIrradianceCorrection

Turns voxel-averaged **interior surface temperatures** into a **per-voxel
U-value**, with the solar gain accounted for by the physics appropriate to each
surface's geometry.

Input comes from `../../EmissivityCalculation/voxel_consensus.py --stage thermal`
(`thermal_voxels.csv`: per-voxel `x,y,z`, corrected `t_mean_c`, consensus
`material` and `solar_absorptance`). Output feeds
`../../3DModelPointCloudExtraction/assign_u_to_osm.py`, which bakes the U values
onto the OpenStudio model's surfaces.

## Pipeline

Run in order:

1. **`parse_arpav.py`** — scrapes the ARPAV weather page export into a tidy
   GHI time series.
   ```powershell
   python parse_arpav.py --in Legnaro.htm --out legnaro_ghi.csv
   ```

2. **`voxel_u_value.py`** — the base U per voxel, no solar term:
   `U = hsi * (Tint - Tsurf) / (Tint - Text)`. Also flags voxels whose raw
   Tsurf falls outside `[min(Tint,Text), max(Tint,Text)]`, which is physically
   implausible for a pure-conduction surface — `solar_suspected`,
   `solar_possible` or `implausible` depending on the material.
   Writes `thermal_voxels_u.csv`.

3. **`voxel_solar_ns.py`** — the solar correction that is actually used, for
   the exterior walls. Places the gain where the sun physically is, on the
   OUTSIDE face, via the sol-air temperature:
   ```
   T_sol_air = Text + (alpha * I) / he
   U         = hsi * (Tint - Tsurf) / (Tint - T_sol_air)
   ```
   `alpha` is per voxel from the input CSV, `he` is normative (25 W/m²K,
   UNI EN ISO 6946), and `I` comes from `sun_incidence.py`, which this script
   invokes as a subprocess per wall orientation — so there is no free parameter
   left to tune. Writes `thermal_voxels_u_solair.csv`, the file
   `assign_u_to_osm.py` consumes (`u_value_corrected_w_m2k`).
   ```powershell
   py voxel_solar_ns.py --in <run>/thermal_voxels_u.csv --tint 29 --text 36.9 --hsi 8
   ```
   Surfaces it declares unmeasurable (far side not outdoor air) get an **empty**
   `u_value_corrected_w_m2k` on purpose — that is a "no valid U exists", not a
   missing value, and downstream must not fill it in.

4. **`sun_incidence.py`** — plane-of-array irradiance for one plane at one
   instant, via pvlib (Erbs decomposition + Perez transposition + IAM). Called
   by `voxel_solar_ns.py`; also runnable alone to inspect a wall.
   ```powershell
   python sun_incidence.py --planes planes.json --list          # pick a --plane-id
   python sun_incidence.py --planes planes.json --plane-id 1 --north-offset-deg 173 ...
   ```

## Removed: `voxel_solar_floor.py` — solar path through the windows

Removed from the working tree, recoverable from git:

```powershell
git show 9e799fc:PointCloudElaboration/SolarIrradianceCorrection/voxel_solar_floor.py > voxel_solar_floor.py
```

**It is the script that computed where the sun actually reached through the
glazing**, and it produced the thesis figure `_ImgTesi/sunpatch_final.png`
("Floor voxels colored by measured Tsurf, with beam sunpatches"). What it did:

1. Parsed the `FixedWindow` SubSurfaces out of an OpenStudio `.osm`
   (`session9_openings.osm`) and translated them back into the SLAM/voxel frame
   via `--osm-origin`. `--assumed-window` let a rectangle the scan had failed to
   reconstruct be added by hand; anything added that way was tagged `assumed` in
   the output's `window_source` column, never silently mixed with surveyed
   geometry.
2. Computed the sun vector in the SLAM frame for the capture instant
   (session 9: azimuth 271.9°, elevation 24.5°).
3. **Projected each window aperture along the sun ray onto the floor plane** —
   the resulting quadrilateral is the sunpatch. A floor voxel counted as sunlit
   if its (x,y) fell inside one. Windows whose ray left the room (sun behind
   that wall — the whole north row at this instant) were dropped. No occlusion
   from furniture, pillars or reveal depth was modelled: flagged, not corrected.
4. Computed the transmitted beam flux,
   `I_transmitted = tau_n * IAM(aoi_window) * DNI * cos(zenith)`
   (`tau_n` 0.85, clear single glazing; IAM and DNI from the same pvlib/Erbs
   chain as the rest of this module). **Beam only** — diffuse light also enters
   through the glazing but arrives from the whole sky hemisphere and needs view
   factors, so the script explains the sunpatch *contrast*, not the room's total
   transmitted gain.
5. Applied the ADDITIVE form
   `U = [hsi*(Tint - Tsurf) + alpha * I_transmitted] / (Tint - Text)`,
   deliberately not sol-air: a sunpatch lands on the very face the camera sees,
   which is the one case `voxel_solar_ns.py`'s docstring concedes the additive
   form is right for.

**What worked — the geometry and the descriptive result.** 33 of 453 floor
voxels fell inside a patch (22 of them from the assumed window w13, 11 from
surveyed w11/w12). Sunlit voxels sit **+0.95 K** above the rest of the floor
raw, **+0.60 K** after controlling for the y-band, i.e. a real but modest
measured effect. That comparison is what the figure shows and it stands.

**What did not work — the U correction built on top.** With `alpha * I = 119
W/m²` added to the numerator and a *negative* denominator (`Tint - Text =
29 - 34.8 = -5.8`, cooling season), every one of the 33 voxels came out at a
negative U (median **-8.45**, range -10.51 … -7.47). That is the same sign
failure `voxel_solar_ns.py` documents for the walls, reached from the other
direction. Nothing downstream ever consumed the result, and
`assign_u_to_osm.py` excludes `Floor` surfaces anyway.

The plotting code behind `sunpatch_final.png` was ad hoc and never saved — the
script wrote only a CSV and a `.ply`. Regenerating the figure means rewriting
the plot around the recovered script.

## `Vostok/`

A separate, **currently unconnected** branch: `solar_shadow_voxel.py` drives the
VOSTOK raycaster to produce a per-voxel sunlit/occluded mask. Nothing in the
pipeline above reads it. Kept because that mask is the missing input for a
shadow-aware irradiance term. See `Vostok/README.md`, which also documents
building `vostok.exe` and the placeholder-bearing caveat on the run currently
on disk.

## Environment

```powershell
py -3.12 -m venv C:\venvs\planefit
C:\venvs\planefit\Scripts\python.exe -m pip install -r requirements.txt
```

Shared with `../OcTreeVoxel` and with `assign_u_to_osm.py` (which additionally
needs `openstudio`), so the chain from voxels to `.osm` runs from one
interpreter. `Vostok/` has its own, smaller `requirements.txt`.

## Known issues

- `voxel_solar_ns.py` lines 139–141: `DEFAULT_PLANES`, `DEFAULT_SUN` and
  `DEFAULT_GHI` all point at `../OpenStudioModel/`, which **does not exist** —
  the real one is under `../../3DModelPointCloudExtraction/`. Pass all three
  explicitly until this is fixed.
- No per-orientation irradiance for floors/ceilings: the sol-air form is applied
  to exterior walls only.

## Output (not tracked, see `.gitignore`)

`solar_shadow_voxel_out/<run>/` — `thermal_voxels_u.csv`,
`thermal_voxels_u_solair.csv` and their `.ply` previews, plus the ARPAV extract.
The directory name is a leftover from when this module was VOSTOK-only.
