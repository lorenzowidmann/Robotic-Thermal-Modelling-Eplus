# Radiometric Calibration

Corrects the **apparent temperature** measured by the thermal camera into the
**true object temperature**, per pixel. Each pixel of the thermal image sees
a different distance (the ground just in front of the rover vs. terrain many
metres ahead), so each pixel gets its own atmospheric transmission computed
from the LiDAR **distance map**, and its own **emissivity** (from the ZED
material classification in `../EmissivityCalculation`). Relative humidity and
air temperature are global values for the frame.

## Physics

Standard thermography measurement chain (FLIR-style):

1. Water vapour content of the air (g/m³) from relative humidity RH (%) and
   air temperature T_atm (°C):

   `ω = (RH/100) · exp(h₁ + h₂·T_atm + h₃·T_atm² + h₄·T_atm³)`

2. Atmospheric transmission per pixel over the LiDAR distance d (m)
   (LOWTRAN-derived constants used by FLIR):

   `τ(d) = X·exp[−√d·(α₁ + β₁·√ω)] + (1−X)·exp[−√d·(α₂ + β₂·√ω)]`

3. The camera reports T_app assuming ε = 1 and τ = 1. Inverting the radiation
   balance per pixel gives the true temperature:

   `T_obj = W⁻¹( [W(T_app) − (1−ε)·τ·W(T_refl) − (1−τ)·W(T_atm)] / (ε·τ) )`

The radiance model W(T) is the Stefan–Boltzmann T⁴ form (draft default);
`radiometric/radiance.py` is structured so the camera-specific Planck
constants (R1, R2, B, F, O) can be plugged in once the thermal camera model
is fixed.

**Assumption:** the thermal, distance, and emissivity maps are co-registered
(same H×W, pixel-aligned). Projecting the LiDAR point cloud / ZED
classification onto the thermal image (extrinsic calibration between the
sensors) is handled separately, like the viewing-angle correction was
deferred in EmissivityCalculation. Once that projection exists, the corrected
per-pixel true temperature can be attached to the LiDAR points and averaged
into the per-wall 2-D raster produced by `../PointCloudElaboration/OcTree`
(its `planes.py` already consumes a per-point temperature scalar, and falls
back to a synthetic field until then).

## Synchronization (design note — not yet implemented)

The current draft processes a single, already-matched set of maps. On the
moving rover the inputs arrive as continuous streams at different rates, so
before correction each thermal frame must be paired with the LiDAR distance
map (and emissivity map) captured at the *same instant*. This is the temporal
counterpart to the spatial co-registration above: co-registration answers
"which pixel", synchronization answers "which moment".

Why it matters: the rover is moving, so the scene geometry changes
continuously. Pairing a thermal frame with a LiDAR sweep even ~50–100 ms
older applies τ with stale distances, and the error is largest for the far
pixels — exactly where the atmospheric correction is already largest.

The sensors do not tick together, and they are not equally time-critical:

| Sensor        | Typical rate | Provides       | Time-criticality                     |
|---------------|--------------|----------------|--------------------------------------|
| Thermal cam   | ~9 / 30 Hz   | apparent T     | reference clock (frames drive output)|
| LiDAR         | ~10–20 Hz    | distance map   | high — geometry changes fast         |
| ZED + CLIP    | ~15–30 Hz    | emissivity map | low — material class is stable       |
| Hygrometer    | ~1 Hz        | RH, air temp   | negligible — ~constant over a run    |

So the tight pair is **thermal ↔ LiDAR**; emissivity may lag by several
frames without harm, and humidity/air-temp are effectively constant.

Approaches, in order of preference:

1. **Hardware trigger / PTP clock** — a shared trigger line or PTP timestamps
   make the sensors capture at the same instant, reducing the software to a
   pass-through. Preferred when the hardware supports it.
2. **Timestamp + nearest-match (software default)** — every measurement is
   tagged with one common host clock; keep a short ring buffer of recent
   LiDAR frames and, for each thermal frame, take the LiDAR frame with the
   closest timestamp within a tolerance window (e.g. ±20 ms), dropping the
   frame if nothing falls inside. Realistic for the thesis rig.
3. **Within-sweep de-skew** (refinement) — a spinning LiDAR captures its
   points across one rotation; rover odometry/IMU can correct the intra-sweep
   skew. Likely out of scope for the draft.

Planned architecture: the correction stage (`correction.py`) is pure array
math and stays untouched. A new synchronizer layer sits *in front* of it,
ingesting `(timestamp, data)` from each source (the extended `sensors.py`
classes) and emitting a matched bundle

```
(t_ref, thermal_map, distance_map, emissivity_map, RH, air_temp)
```

which is exactly the input `main.py` assembles by hand from files today. All
timing and buffering logic is isolated in that one layer; the physics stays
clean. It is left unimplemented until the real sensors and their clocks are
available, since it cannot be validated without them.

## Setup

```powershell
pip install -r requirements.txt   # numpy (+ matplotlib for --show)
```

## Usage

```powershell
# Full per-pixel mode: thermal map + LiDAR distance map, uniform material
python main.py --thermal apparent.npy --distance-map distance.npy `
               --material brick --humidity 60 --air-temp 20 --out corrected.npy --show

# Per-pixel emissivity too (map produced upstream from the ZED classification)
python main.py --thermal apparent.npy --distance-map distance.npy `
               --emissivity-map emissivity.npy --humidity 60 --air-temp 20

# Quick single-point check (scalars everywhere)
python main.py --thermal 34.2 --distance 5.2 --emissivity 0.93 --humidity 60 --air-temp 20
```

Maps are 2-D float arrays as `.npy` or `.csv`. Invalid pixels (LiDAR holes:
NaN or non-positive distance) propagate as NaN in the output.

Emissivity can be given three ways: `--emissivity 0.93` (value),
`--material brick` (looked up in
`../EmissivityCalculation/emissivity_table.csv`, override with `--table`), or
`--emissivity-map file.npy` (per-pixel).

`--reflected-temp` sets the reflected apparent temperature; it defaults to
the air temperature (the usual field assumption).

## Hardware (later)

`radiometric/sensors.py` holds placeholder classes for the real inputs —
`ThermalCameraSource`, `LidarSource` (distance map projected onto the thermal
image), `HygrometerSource` (RH + air temperature) — following the same
stub-until-SDK-installed pattern as `ZedSource` in EmissivityCalculation.
Until then the CLI takes files/values.

## Structure

```
RadiometricCalibration/
├── main.py                  # CLI entry point (map mode + scalar point mode)
├── radiometric/
│   ├── atmosphere.py        # water vapour content + transmittance tau(d, RH, T)
│   ├── radiance.py          # RadianceModel: T <-> W (T^4 draft, Planck-ready)
│   ├── correction.py        # correct_temperature(): radiation balance inversion
│   ├── io_maps.py           # .npy/.csv map I/O + emissivity table lookup
│   └── sensors.py           # hardware stubs (thermal cam, LiDAR, hygrometer)
└── requirements.txt
```
