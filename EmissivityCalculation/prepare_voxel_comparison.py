"""Per-frame inputs for the Before / emissivity-correction voxel maps.

Why
---
voxel_consensus.py --stage thermal voxelizes whatever per-frame .npy it is
told to read with --corrected-name, but it only looks inside
emissivity_map/<stem>/. The apparent (uncorrected) FLIR temperature lives in
the rot180 FLIR folder instead, so there is no "before" map to voxelize. This
script writes, into every emissivity_map/<stem>/ that has a correction:

  apparent_temperature_masked.npy  apparent T (the same rot180 .npy that
                                   correct_session.py reads), set to NaN where
                                   the corrected map is NaN, so that before,
                                   after and correction cover the same voxels
  emissivity_correction.npy        corrected - apparent (deg C): the radiometric
                                   correction alone, without the sensor offset
                                   subtracted by estimate_offset.py --apply

Existing files are never overwritten unless --overwrite is given.

Venv: numpy only, any venv works (the SensorFusion one is used for the next
step anyway).

Usage:
    py prepare_voxel_comparison.py --session-dir ...\\fullrate
        --flir-dir ...\\Flir\\session_A1_rot180
    py voxel_consensus.py --stage thermal --session-dir ...\\fullrate --bag ...
        --voxel 0.20 --corrected-name apparent_temperature_masked.npy
        --out-dir ...\\fullrate\\voxel_map_before
    (same with emissivity_correction.npy -> voxel_map_emis_corr), then
    VoxelCorrectionTable.m in MATLAB.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

APPARENT_NAME = "apparent_temperature_masked.npy"
DELTA_NAME = "emissivity_correction.npy"


def parse_args():
    p = argparse.ArgumentParser(description="Before / emissivity-correction inputs for voxel_consensus.py")
    p.add_argument("--session-dir", required=True, metavar="DIR",
                    help="ZED session folder (the one with emissivity_map/)")
    p.add_argument("--flir-dir", required=True, metavar="DIR",
                    help="Folder with the rot180 FLIR .npy apparent-temperature frames")
    p.add_argument("--emissivity-map-dir", default=None, metavar="DIR",
                    help="Default <session>/emissivity_map")
    p.add_argument("--corrected-name", default="corrected_temperature.npy",
                    help="Radiometric correction written by correct_session.py "
                         "(default corrected_temperature.npy, i.e. without sensor offset)")
    p.add_argument("--overwrite", action="store_true",
                    help="Replace existing output files")
    return p.parse_args()


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    emis_dir = Path(args.emissivity_map_dir) if args.emissivity_map_dir else session_dir / "emissivity_map"
    flir_dir = Path(args.flir_dir)
    if not emis_dir.is_dir():
        print(f"emissivity_map folder not found: {emis_dir}", file=sys.stderr)
        return 1
    if not flir_dir.is_dir():
        print(f"FLIR folder not found: {flir_dir}", file=sys.stderr)
        return 1

    n_ok = n_skip = 0
    deltas = []
    for frame_dir in sorted(p for p in emis_dir.iterdir() if p.is_dir()):
        corr_path = frame_dir / args.corrected_name
        # Same naming rule as correct_session.py: the FLIR .npy has no _R suffix.
        app_path = flir_dir / f"{frame_dir.name.replace('_R', '')}.npy"
        if not (corr_path.exists() and app_path.exists()):
            n_skip += 1
            continue
        if not args.overwrite:
            existing = [frame_dir / n for n in (APPARENT_NAME, DELTA_NAME) if (frame_dir / n).exists()]
            if existing:
                print(f"{existing[0]} exists, use --overwrite to replace it", file=sys.stderr)
                return 1

        corrected = np.load(corr_path).astype(np.float64)
        apparent = np.load(app_path).astype(np.float64)
        if corrected.shape != apparent.shape:
            print(f"{frame_dir.name}: shape {corrected.shape} vs apparent {apparent.shape}", file=sys.stderr)
            return 1

        apparent[~np.isfinite(corrected)] = np.nan
        delta = corrected - apparent
        np.save(frame_dir / APPARENT_NAME, apparent.astype(np.float32))
        np.save(frame_dir / DELTA_NAME, delta.astype(np.float32))
        deltas.append(delta[np.isfinite(delta)])
        n_ok += 1

    if not n_ok:
        print(f"No frame has both {args.corrected_name} and a FLIR .npy", file=sys.stderr)
        return 1
    d = np.concatenate(deltas)
    print(f"{n_ok} frame(s) written, {n_skip} skipped")
    print(f"Pixel correction (C): min {d.min():.2f}  mean {d.mean():.2f}  max {d.max():.2f}")
    print(f"Next: voxel_consensus.py --stage thermal --corrected-name {APPARENT_NAME} "
          f"(and {DELTA_NAME}) with a separate --out-dir each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
