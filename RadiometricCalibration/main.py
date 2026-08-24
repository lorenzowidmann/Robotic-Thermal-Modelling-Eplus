"""Radiometric calibration of the thermal camera (draft).

Corrects the apparent-temperature map reported by the thermal camera into
true object temperatures, per pixel: each pixel uses its own LiDAR distance
(atmospheric transmission) and its own emissivity (from the ZED material
classification), while relative humidity and air temperature are global.

Usage:
    py main.py --thermal apparent.npy --distance-map distance.npy
               --material brick --humidity 60 --air-temp 20 --out corrected.npy --show
    py main.py --thermal apparent.npy --distance-map distance.npy
               --emissivity-map emissivity.npy --humidity 60 --air-temp 20
    py main.py --thermal 34.2 --distance 5.2 --emissivity 0.93 --humidity 60 --air-temp 20
"""

import argparse
import sys

import numpy as np

from radiometric import correct_temperature, transmittance, water_vapour_content
from radiometric.io_maps import (
    DEFAULT_EMISSIVITY_TABLE,
    check_same_shape,
    load_map,
    lookup_emissivity,
    save_map,
)


def parse_args():
    p = argparse.ArgumentParser(description="Thermal camera radiometric calibration")
    p.add_argument(
        "--thermal",
        required=True,
        help="Apparent temperature: map file (.npy/.csv, deg C) or a single value",
    )

    dist = p.add_mutually_exclusive_group(required=True)
    dist.add_argument(
        "--distance-map", help="LiDAR distance map file (.npy/.csv, metres)"
    )
    dist.add_argument("--distance", type=float, help="Single distance in metres")

    eps = p.add_mutually_exclusive_group(required=True)
    eps.add_argument("--emissivity", type=float, help="Emissivity value (0-1)")
    eps.add_argument(
        "--material",
        help="Material name, looked up in the EmissivityCalculation table",
    )
    eps.add_argument(
        "--emissivity-map", help="Per-pixel emissivity map file (.npy/.csv)"
    )

    p.add_argument(
        "--humidity", type=float, required=True, help="Relative humidity in percent"
    )
    p.add_argument(
        "--air-temp", type=float, required=True, help="Air temperature in deg C"
    )
    p.add_argument(
        "--reflected-temp",
        type=float,
        default=None,
        help="Reflected apparent temperature in deg C (default: air temperature)",
    )
    p.add_argument(
        "--table",
        default=None,
        help="Path to a custom emissivity CSV (used with --material)",
    )
    p.add_argument("--out", help="Save the corrected map to this file (.npy/.csv)")
    p.add_argument(
        "--show", action="store_true", help="Display maps with matplotlib (map mode)"
    )
    return p.parse_args()


def load_scalar_or_map(value: str):
    """Interpret a CLI argument as a float, or else as a map file path."""
    try:
        return float(value)
    except ValueError:
        return load_map(value)


def stats(name: str, arr) -> str:
    if np.isscalar(arr):
        return f"{name}: {arr:.3f}"
    return (
        f"{name}: min={np.nanmin(arr):.3f}  max={np.nanmax(arr):.3f}"
        f"  mean={np.nanmean(arr):.3f}"
    )


def show_maps(apparent, tau, corrected):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (apparent, "Apparent temperature (deg C)", "inferno"),
        (tau, "Atmospheric transmission tau", "viridis"),
        (corrected, "Corrected temperature (deg C)", "inferno"),
        (corrected - apparent, "Correction (corrected - apparent, K)", "coolwarm"),
    ]
    for ax, (data, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(np.atleast_2d(data), cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    plt.show()


def main():
    args = parse_args()

    apparent = load_scalar_or_map(args.thermal)
    distance = args.distance if args.distance is not None else load_map(args.distance_map)

    if args.emissivity is not None:
        emissivity = args.emissivity
    elif args.material:
        table = args.table if args.table else DEFAULT_EMISSIVITY_TABLE
        emissivity = lookup_emissivity(args.material, table)
        print(f"Emissivity of '{args.material}' from table: {emissivity}")
    else:
        emissivity = load_map(args.emissivity_map)

    check_same_shape(thermal=apparent, distance=distance, emissivity=emissivity)

    reflected_temp = (
        args.reflected_temp if args.reflected_temp is not None else args.air_temp
    )

    omega = water_vapour_content(args.humidity, args.air_temp)
    tau = transmittance(distance, args.humidity, args.air_temp)
    corrected = correct_temperature(
        apparent, emissivity, tau, reflected_temp, args.air_temp
    )

    print("\nGlobal conditions")
    print("-" * 60)
    print(f"Relative humidity:     {args.humidity:.1f} %")
    print(f"Air temperature:       {args.air_temp:.1f} deg C")
    print(f"Reflected temperature: {reflected_temp:.1f} deg C")
    print(f"Water vapour content:  {omega:.2f} g/m^3")

    print("\nPer-pixel results")
    print("-" * 60)
    print(stats("Distance (m)      ", distance))
    print(stats("Transmission tau  ", tau))
    print(stats("Apparent T (deg C)", apparent))
    print(stats("Corrected T (deg C)", corrected))

    if args.out:
        save_map(args.out, np.atleast_2d(corrected))
        print(f"\nCorrected map saved to {args.out}")

    if args.show:
        if np.isscalar(corrected):
            print("\n--show ignored: scalar inputs, nothing to display.")
        else:
            show_maps(np.atleast_2d(np.asarray(apparent, dtype=float)), tau, corrected)


if __name__ == "__main__":
    sys.exit(main())
