"""Loading/saving 2-D maps and emissivity lookup from the ZED pipeline table.

Maps are 2-D float arrays stored as .npy or .csv. The thermal, distance and
emissivity maps must be co-registered (same HxW, pixel-aligned): projecting
the LiDAR point cloud / ZED classification onto the thermal image (extrinsic
calibration between the sensors) is handled separately.
"""

import csv
from pathlib import Path

import numpy as np

# Table shipped with the EmissivityCalculation module (sibling folder).
DEFAULT_EMISSIVITY_TABLE = (
    Path(__file__).resolve().parent.parent.parent
    / "EmissivityCalculation"
    / "emissivity_table.csv"
)


def load_map(path: str | Path) -> np.ndarray:
    """Load a 2-D float map from a .npy or .csv file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Map file not found: {path}")
    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",", dtype=float)
    else:
        raise ValueError(f"Unsupported map format '{path.suffix}' (use .npy or .csv)")
    arr = np.atleast_2d(np.asarray(arr, dtype=float))
    if arr.ndim != 2:
        raise ValueError(f"{path} is not a 2-D map (shape {arr.shape})")
    return arr


def save_map(path: str | Path, arr: np.ndarray) -> None:
    """Save a 2-D float map to a .npy or .csv file."""
    path = Path(path)
    if path.suffix == ".npy":
        np.save(path, arr)
    elif path.suffix == ".csv":
        np.savetxt(path, arr, delimiter=",", fmt="%.3f")
    else:
        raise ValueError(f"Unsupported map format '{path.suffix}' (use .npy or .csv)")


def check_same_shape(**named_maps) -> None:
    """Raise if the 2-D maps among the inputs have different shapes."""
    shapes = {
        name: m.shape
        for name, m in named_maps.items()
        if isinstance(m, np.ndarray) and m.ndim == 2
    }
    if len(set(shapes.values())) > 1:
        detail = ", ".join(f"{name}: {shape}" for name, shape in shapes.items())
        raise ValueError(f"Maps are not co-registered (shapes differ): {detail}")


def lookup_emissivity(
    material: str, table_path: str | Path = DEFAULT_EMISSIVITY_TABLE
) -> float:
    """Tabulated emissivity of `material` from the EmissivityCalculation CSV."""
    table_path = Path(table_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Emissivity table not found: {table_path}")
    with open(table_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["material"] == material:
            return float(row["emissivity"])
    available = ", ".join(row["material"] for row in rows)
    raise KeyError(
        f"Material '{material}' not in emissivity table. Available: {available}"
    )
