"""Tabulated emissivity and solar-absorptance values loaded from a CSV file.

The CSV columns are: material, emissivity, emissivity_range, solar_absorptance,
solar_absorptance_range, prompt, notes. The `prompt` column is the text prompt
used by the zero-shot classifier, so the classifier's classes always stay in
sync with the table.

solar_absorptance is the fraction of incident shortwave (sunlight) a surface
absorbs -- independent of emissivity (which governs longwave/thermal-IR
radiation). It is what makes a sun-exposed surface run hotter than a passive
conduction model predicts (see EmissivityCalculation/voxel_u_value.py). Many
of these values are color-dependent (paint, plastic, fabric) -- the table
value is a literature-typical placeholder, see notes/solar_absorptance_range
per material.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "emissivity_table.csv"


@dataclass(frozen=True)
class EmissivityRecord:
    material: str
    emissivity: float
    emissivity_range: str
    solar_absorptance: float
    solar_absorptance_range: str
    prompt: str
    notes: str


class EmissivityTable:
    def __init__(self, csv_path: str | Path = DEFAULT_TABLE):
        df = pd.read_csv(csv_path)
        required = {"material", "emissivity", "emissivity_range",
                    "solar_absorptance", "solar_absorptance_range", "prompt", "notes"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Emissivity table is missing columns: {missing}")
        self._records: dict[str, EmissivityRecord] = {
            row.material: EmissivityRecord(
                material=row.material,
                emissivity=float(row.emissivity),
                emissivity_range=str(row.emissivity_range),
                solar_absorptance=float(row.solar_absorptance),
                solar_absorptance_range=str(row.solar_absorptance_range),
                prompt=str(row.prompt),
                notes=str(row.notes),
            )
            for row in df.itertuples(index=False)
        }

    @property
    def materials(self) -> list[str]:
        return list(self._records.keys())

    @property
    def prompts(self) -> list[str]:
        return [r.prompt for r in self._records.values()]

    def lookup(self, material: str) -> EmissivityRecord:
        try:
            return self._records[material]
        except KeyError:
            raise KeyError(
                f"Material '{material}' not in emissivity table. "
                f"Available: {', '.join(self.materials)}"
            ) from None
