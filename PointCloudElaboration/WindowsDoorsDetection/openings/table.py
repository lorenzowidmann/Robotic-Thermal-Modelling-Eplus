"""The door/window/other class list and its ADE20K mapping, loaded from
opening_table.csv.

Same role EmissivityCalculation/emissivity/table.py plays for materials: the
taxonomy lives in the file, so a change (door_open/door_closed,
window_shutter, ...) is a CSV edit, not a code edit.

Columns
-------
class   the class name segments.json and the consensus stage use
ade     the ADE20K-150 label Mask2Former must emit for a region to become this
        class, matched on the first comma-separated synonym, lowercased.
        Blank means "no ADE class maps here" -- which is correct for `other`,
        the catch-all every unmapped ADE label falls into.
prompt  the CLIP prompt. DEAD as of the Mask2Former switch: stage 1 no longer
        runs a zero-shot classifier, and openings/classifier.py -- the module
        that used to read it -- has been deleted as unused. Kept anyway
        because the column documents what each class means.
notes   free text.

Two deliberate differences from EmissivityTable:

  * stdlib csv, not pandas. voxel-consensus runs under the rosbags venv
    (no pandas, no torch -- see EmissivityCalculation/voxel_consensus.py's
    header), and it needs the class list to pool votes. Keeping this module
    dependency-free means both stages import the same file.

  * class names must be UNIQUE. There is no multi-prompt-per-class pooling;
    one row is one class is one prompt. Duplicate rows raise rather than
    silently letting the last one win.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TABLE = Path(__file__).resolve().parent.parent / "opening_table.csv"

# The class every segment falls back to. Referenced by the zone prior
# (a floor or ceiling patch cannot be an opening) and by the consensus
# export (only door/window voxels are written out).
OTHER_CLASS = "other"


@dataclass(frozen=True)
class OpeningRecord:
    cls: str
    prompt: str
    notes: str
    ade: str = ""


class OpeningTable:
    def __init__(self, csv_path: str | Path = DEFAULT_TABLE):
        path = Path(csv_path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Opening table is empty: {path}")
        required = {"class", "prompt", "notes"}
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"Opening table is missing columns: {missing}")

        self._records: dict[str, OpeningRecord] = {}
        for row in rows:
            name = row["class"].strip()
            if name in self._records:
                raise ValueError(
                    f"Duplicate class {name!r} in {path}. One row is one class is one "
                    "prompt -- multi-prompt pooling is not implemented.")
            self._records[name] = OpeningRecord(
                cls=name, prompt=str(row["prompt"]), notes=str(row["notes"]),
                ade=str(row.get("ade") or "").strip().lower())

        if OTHER_CLASS not in self._records:
            raise ValueError(
                f"Opening table must define a {OTHER_CLASS!r} class: it is the class the "
                "zone prior forces floor/ceiling segments to, and the one everything that "
                "is not an opening has to land in.")

    @property
    def classes(self) -> list[str]:
        return list(self._records.keys())

    @property
    def prompts(self) -> list[str]:
        return [r.prompt for r in self._records.values()]

    @property
    def ade_to_class(self) -> dict[str, str]:
        """{ade label -> our class}, for the classes that declare one.

        Deliberately NOT defaulted: an ADE label absent from this dict is
        `other` by omission, which is what makes `other` a genuine catch-all
        over all 150 ADE classes instead of one CLIP prompt trying to cover
        walls, floors, radiators, pillars, people and clutter at once. That
        prompt was the weakest part of the old pipeline.

        Several ADE labels may map to one class -- `ade` accepts a
        semicolon-separated list -- but a label may not map to two classes,
        which would make the raster ambiguous.
        """
        out: dict[str, str] = {}
        for rec in self._records.values():
            for name in (n.strip() for n in rec.ade.split(";")):
                if not name:
                    continue
                if name in out and out[name] != rec.cls:
                    raise ValueError(
                        f"ADE label {name!r} is claimed by both {out[name]!r} and "
                        f"{rec.cls!r}. One ADE label maps to at most one class.")
                out[name] = rec.cls
        return out

    @property
    def opening_classes(self) -> list[str]:
        """Everything that is an actual opening -- i.e. not OTHER_CLASS. These
        are the classes the consensus stage exports as voxels."""
        return [c for c in self._records if c != OTHER_CLASS]

    def lookup(self, cls: str) -> OpeningRecord:
        try:
            return self._records[cls]
        except KeyError:
            raise KeyError(
                f"Class '{cls}' not in opening table. Available: {', '.join(self.classes)}"
            ) from None
