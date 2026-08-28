"""The ADE20K-150 -> allowed-materials prior, loaded from ade_material_prior.csv.

What this replaces
------------------
../emissivity/zones.py::zone_of guesses a region's kind from its BBOX SHAPE --
wide and low is a floor, tall is vertical -- and ZONE_CANDIDATES then restricts
CLIP to the materials that kind of surface can be made of. The restriction is
the right idea and is kept verbatim below; the guess is not. Its documented
failure (WindowsDoorsDetection/openings/zone_prior.py, "The ceiling rule eats
high windows") is that a wide region high in the frame is called `ceiling`
whatever it actually is, so a clerestory window is misconstrained by
construction and no amount of confidence gating downstream can see it.

Here the kind of surface comes from Mask2Former: a region is a column because
the segmenter called it `column`, and a column is not made of glass. That is
the whole idea -- a categorical prior with a semantic source instead of a
geometric one.

Columns
-------
group      free-text name for the row, echoed into segments.json for audit.
ade        semicolon-separated ADE20K-150 labels, matched on the FIRST
           comma-separated synonym, lowercased -- so `floor;flooring` in the
           checkpoint matches `floor` here (see segmentation_m2f._first_name).
zone       floor / ceiling / vertical / any. NOT used to pick materials; it is
           written into segments.json so ../voxel_consensus.py keeps working.
           See the subset invariant below.
materials  semicolon-separated names from ../emissivity_table.csv.
notes      free text -- why this row allows what it allows.

An ADE label absent from every row is unmapped BY OMISSION and gets no
categorical list, only the emissivity floor. Same fallback ZONE_CANDIDATES
["any"] has, and the same reason: "no prior" must not mean "anything goes",
because a bare-metal call (eps=0.07) turns a 37 degC reading into ~156 degC
while every ordinary confusion in this table costs under 1 degC.

The zone subset invariant
-------------------------
../voxel_consensus.py:270 re-applies the GEOMETRIC prior when it pools votes
across frames: `restrict_ranking(ranked, seg.get("zone", "any"))`. If a row
allowed a material that ZONE_CANDIDATES[row.zone] forbids, stage 2 would strip
stage 1's answer back out and the two stages would disagree silently. So a row
with zone != "any" must list a SUBSET of ZONE_CANDIDATES[zone], and that is
checked at load time rather than left as a comment. A row whose materials do
not fit any geometric zone (a rug is fabric; ZONE_CANDIDATES["floor"] has no
fabric) declares zone `any` and is left alone by stage 2.

stdlib csv, not pandas -- same reasoning as
WindowsDoorsDetection/openings/table.py: the module has to stay importable
from the rosbags venv, which has no pandas.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PRIOR_TABLE = Path(__file__).resolve().parent.parent / "ade_material_prior.csv"

VALID_ZONES = ("floor", "ceiling", "vertical", "any")


@dataclass(frozen=True)
class PriorRecord:
    group: str
    zone: str
    materials: tuple[str, ...]
    notes: str


class AdeMaterialPrior:
    """{ade label -> allowed materials}, validated against the emissivity table.

    `table` is the ../emissivity/table.py EmissivityTable the run will use, so a
    material name that does not exist -- a typo, or a row left behind by a table
    edit -- fails here instead of silently dropping out of the filter and
    widening the prior it was meant to narrow.
    """

    def __init__(self, table, csv_path: str | Path = DEFAULT_PRIOR_TABLE):
        path = Path(csv_path)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"Material prior table is empty: {path}")
        required = {"group", "ade", "zone", "materials", "notes"}
        missing = required - set(rows[0].keys())
        if missing:
            raise ValueError(f"Material prior table is missing columns: {missing}")

        # Imported here rather than at module scope: this file is meant to stay
        # importable without the parent package on sys.path (the driver puts it
        # there), and only the subset check needs ZONE_CANDIDATES.
        from emissivity.zones import ZONE_CANDIDATES

        known = set(table.materials)
        self.path = path
        self._by_ade: dict[str, PriorRecord] = {}
        self._groups: list[PriorRecord] = []

        for row in rows:
            group = row["group"].strip()
            zone = (row["zone"] or "any").strip().lower()
            if zone not in VALID_ZONES:
                raise ValueError(
                    f"{path}: group {group!r} declares zone {zone!r}; must be one of "
                    f"{', '.join(VALID_ZONES)}.")

            materials = tuple(m.strip() for m in row["materials"].split(";") if m.strip())
            if not materials:
                raise ValueError(
                    f"{path}: group {group!r} lists no materials. A row that allows nothing "
                    "would fall back to the unrestricted ranking, i.e. the opposite of what "
                    "it says. Delete the row instead -- omission is the neutral case.")
            unknown = [m for m in materials if m not in known]
            if unknown:
                raise ValueError(
                    f"{path}: group {group!r} lists material(s) {unknown} that are not in "
                    f"the emissivity table. Available: {', '.join(sorted(known))}")

            # The invariant the module docstring explains: stage 2 re-applies
            # the geometric prior, so this row must not allow what that prior
            # forbids.
            allowed_by_zone = ZONE_CANDIDATES.get(zone)
            if allowed_by_zone is not None:
                conflict = [m for m in materials if m not in allowed_by_zone]
                if conflict:
                    raise ValueError(
                        f"{path}: group {group!r} declares zone {zone!r} but allows "
                        f"{conflict}, which ZONE_CANDIDATES[{zone!r}] forbids. "
                        "voxel_consensus.py re-applies the geometric prior on this zone "
                        "and would strip those materials back out, so the two stages "
                        "would disagree. Either drop them, or set zone to 'any'.")

            rec = PriorRecord(group=group, zone=zone, materials=materials,
                              notes=str(row["notes"]))
            self._groups.append(rec)
            for name in (n.strip().lower() for n in row["ade"].split(";")):
                if not name:
                    continue
                prev = self._by_ade.get(name)
                if prev is not None and prev.group != group:
                    raise ValueError(
                        f"{path}: ADE label {name!r} is claimed by both {prev.group!r} and "
                        f"{group!r}. One ADE label maps to at most one group, otherwise the "
                        "prior applied to a region would depend on row order.")
                self._by_ade[name] = rec

    @property
    def ade_labels(self) -> list[str]:
        """Every ADE label the table mentions -- the driver checks these against
        the loaded checkpoint's id2label so a typo cannot silently disable a row."""
        return sorted(self._by_ade)

    @property
    def groups(self) -> list[PriorRecord]:
        return list(self._groups)

    def candidates(self, ade_name: str) -> tuple[str, ...] | None:
        """Allowed materials for a region Mask2Former called `ade_name`, or None
        when the label is unmapped (-> emissivity floor only)."""
        rec = self._by_ade.get(ade_name)
        return None if rec is None else rec.materials

    def group_of(self, ade_name: str) -> str | None:
        rec = self._by_ade.get(ade_name)
        return None if rec is None else rec.group

    def zone(self, ade_name: str) -> str:
        """The geometric zone written into segments.json for ../voxel_consensus.py.
        `any` for an unmapped label, which is what that script defaults to anyway."""
        rec = self._by_ade.get(ade_name)
        return "any" if rec is None else rec.zone


def restrict_to_candidates(ranked, allowed, eps_of=None, min_eps: float = 0.5):
    """Drop the materials this region's ADE class forbids, renormalising the rest.

    Same computation as ../emissivity/zones.py::restrict_ranking -- and the same
    justification: restricting a softmax to a subset and renormalising does not
    change the ordering, so this is exactly equivalent to having scored only the
    allowed classes. No extra CLIP forward pass, no extra cost. The only
    difference is where the candidate list comes from: the ADE class here, the
    bbox shape there. The parent module is left untouched.

    `allowed` is None for an ADE label with no row, in which case candidates
    below `min_eps` are dropped instead -- see the module docstring for why the
    unmapped case still needs a floor.

    Falls back to the unrestricted ranking if the filter would empty it, so an
    unusual emissivity table can never produce an empty result.
    """
    if allowed:
        keep = [(m, p) for m, p in ranked if m in allowed]
    elif eps_of is not None:
        keep = [(m, p) for m, p in ranked if eps_of.get(m, 1.0) >= min_eps]
    else:
        return ranked
    if not keep:
        return ranked
    total = sum(p for _m, p in keep) or 1.0
    return [(m, p / total) for m, p in keep]
