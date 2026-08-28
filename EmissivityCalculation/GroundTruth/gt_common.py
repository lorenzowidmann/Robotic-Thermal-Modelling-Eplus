"""Schema reading shared by annotate.py and evaluate.py.

One place that knows what a material_map/v1 run looks like on disk, so the two
scripts cannot drift apart in what they consider a valid run or a valid label.

The region-identity problem
---------------------------
A ground-truth row says "frame 20250906_233144_R, region 3 is glass". That
sentence is only true for a run whose `labels.npy` puts region 3 on the same
pixels. Region ids are positional, not semantic: they are handed out by
connected-component order, so a different segmenter -- or the same segmenter on
a different source image -- gives region 3 to a different object without any
error being raised anywhere.

Measured on frame 20250906_233144_R of session 20260730_161223, of the four
runs on disk only material_map_sam and material_map_consensus have identical
labels.npy (the consensus stage re-votes materials over SAM's segmentation and
leaves it alone). Every other pair differs.

So: annotation is per SEGMENTER, not per run. Prior-on/off pairs share a
segmentation because segmentation happens before and independently of the prior
(classify_session_m2f.py:405 vs :440; classify_session.py:303 vs :311), and
that is what makes one annotation pass cover both. `labels_md5` below is how
that is proven at runtime rather than assumed -- see assert_compatible.
"""

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
DEFAULT_FRAMES = _HERE / "eval_frames.txt"
DEFAULT_TABLE = _HERE.parent / "emissivity_table.csv"

# The two answers that are not materials. `mixed` means the region genuinely
# spans more than one material, so no single label is correct; `unclear` means
# it cannot be judged -- too small, too dark, too blurred. Both are recorded
# rather than guessed, and evaluate.py drops them from the metrics: a region
# with no right answer must not be able to score as a classifier error.
STATUSES = ("mixed", "unclear")
LABELED = "labeled"

# The keys every material_map/v1 run has, checked on all four run variants on
# disk. `zone`, `ade`, `ade_confidence`, `crop`, `prior_group`,
# `allowed_materials`, `solar_absorptance`, `prior` and `gated` exist only on
# newer runs, so nothing here may require them -- the oldest run
# (classify_session.py, no FLIR crop) has only these seven.
REQUIRED_SEGMENT_KEYS = ("id", "centroid_px", "area_px", "top_material",
                         "confidence", "emissivity", "top_k")

GT_COLUMNS = ("session", "run_used", "frame", "region_id", "area_px", "ade",
              "status", "ground_truth", "labels_md5", "annotated_utc")

# How many mismatching frames assert_compatible lists before summarising.
SHOW_N_BAD = 4


def labels_md5(arr: np.ndarray) -> str:
    """Identity of a segmentation, as stored in ground_truth.csv.

    Over the raw buffer plus the shape, because two arrays of different shape
    can share a buffer once flattened and they are not the same segmentation.
    """
    h = hashlib.md5()
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr, dtype=np.int32).tobytes())
    return h.hexdigest()


def load_eval_frames(path: Path = DEFAULT_FRAMES) -> list[str]:
    """FLIR frame stems, one per line; blank lines and #-comments skipped."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} does not exist. Create it once with:\n"
            f"    py annotate.py --init-frames --run <a material_map dir> --n 20")
    stems = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            stems.append(line)
    if not stems:
        raise ValueError(f"{path} lists no frames.")
    dupes = {s for s in stems if stems.count(s) > 1}
    if dupes:
        raise ValueError(f"{path} lists duplicate frames: {sorted(dupes)}")
    return stems


def load_zed_frames_dir(session_dir: Path) -> Path:
    """Where the ZED PNGs live.

    Same convention as classify_session_m2f.py:115 and sync_manifest.py --
    metadata.json's recording.frames_dir, relative to the session dir. Note the
    session dir has BOTH a `frames/` and (one level up) a smaller non-fullrate
    `frames/`; going through metadata.json is what keeps this pointing at the
    right one.
    """
    meta_path = Path(session_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json in {session_dir} -- expected a ZED session dir "
            "(the one holding frames/ and sync_manifest.json).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return Path(session_dir) / (meta.get("recording", {}).get("frames_dir") or "frames")


@dataclass
class RunFrame:
    """One frame of one classification run, as it exists on disk."""
    run_dir: Path
    frame: str
    doc: dict
    segments: list[dict]
    labels: np.ndarray
    md5: str

    @property
    def source_zed_frame(self) -> str | None:
        return self.doc.get("source_zed_frame")

    @property
    def crop_box(self):
        c = self.doc.get("flir_fov_crop")
        return None if not c else (c["x0"], c["y0"], c["x1"], c["y1"])

    @property
    def by_id(self) -> dict[int, dict]:
        return {int(s["id"]): s for s in self.segments}

    def mask(self, region_id: int) -> np.ndarray:
        return self.labels == int(region_id)


def load_run_frame(run_dir: Path, frame: str) -> RunFrame:
    """Read one <run_dir>/<frame>/ pair, validating only what is universal."""
    d = Path(run_dir) / frame
    seg_path, lab_path = d / "segments.json", d / "labels.npy"
    if not seg_path.exists() or not lab_path.exists():
        raise FileNotFoundError(
            f"{d} is missing segments.json and/or labels.npy -- "
            f"does this run cover frame {frame}?")

    doc = json.loads(seg_path.read_text(encoding="utf-8"))
    if doc.get("schema") != "material_map/v1":
        raise ValueError(f"{seg_path} has schema {doc.get('schema')!r}, "
                         "expected 'material_map/v1'.")
    segments = doc["segments"]
    for s in segments:
        missing = [k for k in REQUIRED_SEGMENT_KEYS if k not in s]
        if missing:
            raise ValueError(f"{seg_path} segment {s.get('id')} lacks {missing}.")

    labels = np.load(lab_path)
    # Cheap invariant, and it has held on all 107 m2f frames: the id set in
    # segments.json is exactly the non-negative value set in labels.npy. If it
    # ever stops holding, a region got labelled that is not on screen.
    ids = {int(s["id"]) for s in segments}
    present = {int(v) for v in np.unique(labels) if v >= 0}
    if ids != present:
        raise ValueError(
            f"{d}: segments.json ids and labels.npy disagree "
            f"(only in json: {sorted(ids - present)}, "
            f"only in npy: {sorted(present - ids)}).")

    return RunFrame(Path(run_dir), frame, doc, segments, labels, labels_md5(labels))


def describe_run_frame(rf: RunFrame) -> str:
    """The three things that differ when two runs are not comparable."""
    return (f"n_segments={len(rf.segments):<4} "
            f"crop={rf.crop_box} "
            f"source_zed_frame={rf.source_zed_frame}")


class IncompatibleRun(Exception):
    """A run's segmentation is not the one the ground truth was drawn on."""


def assert_compatible(run_dir: Path, frames: list[str],
                      expected_md5: dict[str, str],
                      reference_dir: Path | None = None) -> dict[str, RunFrame]:
    """Load `frames` from `run_dir`, refusing if the segmentation differs.

    `expected_md5` maps frame -> the labels_md5 recorded in ground_truth.csv at
    annotation time. A mismatch means the region ids in this run point at
    different pixels than the ones that were looked at, so every score computed
    from it would be wrong -- and wrong in a way that still produces a
    plausible-looking accuracy number. That is worse than no number, which is
    why this raises and there is no --force to get past it.

    The diagnostic prints n_segments, the crop box and source_zed_frame,
    because those are what actually differ in the failure this is built to
    catch: a run made before commit a664518 fixed the ZED intrinsics used a
    751x733 FLIR-FOV box where the corrected one is 979x952, and a run made
    before the sync manifest was regenerated classified a different ZED image
    for the same FLIR stem.
    """
    loaded, bad = {}, []
    for f in frames:
        rf = load_run_frame(run_dir, f)
        loaded[f] = rf
        if expected_md5.get(f) and rf.md5 != expected_md5[f]:
            bad.append(rf)
    if bad:
        lines = [
            f"{Path(run_dir).name} was segmented differently from the run the "
            f"ground truth was annotated on -- {len(bad)}/{len(frames)} frame(s) "
            "differ. Region ids therefore refer to different pixels and cannot "
            "be scored against these labels.",
            "",
        ]
        # A few examples, not all 20: the pattern is what diagnoses this, and a
        # screenful of identical-looking rows buries the closing advice.
        shown = bad[:SHOW_N_BAD]
        for rf in shown:
            lines.append(f"  {rf.frame}")
            lines.append(f"    this run : {describe_run_frame(rf)}")
            if reference_dir is not None:
                try:
                    ref = load_run_frame(reference_dir, rf.frame)
                    lines.append(f"    annotated: {describe_run_frame(ref)}")
                except (FileNotFoundError, ValueError) as exc:
                    lines.append(f"    annotated: <{Path(reference_dir).name}: {exc}>")
        if len(bad) > len(shown):
            lines.append(f"  ... and {len(bad) - len(shown)} more frame(s).")
        lines += [
            "",
            "Annotate a separate ground_truth file against this run, or "
            "regenerate it so its segmentation matches.",
        ]
        raise IncompatibleRun("\n".join(lines))
    return loaded


def read_ground_truth(path: Path) -> list[dict]:
    """ground_truth.csv -> rows, with region_id/area_px as ints."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist -- run annotate.py first.")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no rows.")
    missing = [c for c in GT_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}.")
    for r in rows:
        r["region_id"] = int(r["region_id"])
        r["area_px"] = int(r["area_px"]) if r["area_px"] else 0
    return rows


def write_ground_truth(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite: temp file then replace.

    Not an append, because annotate.py rewrites the whole file after every
    single label. A process killed mid-append leaves a truncated final line
    that csv silently reads as a short row; a killed replace leaves the
    previous complete file.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(GT_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in GT_COLUMNS})
    tmp.replace(path)


def load_material_table(table_path: str | Path | None = None) -> tuple[list[str], dict[str, float]]:
    """The material list and emissivity per material, read from the same CSV.

    Returns (materials in file order, {material: emissivity}).

    Read with the stdlib `csv` rather than reusing
    ../emissivity/table.py::EmissivityTable, deliberately. That class is fine,
    but importing it runs ../emissivity/__init__.py, which imports .classifier
    (torch, transformers) and .segmentation (scikit-image). Neither script here
    loads a model -- annotate.py shows you pixels, evaluate.py compares strings
    -- and making them drag in a ~4 GB dependency chain to read a 20-row CSV
    would tie them to C:\\venvs\\emissivity for no reason. Same call, same
    reasoning, as m2f_materials/category_prior.py, which also reads its table
    with the stdlib.

    The contract is only the `material` and `emissivity` columns, so an
    emissivity_table.csv edit that EmissivityTable accepts is accepted here too.
    """
    path = Path(table_path) if table_path else DEFAULT_TABLE
    if not path.exists():
        raise FileNotFoundError(f"No emissivity table at {path}.")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for col in ("material", "emissivity"):
        if not rows or col not in rows[0]:
            raise ValueError(f"{path} has no '{col}' column.")

    materials, eps = [], {}
    for r in rows:
        name = (r["material"] or "").strip()
        if not name:
            continue
        if name in eps:
            raise ValueError(f"{path} lists material {name!r} twice.")
        materials.append(name)
        eps[name] = float(r["emissivity"])

    # A material called `mixed` would make ground_truth.csv ambiguous between
    # "this surface is mixed" and "the annotator could not pick one", so this
    # is a load-time failure rather than a comment.
    clash = sorted(set(materials) & set(STATUSES))
    if clash:
        raise ValueError(
            f"{path} defines material(s) {clash}, which collide with the "
            f"annotation statuses {list(STATUSES)}. Rename the material.")
    return materials, eps
