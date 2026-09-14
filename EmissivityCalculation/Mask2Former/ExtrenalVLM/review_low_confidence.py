"""Second opinion from a VLM on the regions CLIP was least sure about.

A fallback stage that runs AFTER a finished ../classify_session_m2f.py run and
changes nothing about it. It reads the run's segments.json files, takes only the
regions whose CLIP `confidence` is below --min-confidence, asks a vision model to
confirm or override each one, and writes the whole run out as a CSV with the
reviewed rows updated. The run directory is never touched -- not segments.json,
not labels.npy -- so the M2F+CLIP output stays exactly what it was and a rerun
here is free.

Why a VLM at all, and only here
--------------------------------
CLIP scores a crop against 20 fixed prompts from ../../emissivity_table.csv. It
has no access to the rest of the frame, which is where the answer often is: a
worn metallic-looking shape is `steel_oxidized` on texture alone and
`painted_metal` the moment you can see it is a radiator under a window. The
category prior (ade_material_prior.csv) already fixes the cases the ADE class
settles; this fixes the cases where the ADE class is right but the texture is
ambiguous, and those are exactly the low-confidence ones. Above the threshold
nothing is asked, so a confident run costs nothing.

Four engines, one downstream contract
---------------------------------------
LOCAL (default): Ollama + moondream, on your own machine. Free, no API key, no
data leaves the machine -- see "Local engine" below for the (real) quality
trade-off.

--use-api: a cloud VLM instead, one sequential call per region. WHICH provider
runs is automatic: whichever of ANTHROPIC_API_KEY / GEMINI_API_KEY /
OPENAI_API_KEY is set in the environment picks it -- paste in a Gemini key and
this calls Gemini, an OpenAI key and it calls OpenAI, no other flag needed.
--api-provider forces one explicitly, for when more than one key happens to be
set. See "Cloud engines" below.

All four engines produce the exact same result shape --
{"material", "confidence", "reasoning"} or {"error": str} per region key -- so
render, merge, the low-emissivity gate and the CSV are ONE code path regardless
of which one ran. Adding a fifth provider means adding one make_client_X + one
run_X + one line in API_PROVIDERS; nothing else changes.

What the model is shown
------------------------
The FULL frame with the region outlined in red, which is the right-hand panel of
../../GroundTruth/annotate.py rendered through `annotate.render_context()`
itself -- same function, same red, same blend. The one difference is what is
handed to it: annotate.py passes the region mask, so the region is tint-FILLED;
here the mask passed is a dilated boundary band, so only the outline is tinted
and the surface the model has to judge keeps its true colour. Nothing in
GroundTruth/ is modified or copied.

Full frame rather than annotate.py's zoomed `render_region()` crop on purpose:
the scene context is the entire point of asking a VLM, and a padded crop is
where it gets thrown away. Each rendered image is also saved to
<out>_work/images/<key>.jpg -- every engine reads it from there (Ollama needs a
path; the three cloud engines re-read the same file to base64 it), and it
doubles as an audit trail of exactly what was shown.

Local engine: moondream via Ollama
------------------------------------
`ollama pull moondream` first. moondream is a 1B-parameter model with a 2048-
token context window -- fast and free, but genuinely weak at the nuanced
judgment this task sometimes needs. Measured while building this: given a
prompt example ("a radiator near a window is painted_metal"), it echoed that
example back as its "reasoning" for an unrelated wall region instead of
describing what was actually in the photo. The local prompt below is
deliberately short and avoids handing it a quotable phrase, but treat
`vlm_reasoning` on local runs as a hint, not a citation -- spot-check the
`overridden` rows in the output CSV before trusting them, especially anything
that flips to or from a bare metal. --use-api exists for when that matters.

Cloud engines (--use-api)
---------------------------
Vision + a JSON-schema-constrained response in one call per region, no batch
job (batch adds up to 24 h turnaround for a handful of images, which defeats
the point of --use-api). Three providers, auto-selected by whichever key is
set:

    ANTHROPIC_API_KEY  -> Claude API      (claude-opus-5)
    GEMINI_API_KEY     -> Gemini API      (gemini-flash-latest)
    OPENAI_API_KEY     -> OpenAI API      (gpt-5.6-luna)

--api-provider {claude,gemini,openai} forces one when more than one key is
set. --model overrides that provider's default. None of the three keys is
ever read from a file or an argument -- environment only.

The override is still gated
-----------------------------
Either engine may answer with any of the 20 materials in
../../emissivity_table.csv -- the same list ../../GroundTruth/annotate.py binds
its keys to. Seven of those are BARE metals, and this stage lands after
../classify_session_m2f.py's low-emissivity gate has already run, so an
unchecked override here would reintroduce exactly the failure that gate and the
ADE prior exist to prevent: the radiometric correction divides by emissivity,
so a wrong `aluminum_polished` (e=0.05) turns a 37 degC reading into ~156 degC,
while confusing any two ordinary indoor materials costs under 1 K.

So the gate's own two parameters are reapplied to the model's answer, with the
same defaults as ../classify_session_m2f.py: an override to a material below
--low-emissivity-max is accepted only if the model's own confidence reaches
--low-emissivity-min-conf. A refused override keeps the pipeline's label and is
recorded as `rejected_low_emissivity` in vlm_status -- never silently dropped.

Merge rule
-----------
    agree     -> keep the label, raise confidence to max(clip, model)
    disagree  -> take the model's label AND its confidence
    error, or the region was never flagged -> row passes through untouched

Every row keeps the pipeline's original answer in `pipeline_material` /
`pipeline_confidence` alongside the final one, so the stage is always reversible
from its own output and `vlm_status` explains every row that moved.

stdlib csv, not pandas
------------------------
Same call, and the same reason, as ../../GroundTruth/gt_common.py and
../m2f_materials/category_prior.py: nothing here loads a segmentation or CLIP
model, and reading a 20-row table must not drag torch/transformers/pandas in
behind it.

Usage:
    # local, default -- needs `ollama pull moondream` and the Ollama app running
    py review_low_confidence.py --run ...\\fullrate\\material_map_m2f ^
                                --session-dir ...\\ZED\\20260730_161223\\fullrate

    # what would be rendered and sent, without calling any model
    py review_low_confidence.py --run ... --session-dir ... --dry-run

    # a cloud engine instead -- whichever key you set picks the provider
    # (ANTHROPIC_API_KEY -> Claude, GEMINI_API_KEY -> Gemini, OPENAI_API_KEY -> OpenAI)
    set ANTHROPIC_API_KEY=...
    py review_low_confidence.py --run ... --session-dir ... --use-api

    # more than one key set? name the provider explicitly
    py review_low_confidence.py --run ... --session-dir ... ^
                                --use-api --api-provider gemini

    # only the pinned eval frames, so the result is scoreable by
    # ../../GroundTruth/evaluate.py against ground_truth_m2f.csv
    py review_low_confidence.py --run ... --session-dir ... ^
                                --frames ..\\..\\GroundTruth\\eval_frames.txt
"""

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

# --- the annotation tool's renderer -----------------------------------------
# ../../GroundTruth/ is imported, not copied: render_context() below IS
# annotate.py's right-hand panel, so the image the model sees and the image the
# human annotator saw cannot drift apart. Both modules are import-safe -- they
# pull in numpy and the stdlib at module scope, and tkinter/PIL only inside the
# functions that need them -- so this costs nothing at import time.
_HERE = Path(__file__).resolve().parent
_GROUND_TRUTH = _HERE.parent.parent / "GroundTruth"
if not (_GROUND_TRUTH / "annotate.py").exists():
    raise SystemExit(f"Expected the annotation tool at {_GROUND_TRUTH} -- this module "
                     "reuses its renderer and its material-table reader.")
sys.path.insert(0, str(_GROUND_TRUTH))

import annotate                                                   # noqa: E402
import gt_common as gt                                            # noqa: E402

DEFAULT_MODEL_LOCAL = "moondream"

# One entry per cloud provider --use-api can drive. `env` is the key this
# module reads (never a file, never an argument -- see make_client_*).
# `default_model` is only used when --model is not passed.
#
# Auto-detection order (see detect_api_provider): the FIRST of these whose
# env var is set wins. If more than one is set, --api-provider picks one
# explicitly rather than silently guessing which key you meant to use.
API_PROVIDERS = {
    "claude": {
        "env": "ANTHROPIC_API_KEY",
        "default_model": "claude-opus-5",   # mandated default for this model family
        "signup": "https://platform.claude.com (new accounts: ~$5 free credit, no card)",
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        # A rolling alias, not a dated snapshot -- gemini-2.5-flash (this
        # module's original default) was retired for new-project keys
        # partway through building this, with no warning beyond a 404 at
        # request time. -latest is Google's own answer to that problem.
        "default_model": "gemini-flash-latest",
        "signup": "https://aistudio.google.com/apikey",
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        # Cheapest current vision-capable tier ($0.20/$1.20 per MTok in/out),
        # not the flagship -- this is a short classification call, not a
        # reasoning task, the same reasoning behind Claude's --effort low.
        "default_model": "gpt-5.6-luna",
        "signup": "https://platform.openai.com/api-keys",
    },
}
API_PROVIDER_ORDER = ("claude", "gemini", "openai")   # detection priority

# vlm_status values. Everything except "" and "confirmed"/"overridden" means the
# row kept the pipeline's answer, and says why.
ST_NOT_REVIEWED = ""
ST_CONFIRMED = "confirmed"
ST_OVERRIDDEN = "overridden"
ST_REJECTED_LOW_E = "rejected_low_emissivity"
ST_ERROR = "error"
ST_MISSING = "missing_from_results"

CSV_COLUMNS = (
    # Identity -- the first five deliberately match ../../GroundTruth/
    # ground_truth_m2f.csv, so this file joins to it on (frame, region_id)
    # without a rename.
    "session", "run_used", "frame", "region_id", "area_px", "ade", "labels_md5",
    # What ../classify_session_m2f.py decided, preserved whatever happens below.
    "pipeline_material", "pipeline_confidence",
    # The answer after this stage. `material`/`confidence` are the columns a
    # downstream consumer should read.
    "material", "confidence", "emissivity", "solar_absorptance",
    # The review itself.
    "vlm_reviewed", "vlm_status", "vlm_material", "vlm_confidence", "vlm_reasoning",
    "reviewed_utc",
)

# Full prompt: used with --use-api, where a strong model and a 200K+ token
# context window can use a concrete example without just repeating it.
PROMPT_API = """\
This is a full frame from an indoor building thermal survey. Exactly one \
segmented region is outlined in red.

Identify the MATERIAL OF THE OUTLINED REGION ONLY. Judge it from the whole \
scene, not from the texture inside the outline alone: an outlined shape mounted \
low on a wall under a window, with horizontal fins, is a radiator, and a \
radiator in a building like this is painted_metal even when its surface looks \
worn, grey and metallic.

The bare-metal labels (aluminum_polished, aluminum_oxidized, steel_polished, \
steel_oxidized, iron_rusted, copper_polished, copper_oxidized) mean UNPAINTED, \
exposed metal. A painted radiator, a painted pipe or a painted panel is \
painted_metal, not one of those. Only choose a bare metal if you can see raw \
unpainted metal.

The segmenter classified this region as "{ade}".
The current pipeline guess is "{material}", with confidence {confidence:.2f}.

Confirm that guess or override it. Give your own confidence in [0, 1] and one \
sentence of reasoning."""

# Short prompt: used locally. moondream's context window is 2048 tokens total
# (prompt + image + answer), and it is small enough to echo back a vivid
# example verbatim as its "reasoning" instead of describing the photo --
# measured while building this (see the module docstring). So: no quotable
# example sentence here, and an explicit instruction to describe only what is
# actually visible.
PROMPT_LOCAL = """\
One region of this photo is outlined in red. Look at the whole photo, not just \
the outline, to judge what the object really is.

What material is the outlined region? Bare-metal answers mean unpainted, \
exposed metal -- if the outlined object is more likely something ordinarily \
painted (a radiator, a pipe, a panel), prefer painted_metal instead, even if \
the surface looks worn or grey.

Segmenter's guess: "{ade}". Pipeline's material guess: "{material}" \
(confidence {confidence:.2f}). Confirm or correct it.

material: one word from the allowed list. confidence: your own, 0 to 1. \
reasoning: one short sentence describing what you actually see, not an example."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_solar_absorptance(table_path=None) -> dict[str, float]:
    """material -> solar_absorptance, with the stdlib.

    gt.load_material_table returns the validated material list and emissivity
    but not this column, and it is needed whenever an override changes the
    material. Read the same way ../../voxel_consensus.py:77 reads it, for the
    same reason -- see this module's docstring.
    """
    path = Path(table_path) if table_path else gt.DEFAULT_TABLE
    with path.open(newline="", encoding="utf-8") as fh:
        return {r["material"]: float(r["solar_absorptance"])
                for r in csv.DictReader(fh) if (r["material"] or "").strip()}


def build_review_model(materials: list[str]) -> type[BaseModel]:
    """The structured-output schema, built once from the live material list --
    never hardcoded, so an emissivity_table.csv edit can't leave it stale (the
    same rule ../m2f_materials/category_prior.py is validated against).

    One Pydantic model feeds BOTH engines: Ollama's `format=` wants a JSON
    Schema dict (`.model_json_schema()` gives it one, with the material field
    rendered as a proper `enum`), and the Claude API's `output_config.format`
    wants the same shape. Building it once is what keeps the two engines from
    drifting apart on what a valid answer looks like.
    """
    MaterialName = Literal[tuple(materials)]          # type: ignore[valid-type]

    class MaterialReview(BaseModel):
        material: MaterialName
        confidence: float
        reasoning: str

    return MaterialReview


# --- the image ---------------------------------------------------------------

def outline_band(mask: np.ndarray, width_px: int) -> np.ndarray:
    """The region's boundary as a `width_px`-thick band, as a bool mask.

    The first three lines are ../../GroundTruth/annotate.py:155-159's outline
    idiom -- mark every pixel whose right or lower neighbour is on the other
    side of the mask. The dilation is the part annotate.py does not need: it
    draws into a zoomed crop shown at screen size, where one pixel is visible,
    while this frame is downscaled to --image-max-px and JPEG-encoded before
    anything sees it, and a 1 px line does not survive either.

    Dilated into a fresh array each pass rather than OR-ing shifted views of the
    band into itself, which aliases and would smear the band across the frame in
    one direction.

    Each pass grows the band on BOTH sides, so it takes (width_px - 1) // 2
    passes to reach a `width_px`-thick line, not width_px - 1 -- which is what
    the parameter has to mean for --outline-width to be readable as pixels.
    """
    m = np.asarray(mask, dtype=bool)
    band = np.zeros_like(m)
    band[:-1, :] |= m[:-1, :] ^ m[1:, :]
    band[:, :-1] |= m[:, :-1] ^ m[:, 1:]
    for _ in range(max(0, (width_px - 1) // 2)):
        nxt = band.copy()
        nxt[1:, :] |= band[:-1, :]
        nxt[:-1, :] |= band[1:, :]
        nxt[:, 1:] |= band[:, :-1]
        nxt[:, :-1] |= band[:, 1:]
        band = nxt
    return band


def render_outlined(image: np.ndarray, mask: np.ndarray, width_px: int) -> np.ndarray:
    """Full frame, region outlined in red -- annotate.py's own renderer.

    annotate.render_context tints whatever mask it is given. Handing it the
    boundary band instead of the region turns its filled tint into an outline
    and leaves the region's real colour intact, which matters here in a way it
    does not for a human: the model is being asked what colour and finish that
    surface has, and a 55% red blend over it would be answering for it.
    """
    return annotate.render_context(image, outline_band(mask, width_px))


def encode_jpeg(arr: np.ndarray, max_px: int, quality: int) -> bytes:
    """Downscale to `max_px` on the long side and JPEG-encode.

    JPEG, not PNG: these are photographs, the frame is 1920x1080. PNG runs ~8x
    larger for no gain either engine can use, and every extra pixel is either
    tokens (Claude) or context budget moondream does not have to spare (2048
    tokens total).
    """
    from PIL import Image

    img = Image.fromarray(arr)
    w, h = img.size
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# --- the run -----------------------------------------------------------------

def region_key(frame: str, region_id: int) -> str:
    """Identity used for the image filename and the results dict. Parsed back
    by split_key, so the separator must not occur in a FLIR frame stem (they
    are digits, underscores and a side letter, e.g. 20250906_233144_R)."""
    return f"{frame}-r{region_id}"


def split_key(key: str) -> tuple[str, int]:
    frame, rid = key.rsplit("-r", 1)
    return frame, int(rid)


def collect_rows(run_dir: Path, frames: list[str], session: str) -> list[dict]:
    """Every region of every frame, as CSV rows, pipeline answer intact.

    Loaded through gt.load_run_frame so this stage validates the run exactly the
    way the scorer does -- schema, required keys, and the segments.json/
    labels.npy id agreement -- rather than trusting the JSON.
    """
    rows, run_frames = [], {}
    for frame in frames:
        rf = gt.load_run_frame(run_dir, frame)
        run_frames[frame] = rf
        for seg in sorted(rf.segments, key=lambda s: int(s["id"])):
            rows.append({
                "session": session,
                "run_used": run_dir.name,
                "frame": frame,
                "region_id": int(seg["id"]),
                "area_px": int(seg["area_px"]),
                "ade": seg.get("ade", ""),
                "labels_md5": rf.md5,
                "pipeline_material": seg["top_material"],
                "pipeline_confidence": float(seg["confidence"]),
                "material": seg["top_material"],
                "confidence": float(seg["confidence"]),
                "emissivity": float(seg["emissivity"]),
                "solar_absorptance": seg.get("solar_absorptance", ""),
                "vlm_reviewed": False,
                "vlm_status": ST_NOT_REVIEWED,
                "vlm_material": "",
                "vlm_confidence": "",
                "vlm_reasoning": "",
                "reviewed_utc": "",
            })
    return rows, run_frames


def render_review_items(flagged: list[dict], run_frames: dict, frames_dir: Path,
                        images_dir: Path, args) -> list[dict]:
    """Render + save one outlined JPEG per flagged region.

    Written to disk rather than kept as in-memory bytes because the local
    engine's `images=` parameter wants a file path (that is the documented
    Ollama Python client contract, not a convenience choice); the cloud engine
    reads the same file back and base64s it. One rendering, one file, both
    engines and the audit trail read from it.

    ZED frames are 6 MB each and are loaded one at a time, in frame order, the
    same way ../../GroundTruth/annotate.py:232 does it -- a 20-frame session
    held in memory at once is 120 MB for no reason.
    """
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    items = []
    image, current = None, None
    for row in sorted(flagged, key=lambda r: (r["frame"], r["region_id"])):
        frame = row["frame"]
        if frame != current:
            rf = run_frames[frame]
            name = rf.source_zed_frame
            if not name:
                raise SystemExit(f"{frame}: segments.json has no source_zed_frame.")
            path = frames_dir / name
            if not path.exists():
                raise SystemExit(f"{frame}: ZED image {path} does not exist.")
            image = np.asarray(Image.open(path).convert("RGB"))
            current = frame

        mask = run_frames[frame].mask(row["region_id"])
        # Auto: thick enough at source resolution that it lands ~3 px wide once
        # encode_jpeg has downscaled to --image-max-px. 1920 -> 1024 gives 6.
        width = args.outline_width or max(3, round(3 * max(image.shape[:2]) / args.image_max_px))
        jpeg = encode_jpeg(render_outlined(image, mask, width),
                           args.image_max_px, args.jpeg_quality)
        key = region_key(frame, row["region_id"])
        img_path = images_dir / f"{key}.jpg"
        if not img_path.exists():          # cheap skip on a resumed run
            img_path.write_bytes(jpeg)
        items.append({
            "key": key,
            "ade": row["ade"] or "unknown",
            "pipeline_material": row["pipeline_material"],
            "pipeline_confidence": row["pipeline_confidence"],
            "image_path": img_path,
        })
    return items


# --- local engine: Ollama + moondream -----------------------------------------

def check_ollama(model: str) -> None:
    """One clear failure before the loop, instead of N identical ones inside it.

    ollama.chat raises ollama.ResponseError with a status_code, but a server
    that is not running at all raises a plain connection error from the
    underlying httpx client -- caught here as the generic case so both surface
    the same two fixes (start Ollama; pull the model) instead of a traceback.
    """
    import ollama

    try:
        names = {m.model for m in ollama.list().models}
    except Exception as exc:
        raise SystemExit(
            f"Can't reach the Ollama server ({type(exc).__name__}: {exc}).\n"
            f"Start it (the Ollama app, or `ollama serve`) and rerun.")
    if model not in names and f"{model}:latest" not in names:
        raise SystemExit(
            f"Model {model!r} is not pulled. Visible models: {sorted(names) or '(none)'}\n"
            f"Pull it with:\n    ollama pull {model}")


def load_partial_results(results_path: Path) -> dict[str, dict]:
    if results_path.exists():
        try:
            return json.loads(results_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # a partial write from a hard crash; start over rather than fail the run
    return {}


def still_todo(items: list[dict], out: dict[str, dict]) -> list[dict]:
    """Which items a resume still has to answer.

    An item is done only if `out` holds a CLEAN answer for its key -- one
    without "error". An errored entry is retried automatically on every
    rerun, with no --redo needed: "log and skip" (the module's own promise)
    means the run keeps going past a bad item, not that the item is given up
    on forever the next time this --out is reused.

    This also self-heals a real incident: results.json is keyed only by
    `<frame>-r<region_id>`, with no record of which engine or script version
    produced an entry. A run of the pre-rewrite Gemini version of this module
    against the same --out left 28 entries recording
    `ClientError: 429 RESOURCE_EXHAUSTED ... prepayment credits are
    depleted` -- a completely different engine's billing failure, nothing to
    do with the local run reusing that path. Skipping only clean successes
    means those 28 get retried by the current engine without anyone having
    to notice, diagnose, or pass --redo.
    """
    return [it for it in items if it["key"] not in out or "error" in out[it["key"]]]


def save_results(results_path: Path, results: dict) -> None:
    """Atomic rewrite after every item, not just at the end.

    Motivated by a real incident, not a hypothetical: a 151-region local run
    was OOM-killed by Windows after ~30+ minutes, and because results were
    only ever written once at the end, EVERY completed call was lost --
    nothing to resume from, nothing to merge, the whole run wasted. Writing
    after each item costs one small JSON dump (results stay a few KB even at
    hundreds of regions) and means a kill -- OOM, Ctrl+C, closing the terminal
    -- loses at most the one call in flight.
    """
    tmp = results_path.with_suffix(results_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(results_path)


def run_local(items: list[dict], model_cls: type[BaseModel], args,
             results_path: Path) -> dict[str, dict]:
    """One ollama.chat call per region. Sequential, on purpose: this is your
    own machine, there is no quota to batch around and no per-call cost, so the
    only thing batching would buy here is complexity.

    Resumes from `results_path` if it already holds answers for some of
    `items` (from a run that was interrupted) -- see save_results.

    Returns {key: {"material", "confidence", "reasoning"}} or {key: {"error"}},
    the same contract every run_<provider> in RUN_API produces -- merge() does
    not know or care which engine ran.
    """
    import ollama

    check_ollama(args.model)
    schema = model_cls.model_json_schema()
    out = {} if args.redo else load_partial_results(results_path)
    todo = still_todo(items, out)
    n_done, n = len(items) - len(todo), len(items)
    if n_done:
        print(f"Resuming: {n_done}/{n} already answered in {results_path.name}, "
              f"{len(todo)} left (--redo to ignore and start over)")
    print(f"Local engine: {len(todo)} call(s) to {args.model} via Ollama "
          f"(no key, no cost, no cap)")

    for i, item in enumerate(todo, 1):
        key = item["key"]
        text = PROMPT_LOCAL.format(ade=item["ade"], material=item["pipeline_material"],
                                   confidence=item["pipeline_confidence"])
        t0 = time.time()
        try:
            response = ollama.chat(
                model=args.model,
                messages=[{"role": "user", "content": text,
                          "images": [str(item["image_path"])]}],
                format=schema,
                options={"temperature": args.temperature, "num_predict": 200},
                keep_alive=args.keep_alive,
            )
            parsed = model_cls.model_validate_json(response.message.content)
            out[key] = {"material": parsed.material, "confidence": float(parsed.confidence),
                       "reasoning": parsed.reasoning.replace("\n", " ").strip()}
        except Exception as exc:
            # A malformed/truncated JSON from a 1B model is exactly as
            # survivable as a cloud error -- log and keep the pipeline's answer.
            out[key] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            print(f"  [{i}/{len(todo)}] {key}: FAILED, {type(exc).__name__}: {str(exc)[:160]}",
                  file=sys.stderr)
        save_results(results_path, out)
        dt = time.time() - t0
        n_err = sum(1 for v in out.values() if "error" in v)
        print(f"  [{i}/{len(todo)}] {key}: {dt:.1f}s ({n_err} failed so far)")
    return out


# --- cloud engines (--use-api) ------------------------------------------------
#
# Three providers, one contract: each run_<provider> takes the same
# (items, model_cls, args, results_path) and returns the same
# {key: {"material", "confidence", "reasoning"}} or {key: {"error"}} dict that
# run_local does. merge() reads that dict and has no idea which provider, or
# even whether local or cloud, produced it. Adding a fourth provider later
# means adding one make_client_X + one run_X to this section and one line to
# API_PROVIDERS -- nothing downstream changes.

def detect_api_provider(explicit: str | None) -> str:
    """Which cloud provider --use-api should drive.

    --api-provider wins outright. Otherwise the first of API_PROVIDER_ORDER
    whose env var is actually set wins -- not an arbitrary "first configured"
    guess, but also not a silent one: the caller always prints which key was
    found and why, so an unexpected provider is never a surprise.
    """
    if explicit:
        return explicit
    for name in API_PROVIDER_ORDER:
        if os.environ.get(API_PROVIDERS[name]["env"]):
            return name
    lines = [f"  {p['env']:<20} -> --api-provider {name}"
             for name, p in API_PROVIDERS.items()]
    raise SystemExit(
        "--use-api needs one API key in the environment. None of these are set:\n"
        + "\n".join(lines) +
        "\n\nSet exactly one (or pass --api-provider to pick when more than one "
        "is set) and rerun. Get a key at:\n"
        + "\n".join(f"  {name}: {p['signup']}" for name, p in API_PROVIDERS.items()))


def _missing_key_error(provider: str) -> str:
    p = API_PROVIDERS[provider]
    return (f"{p['env']} is not set. Get a key at:\n    {p['signup']}\n"
           f"then set it and rerun -- it is never read from a file or an argument.")


def _missing_package_error(package: str, exc: Exception) -> str:
    return (f"{package} is not importable in {sys.executable}\n"
           f"  ({type(exc).__name__}: {exc})\n"
           f"Install it into THIS interpreter with:\n"
           f'  "{sys.executable}" -m pip install {package}\n'
           f"or rerun with the interpreter it is already installed in.")


def make_client_claude():
    """anthropic.Anthropic() from the environment, failing separately on the
    two things that actually go wrong: no key, and the package installed into
    a different interpreter than the one running this."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit(_missing_key_error("claude"))
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit(_missing_package_error("anthropic", exc))
    return anthropic.Anthropic()


def make_client_gemini():
    """genai.Client() from the environment -- same two-way failure split."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(_missing_key_error("gemini"))
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(_missing_package_error("google-genai", exc))
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def make_client_openai():
    """openai.OpenAI() from the environment -- same two-way failure split."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(_missing_key_error("openai"))
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(_missing_package_error("openai", exc))
    return OpenAI()


def run_claude(items: list[dict], model_cls: type[BaseModel], args,
              results_path: Path) -> dict[str, dict]:
    """One Claude Messages API call per region, vision + structured output.

    Sequential rather than the Message Batches API: batching trades up to 24 h
    turnaround for a 50% discount, which is the wrong trade for a handful of
    images that cost cents either way. If a future run reviews thousands of
    regions in one pass, that trade flips -- worth revisiting then.
    """
    import anthropic

    client = make_client_claude()
    schema = model_cls.model_json_schema()
    out = {} if args.redo else load_partial_results(results_path)
    todo = still_todo(items, out)
    n_done, n = len(items) - len(todo), len(items)
    if n_done:
        print(f"Resuming: {n_done}/{n} already answered in {results_path.name}, "
              f"{len(todo)} left (--redo to ignore and start over)")
    print(f"Cloud engine: {len(todo)} call(s) to {args.model} via the Claude API")

    for i, item in enumerate(todo, 1):
        key = item["key"]
        text = PROMPT_API.format(ade=item["ade"], material=item["pipeline_material"],
                                 confidence=item["pipeline_confidence"])
        jpeg = item["image_path"].read_bytes()
        for attempt in range(args.max_retries + 1):
            try:
                response = client.messages.create(
                    model=args.model,
                    max_tokens=1024,
                    output_config={
                        "effort": "low",  # a short classification, not a reasoning task
                        "format": {"type": "json_schema", "schema": schema},
                    },
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {
                                "type": "base64", "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(jpeg).decode("ascii")}},
                            {"type": "text", "text": text},
                        ],
                    }],
                )
                body = next(b.text for b in response.content if b.type == "text")
                parsed = model_cls.model_validate_json(body)
                out[key] = {"material": parsed.material, "confidence": float(parsed.confidence),
                           "reasoning": parsed.reasoning.replace("\n", " ").strip()}
                break
            except anthropic.RateLimitError as exc:
                transient, label, detail = True, "RateLimitError", str(exc)
            except anthropic.APIStatusError as exc:
                transient = exc.status_code >= 500
                label, detail = f"APIStatusError {exc.status_code}", str(exc)
            except anthropic.APIConnectionError as exc:
                transient, label, detail = True, "APIConnectionError", str(exc)
            except Exception as exc:
                # BadRequestError, NotFoundError, AuthenticationError,
                # PermissionDeniedError, and a schema/JSON mismatch: none of
                # these get better on retry.
                transient, label, detail = False, type(exc).__name__, str(exc)

            if transient and attempt < args.max_retries:
                wait = args.retry_base_seconds * (2 ** attempt)
                print(f"  [{i}/{len(todo)}] {key}: {label}, retry in {wait:.0f}s\n"
                      f"        {detail[:300]}")
                time.sleep(wait)
                continue
            out[key] = {"error": f"{label}: {detail[:300]}"}
            print(f"  [{i}/{len(todo)}] {key}: FAILED, {label}: {detail[:200]}", file=sys.stderr)
            break

        save_results(results_path, out)
        n_err = sum(1 for v in out.values() if "error" in v)
        print(f"  [{i}/{len(todo)}] {key}: done ({n_err} failed so far)")
        if i < len(todo):
            time.sleep(args.request_interval)
    return out


def run_gemini(items: list[dict], model_cls: type[BaseModel], args,
              results_path: Path) -> dict[str, dict]:
    """One Gemini generateContent call per region. Sequential, matching the
    other two cloud engines -- see run_claude's docstring for why not batch.

    Request/response shape verified against the current Batch Mode docs
    (https://ai.google.dev/gemini-api/docs/batch-api) while building this
    module's first version: a batch line's "request" field IS a
    GenerateContentRequest, so the same generation_config/inline_data shape
    applies here unchanged.
    """
    from google.genai import types  # noqa: F401  (import surfaces a clear error early)

    client = make_client_gemini()
    schema = model_cls.model_json_schema()
    out = {} if args.redo else load_partial_results(results_path)
    todo = still_todo(items, out)
    n_done, n = len(items) - len(todo), len(items)
    if n_done:
        print(f"Resuming: {n_done}/{n} already answered in {results_path.name}, "
              f"{len(todo)} left (--redo to ignore and start over)")
    print(f"Cloud engine: {len(todo)} call(s) to {args.model} via the Gemini API")

    for i, item in enumerate(todo, 1):
        key = item["key"]
        text = PROMPT_API.format(ade=item["ade"], material=item["pipeline_material"],
                                 confidence=item["pipeline_confidence"])
        jpeg = item["image_path"].read_bytes()
        for attempt in range(args.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=[{
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg",
                                             "data": base64.standard_b64encode(jpeg).decode("ascii")}},
                            {"text": text},
                        ],
                    }],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                        "temperature": args.temperature,
                    },
                )
                if not response.text:
                    cands = getattr(response, "candidates", None) or []
                    fr = getattr(cands[0], "finish_reason", None) if cands else None
                    raise ValueError(f"empty response (finish_reason={fr})")
                parsed = model_cls.model_validate_json(response.text)
                out[key] = {"material": parsed.material, "confidence": float(parsed.confidence),
                           "reasoning": parsed.reasoning.replace("\n", " ").strip()}
                break
            except Exception as exc:
                # Off the exception's own status where the SDK provides one
                # (google.genai.errors.ClientError has .code/.status), not a
                # substring search of the message -- an id or a quota value
                # containing "429" would otherwise be misread as a status.
                # Verified against a real 429 RESOURCE_EXHAUSTED while
                # building this module: .code == 429, .status ==
                # "RESOURCE_EXHAUSTED" on the actual exception.
                code = getattr(exc, "code", None)
                status = str(getattr(exc, "status", "") or "")
                transient = code in (429, 500, 503) or status in (
                    "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL")
                label = type(exc).__name__ + (f" {code}" if code else "") + \
                       (f" {status}" if status else "")
                detail = str(exc)

                if transient and attempt < args.max_retries:
                    wait = args.retry_base_seconds * (2 ** attempt)
                    print(f"  [{i}/{len(todo)}] {key}: {label}, retry in {wait:.0f}s\n"
                          f"        {detail[:300]}")
                    time.sleep(wait)
                    continue
                out[key] = {"error": f"{label}: {detail[:300]}"}
                print(f"  [{i}/{len(todo)}] {key}: FAILED, {label}: {detail[:200]}",
                      file=sys.stderr)
                break

        save_results(results_path, out)
        n_err = sum(1 for v in out.values() if "error" in v)
        print(f"  [{i}/{len(todo)}] {key}: done ({n_err} failed so far)")
        if i < len(todo):
            time.sleep(args.request_interval)
    return out


def run_openai(items: list[dict], model_cls: type[BaseModel], args,
              results_path: Path) -> dict[str, dict]:
    """One Responses API call per region, vision + a Pydantic-typed parse.

    client.responses.parse(text_format=model_cls) is the current (2026)
    structured-output entry point -- verified against
    https://developers.openai.com/api/docs/guides/structured-outputs while
    building this. It returns the schema-validated object directly on
    response.output_parsed, so unlike the other two engines there is no
    manual json.loads/model_validate_json step here.
    """
    import openai

    client = make_client_openai()
    out = {} if args.redo else load_partial_results(results_path)
    todo = still_todo(items, out)
    n_done, n = len(items) - len(todo), len(items)
    if n_done:
        print(f"Resuming: {n_done}/{n} already answered in {results_path.name}, "
              f"{len(todo)} left (--redo to ignore and start over)")
    print(f"Cloud engine: {len(todo)} call(s) to {args.model} via the OpenAI API")

    for i, item in enumerate(todo, 1):
        key = item["key"]
        text = PROMPT_API.format(ade=item["ade"], material=item["pipeline_material"],
                                 confidence=item["pipeline_confidence"])
        jpeg = item["image_path"].read_bytes()
        b64 = base64.standard_b64encode(jpeg).decode("ascii")
        for attempt in range(args.max_retries + 1):
            try:
                response = client.responses.parse(
                    model=args.model,
                    input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": text},
                            {"type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}"},
                        ],
                    }],
                    text_format=model_cls,
                )
                parsed = response.output_parsed
                out[key] = {"material": parsed.material, "confidence": float(parsed.confidence),
                           "reasoning": parsed.reasoning.replace("\n", " ").strip()}
                break
            except openai.RateLimitError as exc:
                transient, label, detail = True, "RateLimitError", str(exc)
            except openai.APIStatusError as exc:
                transient = exc.status_code >= 500
                label, detail = f"APIStatusError {exc.status_code}", str(exc)
            except openai.APIConnectionError as exc:
                transient, label, detail = True, "APIConnectionError", str(exc)
            except Exception as exc:
                # BadRequestError, AuthenticationError, PermissionDeniedError,
                # NotFoundError, and a schema/refusal mismatch: none of these
                # get better on retry.
                transient, label, detail = False, type(exc).__name__, str(exc)

            if transient and attempt < args.max_retries:
                wait = args.retry_base_seconds * (2 ** attempt)
                print(f"  [{i}/{len(todo)}] {key}: {label}, retry in {wait:.0f}s\n"
                      f"        {detail[:300]}")
                time.sleep(wait)
                continue
            out[key] = {"error": f"{label}: {detail[:300]}"}
            print(f"  [{i}/{len(todo)}] {key}: FAILED, {label}: {detail[:200]}", file=sys.stderr)
            break

        save_results(results_path, out)
        n_err = sum(1 for v in out.values() if "error" in v)
        print(f"  [{i}/{len(todo)}] {key}: done ({n_err} failed so far)")
        if i < len(todo):
            time.sleep(args.request_interval)
    return out


RUN_API = {"claude": run_claude, "gemini": run_gemini, "openai": run_openai}


# --- merge -------------------------------------------------------------------

def merge(rows: list[dict], flagged_keys: set[str], results: dict[str, dict],
          eps: dict[str, float], alpha: dict[str, float], materials: set[str],
          args) -> dict[str, int]:
    """Apply the review to the flagged rows, in place. Returns a status tally."""
    tally = {}
    by_key = {region_key(r["frame"], r["region_id"]): r for r in rows}
    now = utc_now_iso()

    for key in sorted(flagged_keys):
        row = by_key[key]
        row["vlm_reviewed"] = True
        row["reviewed_utc"] = now
        res = results.get(key)

        if res is None:
            row["vlm_status"] = ST_MISSING
        elif "error" in res:
            row["vlm_status"] = ST_ERROR
            row["vlm_reasoning"] = res["error"]
        elif res["material"] not in materials:
            # The schema enum should make this impossible; if it happens the
            # run must not silently write a material with no emissivity.
            row["vlm_status"] = ST_ERROR
            row["vlm_reasoning"] = f"material {res['material']!r} is not in the table"
        else:
            g_mat = res["material"]
            g_conf = min(1.0, max(0.0, res["confidence"]))
            row["vlm_material"] = g_mat
            row["vlm_confidence"] = round(g_conf, 4)
            row["vlm_reasoning"] = res["reasoning"]

            if g_mat == row["pipeline_material"]:
                row["vlm_status"] = ST_CONFIRMED
                row["confidence"] = round(max(row["pipeline_confidence"], g_conf), 4)
            elif eps[g_mat] < args.low_emissivity_max and g_conf < args.low_emissivity_min_conf:
                # ../classify_session_m2f.py's gate, reapplied to the override.
                row["vlm_status"] = ST_REJECTED_LOW_E
            else:
                row["vlm_status"] = ST_OVERRIDDEN
                row["material"] = g_mat
                row["confidence"] = round(g_conf, 4)
                row["emissivity"] = eps[g_mat]
                row["solar_absorptance"] = alpha.get(g_mat, "")

        tally[row["vlm_status"]] = tally.get(row["vlm_status"], 0) + 1
    return tally


def write_csv(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite, same as ../../GroundTruth/gt_common.py::write_ground_truth."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-review the low-confidence regions of a finished "
                    "classify_session_m2f.py run with a VLM (local by default, "
                    "or --use-api for Claude/Gemini/OpenAI, auto-selected by "
                    "whichever API key is set). The run directory is never "
                    "modified.")
    # Not required=True so argparse errors read the same way in every mode;
    # checked explicitly in main().
    p.add_argument("--run", metavar="DIR",
                   help="A material_map/v1 run directory, e.g. "
                        "...\\fullrate\\material_map_m2f. Read only.")
    p.add_argument("--session-dir", metavar="DIR",
                   help="ZED session folder holding metadata.json + frames/ -- the "
                        "frames the outlined images are drawn on.")
    p.add_argument("--out", default=None, metavar="CSV",
                   help="Output CSV (default: <run>_vlm_review.csv beside the run). "
                        "Never the run's own files.")
    p.add_argument("--frames", default=None, metavar="TXT",
                   help="Restrict to the frame stems listed in this file (e.g. "
                        "..\\..\\GroundTruth\\eval_frames.txt, which makes the output "
                        "scoreable against ground_truth_m2f.csv). Default: every frame "
                        "in --run.")
    p.add_argument("--min-confidence", type=float, default=0.5, metavar="P",
                   help="Regions with CLIP confidence below this are sent for review "
                        "(default 0.5 -- the same floor as classify_session_m2f.py's "
                        "--low-emissivity-min-conf and voxel_consensus.py's "
                        "--min-vote-confidence).")
    p.add_argument("--table", default=None, metavar="CSV",
                   help="Alternative emissivity_table.csv. Defines the enum the model "
                        "must answer from.")
    # --- the override gate, defaults identical to ../classify_session_m2f.py --
    p.add_argument("--low-emissivity-max", type=float, default=0.5, metavar="E",
                   help="An override to a material below this emissivity is only "
                        "accepted on strong evidence (default 0.5).")
    p.add_argument("--low-emissivity-min-conf", type=float, default=0.50, metavar="P",
                   help="Confidence the model must reach for such an override to be "
                        "accepted (default 0.50). Below it the pipeline's label is "
                        "kept and the row is marked rejected_low_emissivity.")
    # --- the engine -----------------------------------------------------------
    p.add_argument("--use-api", action="store_true",
                   help="Use a cloud VLM instead of the local Ollama model -- stronger "
                        "reasoning, costs cents, runs unattended. See the module "
                        "docstring for the local engine's known weaknesses before "
                        "trusting a large local-only run. Which provider runs is "
                        "AUTOMATIC: whichever of ANTHROPIC_API_KEY / GEMINI_API_KEY / "
                        "OPENAI_API_KEY is set in the environment picks it -- paste in "
                        "a Gemini key and it calls Gemini, an OpenAI key and it calls "
                        "OpenAI, no other flag needed. See --api-provider for what "
                        "happens if more than one is set.")
    p.add_argument("--api-provider", choices=tuple(API_PROVIDERS), default=None,
                   metavar="NAME",
                   help="Force a specific --use-api provider (claude / gemini / "
                        "openai) instead of auto-detecting from which key is set. "
                        "Needed only when more than one API key is present in the "
                        "environment at once and they don't all mean 'use this one'.")
    p.add_argument("--model", default=None, metavar="NAME",
                   help=f"Model name. Default: {DEFAULT_MODEL_LOCAL!r} locally; with "
                        f"--use-api, whichever provider was selected supplies its own "
                        f"default ({', '.join(f'{n}={p['default_model']!r}' for n, p in API_PROVIDERS.items())}).")
    p.add_argument("--image-max-px", type=int, default=1024, metavar="PX",
                   help="Long side of the frame sent per region (default 1024). The ZED "
                        "frame is 1920x1080.")
    p.add_argument("--jpeg-quality", type=int, default=85, metavar="Q",
                   help="JPEG quality of that image (default 85).")
    p.add_argument("--outline-width", type=int, default=0, metavar="PX",
                   help="Outline thickness in SOURCE pixels (default 0 = auto, scaled so "
                        "it lands ~3 px wide after the downscale to --image-max-px).")
    p.add_argument("--temperature", type=float, default=0.0, metavar="T",
                   help="Sampling temperature (default 0.0 -- this is a classification, "
                        "and a rerun should give the same answer). Local engine only; "
                        "--use-api controls thoroughness with --effort instead.")
    p.add_argument("--keep-alive", default="0", metavar="SECONDS",
                   help="Local engine only: how long Ollama keeps moondream loaded "
                        "after each call (default '0' = unload immediately). Measured "
                        "on this project's 16 GB/no-GPU machine: leaving the default "
                        "keep-alive (model stays resident between calls) let the "
                        "Ollama server's memory climb to ~9.5 GB over ~30 calls and the "
                        "run was OOM-killed by Windows. Unloading every call costs a "
                        "few seconds of reload per region but has run stably for a "
                        "150+ region session. Raise it (e.g. '5m') only if your machine "
                        "has RAM to spare and you want the loop to run faster.")
    # --- --use-api's retry/pacing (no-ops locally: no quota, no cost) ----------
    p.add_argument("--request-interval", type=float, default=0.5, metavar="S",
                   help="Seconds between --use-api calls (default 0.5).")
    p.add_argument("--max-retries", type=int, default=4, metavar="N",
                   help="--use-api retries per request on 429/5xx (default 4, "
                        "exponential backoff). Other errors are not retried -- logged "
                        "and the row keeps the pipeline's answer.")
    p.add_argument("--retry-base-seconds", type=float, default=2.0, metavar="S",
                   help="First backoff wait for --use-api retries, doubling each time "
                        "(default 2.0).")
    p.add_argument("--dry-run", action="store_true",
                   help="Render and save the images, report how many regions and how "
                        "much disk they use, and exit without calling any model. No key "
                        "needed, nothing billed, Ollama does not need to be running.")
    p.add_argument("--redo", action="store_true",
                   help="Ignore any results.json already sitting in the work dir from an "
                        "earlier, interrupted run of this same --out, and answer every "
                        "flagged region again. Default: resume, skipping regions already "
                        "answered.")
    return p.parse_args()


def main():
    # Line-buffer stdout even when redirected to a file. Python block-buffers a
    # non-tty stream by default, so a run piped to a log (as any run over 100+
    # regions should be, given each call takes ~20s) writes NOTHING until
    # either the buffer fills or the process exits cleanly -- if it is killed
    # first (OOM, Ctrl+C, closing the terminal), the whole log is empty and
    # there is no way to tell how far it got. Measured: a 151-region run
    # OOM-killed after ~30+ min left a completely empty log file.
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.api_provider:              # naming a provider implies --use-api
        args.use_api = True
    provider = detect_api_provider(args.api_provider) if args.use_api else None
    if args.model is None:
        args.model = (API_PROVIDERS[provider]["default_model"] if args.use_api
                      else DEFAULT_MODEL_LOCAL)
    if not args.run or not args.session_dir:
        print("--run and --session-dir are required.", file=sys.stderr)
        return 1
    run_dir = Path(args.run)
    session_dir = Path(args.session_dir)
    if not run_dir.is_dir():
        print(f"{run_dir} is not a directory.", file=sys.stderr)
        return 1

    out_path = (Path(args.out) if args.out
                else run_dir.parent / f"{run_dir.name}_vlm_review.csv")
    if not out_path.is_absolute():
        out_path = _HERE / out_path
    if out_path.resolve() == run_dir.resolve() or run_dir in out_path.resolve().parents:
        print(f"--out {out_path} is inside the run directory. This stage never writes "
              f"into a run.", file=sys.stderr)
        return 1
    work_dir = out_path.parent / f"{out_path.stem}_work"

    materials, eps = gt.load_material_table(args.table)
    alpha = load_solar_absorptance(args.table)
    model_cls = build_review_model(materials)
    frames_dir = gt.load_zed_frames_dir(session_dir)

    if args.frames:
        frames = gt.load_eval_frames(Path(args.frames))
    else:
        frames = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    if not frames:
        print(f"No frame subdirectories in {run_dir}.", file=sys.stderr)
        return 1

    if args.use_api:
        engine = f"Cloud ({provider}, via {API_PROVIDERS[provider]['env']})"
    else:
        engine = "Local (Ollama)"
    print(f"Run      : {run_dir}")
    print(f"Frames   : {len(frames)}")
    print(f"Engine   : {engine}, model {args.model!r}")
    rows, run_frames = collect_rows(run_dir, frames, session=session_dir.parent.name)
    flagged = [r for r in rows if r["pipeline_confidence"] < args.min_confidence]
    flagged_keys = {region_key(r["frame"], r["region_id"]) for r in flagged}
    print(f"Regions  : {len(rows)}, of which {len(flagged)} below "
          f"--min-confidence {args.min_confidence} "
          f"({100.0 * len(flagged) / max(1, len(rows)):.1f}%)")
    print(f"Enum     : {len(materials)} materials from "
          f"{Path(args.table).name if args.table else gt.DEFAULT_TABLE.name}")

    if not flagged:
        write_csv(out_path, rows)
        print(f"Nothing below the threshold. Wrote {out_path} unchanged from the run.")
        return 0

    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(flagged)} outlined frame(s) at "
          f"{args.image_max_px} px, q{args.jpeg_quality} ...")
    items = render_review_items(flagged, run_frames, frames_dir, work_dir / "images", args)
    total_kb = sum(it["image_path"].stat().st_size for it in items) / 1024
    print(f"Saved {len(items)} image(s) to {work_dir / 'images'} ({total_kb:.0f} KB)")
    if args.dry_run:
        print("--dry-run: no model called, nothing billed. Inspect the images above.")
        return 0

    # results.json is written incrementally, after every single item -- see
    # save_results -- so a kill mid-run (OOM, Ctrl+C, closed terminal) loses at
    # most the one call in flight, and rerunning the same --out resumes rather
    # than starting over.
    results_path = work_dir / "results.json"
    results = (RUN_API[provider](items, model_cls, args, results_path) if args.use_api
              else run_local(items, model_cls, args, results_path))
    print(f"{len(results)} result(s) in {results_path}")

    # --- merge and write -----------------------------------------------------
    tally = merge(rows, flagged_keys, results, eps, alpha, set(materials), args)
    write_csv(out_path, rows)

    print("\nReview outcome")
    for status in (ST_CONFIRMED, ST_OVERRIDDEN, ST_REJECTED_LOW_E, ST_ERROR, ST_MISSING):
        if tally.get(status):
            print(f"  {status:<26} {tally[status]:>5}")
    changed = [r for r in rows if r["vlm_status"] == ST_OVERRIDDEN]
    if changed:
        moved = [r for r in changed if abs(eps[r["material"]] - eps[r["pipeline_material"]]) > 0.10]
        print(f"\n{len(changed)} label(s) changed, {len(moved)} of them by more than "
              f"0.10 emissivity (the threshold ../../GroundTruth/evaluate.py calls "
              f"materially wrong):")
        for r in changed[:15]:
            print(f"  {r['frame']} r{r['region_id']:<4} {r['ade']:<14} "
                  f"{r['pipeline_material']} {r['pipeline_confidence']:.2f} -> "
                  f"{r['material']} {r['confidence']:.2f}   "
                  f"e {eps[r['pipeline_material']]:.2f}->{eps[r['material']]:.2f}")
        if len(changed) > 15:
            print(f"  ... and {len(changed) - 15} more (all in the CSV).")
        if not args.use_api:
            print("  (local engine -- spot-check these against the image before trusting "
                  "them; see the module docstring.)")

    print(f"\nWrote {out_path}")
    print(f"      {run_dir} untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
