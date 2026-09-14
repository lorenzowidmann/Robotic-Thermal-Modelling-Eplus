# VLM review of low-confidence regions

`review_low_confidence.py` is a **fallback stage**, not a classifier. It runs after a
finished `../classify_session_m2f.py` run, takes only the regions CLIP was least sure
about, asks a vision model to confirm or override each one, and writes the whole run out
as a CSV with those rows updated.

Nothing in the run directory is modified — not `segments.json`, not `labels.npy`. The
M2F+CLIP logic is untouched and unimported, so a review pass is free to rerun and its
output is always diffable against the run it came from.

```
# local, default — free, no key, needs Ollama running
py review_low_confidence.py --run ...\fullrate\material_map_m2f ^
                            --session-dir ...\ZED\20260730_161223\fullrate

# what would be rendered, without calling any model
py review_low_confidence.py --run ... --session-dir ... --dry-run

# a cloud engine instead -- whichever key you set picks the provider
set ANTHROPIC_API_KEY=...          :: or GEMINI_API_KEY, or OPENAI_API_KEY
py review_low_confidence.py --run ... --session-dir ... --use-api
```

No segmentation or CLIP model is loaded — this stage only reads a finished run.

## Why only the low-confidence ones

CLIP scores a crop against 20 fixed prompts from `../../emissivity_table.csv`. It never
sees the rest of the frame, and that is where the answer often is: a worn metallic-looking
shape is `steel_oxidized` on texture alone and `painted_metal` the moment you can see it is
a radiator mounted under a window.

`ade_material_prior.csv` already fixes everything the ADE class settles — a `column` is not
glass. This stage covers the other case: the ADE class is right, the texture is ambiguous,
and CLIP says so by returning a low confidence. Above the threshold nothing is sent, so a
confident run costs nothing.

## Four engines, one downstream contract

| | Local (default) | `--use-api` |
|---|---|---|
| Model | moondream, via Ollama | Claude / Gemini / OpenAI |
| Cost | free | cents (see below) |
| Setup | `ollama pull moondream`, Ollama running | one API key, in the environment |
| Data leaves the machine | no | yes |
| Reasoning quality | weak (1B params, 2K context) — see caveat below | strong |

All four produce the exact same result shape — `{material, confidence, reasoning}` or
`{error}` per region — so render, merge, the low-emissivity gate and the CSV are **one
code path** regardless of which engine ran.

### Local engine — the honest caveat

moondream is a 1-billion-parameter model with a 2048-token context window: fast and free,
but genuinely weak at the nuanced judgment this task sometimes needs. Measured while
building this stage: given a prompt example ("a radiator near a window is
`painted_metal`"), it echoed that example back **verbatim as its reasoning for an
unrelated wall region**, instead of describing the photo. On a 28-region test run, ~40% of
answers had `vlm_reasoning` copied straight from the prompt's own instruction text, and the
one `overridden` region checked against its source image by eye turned out to be **wrong**
(a wall-mounted painting called `painted_metal`). The local prompt is deliberately short
and gives it nothing quotable to copy, but treat `vlm_reasoning` on local runs as a hint,
not a citation. **Spot-check the `overridden` rows before trusting them** — the script's
own summary flags this every run.

Use `--use-api` for anything where a wrong material actually matters — a large run, or one
feeding real thermal-correction numbers.

### Cloud engines — `--use-api`

One call per region (vision + JSON-schema-constrained output), not a batch job: batching
trades up to 24 h turnaround for a 50% discount, the wrong trade for a handful of images
that cost cents either way.

**Which provider runs is automatic** — whichever of these three keys is set in the
environment picks it, with no other flag needed:

| env var | provider | default model | get a key |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Claude | `claude-opus-5` | https://platform.claude.com (~$5 free, no card) |
| `GEMINI_API_KEY` | Gemini | `gemini-flash-latest` | https://aistudio.google.com/apikey |
| `OPENAI_API_KEY` | OpenAI | `gpt-5.6-luna` | https://platform.openai.com/api-keys |

If more than one key happens to be set, the priority order above wins (`claude` first);
`--api-provider {claude,gemini,openai}` picks one explicitly instead. `--model` overrides
that provider's default. Whichever key is used, it is never read from a file or an
argument — environment only.

One caveat learned building this against a live Gemini key: a *dated* model name
(`gemini-2.5-flash`) got retired for new-project keys mid-project, with only a 404 at
request time as warning — which is why the Gemini default here is the rolling
`-latest` alias, not a snapshot.

## Input

| what | where it comes from |
|---|---|
| regions, materials, confidences | `<run>/<flir_stem>/segments.json`, schema `material_map/v1` |
| region masks | `<run>/<flir_stem>/labels.npy` |
| the frame to draw on | `--session-dir`'s `metadata.json` → `recording.frames_dir` |
| the label enum | `../../emissivity_table.csv`, `material` column |

Loaded through `../../GroundTruth/gt_common.py::load_run_frame`, so this stage validates a
run exactly the way `evaluate.py` does — schema, required keys, and the
`segments.json` ↔ `labels.npy` id agreement — rather than trusting the JSON.

## What the model is shown

The **full frame with the region outlined in red** — literally
`../../GroundTruth/annotate.py::render_context()`, the right-hand panel of the manual
annotation tool, called rather than copied. The human annotator and the model look at the
same picture.

One difference, and it is in what is passed to that function, not in the function:
`annotate.py` hands it the region mask, so the region is tint-**filled** red; here it is
handed a dilated boundary band, so only the outline is tinted and the surface keeps its
true colour. That matters for a model in a way it does not for a person — the question is
what colour and finish that surface has, and a 55 % red wash over it would be answering it.

Full frame rather than `annotate.py`'s zoomed `render_region()` crop on purpose: scene
context is the entire reason for asking a VLM, and a padded crop is where it gets thrown
away.

Every rendered image is saved to `<out>_work/images/<frame>-r<region_id>.jpg` — the local
engine reads it from there (Ollama's `images=` parameter wants a file path), every cloud
engine re-reads the same file to base64 it, and it doubles as an audit trail of exactly
what the model saw.

## The schema, shared by all four engines

One Pydantic model, built at runtime from `../../emissivity_table.csv`'s live material
list (never hardcoded — a table edit can't leave it stale, the same rule
`../m2f_materials/category_prior.py` is validated against):

```python
class MaterialReview(BaseModel):
    material: Literal[tuple(materials)]
    confidence: float
    reasoning: str
```

`.model_json_schema()` feeds Ollama's `format=`, Claude's `output_config.format`, Gemini's
`response_schema`, and OpenAI's `text_format=` (passed as the Pydantic class itself, which
`client.responses.parse` reads directly) — four different parameter names over the same
JSON Schema, so a valid answer means the same thing on every engine. Adding a fifth
provider is one `make_client_X` + one `run_X` off this same schema, no changes anywhere
else.

## The merge, and the gate on it

| model's answer vs the pipeline | result | `vlm_status` |
|---|---|---|
| same material | keep the label, confidence → `max(clip, model)` | `confirmed` |
| different material | take the model's label **and** its confidence | `overridden` |
| different, but low-ε and weakly held | keep the pipeline's label | `rejected_low_emissivity` |
| item errored, or absent from the results | keep the pipeline's label | `error` / `missing_from_results` |
| never flagged | row passes through untouched | *(empty)* |

That third row is the one worth reading. The model may answer with any of the 20
materials — the same list `annotate.py` binds its keys to — and 7 of them are **bare
metals**. This stage lands *after* `classify_session_m2f.py`'s low-emissivity gate has
already run, so an unchecked override would reintroduce exactly what that gate and the ADE
prior exist to prevent: the radiometric correction divides by ε, so a wrong
`aluminum_polished` (ε = 0.05) turns a 37 °C reading into ~156 °C, while confusing any two
ordinary indoor materials costs under 1 K.

So the gate's own two parameters are reapplied to the model's answer, with the **same
defaults** as `classify_session_m2f.py` — `--low-emissivity-max 0.5`,
`--low-emissivity-min-conf 0.50`. A refused override is recorded, never silently dropped.
Both prompts also spell out that the bare-metal labels mean *unpainted* metal, which is the
confusion that produces them.

## Thresholds

`--min-confidence` defaults to **0.5**, which is not a new number: it is the floor
`classify_session_m2f.py --low-emissivity-min-conf` and `../../voxel_consensus.py
--min-vote-confidence` both already use.

## Output

`<run>_vlm_review.csv`, beside the run, never inside it:

| column group | columns |
|---|---|
| identity | `session`, `run_used`, `frame`, `region_id`, `area_px`, `ade`, `labels_md5` |
| what the pipeline said | `pipeline_material`, `pipeline_confidence` |
| the answer after this stage | `material`, `confidence`, `emissivity`, `solar_absorptance` |
| the review | `vlm_reviewed`, `vlm_status`, `vlm_material`, `vlm_confidence`, `vlm_reasoning`, `reviewed_utc` |

`material` / `confidence` are what a consumer should read. The pipeline's original answer is
kept on every row, so the stage is fully reversible from its own output and `vlm_status`
explains every row that moved.

The first five identity columns match `../../GroundTruth/ground_truth_m2f.csv` so the two
join on `(frame, region_id)` with no rename — which is how you measure whether this stage
helped. `labels_md5` travels with every row for the reason
`../../GroundTruth/README.md` sets out at length: region ids are **positional**, so a row
only means anything against the segmentation it was written from.

Alongside it, `<run>_vlm_review_work/`:

- `images/<frame>-r<region_id>.jpg` — every rendered image, the audit trail and what
  `--dry-run` stops at
- `results.json` — the parsed per-key responses

## Failure handling and resuming

Per-item failures never fail the run. A malformed response, a connection hiccup, a rate
limit — each is logged, recorded in `vlm_status` / `vlm_reasoning`, and the row keeps the
pipeline's answer. `--use-api` retries 429/5xx with exponential backoff
(`--request-interval`, `--max-retries`, `--retry-base-seconds`); the local engine has no
quota to retry around, but does need `--keep-alive` (default `0`, unload after every call)
-- on one 16 GB/no-GPU machine, leaving Ollama's default keep-alive on let its server
process climb to ~9.5 GB resident over ~30 calls and Windows OOM-killed the run.

`<out>_work/results.json` is written after **every single region**, not once at the end.
Rerunning the exact same command resumes: it skips any region already answered
successfully and retries only the ones still missing or still marked `error` — including
errors left behind by a *different* engine's earlier, abandoned attempt at the same `--out`
(results are keyed only by region, with no record of which engine produced an entry, so an
old failure is always fair game for a retry). `--redo` ignores all of that and answers
every flagged region again from scratch.

This survives exactly the failure that motivated it: a 151-region local run was
OOM-killed after ~30 minutes with a completely empty log (Python block-buffers stdout when
it is not a terminal, so a killed process can leave nothing on disk at all -- fixed here by
line-buffering stdout at startup). Rerunning the same command picked up at region 29
instead of starting over.

## Scoring it

`--frames ..\..\GroundTruth\eval_frames.txt` restricts the pass to the pinned eval set, so
the output covers exactly the regions `ground_truth_m2f.csv` has labels for and the
before/after is measurable rather than an impression.

## Layout

```
ExtrenalVLM/
  review_low_confidence.py    the whole stage, all four engines
  requirements.txt
  README.md
```

`../../GroundTruth/annotate.py` (the renderer) and `gt_common.py` (the run loader and the
material table) are **imported, not copied**. Both are import-safe: they pull in numpy and
the stdlib at module scope and tkinter/PIL only inside the functions that need them, so
importing them here loads no GUI and no model.
