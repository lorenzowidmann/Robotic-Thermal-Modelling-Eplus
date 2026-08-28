"""Score one or more classification runs against a hand-made ground truth.

Reads ground_truth.csv (from annotate.py) and any number of material_map/v1 run
directories, and reports accuracy, per-class precision/recall/F1, emissivity
error and a confusion matrix for each, side by side, so prior-on vs prior-off
(or any other pair) can be read off directly.

Read the emissivity error, not only the accuracy
------------------------------------------------
Accuracy counts every mistake once. The thermal pipeline does not: it divides by
emissivity, and every non-metal in emissivity_table.csv sits between 0.90 and
0.95. Calling a painted wall `ceramic` is an accuracy miss worth ~2 K; calling a
window `aluminum_polished` is one accuracy miss worth hundreds. Measured on the
first real 20-frame ground truth, prior-on and prior-off differ by 17 accuracy
points but by 0 vs 12 regions with a materially wrong emissivity, and by 3 K vs
316 K in the worst case. The accuracy number badly understates what the prior
does.

What it refuses to do
---------------------
Score a run whose segmentation is not the one that was annotated. A ground-truth
row means "region 3 of frame X"; region ids are positional, so against a
different segmentation that row silently points at a different object and the
resulting accuracy is a plausible-looking fiction. Every frame's labels.npy is
therefore checked against the md5 recorded at annotation time, and a mismatch is
fatal. There is deliberately no --force.

This is not hypothetical: material_map_sam on this session was produced before
commit a664518 fixed the ZED intrinsics (so its FLIR-FOV crop is 751x733 where
the corrected box is 979x952) and before the sync manifest was regenerated (so
it classified right_000148.png where the m2f run classified right_000149.png).
Nothing about it looks wrong from the outside.

Macro-averaging
---------------
The macro average is taken over classes PRESENT IN THE GROUND TRUTH, not over
all 20 in emissivity_table.csv. Most of the table never occurs indoors here --
the m2f run predicts zero bare metals across all 1192 of its regions -- and a
macro over 20 classes would be dominated by classes with no support, moving for
reasons unrelated to either classifier. Classes a run predicts that never occur
in the ground truth are false positives; they are counted in accuracy and in the
confusion matrix, and listed separately, since they have a precision but no
meaningful recall.

Usage:
    py evaluate.py --ground-truth ground_truth_m2f.csv \\
                   --runs ...\\fullrate\\material_map_m2f ...\\fullrate\\material_map_m2f_noprior \\
                   --names prior-on prior-off
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import gt_common as gt                                            # noqa: E402


def frame_md5_map(rows: list[dict]) -> dict[str, str]:
    """frame -> labels_md5, checking the ground truth is internally consistent."""
    out = {}
    for r in rows:
        f, m = r["frame"], r["labels_md5"]
        if f in out and out[f] != m:
            raise SystemExit(
                f"{f} appears in the ground truth with two different labels_md5 "
                f"({out[f][:10]} and {m[:10]}). The file mixes two segmentations "
                "of the same frame; re-annotate rather than scoring it.")
        out[f] = m
    return out


def score(gt_rows: list[dict], frames: dict[str, gt.RunFrame]) -> dict:
    """Predicted vs true material for every `labeled` row."""
    y_true, y_pred, area, missing = [], [], [], []
    for r in gt_rows:
        seg = frames[r["frame"]].by_id.get(r["region_id"])
        if seg is None:
            # Unreachable while the md5 matches (ids come from labels.npy), so
            # if it fires the invariant in load_run_frame has been broken.
            missing.append((r["frame"], r["region_id"]))
            continue
        y_true.append(r["ground_truth"])
        y_pred.append(seg["top_material"])
        area.append(r["area_px"])
    if missing:
        raise SystemExit(f"Regions in the ground truth but not in the run: {missing[:10]}")
    return {"y_true": y_true, "y_pred": y_pred, "area": np.array(area, dtype=float)}


# A surface at this temperature is assumed when converting an emissivity error
# into the temperature error it would cause. 22 degC is an ordinary indoor wall.
T_REF_K = 273.15 + 22.0
# Above this, an emissivity error stops being cosmetic. Every non-metal in
# emissivity_table.csv sits between 0.90 and 0.95, so anything past 0.1 means a
# metal was proposed for something that is not one.
EPS_SIGNIFICANT = 0.10


def emissivity_error(y_true, y_pred, area, eps):
    """How wrong the emissivity is, which is what the thermal pipeline divides by.

    Accuracy counts paint->plastic and paint->aluminum_polished as one error
    each. The first shifts the corrected temperature by ~0 K, the second by
    hundreds. This separates them.

    `worst_dT` inverts the graybody relation L = eps*sigma*T^4: assuming eps_p
    where the truth is eps_t scales the recovered temperature by
    (eps_t/eps_p)**0.25. It IGNORES reflected ambient radiation, which in a
    corridor where walls, air and objects sit at similar temperatures cancels
    much of the error -- ../RadiometricCalibration/correct_session.py does model
    it. So this is an order-of-magnitude upper bound for ranking runs against
    each other, not a prediction of what the pipeline will output.
    """
    de = np.array([abs(eps[p] - eps[t]) for t, p in zip(y_true, y_pred)])
    ratio = np.array([(eps[t] / eps[p]) ** 0.25 for t, p in zip(y_true, y_pred)])
    dT = (ratio - 1.0) * T_REF_K
    return {
        "mean_abs_deps": float(de.mean()),
        "area_weighted_abs_deps": float(np.average(de, weights=area)) if area.sum() else 0.0,
        "n_deps_over_threshold": int((de > EPS_SIGNIFICANT).sum()),
        "worst_abs_dT_K": float(np.abs(dT).max()),
        "mean_abs_dT_K": float(np.abs(dT).mean()),
    }


def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def per_class(cm: np.ndarray):
    """Precision, recall, F1 per class from a (true x pred) confusion matrix.

    By hand rather than via sklearn: it is a dozen lines, and the repo rule is
    not to add a dependency unless it is necessary. Zero denominators give 0.0,
    the same convention as sklearn's zero_division=0.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return prec, rec, f1, cm.sum(axis=1)          # support = true count


def fmt(v, width=14):
    return f"{v:>{width}.3f}" if isinstance(v, float) else f"{str(v):>{width}}"


def main():
    p = argparse.ArgumentParser(
        description="Score material_map/v1 runs against a ground_truth.csv.")
    p.add_argument("--ground-truth", required=True, metavar="CSV")
    p.add_argument("--runs", required=True, nargs="+", metavar="DIR",
                   help="One or more run directories to compare.")
    p.add_argument("--names", nargs="+", metavar="NAME",
                   help="Column labels (default: the directory names).")
    p.add_argument("--out", default="comparison.csv", metavar="CSV",
                   help="Comparison table (default comparison.csv).")
    p.add_argument("--confusion-dir", default=None, metavar="DIR",
                   help="Write one confusion matrix CSV per run here "
                        "(default: alongside --out).")
    p.add_argument("--table", default=None, metavar="CSV",
                   help="Alternative emissivity_table.csv (validates the labels).")
    args = p.parse_args()

    materials, _eps = gt.load_material_table(args.table)
    rows = gt.read_ground_truth(Path(args.ground_truth))

    names = args.names or [Path(r).name for r in args.runs]
    if len(names) != len(args.runs):
        print(f"--names has {len(names)} entries for {len(args.runs)} runs.",
              file=sys.stderr)
        return 1

    # --- split the ground truth -------------------------------------------
    labeled = [r for r in rows if r["status"] == gt.LABELED]
    n_mixed = sum(1 for r in rows if r["status"] == "mixed")
    n_unclear = sum(1 for r in rows if r["status"] == "unclear")
    bad = sorted({r["ground_truth"] for r in labeled} - set(materials))
    if bad:
        print(f"Ground truth uses label(s) not in the emissivity table: {bad}",
              file=sys.stderr)
        return 1
    if not labeled:
        print("No rows with status 'labeled' -- nothing to score.", file=sys.stderr)
        return 1

    frames = list(dict.fromkeys(r["frame"] for r in rows))
    expected = frame_md5_map(rows)
    annotated_on = sorted({r["run_used"] for r in rows})

    print(f"Ground truth: {Path(args.ground_truth).name}")
    print(f"  annotated on : {', '.join(annotated_on)}")
    print(f"  {len(frames)} frame(s), {len(labeled)} region(s) labelled, "
          f"{n_mixed} mixed + {n_unclear} unclear excluded")

    # --- load every run, refusing incompatible segmentation ----------------
    # The ground truth records run_used as a bare directory name. If a sibling
    # of the run being scored has that name, hand it over so the failure can
    # print what the annotated segmentation looked like next to this one --
    # which is what makes the mismatch diagnosable rather than just fatal.
    def reference_for(run: Path) -> Path | None:
        for cand in annotated_on:
            sib = Path(run).parent / cand
            if sib.is_dir() and sib != Path(run):
                return sib
        return None

    results = {}
    for run, name in zip(args.runs, names):
        try:
            loaded = gt.assert_compatible(Path(run), frames, expected,
                                          reference_dir=reference_for(Path(run)))
        except gt.IncompatibleRun as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(f"\n{name}: {exc}", file=sys.stderr)
            return 2
        results[name] = score(labeled, loaded)
    print(f"  {len(args.runs)} run(s) verified against the annotated segmentation\n")

    # --- classes ------------------------------------------------------------
    gt_classes = sorted({r["ground_truth"] for r in labeled})
    pred_only = sorted({m for res in results.values() for m in res["y_pred"]}
                       - set(gt_classes))
    classes = gt_classes + pred_only          # gt classes first, then FP-only

    conf_dir = Path(args.confusion_dir) if args.confusion_dir else Path(args.out).parent
    if not conf_dir.is_absolute():
        conf_dir = _HERE / conf_dir
    conf_dir.mkdir(parents=True, exist_ok=True)

    table_rows, metrics = [], {}
    for name in names:
        y_true, y_pred = results[name]["y_true"], results[name]["y_pred"]
        cm = confusion(y_true, y_pred, classes)
        prec, rec, f1, support = per_class(cm)
        keep = [classes.index(c) for c in gt_classes]        # macro over GT-present only
        metrics[name] = {
            "n_scored": len(y_true),
            "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
            "macro_precision": float(prec[keep].mean()),
            "macro_recall": float(rec[keep].mean()),
            "macro_f1": float(f1[keep].mean()),
            # Predictions naming a class that occurs nowhere in the ground
            # truth. Every one is necessarily wrong, and NONE of them appear in
            # macro precision, because the classes they name are not in the
            # average. Without this row a high macro precision reads as "the
            # model is precise" when it can mean "the model's mistakes were
            # filed under classes this metric does not look at".
            "n_pred_outside_gt": int(sum(1 for p in y_pred if p not in gt_classes)),
            **emissivity_error(y_true, y_pred, results[name]["area"], _eps),
            "per_class": {c: (prec[i], rec[i], f1[i], int(support[i]))
                          for i, c in enumerate(classes)},
        }
        out_cm = conf_dir / f"confusion_{name}.csv"
        with out_cm.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["true\\pred"] + classes)
            for i, c in enumerate(classes):
                w.writerow([c] + list(cm[i]))
        table_rows.append(out_cm)

    # --- printed comparison -------------------------------------------------
    w0 = max(24, max(len(c) for c in classes) + 12)
    head = " " * w0 + "".join(fmt(n) for n in names)
    print(head)
    print("-" * len(head))
    for key, label in (("n_scored", "regions scored"),
                       ("accuracy", "accuracy"),
                       ("macro_precision", f"macro precision ({len(gt_classes)} GT cls)"),
                       ("macro_recall", f"macro recall    ({len(gt_classes)} GT cls)"),
                       ("macro_f1", f"macro F1        ({len(gt_classes)} GT cls)"),
                       ("n_pred_outside_gt", "predictions outside GT cls")):
        print(f"{label:<{w0}}" + "".join(fmt(metrics[n][key]) for n in names))

    print(f"\n# emissivity error -- what the radiometric correction divides by")
    print(f"{'':<{w0}}" + "".join(fmt(n) for n in names))
    print("-" * len(head))
    for key, label in (("mean_abs_deps", "mean |d eps|"),
                       ("area_weighted_abs_deps", "area-weighted |d eps|"),
                       ("n_deps_over_threshold", f"regions |d eps| > {EPS_SIGNIFICANT:.2f}"),
                       ("mean_abs_dT_K", "mean |dT| K *"),
                       ("worst_abs_dT_K", "worst |dT| K *")):
        print(f"{label:<{w0}}" + "".join(fmt(metrics[n][key]) for n in names))
    print("* naive graybody bound, (eps_true/eps_pred)^0.25 at "
          f"{T_REF_K - 273.15:.0f} degC, ignoring reflected ambient.")
    print("  Use it to rank runs, not to predict what correct_session.py outputs.")

    print(f"\n{'per-class F1 (support)':<{w0}}" + "".join(fmt(n) for n in names))
    print("-" * len(head))
    for c in gt_classes:
        sup = metrics[names[0]]["per_class"][c][3]
        print(f"{c + f' ({sup})':<{w0}}"
              + "".join(fmt(float(metrics[n]['per_class'][c][2])) for n in names))
    if pred_only:
        print(f"\npredicted but never in the ground truth (false positives only, "
              f"no recall to report):")
        for c in pred_only:
            counts = " ".join(
                f"{n}={int(np.sum([p == c for p in results[n]['y_pred']]))}" for n in names)
            flag = "  <-- LOW EMISSIVITY" if _eps[c] < 0.5 else ""
            print(f"  {c:<20} eps={_eps[c]:<5} {counts}{flag}")

    # Macro precision sitting well above accuracy is the signature of errors
    # hiding in classes the macro does not average over. Say so rather than
    # leaving it to be noticed.
    for n in names:
        if (metrics[n]["macro_precision"] - metrics[n]["accuracy"] > 0.10
                and metrics[n]["n_pred_outside_gt"]):
            print(f"\nNOTE  {n}: macro precision ({metrics[n]['macro_precision']:.3f}) sits well "
                  f"above accuracy ({metrics[n]['accuracy']:.3f}) because "
                  f"{metrics[n]['n_pred_outside_gt']} prediction(s) name classes absent from the "
                  f"ground truth. Every one of those is wrong, and none enter the macro average, "
                  f"which only covers the {len(gt_classes)} classes that do occur. Read accuracy "
                  f"and macro F1 here, not macro precision.")

    # --- comparison.csv -----------------------------------------------------
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _HERE / out_path
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "class", "support"] + names)
        for key in ("n_scored", "accuracy", "macro_precision", "macro_recall",
                    "macro_f1", "n_pred_outside_gt", "mean_abs_deps",
                    "area_weighted_abs_deps", "n_deps_over_threshold",
                    "mean_abs_dT_K", "worst_abs_dT_K"):
            w.writerow([key, "", ""] + [metrics[n][key] for n in names])
        for c in classes:
            sup = metrics[names[0]]["per_class"][c][3]
            for j, stat in enumerate(("precision", "recall", "f1")):
                w.writerow([stat, c, sup]
                           + [float(metrics[n]["per_class"][c][j]) for n in names])

    print(f"\nWrote {out_path}")
    for pth in table_rows:
        print(f"      {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
