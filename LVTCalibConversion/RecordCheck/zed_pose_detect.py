"""
Rileva le pose statiche della board in una sessione registrata con
zed_record.py (o zed_record_dual.py), per differenza tra frame consecutivi.

Idea (la stessa di Thesis/Calibration/detect_board_poses.py, gia' usata su
Exttr_tryN): mentre la board e' ferma la differenza media assoluta tra due
frame consecutivi (in scala di grigi) e' bassa; quando la board viene spostata
compare un picco. I tratti stabili tra due spostamenti sono le pose.

Soglia di movimento automatica = mediana + K*MAD delle differenze (--mad-k),
oppure fissa con --threshold. Per non spezzare una posa a causa di picchi brevi
(persone che passano, autoesposizione, board che oscilla) conta come
spostamento solo una sequenza di almeno --min-move-frames frame sopra soglia
(debounce). Le pose piu' corte di --min-duration secondi vengono scartate.
Su Exttr_tryN (~2.5 frame/s) --min-move-frames 12 --min-duration 45 ha dato
12 pose pulite.

Tempi: da metadata.json (frames[].t_offset_s = offset dall'inizio della
sessione). Senza manifest dei frame (metadata.json assente, oppure sessione
interrotta prima della scrittura finale) si ricade su indice / --fps.
Frame: per zed_record.py (schema zed_record/v1) le voci frames[].file; per
zed_record_dual.py le voci frames[].right / frames[].left, scelte con --eye.

A 30 fps una sessione produce decine di migliaia di PNG: --frame-stride N usa
un frame ogni N. --min-move-frames e --min-pose-frames si contano sui frame
campionati (stride 12 a 30 fps ~ i 2.5 frame/s di Exttr_tryN), mentre n_frame
nella tabella conta tutti i frame della sessione che cadono nella posa.

Lo script legge solo: nessun file viene copiato, spostato o modificato.

Uso tipico:
    py zed_pose_detect.py <sessione_zed>
    py zed_pose_detect.py <sessione_zed> --min-move-frames 12 --min-duration 45
    py zed_pose_detect.py <sessione_zed> --frame-stride 12 --csv-out zed_pose.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Serve opencv-python:  py -m pip install opencv-python")

_INDEX_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


def _frame_index(path):
    m = _INDEX_RE.search(path.name)
    return int(m.group(1)) if m else -1


def load_session(session, eye, fps):
    """Frame di un occhio come lista ordinata di (t_offset_s, Path).

    Restituisce (entries, started_utc, sorgente_tempi)."""
    session = Path(session)
    frames_dir = session / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"Cartella frames/ non trovata in {session}")
    on_disk = sorted(frames_dir.glob(f"{eye}_*.png"), key=_frame_index)

    meta_path = session / "metadata.json"
    started, entries = None, []
    if meta_path.is_file():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            print(f"ATTENZIONE: metadata.json illeggibile ({exc})")
            data = {}
        sess = data.get("session") or {}
        started = sess.get("started_utc")
        if data and sess.get("stopped_utc") is None:
            print("ATTENZIONE: metadata.json senza stopped_utc: la registrazione "
                  "non si e' chiusa correttamente.")
        for fr in data.get("frames") or []:
            name = fr.get(eye) or fr.get("file")
            if name and name.startswith(f"{eye}_") and "t_offset_s" in fr:
                entries.append((float(fr["t_offset_s"]), frames_dir / name))
    else:
        print("ATTENZIONE: metadata.json assente.")

    if entries:
        source = "metadata.json (t_offset_s)"
        if len(on_disk) != len(entries):
            print(f"ATTENZIONE: {len(entries)} frame '{eye}' nel manifest, "
                  f"{len(on_disk)} su disco.")
    else:
        entries = [(i / fps, f) for i, f in enumerate(on_disk)]
        source = f"indice / {fps:g} fps (nessun manifest frame in metadata.json)"
    entries.sort(key=lambda e: e[0])
    return entries, started, source


def diff_profile(entries, stride, downscale):
    """Differenza media tra frame campionati consecutivi.

    Restituisce (values, idx): values[k] confronta il frame campionato
    precedente con entries[idx[k]]."""
    values, idx = [], []
    prev = None
    skipped = 0
    sampled = range(0, len(entries), stride)
    for n, i in enumerate(sampled, start=1):
        img = cv2.imread(str(entries[i][1]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        if downscale > 1:
            img = cv2.resize(img, (img.shape[1] // downscale,
                                   img.shape[0] // downscale))
        if prev is not None:
            values.append(float(np.mean(cv2.absdiff(prev, img))))
            idx.append(i)
        prev = img
        if n % 200 == 0:
            print(f"\r  {n}/{len(sampled)} frame letti", end="", flush=True)
    print()
    if skipped:
        print(f"ATTENZIONE: {skipped}/{len(sampled)} frame illeggibili o mancanti, saltati.")
    return np.array(values), idx


def segment(values, threshold, min_move_frames, min_pose_frames):
    """Segmentazione con debounce (da detect_board_poses.py).

    Restituisce (poses, transitions): intervalli di indici in `values`."""
    moving = values > threshold
    n = len(moving)
    transitions = []
    i = 0
    while i < n:
        if moving[i]:
            j = i
            while j < n and moving[j]:
                j += 1
            if j - i >= min_move_frames:
                transitions.append((i, j))
            i = j
        else:
            i += 1
    poses = []
    prev_end = 0
    for a, b in transitions:
        if a - prev_end >= min_pose_frames:
            poses.append((prev_end, a))
        prev_end = b
    if n - prev_end >= min_pose_frames:
        poses.append((prev_end, n))
    return poses, transitions


def print_table(header, rows):
    cells = [[str(c) for c in r] for r in rows]
    widths = [len(h) for h in header]
    for r in cells:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in cells:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"CSV salvato: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Rileva le pose statiche della board in una sessione zed_record.py."
    )
    ap.add_argument("session",
                    help="Cartella sessione di zed_record.py (con metadata.json e frames/)")
    ap.add_argument("--eye", choices=("right", "left"), default="right",
                    help="Occhio da analizzare (default right, quello di zed_record.py)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Soglia di movimento fissa. Se omessa: mediana + K*MAD (--mad-k)")
    ap.add_argument("--mad-k", type=float, default=6.0,
                    help="Fattore K della soglia automatica (default 6)")
    ap.add_argument("--min-move-frames", type=int, default=3,
                    help="Frame campionati consecutivi sopra soglia per contare come "
                         "spostamento reale (default 3; Exttr_tryN: 12)")
    ap.add_argument("--min-pose-frames", type=int, default=15,
                    help="Frame campionati stabili minimi per una posa (default 15)")
    ap.add_argument("--min-duration", type=float, default=None,
                    help="Tiene solo le pose lunghe almeno N secondi (Exttr_tryN: 45)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="Usa un frame ogni N (default 1 = tutti)")
    ap.add_argument("--downscale", type=int, default=2,
                    help="Fattore di riduzione delle immagini per il confronto (default 2)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Frame rate di ripiego se metadata.json non ha il manifest "
                         "dei frame (default 30)")
    ap.add_argument("--csv-out", default=None, metavar="CSV",
                    help="Salva anche la tabella delle pose in CSV")
    args = ap.parse_args()
    if args.frame_stride < 1:
        ap.error("--frame-stride deve essere >= 1")

    entries, started, source = load_session(args.session, args.eye, args.fps)
    if len(entries) < 2:
        sys.exit(f"Servono almeno 2 frame '{args.eye}' in {args.session}.")
    print(f"Frame '{args.eye}': {len(entries)}   Tempi: {source}")
    print(f"Primo: {entries[0][1].name}   Ultimo: {entries[-1][1].name}")
    dur_total = entries[-1][0] - entries[0][0]
    print(f"Durata sessione: {dur_total:.0f} s ({dur_total / 60:.1f} min)"
          + (f"   Inizio: {started}" if started else ""))
    print(f"Confronto tra frame (stride {args.frame_stride}, downscale {args.downscale})...")

    values, idx = diff_profile(entries, args.frame_stride, args.downscale)
    if values.size == 0:
        sys.exit("Nessuna differenza calcolabile.")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = (args.threshold if args.threshold is not None
                 else median + args.mad_k * (mad if mad > 1e-6 else 1.0))
    print(f"Differenze: mediana={median:.3f}  MAD={mad:.3f}  soglia={threshold:.3f}"
          + ("  (fissa)" if args.threshold is not None else f"  (mediana+{args.mad_k:g}*MAD)"))
    print()

    poses, transitions = segment(values, threshold, args.min_move_frames,
                                 args.min_pose_frames)

    rows_raw = []
    for a, b in poses:
        i0, i1 = idx[a], idx[b - 1]
        rows_raw.append((i0, i1, entries[i0][0], entries[i1][0]))
    dropped = 0
    if args.min_duration is not None:
        kept = [r for r in rows_raw if r[3] - r[2] >= args.min_duration]
        dropped = len(rows_raw) - len(kept)
        rows_raw = kept

    header = ["posa", "primo_file", "ultimo_file", "offset_inizio_s",
              "offset_fine_s", "durata_s", "n_frame"]
    rows = []
    for k, (i0, i1, t0, t1) in enumerate(rows_raw, start=1):
        rows.append([k, entries[i0][1].name, entries[i1][1].name,
                     f"{t0:.1f}", f"{t1:.1f}", f"{t1 - t0:.1f}", i1 - i0 + 1])

    print(f"POSE: {len(rows)}   (spostamenti: {len(transitions)}"
          + (f"; {dropped} pose piu' corte di {args.min_duration:g} s scartate" if dropped else "")
          + ")")
    print_table(header, rows)
    print()
    print("Se il numero di pose non torna: --min-move-frames piu' alto unisce pose "
          "spezzate da picchi brevi, --mad-k piu' alto = meno spostamenti, "
          "--min-duration scarta i segmenti spuri.")

    if args.csv_out:
        print()
        write_csv(args.csv_out, header, rows)


if __name__ == "__main__":
    main()
