"""
Report delle pose della board in una sessione FLIR Vue Pro R, basato sui gap
temporali tra i file radiometrici (*_R.jpg).

La FLIR scrive un file per frame con il timestamp nel nome
(YYYYMMDD_HHMMSS_R.jpg). Se la registrazione viene fermata tra una posa e
l'altra, tra l'ultimo file di una posa e il primo della successiva resta un
buco: ogni gap piu' lungo di --gap-threshold secondi apre una nuova posa.

Il clock della FLIR puo' essere sbagliato (data/ora della camera non
impostate): gli orari stampati sono quelli del clock FLIR e valgono solo in
relativo. L'allineamento con ZED/LiDAR va fatto per ordine delle pose, non per
ora assoluta.

Una sessione lunga puo' essere spezzata dalla SD su piu' cartelle
(100_FLIR, 101_FLIR, ...): passarle tutte, vengono unite in un unico flusso
ordinato per timestamp.

Lo script legge solo i NOMI dei file: nessun file viene aperto, copiato,
spostato o modificato.

NOTA: se la FLIR ha registrato in continuo (senza stop tra le pose) non ci sono
gap e il report trova una sola posa. In quel caso le pose vanno cercate per
differenza tra frame (Thesis/Calibration/detect_board_poses.py).

Uso tipico:
    py flir_gap_report.py D:\\DCIM\\100_FLIR D:\\DCIM\\101_FLIR
    py flir_gap_report.py <cartella_flir> --gap-threshold 3 --min-frames 10
    py flir_gap_report.py <cartella_flir> --csv-out flir_pose.csv
"""

import argparse
import csv
import fnmatch
import re
import sys
from datetime import datetime
from pathlib import Path

_TS_RE = re.compile(r"(\d{8})_(\d{6})")


def parse_flir_timestamp(name):
    """datetime dal nome FLIR YYYYMMDD_HHMMSS_R.jpg, None se non parsabile."""
    m = _TS_RE.search(name)
    if m is None:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def collect_frames(dirs, pattern):
    """(timestamp, Path) dei file che rispettano `pattern` in tutte le cartelle,
    uniti e ordinati per timestamp. Restituisce anche i file senza timestamp."""
    frames, unparsed, seen = [], [], set()
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            sys.exit(f"Cartella non trovata: {p}")
        if p.resolve() in seen:
            print(f"ATTENZIONE: cartella passata due volte, ignorata: {p}")
            continue
        seen.add(p.resolve())
        found = [f for f in p.iterdir()
                 if f.is_file() and fnmatch.fnmatch(f.name.lower(), pattern.lower())]
        if not found:
            print(f"ATTENZIONE: nessun file {pattern} in {p}")
        for f in found:
            ts = parse_flir_timestamp(f.name)
            if ts is None:
                unparsed.append(f)
            else:
                frames.append((ts, f))
    frames.sort(key=lambda t: (t[0], t[1].name))
    return frames, unparsed


def split_on_gaps(frames, gap_threshold):
    """Spezza il flusso ordinato in pose: un gap > gap_threshold apre una posa."""
    poses = [[frames[0]]]
    for prev, cur in zip(frames, frames[1:]):
        if (cur[0] - prev[0]).total_seconds() > gap_threshold:
            poses.append([])
        poses[-1].append(cur)
    return poses


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
        description="Report delle pose FLIR dai gap temporali tra i file *_R.jpg."
    )
    ap.add_argument("dirs", nargs="+",
                    help="Una o piu' cartelle FLIR (anche una sessione spezzata "
                         "dalla SD), unite in un unico flusso ordinato per timestamp")
    ap.add_argument("--gap-threshold", type=float, default=5.0,
                    help="Gap in secondi tra due file consecutivi oltre il quale "
                         "inizia una nuova posa (default 5)")
    ap.add_argument("--min-frames", type=int, default=1,
                    help="Scarta le pose con meno di N frame, es. scatti isolati "
                         "tra una posa e l'altra (default 1 = tiene tutto)")
    ap.add_argument("--pattern", default="*_R.jpg",
                    help="Pattern dei file radiometrici, case-insensitive "
                         "(default *_R.jpg)")
    ap.add_argument("--csv-out", default=None, metavar="CSV",
                    help="Salva anche la tabella delle pose in CSV")
    args = ap.parse_args()
    if args.gap_threshold <= 0:
        ap.error("--gap-threshold deve essere > 0")

    frames, unparsed = collect_frames(args.dirs, args.pattern)
    if unparsed:
        print(f"ATTENZIONE: {len(unparsed)} file senza timestamp nel nome, "
              f"ignorati (es. {unparsed[0].name})")
    if not frames:
        sys.exit("Nessun file FLIR con timestamp trovato.")

    span = (frames[-1][0] - frames[0][0]).total_seconds()
    print(f"File FLIR: {len(frames)}   Primo: {frames[0][1].name}   "
          f"Ultimo: {frames[-1][1].name}")
    print(f"Durata totale (clock FLIR): {span:.0f} s ({span / 60:.1f} min)")
    print("Orari = clock FLIR: se data/ora della camera sono sbagliate valgono "
          "solo in relativo.")
    print()

    all_poses = split_on_gaps(frames, args.gap_threshold)
    poses = [p for p in all_poses if len(p) >= args.min_frames]
    dropped = len(all_poses) - len(poses)

    header = ["posa", "primo_file", "ultimo_file", "inizio", "fine",
              "durata_s", "n_frame"]
    rows = []
    for k, pose in enumerate(poses, start=1):
        (t0, f0), (t1, f1) = pose[0], pose[-1]
        rows.append([k, f0.name, f1.name, t0.strftime("%H:%M:%S"),
                     t1.strftime("%H:%M:%S"),
                     f"{(t1 - t0).total_seconds():.0f}", len(pose)])

    print(f"POSE: {len(poses)}   (nuova posa dopo un gap > {args.gap_threshold:g} s"
          + (f"; {dropped} con meno di {args.min_frames} frame scartate" if dropped else "")
          + ")")
    print_table(header, rows)

    if len(all_poses) == 1:
        print()
        print("NOTA: nessun gap trovato. Se la FLIR ha registrato in continuo le "
              "pose vanno cercate per differenza tra frame (detect_board_poses.py).")

    if args.csv_out:
        print()
        write_csv(args.csv_out, header, rows)


if __name__ == "__main__":
    main()
