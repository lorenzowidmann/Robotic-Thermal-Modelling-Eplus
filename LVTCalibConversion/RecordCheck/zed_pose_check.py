"""
Per ogni posa rilevata in una sessione ZED (stessa segmentazione di
zed_pose_detect.py), controlla se la board di calibrazione a quattro fori e'
davvero presente e interamente dentro l'inquadratura -- non solo se la scena
e' "ferma" (che zed_pose_detect.py da solo non distingue da un cavalletto
vuoto, vedi il caso reale trovato a mano il 2026-09-11: pose 2/5/13/16 erano
solo il cavalletto senza board).

Equivalente RGB di Thesis/Calibration/flir_pose_check.py. Stessa idea, stessa
selezione dei 4 fori per geometria a rettangolo (_rect_score identica), ma con
DUE differenze rispetto alla FLIR:

1. polarita' di contrasto OPPOSTA:

  - FLIR (termico): pannello e muro sono quasi alla stessa temperatura, i fori
    risultano PIU' SCURI del pannello -> MORPH_BLACKHAT + edge-scan che cerca
    un CALO di intensita' dal pannello verso lo sfondo.
  - ZED (RGB, questa board): il pannello e' nero/molto scuro, i fori mostrano
    lo sfondo chiaro dietro (muro/porta bianca) -> qui i fori sono PIU'
    CHIARI del pannello -> MORPH_TOPHAT + edge-scan che cerca un AUMENTO di
    intensita' dal pannello verso lo sfondo (--edge-rise invece di
    --edge-drop).

2. la FLIR inquadra quasi solo la board (campo stretto): la sola geometria
   "4 punti a rettangolo" basta. La ZED inquadra anche muro/porta/oggetti di
   sfondo (maniglie, finestra, ganci), che a volte passano comunque il test
   del rettangolo -- scoperto sui dati reali il 2026-09-11 (pose senza board
   marcate "INSIDE" per errore). Aggiunto quindi _spacing_ratio: il rapporto
   fra la semi-diagonale del pattern e il raggio dei blob e' un invariante
   FISICO della board reale (fori a +/-0.15,+/-0.15 m, diametro 0.13 m -> atteso
   ~3.26, indipendente da distanza/scala), --hole-spacing-ratio-min/-max.

La segmentazione delle pose e' quella di zed_pose_detect.py (stessa funzione,
importata da qui): stessi flag, stessi default, cosi' i numeri di posa
combaciano tra i due script sulla stessa sessione.

Lo script legge solo: nessun file viene copiato, spostato o modificato.

Uso tipico:
    py zed_pose_check.py <sessione_zed>
    py zed_pose_check.py <sessione_zed> --min-move-frames 12 --min-duration 45
    py zed_pose_check.py <sessione_zed> --debug-dir out_debug --csv-out zed_check.csv
"""

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Serve opencv-python:  py -m pip install opencv-python")

# riusa la segmentazione di zed_pose_detect.py (stessa cartella)
from zed_pose_detect import load_session, diff_profile, segment


def detect_holes(gray, hole_kernel, min_area, max_area, min_circ, rect_tol,
                 ratio_min, ratio_max):
    """Rileva i 4 fori della board. Ritorna (pts(4,2), centro(2,)) o None.

    Top-hat: isola le macchie CHIARE (i fori, che mostrano lo sfondo chiaro)
    piu' piccole del kernel, su un pannello scuro -- l'opposto del black-hat
    usato per la FLIR (li' i fori sono piu' scuri del pannello)."""
    gb = cv2.GaussianBlur(gray, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hole_kernel, hole_kernel))
    th = cv2.morphologyEx(gb, cv2.MORPH_TOPHAT, k)
    th = cv2.normalize(th, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cand = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        circ = area / (np.pi * r * r)
        if circ < min_circ:
            continue
        cand.append((area, x, y, circ, r))
    return select_holes(cand, rect_tol, ratio_min, ratio_max)


def _rect_score(p):
    """Punteggio di 'quanto i 4 punti sembrano i vertici di un rettangolo
    centrato' (identica a flir_pose_check.py: geometria pura, non dipende dal
    sensore). Piu' basso = migliore."""
    m = p.mean(axis=0)
    v = p - m
    r = np.linalg.norm(v, axis=1)
    rmean = float(r.mean())
    if rmean < 1e-3:
        return 1e9
    rad_err = float(r.std()) / rmean
    sym_err = 0.0
    for i in range(4):
        d = np.linalg.norm(v + v[i], axis=1)
        d[i] = np.inf
        sym_err += float(d.min())
    sym_err /= 4.0 * rmean
    return rad_err + sym_err


def _spacing_ratio(p, radii):
    """Rapporto tra la semi-diagonale del pattern (distanza media punti-centro)
    e il raggio medio dei blob. Per la board reale (fori a +/-0.15,+/-0.15 m,
    diametro 0.13 m) vale fisicamente ~3.26, indipendente dalla distanza/scala
    -- quindi separa i 4 fori veri da 4 oggetti di sfondo (maniglie, finestra,
    ganci) che passano il solo _rect_score ma non hanno questa proporzione."""
    ctr = p.mean(axis=0)
    rmean = float(np.linalg.norm(p - ctr, axis=1).mean())
    rad = float(np.mean(radii))
    return rmean / rad if rad > 1e-6 else 1e9


def select_holes(cand, rect_tol, ratio_min, ratio_max):
    """Sceglie i 4 fori della board (geometria di base identica a
    flir_pose_check.py, con l'aggiunta del controllo fisico di spaziatura
    _spacing_ratio -- la FLIR non ne ha bisogno perche' la board occupa quasi
    tutto il campo stretto della termica; la ZED inquadra anche muro/porta/
    oggetti di sfondo che senza questo controllo vengono scambiati per fori).
    Ritorna (pts(4,2), centro(2,)) o None."""
    if len(cand) < 3:
        return None
    cand = sorted(cand, key=lambda t: -t[3])[:8]
    areas = np.array([a for a, _, _, _, _ in cand], dtype=float)
    pts_all = np.array([[x, y] for _, x, y, _, _ in cand], dtype=np.float32)
    radii_all = np.array([r for _, _, _, _, r in cand], dtype=float)
    n = len(cand)

    def area_ok(ii):
        aa = areas[ii]
        return aa.max() / max(aa.min(), 1e-6) <= 3.5

    best = None
    for idx in itertools.combinations(range(n), 4):
        ii = list(idx)
        if not area_ok(ii):
            continue
        p = pts_all[ii]
        if p.std(axis=0).max() < 4.0:
            continue
        ratio = _spacing_ratio(p, radii_all[ii])
        if not (ratio_min <= ratio <= ratio_max):
            continue
        s = _rect_score(p)
        if best is None or s < best[0]:
            best = (s, p)
    if best is not None and best[0] <= rect_tol:
        return best[1], best[1].mean(axis=0)

    best3 = None
    for idx in itertools.combinations(range(n), 3):
        ii = list(idx)
        if not area_ok(ii):
            continue
        p3 = pts_all[ii]
        for vtx in range(3):
            o = [j for j in range(3) if j != vtx]
            va = p3[o[0]] - p3[vtx]
            vb = p3[o[1]] - p3[vtx]
            la, lb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if la < 6 or lb < 6:
                continue
            if abs(float(va @ vb)) / (la * lb) > 0.25:
                continue
            p4 = p3[o[0]] + p3[o[1]] - p3[vtx]
            quad = np.array([p3[0], p3[1], p3[2], p4], dtype=np.float32)
            ratio = _spacing_ratio(quad, radii_all[ii])  # 4a radice ricostruita, ignorata
            if not (ratio_min <= ratio <= ratio_max):
                continue
            s = _rect_score(quad)
            if best3 is None or s < best3[0]:
                best3 = (s, quad)
    if best3 is not None and best3[0] <= rect_tol:
        return best3[1], best3[1].mean(axis=0)
    return None


def _scan_edge(gray, cx, cy, dx, dy, i0, rise, run=3):
    """Cammina da (cx,cy) lungo (dx,dy). Ritorna la distanza dal bordo della
    board (salto di intensita' sopra i0+rise per `run` px -- lo sfondo e' piu'
    CHIARO del pannello, opposto della FLIR) o None se il raggio esce dal
    frame restando ancora sul pannello."""
    h_img, w_img = gray.shape[:2]
    cnt = 0
    t = 1
    while True:
        x = int(round(cx + dx * t))
        y = int(round(cy + dy * t))
        if x < 0 or x >= w_img or y < 0 or y >= h_img:
            return None
        if gray[y, x] > i0 + rise:
            cnt += 1
            if cnt >= run:
                return t - run + 1
        else:
            cnt = 0
        t += 1


def board_extent(gray, ctr, rise):
    """Semi-estensioni (hx, hy) della board attorno al centro fori, tramite
    scansione simmetrica. Ritorna dict con hx, hy (o None), i0."""
    cx, cy = float(ctr[0]), float(ctr[1])
    x0, y0 = int(round(cx)), int(round(cy))
    patch = gray[max(0, y0 - 2):y0 + 3, max(0, x0 - 2):x0 + 3]
    i0 = float(np.median(patch)) if patch.size else float(gray[y0, x0])

    eL = _scan_edge(gray, cx, cy, -1, 0, i0, rise)
    eR = _scan_edge(gray, cx, cy, 1, 0, i0, rise)
    eU = _scan_edge(gray, cx, cy, 0, -1, i0, rise)
    eD = _scan_edge(gray, cx, cy, 0, 1, i0, rise)

    def half(a, b):
        vals = [e for e in (a, b) if e is not None]
        return sum(vals) / len(vals) if vals else None

    return {"i0": i0, "hx": half(eL, eR), "hy": half(eU, eD),
            "edges": (eL, eR, eU, eD)}


def board_inside(gray, ctr, ext, margin):
    """(inside_bool_or_None, lati_tagliati). None = inconcludente."""
    h_img, w_img = gray.shape[:2]
    cx, cy = float(ctr[0]), float(ctr[1])
    hx, hy = ext["hx"], ext["hy"]
    if hx is None or hy is None:
        return None, []
    sides = []
    if cx - hx < margin:
        sides.append("L")
    if cx + hx > w_img - 1 - margin:
        sides.append("R")
    if cy - hy < margin:
        sides.append("U")
    if cy + hy > h_img - 1 - margin:
        sides.append("D")
    return (len(sides) == 0), sides


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
        description="Controlla se la board e' presente e interamente in "
                    "frame in ogni posa ZED rilevata."
    )
    ap.add_argument("session",
                    help="Cartella sessione di zed_record.py (con metadata.json e frames/)")
    ap.add_argument("--eye", choices=("right", "left"), default="right",
                    help="Occhio da analizzare (default right)")
    # --- segmentazione (stessi default di zed_pose_detect.py) ---
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--mad-k", type=float, default=6.0)
    ap.add_argument("--min-move-frames", type=int, default=3)
    ap.add_argument("--min-pose-frames", type=int, default=15)
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--downscale", type=int, default=2,
                    help="Downscale per il PASSO DI SEGMENTAZIONE (frame-diff, default 2)")
    ap.add_argument("--fps", type=float, default=30.0)
    # --- il controllo vero e proprio ---
    ap.add_argument("--min-frames", type=int, default=60,
                    help="Controlla solo le pose con almeno N frame (default 60)")
    ap.add_argument("--step", type=int, default=5,
                    help="Analizza un frame ogni N nella posa (default 5)")
    ap.add_argument("--margin-px", type=int, default=3,
                    help="Margine di bordo richiesto, in pixel (default 3)")
    ap.add_argument("--inside-frac", type=float, default=0.9,
                    help="Posa 'INSIDE' se >= questa frazione dei frame CON fori "
                         "rilevati e' interamente in cornice (default 0.90)")
    ap.add_argument("--detect-frac", type=float, default=0.5,
                    help="Posa inconcludente se i fori sono rilevati in meno di "
                         "questa frazione dei frame analizzati (default 0.50)")
    # --- rilevamento fori (default tarati sulla board reale 1920x1080, "
    #     "fori ~90-100px -> ~45-50px a downscale 2) ---
    ap.add_argument("--hole-downscale", type=int, default=2,
                    help="Downscale delle immagini prima del rilevamento fori (default 2)")
    ap.add_argument("--hole-kernel", type=int, default=61,
                    help="Kernel top-hat (px), deve superare il diametro dei fori "
                         "a downscale --hole-downscale (default 61)")
    ap.add_argument("--hole-min-area", type=float, default=150.0,
                    help="Area minima blob foro in px^2 (default 150)")
    ap.add_argument("--hole-max-area", type=float, default=8000.0,
                    help="Area massima blob foro in px^2 (default 8000, la board "
                         "cambia molto distanza da camera tra le pose)")
    ap.add_argument("--hole-min-circ", type=float, default=0.45,
                    help="Circolarita' minima per un candidato foro (default 0.45)")
    ap.add_argument("--rect-tol", type=float, default=0.35,
                    help="Punteggio 'rettangolo' massimo per i 4 fori (default 0.35)")
    ap.add_argument("--hole-spacing-ratio-min", type=float, default=2.0,
                    help="Rapporto minimo semi-diagonale/raggio-foro (fisico atteso "
                         "~3.26 per la board reale; scarta oggetti di sfondo troppo "
                         "vicini tra loro rispetto alla loro dimensione, default 2.0)")
    ap.add_argument("--hole-spacing-ratio-max", type=float, default=6.0,
                    help="Rapporto massimo semi-diagonale/raggio-foro (scarta oggetti "
                         "di sfondo sparsi su tutta l'inquadratura, default 6.0)")
    ap.add_argument("--edge-rise", type=float, default=40.0,
                    help="Salto di intensita' (0-255) dal pannello (scuro) allo "
                         "sfondo (chiaro) che segna il bordo board (default 40)")
    ap.add_argument("--debug-dir", default=None,
                    help="Se impostato, salva un frame annotato per posa qui")
    ap.add_argument("--csv-out", default=None, metavar="CSV",
                    help="Salva anche la tabella in CSV")
    args = ap.parse_args()
    if args.step < 1 or args.hole_downscale < 1:
        ap.error("--step e --hole-downscale devono essere >= 1")

    entries, started, source = load_session(args.session, args.eye, args.fps)
    if len(entries) < 2:
        sys.exit(f"Servono almeno 2 frame '{args.eye}' in {args.session}.")
    print(f"Frame '{args.eye}': {len(entries)}   Tempi: {source}")

    values, idx = diff_profile(entries, args.frame_stride, args.downscale)
    if values.size == 0:
        sys.exit("Nessuna differenza calcolabile.")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = (args.threshold if args.threshold is not None
                 else median + args.mad_k * (mad if mad > 1e-6 else 1.0))
    poses, _ = segment(values, threshold, args.min_move_frames, args.min_pose_frames)

    pose_ranges = []
    for a, b in poses:
        i0, i1 = idx[a], idx[b - 1]
        t0, t1 = entries[i0][0], entries[i1][0]
        if args.min_duration is not None and t1 - t0 < args.min_duration:
            continue
        pose_ranges.append((i0, i1))

    long_poses = [(i0, i1) for i0, i1 in pose_ranges if (i1 - i0 + 1) >= args.min_frames]
    print(f"Pose rilevate: {len(pose_ranges)}   con >= {args.min_frames} frame: {len(long_poses)}")
    print()

    debug_dir = None
    if args.debug_dir:
        debug_dir = Path(args.debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

    header_screen = ["#", "offset_inizio_s", "offset_fine_s", "n_frame", "det", "inside", "verdetto"]
    print("=" * 100)
    print(f"{'#':>3}  {'inizio_s':>9} {'fine_s':>9} {'frame':>6}  {'det':>6} {'inside':>7}  verdetto")
    print("-" * 100)

    header = ["posa", "primo_file", "ultimo_file", "offset_inizio_s", "offset_fine_s",
              "n_frame", "n_analizzati", "n_rilevati", "n_inside", "verdetto"]
    rows = []
    for k, (i0, i1) in enumerate(long_poses, start=1):
        pose_files = [entries[i][1] for i in range(i0, i1 + 1)]
        sample = pose_files[::args.step] or pose_files

        n_analyzed = len(sample)
        n_detected = 0
        n_inside = 0
        clip_tally = {}
        first_annot = None

        for f in sample:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if args.hole_downscale > 1:
                img = cv2.resize(img, (img.shape[1] // args.hole_downscale,
                                       img.shape[0] // args.hole_downscale))
            hole = detect_holes(img, args.hole_kernel, args.hole_min_area,
                                args.hole_max_area, args.hole_min_circ, args.rect_tol,
                                args.hole_spacing_ratio_min, args.hole_spacing_ratio_max)
            if hole is None:
                continue
            n_detected += 1
            pts, ctr = hole
            ext = board_extent(img, ctr, args.edge_rise)
            inside, sides = board_inside(img, ctr, ext, args.margin_px)
            if inside:
                n_inside += 1
            for s in sides:
                clip_tally[s] = clip_tally.get(s, 0) + 1
            if debug_dir is not None and first_annot is None:
                vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for (px, py) in pts.astype(int):
                    cv2.circle(vis, (px, py), 4, (0, 165, 255), -1)
                col = (0, 200, 0) if inside else (0, 0, 255)
                if ext["hx"] is not None and ext["hy"] is not None:
                    cx, cy = int(ctr[0]), int(ctr[1])
                    hx, hy = int(ext["hx"]), int(ext["hy"])
                    cv2.rectangle(vis, (cx - hx, cy - hy), (cx + hx, cy + hy), col, 2)
                first_annot = (f.stem, vis)

        if n_analyzed == 0 or n_detected < args.detect_frac * n_analyzed:
            verdict = "INCONCLUSIVE (fori raramente rilevati)"
        else:
            frac_inside = n_inside / n_detected
            if frac_inside >= args.inside_frac:
                verdict = "INSIDE"
            else:
                worst = "".join(sorted(clip_tally, key=lambda s: -clip_tally[s]))
                verdict = f"CLIPPED [{worst}]"

        t0, t1 = entries[i0][0], entries[i1][0]
        print(f"{k:>3}  {t0:>9.1f} {t1:>9.1f} {i1 - i0 + 1:>6}  "
              f"{n_detected:>3}/{n_analyzed:<2} {n_inside:>3}/{max(n_detected,1):<3}  {verdict}")

        rows.append([k, pose_files[0].name, pose_files[-1].name, f"{t0:.1f}", f"{t1:.1f}",
                     i1 - i0 + 1, n_analyzed, n_detected, n_inside, verdict])

        if debug_dir is not None and first_annot is not None:
            out = debug_dir / f"pose{k:02d}_{first_annot[0]}.png"
            cv2.imwrite(str(out), first_annot[1])

    print("=" * 100)
    inside_ok = [r[0] for r in rows if r[-1] == "INSIDE"]
    print(f"Pose con target interamente in cornice: {len(inside_ok)} / {len(long_poses)}  -> {inside_ok}")
    print()
    print("NOTE:")
    print("  - 'det'    = frame dove i 4 fori sono stati rilevati / frame analizzati (--step).")
    print("  - 'inside' = frame rilevati la cui board risulta interamente in cornice.")
    print("  - CLIPPED [lati]: L/R/U/D = bordi immagine attraversati dalla board.")
    print("  - INCONCLUSIVE: fori raramente rilevati -> probabile cavalletto vuoto (nessuna "
          "board) o board fuori scala rispetto a --hole-min-area/--hole-max-area.")
    print("  - Verifica con --debug-dir (pallini arancioni = fori, box verde = dentro, "
          "box rosso = tagliata).")

    if args.csv_out:
        print()
        write_csv(args.csv_out, header, rows)


if __name__ == "__main__":
    main()
