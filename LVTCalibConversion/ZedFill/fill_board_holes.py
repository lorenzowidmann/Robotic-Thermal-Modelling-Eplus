"""
Riempie di bianco i 4 fori della board nei frame ZED di una posa, in modo che il
detector RGB di lvt2calib li accetti anche quando dietro un foro si vede la
maniglia (o la fuga fra le ante) dell'armadio bianco.

IL PROBLEMA
-----------
`cam_pattern` (lvt2calib) cerca il pattern con
`findCirclesGrid(2x2, SYMMETRIC_GRID + CLUSTERING)` e un `SimpleBlobDetector`
con `minCircularity 0.8`, `minInertiaRatio 0.1`, `minArea 50` e
`blobColor 255` (la board e' NERA, i fori mostrano il muro chiaro dietro).
Una striscia scura dentro un foro chiaro spezza il blob: la circolarita' crolla
e la posa non viene mai accettata.

PERCHE' NON BASTA "UN CERCHIO PIENO"
------------------------------------
Se la board non e' perfettamente frontale, in immagine i fori sono ELLISSI e i
4 centri NON formano un quadrato: non si puo' riempire con 4 cerchi uguali.
La soluzione e' lavorare nel PIANO DELLA BOARD, dove la geometria e' nota ed
esatta (4 fori di diametro 0.13 m ai vertici di un quadrato +/-0.15 m, lo stesso
modello fisico usato da RecordCheck/zed_pose_check.py::_spacing_ratio).
Immagine e piano sono legati da un'OMOGRAFIA H: stimata H, ogni foro si
ridisegna come proiezione esatta del cerchio del modello -- ellisse giusta,
nella posizione giusta, anche per il foro occluso, che geometricamente e'
determinato dagli altri tre.

COME
----
1. frame di riferimento = mediana di ~15 frame della posa (la board e' ferma
   dentro la posa: drift dei centri misurato <= 0.2 px);
2. detezione iniziale dei 4 centri riusando `detect_holes` di
   RecordCheck/zed_pose_check.py (top-hat + selezione a rettangolo + invariante
   fisico semi-diagonale/raggio ~3.26);
3. H iniziale dai 4 centri (la corrispondenza ciclica e' irrilevante: il
   quadrato e' invariante per il gruppo diedrale, quindi ogni rotazione o
   riflessione produce lo STESSO insieme di 4 ellissi);
4. raffinamento robusto: punti di bordo per scansione radiale (i raggi che
   incontrano l'occlusore vengono scartati da soli), poi fit congiunto di
   H (8 DOF) + scala del raggio s (1 DOF) su TUTTI i punti di bordo, con
   perdita robusta. Serve perche' il centro del blob di un foro occluso e'
   sbilanciato di 5-7 px;
5. riempimento delle 4 ellissi in bianco (sub-pixel, `shift=3`);
6. `--check`: riproduce ESATTAMENTE lvt2calib (undistort con
   zed_right_intrinsic.yaml + stessi parametri blob) e conta i frame in cui il
   grid viene trovato, PRIMA e DOPO il riempimento.

Misurato su NewCalibration\\Zed\\Poses (frame undistorti, parametri lvt2calib):
pose 01,03,04,06,09,10,12,15 -> grid trovato 0% sui frame originali, 100% dopo;
pose 19 -> 100% prima e dopo.

Output: cartella sorella `<pose>_filled` con `frames/` + `metadata.json`
copiato IDENTICO (lista e timing dei frame non cambiano), quindi utilizzabile
subito come `zed_frame_publisher.py --session-dir`. Gli originali non vengono
mai toccati.

Uso:
    py fill_board_holes.py --poses-root <...>\\Zed\\Poses --check
    py fill_board_holes.py --pose-dir  <...>\\Zed\\Poses\\pose_01 --check
    py fill_board_holes.py --pose-dir  <...>\\Poses\\pose_01 --dry-run
    py fill_board_holes.py --pose-dir  <...>\\Poses\\pose_01 --click
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("Serve opencv-python:  py -m pip install opencv-python")

try:
    from scipy.optimize import least_squares
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# riusa il detector dei fori gia' tarato su questa board (stessa polarita',
# stesso invariante fisico) invece di riscriverlo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "RecordCheck"))
from zed_pose_check import detect_holes  # noqa: E402

# --- modello fisico della board (metri) -------------------------------------
# fori a (+/-0.15, +/-0.15), diametro 0.13 -> raggio 0.065.
# Ordinati per angolo attorno al centro: -135, -45, +45, +135 gradi.
MODEL = np.array([[-0.15, -0.15],
                  [0.15, -0.15],
                  [0.15, 0.15],
                  [-0.15, 0.15]], dtype=np.float64)
HOLE_R = 0.065

# parametri blob IDENTICI a lvt2calib/src/camera/cam_pattern.cpp + cfg/Camera.cfg
LVT_MIN_AREA = 50.0
LVT_MIN_INERTIA = 0.1
LVT_MIN_CIRC = 0.8
LVT_BLOB_COLOR = 255


# ---------------------------------------------------------------- utilita' --
def bilinear(img, pts):
    """Campionamento bilineare di `img` (float32, HxW) nei punti (N,2) float."""
    h, w = img.shape[:2]
    x = np.clip(pts[..., 0], 0, w - 1.001)
    y = np.clip(pts[..., 1], 0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    fx = x - x0
    fy = y - y0
    i00 = img[y0, x0]
    i01 = img[y0, x0 + 1]
    i10 = img[y0 + 1, x0]
    i11 = img[y0 + 1, x0 + 1]
    return ((1 - fx) * (1 - fy) * i00 + fx * (1 - fy) * i01 +
            (1 - fx) * fy * i10 + fx * fy * i11)


def project(H, pts):
    """Applica l'omografia H a punti (..., 2)."""
    p = np.asarray(pts, dtype=np.float64)
    shape = p.shape
    p = p.reshape(-1, 2)
    q = H @ np.vstack([p.T, np.ones(len(p))])
    return (q[:2] / q[2]).T.reshape(shape)


def px_per_meter(H):
    """Scala locale (px per metro) al centro del pattern, per pesare i residui."""
    a = project(H, np.array([[0.0, 0.0]]))[0]
    b = project(H, np.array([[1e-3, 0.0]]))[0]
    c = project(H, np.array([[0.0, 1e-3]]))[0]
    return 0.5 * (np.linalg.norm(b - a) + np.linalg.norm(c - a)) / 1e-3


def circle_poly(H, center, radius, n=360):
    """Poligono immagine della proiezione del cerchio (center, radius) del modello."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([center[0] + radius * np.cos(t),
                    center[1] + radius * np.sin(t)], axis=1)
    return project(H, pts)


def ellipse_params(poly):
    """(cx, cy, asse_maggiore, asse_minore, angolo_deg) del poligono proiettato."""
    (cx, cy), (a, b), ang = cv2.fitEllipse(poly.astype(np.float32))
    return float(cx), float(cy), float(max(a, b)), float(min(a, b)), float(ang)


# ------------------------------------------------------- geometria iniziale --
def reference_gray(files, n_ref):
    """Mediana di n_ref frame campionati lungo la posa (grigio, float32)."""
    idx = np.unique(np.linspace(0, len(files) - 1, min(n_ref, len(files))).astype(int))
    stack = []
    for i in idx:
        img = cv2.imread(str(files[i]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        stack.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not stack:
        return None
    return np.median(np.stack(stack), axis=0).astype(np.float32)


def order_cyclic(pts):
    """Ordina 4 punti immagine per angolo attorno al loro centroide."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def initial_centers(gray, args):
    """4 centri fori (px, piena risoluzione) o None."""
    ds = args.hole_downscale
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    small = cv2.resize(g8, (g8.shape[1] // ds, g8.shape[0] // ds))
    res = detect_holes(small, args.hole_kernel, args.hole_min_area,
                       args.hole_max_area, args.hole_min_circ, args.rect_tol,
                       args.hole_spacing_ratio_min, args.hole_spacing_ratio_max)
    if res is None:
        return None
    return order_cyclic(np.asarray(res[0], dtype=np.float64) * ds)


def homography_from_centers(centers):
    H, _ = cv2.findHomography(MODEL.astype(np.float32),
                              np.asarray(centers, np.float32), 0)
    if H is None:
        return None
    return H / H[2, 2]


# --------------------------------------------------------- bordo dei fori ---
def rim_points(gray, H, s, n_rays=180, t0=0.55, t1=1.55, n_samples=48,
               min_contrast=40.0):
    """Punti di bordo (immagine) dei 4 fori + frazione di raggi utilizzati.

    Scansione radiale NEL PIANO DELLA BOARD (cosi' i raggi sono equispaziati
    sul cerchio vero, non sull'ellisse deformata). Su ogni raggio il bordo e'
    l'ULTIMO attraversamento del livello medio fra interno (chiaro) ed esterno
    (scuro): se l'occlusore sta DENTRO il foro il raggio torna chiaro dopo di
    esso, quindi l'ultimo attraversamento e' comunque il bordo vero. Un raggio
    viene scartato solo se non c'e' contrasto (occlusore che copre tutto
    l'interno lungo quel raggio) o se il bordo cade fuori da [0.75, 1.35]*R.
    I pochi raggi in cui l'occlusore invade il bordo restano outlier e li
    assorbe la perdita robusta del fit."""
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    t = np.linspace(t0, t1, n_samples)
    out_pts, fracs = [], []
    for mx, my in MODEL:
        r = s * HOLE_R * t
        mpts = np.stack([mx + np.outer(np.cos(theta), r),
                         my + np.outer(np.sin(theta), r)], axis=-1)
        prof = bilinear(gray, project(H, mpts))                    # (n_rays, n_samples)

        i_in = np.median(prof[:, t <= 0.80], axis=1)
        i_out = np.median(prof[:, t >= 1.35], axis=1)
        contrast = i_in - i_out
        mid = 0.5 * (i_in + i_out)
        above = prof > mid[:, None]

        pts, ok = [], 0
        for j in range(n_rays):
            if contrast[j] < min_contrast:
                continue
            bright = np.flatnonzero(above[j])
            if bright.size == 0:
                continue
            k = bright[-1]                        # ultimo campione sopra il livello medio
            if k + 1 >= n_samples or t[k] < 0.60 or t[k] > 1.35:
                continue
            a, b = prof[j, k], prof[j, k + 1]
            if a <= b:
                continue
            w = (a - mid[j]) / (a - b)
            tc = t[k] + w * (t[k + 1] - t[k])
            rr = s * HOLE_R * tc
            pts.append([mx + rr * np.cos(theta[j]), my + rr * np.sin(theta[j])])
            ok += 1
        mpts_ok = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out_pts.append(project(H, mpts_ok) if len(mpts_ok) else mpts_ok)
        fracs.append(ok / n_rays)
    return out_pts, np.asarray(fracs)


def clear_fraction(gray, H, s, n_rays=180, n_rad=12):
    """Per ogni foro, frazione dell'interno (r < 0.85 R) che e' davvero chiara.

    E' la misura di occlusione: 1.0 = foro completamente libero, valori
    piu' bassi = maniglia / fuga fra le ante visibile dentro il foro."""
    theta = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    t_in = np.linspace(0.10, 0.85, n_rad)
    t_out = np.linspace(1.30, 1.50, 4)
    fracs = []
    for mx, my in MODEL:
        def disc(tt):
            r = s * HOLE_R * tt
            return np.stack([mx + np.outer(np.cos(theta), r),
                             my + np.outer(np.sin(theta), r)], axis=-1)
        inner = bilinear(gray, project(H, disc(t_in)))
        outer = bilinear(gray, project(H, disc(t_out)))
        board = float(np.median(outer))
        hole = float(np.percentile(inner, 90))
        mid = 0.5 * (board + hole)
        fracs.append(float((inner > mid).mean()) if hole - board > 20 else 0.0)
    return np.asarray(fracs)


# ------------------------------------------------------------ raffinamento --
def _pack(H, s):
    return np.concatenate([(H / H[2, 2]).ravel()[:8], [s]])


def _unpack(x):
    H = np.array([[x[0], x[1], x[2]],
                  [x[3], x[4], x[5]],
                  [x[6], x[7], 1.0]])
    return H, x[8]


def refine_scipy(gray, H0, s0, min_pts=40):
    """Fit congiunto di H (8 DOF) + scala del raggio s su tutti i punti di bordo.

    Residuo di ogni punto: retroproiettato nel piano board, deve stare sul
    cerchio del modello -> | H^-1(p) - centro_i | - s*R, in pixel equivalenti.
    `loss='soft_l1'` scarta da sola i punti sopravvissuti ma contaminati."""
    H, s = H0.copy(), s0
    note = ""
    for _ in range(2):
        pts, _ = rim_points(gray, H, s)
        n_tot = sum(len(p) for p in pts)
        if n_tot < min_pts:
            return H, s, "pochi punti di bordo: raffinamento saltato"
        idx = np.concatenate([np.full(len(p), i) for i, p in enumerate(pts)])
        allp = np.vstack([p for p in pts if len(p)])
        k = px_per_meter(H)
        centers = MODEL[idx]

        def resid(x):
            Hx, sx = _unpack(x)
            q = project(np.linalg.inv(Hx), allp)
            return (np.linalg.norm(q - centers, axis=1) - sx * HOLE_R) * k

        sol = least_squares(resid, _pack(H, s), loss="soft_l1",
                            f_scale=2.0, max_nfev=200)
        H, s = _unpack(sol.x)
        med = float(np.median(np.abs(resid(sol.x))))
        note = f"scarto mediano bordo={med:.2f}px su {allp.shape[0]} punti"
    return H, s, note


def _support(gray, H, s, center, offset):
    """Frazione di raggi con anello interno chiaro ed esterno scuro."""
    inner = circle_poly(H, center, 0.80 * s * HOLE_R, 180) + offset
    outer = circle_poly(H, center, 1.25 * s * HOLE_R, 180) + offset
    return float(((bilinear(gray, inner) > 110) & (bilinear(gray, outer) < 90)).mean())


def refine_numpy(gray, H0, s0):
    """Fallback senza scipy: ricerca dell'offset intero (+/-8 px) di ogni foro
    che massimizza il supporto del bordo, poi rifit di H sui centri corretti;
    infine ricerca 1D della scala s."""
    H, s = H0.copy(), s0
    for _ in range(2):
        centers = project(H, MODEL)
        moved = []
        for i, m in enumerate(MODEL):
            best = (-1.0, np.zeros(2))
            for dx in range(-8, 9):
                for dy in range(-8, 9):
                    off = np.array([dx, dy], float)
                    v = _support(gray, H, s, m, off)
                    if v > best[0]:
                        best = (v, off)
            moved.append(centers[i] + best[1])
        Hn = homography_from_centers(np.asarray(moved))
        if Hn is None:
            break
        H = Hn
        best = (-1.0, s)
        for cand in np.arange(0.80, 1.21, 0.01):
            v = float(np.mean([_support(gray, H, cand, m, np.zeros(2)) for m in MODEL]))
            if v > best[0]:
                best = (v, float(cand))
        s = best[1]
    return H, s, "fallback numpy (scipy non installato)"


# ----------------------------------------------------------- riempimento ----
SHIFT = 3  # 1/8 px


def fill_holes(img, H, s, fill_value, n=360):
    """Disegna in `img` (in place) le 4 ellissi piene."""
    color = (fill_value, fill_value, fill_value) if img.ndim == 3 else fill_value
    for m in MODEL:
        poly = circle_poly(H, m, s * HOLE_R, n)
        cv2.fillPoly(img, [np.round(poly * (1 << SHIFT)).astype(np.int32)],
                     color, lineType=cv2.LINE_AA, shift=SHIFT)
    return img


# -------------------------------------------------- verifica stile lvt2calib --
def blob_detector():
    p = cv2.SimpleBlobDetector_Params()
    p.filterByArea = True
    p.minArea = LVT_MIN_AREA
    p.maxArea = 1e9
    p.filterByInertia = True
    p.minInertiaRatio = LVT_MIN_INERTIA
    p.filterByCircularity = True
    p.minCircularity = LVT_MIN_CIRC
    p.blobColor = LVT_BLOB_COLOR
    return cv2.SimpleBlobDetector_create(p)


def load_undistort_maps(path, size=(1920, 1080)):
    """Stessi map1/map2 di cam_pattern.cpp: initUndistortRectifyMap(K,D,None,K)."""
    if path is None or not Path(path).is_file():
        return None
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("CameraMat").mat()
    D = fs.getNode("DistCoeff").mat()
    sz = fs.getNode("ImageSize")          # sequenza [w, h], non una matrice
    if sz is not None and sz.isSeq() and sz.size() == 2:
        size = (int(sz.at(0).real()), int(sz.at(1).real()))
    fs.release()
    if K is None or D is None:
        return None
    return cv2.initUndistortRectifyMap(K, D, None, K, size, cv2.CV_16SC2)


def grid_found(gray_u8, maps, detector):
    g = cv2.remap(gray_u8, maps[0], maps[1], cv2.INTER_LINEAR) if maps else gray_u8
    ok, _ = cv2.findCirclesGrid(
        g, (2, 2),
        flags=cv2.CALIB_CB_SYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
        blobDetector=detector)
    return bool(ok)


# --------------------------------------------------------------- click mode --
def click_centers(bgr):
    """Click sui 4 centri dei fori (ordine libero). ESC/q per annullare."""
    win = "click sui 4 centri dei fori  (r=ricomincia, q=annulla)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, bgr.shape[1]), min(900, bgr.shape[0]))
    pts = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((float(x), float(y)))

    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = bgr.copy()
        for i, (x, y) in enumerate(pts):
            cv2.drawMarker(vis, (int(x), int(y)), (0, 165, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(vis, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            cv2.destroyAllWindows()
            return None
        if key == ord('r'):
            pts.clear()
        if len(pts) == 4:
            cv2.destroyAllWindows()
            return order_cyclic(np.asarray(pts, dtype=np.float64))


# ------------------------------------------------------------------ debug ----
def write_debug(path, bgr, H, s, rim, fracs):
    vis = bgr.copy()
    for i, m in enumerate(MODEL):
        poly = np.round(circle_poly(H, m, s * HOLE_R)).astype(np.int32)
        cv2.polylines(vis, [poly], True, (0, 255, 0), 2, cv2.LINE_AA)
        c = project(H, np.array([m]))[0]
        cv2.drawMarker(vis, (int(round(c[0])), int(round(c[1]))),
                       (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        for p in rim[i]:
            cv2.circle(vis, (int(round(p[0])), int(round(p[1]))), 1, (0, 255, 255), -1)
        cv2.putText(vis, f"{fracs[i]:.2f}", (int(c[0]) + 26, int(c[1]) - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


# ------------------------------------------------------------- una posa -----
def process_pose(pose_dir, args, maps, detector):
    pose_dir = Path(pose_dir)
    frames_dir = pose_dir / "frames"
    meta = pose_dir / "metadata.json"
    if not frames_dir.is_dir():
        return {"pose": pose_dir.name, "esito": "SALTATA (nessuna frames/)"}
    files = sorted(frames_dir.glob("*.png"))
    if not files:
        return {"pose": pose_dir.name, "esito": "SALTATA (nessun PNG)"}

    gray = reference_gray(files, args.ref_frames)
    if gray is None:
        return {"pose": pose_dir.name, "esito": "SALTATA (frame illeggibili)"}

    ref_bgr = cv2.imread(str(files[len(files) // 2]), cv2.IMREAD_COLOR)
    if args.click:
        centers = click_centers(ref_bgr)
        if centers is None:
            return {"pose": pose_dir.name, "esito": "ANNULLATA (click)"}
    else:
        centers = initial_centers(gray, args)
        if centers is None:
            return {"pose": pose_dir.name,
                    "esito": "FALLITA (fori non rilevati; riprovare con --click)"}

    H = homography_from_centers(centers)
    if H is None:
        return {"pose": pose_dir.name, "esito": "FALLITA (omografia degenere)"}

    if args.no_refine:
        s, note = 1.0, "raffinamento disattivato (--no-refine)"
    elif HAVE_SCIPY:
        H, s, note = refine_scipy(gray, H, 1.0)
    else:
        H, s, note = refine_numpy(gray, H, 1.0)
    s_fill = s * args.radius_scale
    clear = clear_fraction(gray, H, s)

    out_dir = (Path(args.outdir) if args.outdir
               else pose_dir.with_name(pose_dir.name + args.suffix))
    out_frames = out_dir / "frames"
    if args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_frames.is_dir() and any(out_frames.glob("*.png")) and not args.force:
            return {"pose": pose_dir.name,
                    "esito": f"FALLITA ({out_frames} gia' piena; usare --force)"}
        out_frames.mkdir(parents=True, exist_ok=True)

    rim, rays = rim_points(gray, H, s)
    write_debug(out_dir / "debug.png", ref_bgr, H, s_fill, rim, clear)

    holes = []
    for i, m in enumerate(MODEL):
        cx, cy, maj, mino, ang = ellipse_params(circle_poly(H, m, s_fill * HOLE_R))
        holes.append({"modello_m": [float(m[0]), float(m[1])],
                      "centro_px": [cx, cy],
                      "assi_px": [maj, mino],
                      "angolo_deg": ang,
                      "frazione_foro_libero": float(clear[i]),
                      "frazione_raggi_bordo": float(rays[i]),
                      "occluso": bool(clear[i] < args.occluded_frac)})
    (out_dir / "holes.json").write_text(json.dumps(
        {"pose": pose_dir.name, "H": H.tolist(), "scala_raggio": float(s),
         "radius_scale": float(args.radius_scale), "raggio_modello_m": HOLE_R,
         "frame_riferimento": files[len(files) // 2].name,
         "nota_fit": note, "fori": holes}, indent=2), encoding="utf-8")

    n_raw = n_fill = n_chk = 0
    written = 0
    for k, f in enumerate(files):
        img = cv2.imread(str(f), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ATTENZIONE: illeggibile, saltato: {f.name}")
            continue
        filled = fill_holes(img.copy(), H, s_fill, args.fill)
        if args.check and k % args.check_step == 0:
            n_chk += 1
            n_raw += grid_found(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), maps, detector)
            n_fill += grid_found(cv2.cvtColor(filled, cv2.COLOR_BGR2GRAY), maps, detector)
        if not args.dry_run:
            cv2.imwrite(str(out_frames / f.name), filled)
            written += 1
    if not args.dry_run and meta.is_file():
        shutil.copy2(meta, out_dir / "metadata.json")

    return {"pose": pose_dir.name,
            "out": str(out_dir),
            "frame": f"{written}/{len(files)}" if not args.dry_run else "dry-run",
            "occlusi": sum(h["occluso"] for h in holes),
            "libero": " ".join(f"{v:.2f}" for v in clear),
            "s": f"{s:.3f}",
            "check": f"{n_raw}/{n_chk} -> {n_fill}/{n_chk}" if args.check else "-",
            "esito": "OK", "nota": note}


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(
        description="Riempie di bianco i 4 fori della board nei frame ZED di una posa.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pose-dir", help="Una cartella posa (frames/ + metadata.json)")
    src.add_argument("--poses-root",
                     help="Cartella con piu' pose: elabora tutte le 'pose_*'")
    ap.add_argument("--outdir", default=None,
                    help="Solo con --pose-dir: cartella di output (default <posa>_filled)")
    ap.add_argument("--suffix", default="_filled",
                    help="Suffisso della cartella di output (default _filled)")
    ap.add_argument("--force", action="store_true",
                    help="Sovrascrive una cartella di output gia' popolata")
    ap.add_argument("--dry-run", action="store_true",
                    help="Calcola geometria, debug.png, holes.json e --check "
                         "senza scrivere i frame")
    # geometria
    ap.add_argument("--ref-frames", type=int, default=15,
                    help="Frame campionati per la mediana di riferimento (default 15)")
    ap.add_argument("--fill", type=int, default=255,
                    help="Valore di riempimento 0-255 (default 255, bianco)")
    ap.add_argument("--radius-scale", type=float, default=1.0,
                    help="Moltiplicatore extra sul raggio stimato (default 1.0)")
    ap.add_argument("--no-refine", action="store_true",
                    help="Usa solo l'omografia dai centri rilevati, senza raffinamento")
    ap.add_argument("--occluded-frac", type=float, default=0.95,
                    help="Sotto questa frazione di interno libero il foro e' marcato "
                         "occluso (default 0.95)")
    ap.add_argument("--click", action="store_true",
                    help="Fallback manuale: click sui 4 centri sul frame di riferimento")
    # detezione iniziale (stessi default di RecordCheck/zed_pose_check.py)
    ap.add_argument("--hole-downscale", type=int, default=2)
    ap.add_argument("--hole-kernel", type=int, default=61)
    ap.add_argument("--hole-min-area", type=float, default=150.0)
    ap.add_argument("--hole-max-area", type=float, default=8000.0)
    ap.add_argument("--hole-min-circ", type=float, default=0.45)
    ap.add_argument("--rect-tol", type=float, default=0.35)
    ap.add_argument("--hole-spacing-ratio-min", type=float, default=2.0)
    ap.add_argument("--hole-spacing-ratio-max", type=float, default=6.0)
    # verifica
    ap.add_argument("--check", action="store_true",
                    help="Conta i frame in cui lvt2calib troverebbe il grid, prima e dopo")
    ap.add_argument("--check-step", type=int, default=20,
                    help="Un frame ogni N per --check (default 20)")
    ap.add_argument("--camera-info", default=None,
                    help="YAML intrinseci per l'undistort di --check "
                         "(default lvt2calib/data/camera_info/zed_right_intrinsic.yaml)")
    args = ap.parse_args()

    if args.outdir and args.poses_root:
        ap.error("--outdir vale solo con --pose-dir")

    maps = detector = None
    if args.check:
        cam = args.camera_info
        if cam is None:
            cam = (Path(__file__).resolve().parents[3] / "lvt2calib" / "data" /
                   "camera_info" / "zed_right_intrinsic.yaml")
        maps = load_undistort_maps(cam)
        if maps is None:
            print(f"ATTENZIONE: intrinseci non trovati ({cam}) -> --check senza undistort")
        else:
            print(f"--check: undistort con {cam}")
        detector = blob_detector()

    if args.pose_dir:
        poses = [Path(args.pose_dir)]
    else:
        poses = sorted(p for p in Path(args.poses_root).iterdir()
                       if p.is_dir() and p.name.startswith("pose_")
                       and not p.name.endswith(args.suffix))
        if not poses:
            sys.exit(f"Nessuna cartella 'pose_*' in {args.poses_root}")

    if not HAVE_SCIPY and not args.no_refine:
        print("ATTENZIONE: scipy non installato -> raffinamento con fallback numpy "
              "(py -m pip install scipy per il fit congiunto)")

    rows = []
    for p in poses:
        print(f"\n=== {p.name} ===")
        r = process_pose(p, args, maps, detector)
        rows.append(r)
        if r["esito"] != "OK":
            print(f"  {r['esito']}")
            continue
        print(f"  fori occlusi: {r['occlusi']}   foro libero: {r['libero']}   "
              f"scala raggio: {r['s']}   ({r['nota']})")
        print(f"  frame scritti: {r['frame']} -> {r['out']}")
        if args.check:
            print(f"  grid lvt2calib (RAW -> FILLED): {r['check']}")

    print("\n" + "=" * 78)
    hdr = f"{'posa':<12} {'esito':<10} {'occl':>4} {'frame':>9}"
    if args.check:
        hdr += "   grid RAW -> FILLED"
    print(hdr)
    print("-" * 78)
    for r in rows:
        line = (f"{r['pose']:<12} {str(r.get('esito', ''))[:10]:<10} "
                f"{str(r.get('occlusi', '')):>4} {str(r.get('frame', '')):>9}")
        if args.check:
            line += f"   {r.get('check', '-')}"
        print(line)


if __name__ == "__main__":
    main()
