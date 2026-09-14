# ZedFill

`fill_board_holes.py` — riempie di bianco i **4 fori della board** nei frame ZED
di una posa, così che il detector RGB di LVT2Calib li accetti anche quando
dietro un foro si vede la **maniglia / fuga fra le ante** dell'armadio bianco.

Gira sull'host Windows (plain `pip`), non nel container.

## Il problema

`cam_pattern` (lvt2calib) cerca il pattern con
`findCirclesGrid(2x2, SYMMETRIC_GRID + CLUSTERING)` e un `SimpleBlobDetector`
configurato in `cfg/Camera.cfg`:

| parametro | valore |
|---|---|
| `minCircularity` | 0.8 |
| `minInertiaRatio` | 0.1 |
| `minArea` | 50 |
| `blobColor` | 255 (`use_darkboard`: la board è NERA, i fori mostrano il muro chiaro) |

Una striscia scura dentro un foro chiaro spezza il blob: la circolarità crolla
sotto 0.8 e la posa non viene **mai** accettata. Nella sessione
`NewCalibration\Zed` succede in 8 pose su 9.

## Perché non basta "riempire con 4 cerchi"

Se la board non è perfettamente frontale, in immagine i fori sono **ellissi** e
i 4 centri **non** formano un quadrato: non esiste un cerchio unico da
stampigliare 4 volte, e il foro occluso non ha nemmeno un contorno completo da
cui misurare la sua ellisse.

La soluzione è non lavorare in immagine ma nel **piano della board**, dove la
geometria è nota ed esatta: 4 fori di ⌀0.13 m ai vertici di un quadrato
±0.15 m (lo stesso modello fisico dell'invariante
`RecordCheck/zed_pose_check.py::_spacing_ratio`, semi-diagonale/raggio ≈ 3.26).
Immagine e piano sono legati da un'**omografia H**. Stimata H:

- ogni foro si ridisegna come **proiezione esatta del cerchio del modello** —
  ellisse giusta, orientata bene, nella posizione giusta;
- il foro occluso è **determinato dagli altri tre**: non serve vederlo per
  sapere dove sta e che forma ha.

## Come funziona

1. **Frame di riferimento**: mediana di ~15 frame campionati lungo la posa.
   Lecito perché dentro una posa la board è ferma — drift dei centri misurato
   ≤ 0.2 px fra primo, centrale e ultimo frame.
2. **Detezione iniziale dei 4 centri**: riusa `detect_holes` di
   `../RecordCheck/zed_pose_check.py` (top-hat su pannello scuro + selezione a
   rettangolo + invariante fisico di spaziatura). Stessi default, stessi flag
   `--hole-*`. Trova i 4 fori in tutte e 9 le pose.
3. **H iniziale** dai 4 centri. L'ordine delle corrispondenze è irrilevante: il
   quadrato è invariante per il gruppo diedrale, quindi ogni rotazione o
   riflessione produce lo **stesso insieme** di 4 ellissi.
4. **Raffinamento robusto** — serve davvero: il centro del blob di un foro
   occluso è sbilanciato di **5–7 px**, e quell'errore finirebbe dritto nella
   posa stimata da lvt2calib.
   - punti di bordo per scansione radiale nel piano board; su ogni raggio il
     bordo è l'**ultimo** attraversamento del livello medio interno/esterno
     (se l'occlusore sta dentro il foro il raggio torna chiaro dopo di esso,
     quindi l'ultimo attraversamento è comunque il bordo vero);
   - fit congiunto di **H (8 DOF) + scala del raggio s (1 DOF)** su tutti i
     punti di bordo: ogni punto retroproiettato nel piano deve stare sul cerchio
     del modello. `scipy.optimize.least_squares(loss="soft_l1")` assorbe i pochi
     raggi in cui l'occlusore invade il bordo.
     Bastano 2 fori puliti (10 vincoli) per i 9 parametri.
   - senza `scipy` parte un fallback solo numpy (ricerca dell'offset intero
     ±8 px che massimizza il supporto del bordo + ricerca 1D della scala).
5. **Riempimento** delle 4 ellissi in bianco, sub-pixel (`cv2.fillPoly` con
   `shift=3` e `LINE_AA`).
6. **`--check`**: riproduce *esattamente* lvt2calib — `cv2.remap` con
   `initUndistortRectifyMap(K, D, None, K)` da `zed_right_intrinsic.yaml`, poi
   `findCirclesGrid` con i parametri blob della tabella qui sopra — e conta i
   frame in cui il grid verrebbe trovato **prima e dopo** il riempimento.

## Risultato su `NewCalibration\Zed\Poses`

Frame undistorti, parametri identici a lvt2calib, un frame ogni 20:

| posa | fori occlusi | grid RAW | grid FILLED |
|---|---|---|---|
| pose_01 | 1 | 0/12 | 12/12 |
| pose_03 | 0 | 0/12 | 12/12 |
| pose_04 | 2 | 0/12 | 12/12 |
| pose_06 | 2 | 0/12 | 12/12 |
| pose_09 | 2 | 0/14 | 14/14 |
| pose_10 | 2 | 0/11 | 11/11 |
| pose_12 | 1 | 0/12 | 12/12 |
| pose_15 | 2 | 0/12 | 12/12 |
| pose_19 | 0 | 12/12 | 12/12 |

La scala del raggio stimata è 0.954–0.961 in tutte le pose (fori reali ⌀ ~0.124 m
invece di 0.130), coerente fra pose indipendenti: è il segno che il fit è
stabile e fisicamente sensato. Scarto mediano dei punti di bordo: 0.12–0.39 px.

## Output

Cartella sorella `<pose>_filled`:

```
pose_01_filled/
  frames/           PNG riempiti, stessi nomi degli originali
  metadata.json     copiato IDENTICO (lista e timing dei frame non cambiano)
  holes.json        H, scala del raggio, 4 ellissi (centro, assi, angolo),
                    frazione di foro libero, flag occluso
  debug.png         overlay: ellissi (verde), centri (rosso), punti di bordo (giallo)
```

Pronta per `zed_frame_publisher.py --session-dir <...>\pose_01_filled`.
**Gli originali non vengono mai toccati.**

## Uso

```bat
:: tutte le pose + verifica
py fill_board_holes.py --poses-root C:\...\NewCalibration\Zed\Poses --check

:: una sola posa
py fill_board_holes.py --pose-dir C:\...\Poses\pose_01 --check

:: solo geometria/diagnostica, senza scrivere i frame
py fill_board_holes.py --pose-dir C:\...\Poses\pose_01 --dry-run --check

:: fallback manuale: click sui 4 centri (se la detezione automatica fallisce)
py fill_board_holes.py --pose-dir C:\...\Poses\pose_01 --click
```

Flag principali:

| flag | default | cosa fa |
|---|---|---|
| `--suffix` | `_filled` | suffisso della cartella di output |
| `--outdir` | — | output esplicito (solo con `--pose-dir`) |
| `--force` | off | sovrascrive una cartella di output già piena |
| `--dry-run` | off | calcola geometria, `debug.png`, `holes.json`, `--check` senza scrivere i frame |
| `--fill` | 255 | valore di riempimento (bianco) |
| `--radius-scale` | 1.0 | moltiplicatore extra sul raggio stimato |
| `--no-refine` | off | usa la sola omografia dai centri rilevati |
| `--occluded-frac` | 0.95 | soglia sulla frazione di foro libero per marcarlo occluso |
| `--check-step` | 20 | un frame ogni N per `--check` |
| `--camera-info` | `lvt2calib/data/camera_info/zed_right_intrinsic.yaml` | intrinseci per l'undistort di `--check` |
| `--hole-*`, `--rect-tol` | come `zed_pose_check.py` | detezione iniziale |

## Dipendenze

`numpy`, `opencv-python`; `scipy` opzionale ma consigliata (senza, parte il
fallback numpy — meno preciso sul foro occluso).

Dipende inoltre da `../RecordCheck/zed_pose_check.py`, importato via
`sys.path` (stesso schema con cui `zed_pose_check.py` importa
`detect_board_poses.py`): **le due cartelle devono restare sorelle**.

## Note

- Il riempimento è una modifica dell'immagine RGB usata dal detector, non della
  board fisica: nuvola LiDAR e frame FLIR sono flussi separati e non ne sono
  toccati. Stessa logica di `SessionSplit/patch_hole_pose.py`, che però
  stampigliava un cerchio fisso scelto a mano su un singolo foro.
- lvt2calib **undistorce prima** di cercare i blob: la verifica `--check` lo
  riproduce, quindi il risultato in tabella è quello che vedrà il nodo ROS.
- Se una posa fallisce la detezione automatica, `--click` permette di indicare i
  4 centri a mano; il raffinamento successivo è identico.
