"""
Controllo di integrita' e continuita' di un bag ROS2 del LiDAR Livox:
verifica che /livox/lidar e /livox/imu siano stati pubblicati SENZA
INTERRUZIONI per tutta la sessione, non solo che il conteggio totale dei
messaggi torni.

Caso da intercettare (ExtCalibration_try4): il driver Livox si riavvia a meta'
registrazione, il publisher muore ma `ros2 bag record` continua a girare e
registra il silenzio senza errori. Il conteggio totale sembra plausibile, ma
nel mezzo c'e' un buco di decine di secondi. Qui ogni intervallo tra due
messaggi consecutivi dello stesso topic viene confrontato con l'intervallo
tipico: tutto quello che supera --gap-threshold-multiplier volte l'intervallo
MEDIANO (default 3x, es. /livox/lidar ~0.21 s -> soglia ~0.63 s) e' un gap,
riportato con inizio, fine, durata e messaggi mancanti stimati. Si usa il
mediano (come check_bag_rate.py) perche' quello medio viene gonfiato proprio
dai gap che si vogliono trovare.

Tempi = timestamp di ricezione del bag (chiave del messaggio), non l'header del
messaggio: e' il clock del recorder, quello su cui un publisher morto lascia il
buco. Offset in secondi dall'inizio del bag, comuni a tutti i topic.

Per ogni topic:
  - messaggi letti vs dichiarati nei metadati del bag (integrita');
  - ritardo del primo messaggio dall'inizio del bag e silenzio finale prima
    della fine del bag (un publisher partito tardi o morto e mai ripartito non
    lascia un gap "interno"; se muoiono TUTTI i topic fino alla fine il bag
    finisce semplicemente prima: confrontare la durata con quella della ZED);
  - rate medio complessivo (Hz) e se rientra nel range atteso (--expected);
  - rate dall'intervallo mediano, jitter, intervallo massimo;
  - elenco dei gap, con i topic che hanno un gap sovrapposto (gap su lidar e imu
    insieme = riavvio del driver; solo su un topic = problema di quel flusso).

Il bag viene letto in streaming con `rosbags`: niente caricamento completo in
RAM, ma il payload di ogni messaggio passa comunque dal disco, quindi su un bag
da ~17 GB servono alcuni minuti. Il bag non viene modificato.

Exit status 1 se c'e' almeno un gap, un rate fuori range, un topic assente o un
errore di lettura: il controllo puo' fare da gate in uno script.

Uso tipico:
    py lidar_quality_check.py --bag <cartella_bag>
    py lidar_quality_check.py --bag <cartella_bag> --csv-out gap.csv
    py lidar_quality_check.py --bag <file.db3> --gap-threshold-multiplier 5
    py lidar_quality_check.py --bag <cartella_bag> --expected /livox/lidar=9.5:10.5
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag2 import Reader
except ImportError:
    sys.exit("Serve il pacchetto rosbags:  py -m pip install rosbags")

DEFAULT_TOPICS = ("/livox/lidar", "/livox/imu")
DEFAULT_EXPECTED = {"/livox/lidar": (4.5, 4.9), "/livox/imu": (195.0, 210.0)}


def resolve_bag(path):
    """Cartella rosbag2 (con metadata.yaml) o file .db3. Se metadata.yaml manca
    (recorder chiuso male) e la cartella ha un solo .db3, usa quello."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"Bag non trovato: {p}")
    if p.is_dir() and not (p / "metadata.yaml").exists():
        db3 = sorted(p.glob("*.db3"))
        if len(db3) != 1:
            sys.exit(f"metadata.yaml assente in {p} e {len(db3)} file .db3: "
                     "passare direttamente il file .db3 con --bag.")
        print(f"ATTENZIONE: metadata.yaml assente in {p}, leggo direttamente {db3[0].name}")
        return db3[0]
    return p


def parse_expected(text):
    """'TOPIC=MIN:MAX' -> (topic, (min_hz, max_hz))."""
    try:
        topic, rng = text.rsplit("=", 1)
        lo, hi = (float(v) for v in rng.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"formato atteso TOPIC=MIN:MAX, es. /livox/lidar=4.5:4.9 (ricevuto {text!r})"
        ) from None
    if not topic or hi < lo:
        raise argparse.ArgumentTypeError(f"range non valido: {text!r}")
    return topic, (lo, hi)


def read_stamps(bag, topics):
    """Timestamp di ricezione (ns) per topic, letti in streaming."""
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        bag_topics = {c.topic: c.msgtype for c in reader.connections}
        declared = {t: 0 for t in topics}
        for c in conns:
            declared[c.topic] += c.msgcount
        info = {
            "bag_start": reader.start_time,
            "bag_end": reader.end_time,
            "bag_topics": bag_topics,
            "declared": declared,
            "error": None,
        }
        stamps = {t: [] for t in topics}
        if not conns:
            return stamps, info
        total = sum(declared.values())
        n = 0
        try:
            for conn, t_ns, _ in reader.messages(connections=conns):
                stamps[conn.topic].append(t_ns)
                n += 1
                if n % 5000 == 0:
                    print(f"\r  {n}/{total} messaggi letti", end="", flush=True)
        except Exception as exc:
            info["error"] = f"lettura interrotta dopo {n} messaggi: {type(exc).__name__}: {exc}"
        print()
    return stamps, info


def analyse_topic(topic, ns, info, multiplier, min_gap_s):
    """Statistiche di continuita' di un topic e lista dei suoi gap."""
    t = np.sort(np.asarray(ns, dtype=np.int64))
    rep = {"topic": topic, "n": int(t.size), "gaps": []}
    if t.size < 2:
        return rep
    bag_start, bag_end = info["bag_start"], info["bag_end"]
    dt = np.diff(t) / 1e9
    med = float(np.median(dt))
    span = (t[-1] - t[0]) / 1e9
    thr = max(multiplier * med, min_gap_s)
    rep.update({
        "first_off": (t[0] - bag_start) / 1e9,
        "last_off": (t[-1] - bag_start) / 1e9,
        "mean_hz": (t.size - 1) / span if span > 0 else 0.0,
        "median_dt": med,
        "median_hz": 1.0 / med if med > 0 else 0.0,
        "jitter_ms": float(np.std(dt) * 1e3),
        "max_dt": float(dt.max()),
        "threshold": thr,
    })

    def gap(kind, a_ns, b_ns, missing):
        return {"topic": topic, "tipo": kind, "start_ns": int(a_ns), "end_ns": int(b_ns),
                "dur": (b_ns - a_ns) / 1e9, "missing": int(missing)}

    lead = (t[0] - bag_start) / 1e9
    if lead > thr:
        rep["gaps"].append(gap("inizio", bag_start, t[0], round(lead / med)))
    for i in np.nonzero(dt > thr)[0]:
        rep["gaps"].append(gap("interno", t[i], t[i + 1], round(dt[i] / med) - 1))
    tail = (bag_end - t[-1]) / 1e9
    if tail > thr:
        rep["gaps"].append(gap("fine", t[-1], bag_end, round(tail / med)))
    return rep


def fmt_utc(ns):
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


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
        description="Controllo di continuita' della pubblicazione in un bag ROS2 Livox."
    )
    ap.add_argument("--bag", required=True,
                    help="Cartella del bag ROS2 (con metadata.yaml) o file .db3")
    ap.add_argument("--topics", nargs="+", default=list(DEFAULT_TOPICS), metavar="TOPIC",
                    help="Topic da controllare (default /livox/lidar /livox/imu)")
    ap.add_argument("--gap-threshold-multiplier", type=float, default=3.0,
                    help="Un intervallo oltre N volte quello mediano del topic e' un "
                         "gap (default 3)")
    ap.add_argument("--min-gap-s", type=float, default=0.0,
                    help="Soglia minima assoluta in secondi per un gap, utile se il "
                         "jitter dell'IMU produce troppi falsi gap (default 0 = "
                         "disattivata)")
    ap.add_argument("--expected", type=parse_expected, action="append", default=None,
                    metavar="TOPIC=MIN:MAX",
                    help="Range atteso del rate medio in Hz, ripetibile (default "
                         "/livox/lidar=4.5:4.9 e /livox/imu=195:210)")
    ap.add_argument("--max-rows", type=int, default=50,
                    help="Gap mostrati a schermo al massimo (il CSV li contiene "
                         "tutti, default 50)")
    ap.add_argument("--csv-out", default=None, metavar="CSV",
                    help="Salva la tabella dei gap in CSV (solo intestazione se "
                         "non ci sono gap)")
    args = ap.parse_args()
    if args.gap_threshold_multiplier <= 1:
        ap.error("--gap-threshold-multiplier deve essere > 1")
    expected = dict(DEFAULT_EXPECTED)
    if args.expected:
        expected.update(dict(args.expected))

    bag = resolve_bag(args.bag)
    print(f"Bag: {bag}")
    print("Lettura timestamp in streaming...")
    stamps, info = read_stamps(bag, args.topics)
    bag_dur = (info["bag_end"] - info["bag_start"]) / 1e9
    print(f"Inizio bag: {fmt_utc(info['bag_start'])} UTC   "
          f"fine: {fmt_utc(info['bag_end'])} UTC   durata {bag_dur:.1f} s ({bag_dur / 60:.1f} min)")
    print("Topic nel bag: " + ", ".join(sorted(info["bag_topics"])))
    print()

    failed = False
    if info["error"]:
        print(f"ERRORE: {info['error']}")
        print()
        failed = True

    reports = []
    for topic in args.topics:
        msgtype = info["bag_topics"].get(topic)
        if msgtype is None:
            print(f"{topic}: ASSENTE nel bag")
            print()
            failed = True
            continue
        rep = analyse_topic(topic, stamps[topic], info, args.gap_threshold_multiplier,
                            args.min_gap_s)
        reports.append(rep)
        declared = info["declared"][topic]
        count_ok = rep["n"] == declared
        print(f"{topic}  [{msgtype}]")
        print(f"  messaggi letti / dichiarati : {rep['n']} / {declared}  "
              f"{'OK' if count_ok else 'DIVERSI'}")
        failed = failed or not count_ok
        if rep["n"] < 2:
            print("  troppo pochi messaggi per un'analisi di continuita'")
            print()
            failed = True
            continue
        rng = expected.get(topic)
        if rng is not None:
            rate_ok = rng[0] <= rep["mean_hz"] <= rng[1]
            rate_note = f"atteso {rng[0]:g}-{rng[1]:g} Hz  {'OK' if rate_ok else 'FUORI RANGE'}"
            failed = failed or not rate_ok
        else:
            rate_note = "nessun range atteso"
        n_gaps = len(rep["gaps"])
        lost = sum(g["dur"] for g in rep["gaps"])
        missing = sum(g["missing"] for g in rep["gaps"])
        print(f"  primo / ultimo messaggio    : {rep['first_off']:+.2f} s / "
              f"{rep['last_off']:+.2f} s dall'inizio del bag")
        print(f"  rate medio complessivo      : {rep['mean_hz']:.2f} Hz   {rate_note}")
        print(f"  rate da intervallo mediano  : {rep['median_hz']:.2f} Hz   "
              f"(intervallo mediano {rep['median_dt'] * 1e3:.1f} ms, jitter "
              f"{rep['jitter_ms']:.1f} ms)")
        print(f"  intervallo massimo          : {rep['max_dt']:.3f} s")
        print(f"  soglia gap                  : {rep['threshold']:.3f} s")
        print(f"  gap                         : {n_gaps}"
              + (f"  (totale {lost:.1f} s, ~{missing} messaggi mancanti)" if n_gaps else "  OK"))
        print()
        failed = failed or n_gaps > 0

    all_gaps = [g for rep in reports for g in rep["gaps"]]
    for g in all_gaps:
        g["others"] = "+".join(sorted({
            o["topic"] for o in all_gaps
            if o["topic"] != g["topic"] and o["start_ns"] < g["end_ns"] and o["end_ns"] > g["start_ns"]
        })) or "-"

    header = ["topic", "tipo", "gap", "inizio_offset_s", "fine_offset_s", "durata_s",
              "inizio_utc", "fine_utc", "msg_mancanti_stimati", "altri_topic"]
    rows = []
    for rep in reports:
        for k, g in enumerate(rep["gaps"], start=1):
            rows.append([g["topic"], g["tipo"], k,
                         f"{(g['start_ns'] - info['bag_start']) / 1e9:.3f}",
                         f"{(g['end_ns'] - info['bag_start']) / 1e9:.3f}",
                         f"{g['dur']:.3f}", fmt_utc(g["start_ns"]), fmt_utc(g["end_ns"]),
                         g["missing"], g["others"]])

    if rows:
        print(f"GAP TROVATI: {len(rows)}   (offset dall'inizio del bag; altri_topic = "
              "topic con un gap sovrapposto)")
        shown = sorted(rows, key=lambda r: -float(r[5]))[:args.max_rows]
        shown.sort(key=lambda r: (r[0], float(r[3])))
        print_table(header, shown)
        if len(rows) > len(shown):
            print(f"... mostrati i {len(shown)} gap piu' lunghi su {len(rows)}: "
                  "l'elenco completo e' nel CSV (--csv-out).")
        print()

    print("VERDETTO: " + ("PROBLEMI TROVATI" if failed else "OK, pubblicazione continua"))

    if args.csv_out:
        print()
        write_csv(args.csv_out, header, rows)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
