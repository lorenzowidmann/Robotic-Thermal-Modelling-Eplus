"""
Converte un bag ROS2 contenente livox_ros_driver2/msg/CustomMsg in formato ROS1.

DUE PROBLEMI RISOLTI DA QUESTO SCRIPT:

1. Il tipo CustomMsg non e' nel typesystem standard di rosbags, quindi va
   registrato esplicitamente prima della conversione, altrimenti si ottiene:
       TypesysError("Type 'livox_ros_driver2/msg/CustomMsg' is unknown.")

2. Mismatch di namespace tra ROS2 e ROS1:
   - Il bag ROS2 (driver HAP) contiene  livox_ros_driver2/msg/CustomMsg
   - Il nodo koide3/livox_to_pointcloud2, nel suo ramo ROS1, si aspetta invece
     livox_ros_driver/msg/CustomMsg  (driver v1, namespace SENZA il "2")
   La struttura dei campi e' identica tra le due versioni, quindi si puo'
   riscrivere il namespace in fase di conversione: il default di questo script
   e' scrivere il bag ROS1 con il namespace v1, cosi' il nodo lo riconosce.
   Comportamento controllabile con --ros1-namespace.

NOTA: lo script NON converte CustomMsg in PointCloud2 (e' un cambio di formato,
non di contenitore). Quella conversione va fatta dopo, dentro ROS1, con:
    rosrun livox_to_pointcloud2 livox_to_pointcloud2_node

Definizione dei messaggi da: github.com/Livox-SDK/livox_ros_driver2/tree/master/msg

Uso tipico:
    py convert_livox_bag.py --src C:\\Users\\loren\\Desktop\\SLAM\\my_scan ^
                            --dst C:\\Users\\loren\\Desktop\\SLAM\\my_scan_ros1.bag
"""

import argparse
from pathlib import Path

from rosbags.rosbag1 import Writer as Writer1
from rosbags.rosbag2 import Reader as Reader2
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

# Definizioni dei messaggi custom Livox (fonte: repo ufficiale Livox-SDK).
# Struttura identica tra livox_ros_driver (v1) e livox_ros_driver2 (v2):
# cambia solo il nome del package, non i campi.
CUSTOM_POINT_MSG = """
uint32 offset_time
float32 x
float32 y
float32 z
uint8 reflectivity
uint8 tag
uint8 line
"""

# Il tipo dell'array points va qualificato col package corretto: viene
# sostituito a runtime in base al namespace scelto (v1 o v2).
CUSTOM_MSG_TEMPLATE = """
std_msgs/Header header
uint64 timebase
uint32 point_num
uint8 lidar_id
uint8[3] rsvd
{pkg}/CustomPoint[] points
"""


def register_livox_types(typestore, pkg: str) -> None:
    """Registra CustomPoint e CustomMsg nel typestore sotto il package `pkg`."""
    typestore.register(
        get_types_from_msg(CUSTOM_POINT_MSG, f"{pkg}/msg/CustomPoint")
    )
    typestore.register(
        get_types_from_msg(
            CUSTOM_MSG_TEMPLATE.format(pkg=pkg), f"{pkg}/msg/CustomMsg"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte un bag ROS2 con CustomMsg Livox in formato ROS1."
    )
    parser.add_argument(
        "--src", required=True,
        help="Cartella del bag ROS2 (quella contenente .db3 e metadata.yaml)",
    )
    parser.add_argument(
        "--dst", required=True,
        help="File .bag ROS1 di output",
    )
    parser.add_argument(
        "--src-namespace", default="livox_ros_driver2",
        help="Package del CustomMsg NEL BAG SORGENTE (default: livox_ros_driver2, "
             "cioe' quello prodotto dal driver HAP in ROS2)",
    )
    parser.add_argument(
        "--ros1-namespace", default="livox_ros_driver",
        help="Package con cui scrivere il CustomMsg NEL BAG ROS1 (default: "
             "livox_ros_driver, cioe' il v1 atteso dal ramo ROS1 di "
             "koide3/livox_to_pointcloud2). Usare livox_ros_driver2 per "
             "preservare il namespace originale.",
    )
    parser.add_argument(
        "--topics", nargs="+", default=None, metavar="TOPIC",
        help="Converti SOLO questi topic (default: tutti). Per il solo passo "
             "livox_to_pointcloud2 basta --topics /livox/lidar, che evita di "
             "riserializzare /cloud_registered (PointCloud2, il grosso del bag) "
             "e velocizza molto la conversione.",
    )
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    src_pkg = args.src_namespace
    dst_pkg = args.ros1_namespace

    if not src.is_dir():
        raise SystemExit(f"Cartella sorgente non trovata: {src}")
    if dst.exists():
        raise SystemExit(f"Il file di destinazione esiste gia', rimuovilo prima: {dst}")

    src_type = f"{src_pkg}/msg/CustomMsg"
    dst_type = f"{dst_pkg}/msg/CustomMsg"

    # Typestore di lettura (ROS2), con i tipi custom del sorgente
    ts_ros2 = get_typestore(Stores.ROS2_HUMBLE)
    register_livox_types(ts_ros2, src_pkg)

    # Typestore di scrittura (ROS1), con i tipi custom nel namespace di destinazione
    ts_ros1 = get_typestore(Stores.ROS1_NOETIC)
    register_livox_types(ts_ros1, dst_pkg)
    if dst_pkg != src_pkg:
        # serve anche il tipo sorgente per la deserializzazione lato ROS1
        register_livox_types(ts_ros1, src_pkg)

    print(f"Lettura da:  {src}")
    print(f"Scrittura a: {dst}")
    if dst_pkg != src_pkg:
        print(f"Rimappatura namespace: {src_type}  ->  {dst_type}")
    print()

    with Reader2(src) as reader, Writer1(dst) as writer:
        conn_map = {}

        for conn in reader.connections:
            if args.topics is not None and conn.topic not in args.topics:
                print(f"  [--]   {conn.topic}  ({conn.msgtype})  (escluso da --topics)")
                continue
            # rimappa il namespace del CustomMsg, lascia intatti gli altri tipi
            out_type = dst_type if conn.msgtype == src_type else conn.msgtype

            try:
                ts_ros1.generate_msgdef(out_type)
            except Exception as exc:
                print(f"  [SKIP] {conn.topic}  ({conn.msgtype}) -> {exc}")
                continue

            conn_map[conn.id] = writer.add_connection(
                conn.topic,
                out_type,
                typestore=ts_ros1,
            )
            suffix = f"  ->  {out_type}" if out_type != conn.msgtype else ""
            print(f"  [OK]   {conn.topic}  ({conn.msgtype}){suffix}")

        # Totale messaggi da scrivere, per la barra di avanzamento (senza il
        # filtro --topics e' tutto il bag; altrimenti solo i topic selezionati).
        total = sum(
            c.msgcount for c in reader.connections if c.id in conn_map
        )
        print(f"Conversione di {total} messaggi (deserialize CDR -> serialize ROS1)...")

        count = 0
        seq_counters = {}  # ROS1 std_msgs/Header.seq, incrementale per topic
        for conn, timestamp, raw in reader.messages():
            if conn.id not in conn_map:
                continue
            # ROS2 (CDR) e ROS1 usano formati di serializzazione diversi: non si
            # possono copiare i byte grezzi. Deserializza dal CDR del sorgente e
            # riserializza in ROS1 sotto il tipo di destinazione (i campi di
            # CustomMsg/CustomPoint sono identici tra v1 e v2, cambia solo il
            # package, quindi la rimappatura del namespace e' trasparente).
            out_type = dst_type if conn.msgtype == src_type else conn.msgtype
            msg = ts_ros2.deserialize_cdr(raw, conn.msgtype)
            # std_msgs/Header ha il campo `seq` in ROS1 ma non in ROS2: senza
            # questo serialize_ros1 fallisce con
            #   AttributeError: 'std_msgs__msg__Header' object has no attribute 'seq'
            # Lo aggiungiamo con un contatore incrementale per topic (come farebbe
            # ROS1). Vale per qualsiasi messaggio con header (Imu, PointCloud2,
            # Odometry, CustomMsg, ...).
            hdr = getattr(msg, "header", None)
            if hdr is not None and not hasattr(hdr, "seq"):
                seq = seq_counters.get(conn.id, 0)
                hdr.seq = seq
                seq_counters[conn.id] = seq + 1
            raw1 = ts_ros1.serialize_ros1(msg, out_type)
            writer.write(conn_map[conn.id], timestamp, raw1)
            count += 1
            if count % 500 == 0 or count == total:
                print(f"\r  {count}/{total} messaggi", end="", flush=True)
        print()

    print(f"Fatto. {count} messaggi scritti in {dst}")
    print()
    print("Prossimo passo, dentro il container Docker ROS1:")
    print("  rosbag info /data/bags/" + dst.name)
    print("  rosrun livox_to_pointcloud2 livox_to_pointcloud2_node")


if __name__ == "__main__":
    main()
