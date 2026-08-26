# ROS1Conversion

Converte un bag ROS2 contenente `livox_ros_driver2/msg/CustomMsg` in formato ROS1,
rimappando il namespace del messaggio al v1 (`livox_ros_driver`) atteso dal ramo
ROS1 di `koide3/livox_to_pointcloud2`.

Lo script non converte `CustomMsg` in `PointCloud2`: quel passo va fatto dopo,
dentro ROS1, con `rosrun livox_to_pointcloud2 livox_to_pointcloud2_node`.

## Setup

```
pip install -r requirements.txt
```

## Uso

```
py convert_livox_bag.py --src C:\path\to\ros2_bag_dir --dst C:\path\to\output_ros1.bag
```

Opzioni principali:
- `--src-namespace` — package del `CustomMsg` nel bag sorgente (default `livox_ros_driver2`)
- `--ros1-namespace` — package con cui scrivere il `CustomMsg` nel bag ROS1 (default `livox_ros_driver`)
- `--topics` — converti solo i topic indicati (es. `--topics /livox/lidar` evita di
  riserializzare `/cloud_registered`, il grosso del bag)
