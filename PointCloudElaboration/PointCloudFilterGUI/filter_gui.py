"""GUI: load a rosbag2 point cloud, tune filters interactively with a live
preview, save the filtered cloud as a new rosbag2 .db3.

Same filters/order as PointCloudView/view_pointcloud.py (SOR -> declutter ->
voxel), plus the ROI crop from
MATLAB_PointCloudVisualization/ROS2_PointVisualization.m, applied first:

    read all frames -> merge -> ROI crop -> SOR -> declutter -> voxel downsample

Two preview paths, both after "Apply filters / Preview" filters in a
background thread:
  - Embedded matplotlib view (right panel, CPU-rendered) updates immediately
    -- quick to glance at, but no GPU accel, so rotate/pan gets sluggish past
    a few tens of thousands of points (see "max points shown").
  - "Open smooth 3D view (PyVista)..." opens the full-resolution cloud in a
    SEPARATE PROCESS with PyVista/VTK -- same GPU-accelerated renderer
    view_pointcloud.py uses, so rotate/zoom stays smooth even on dense
    clouds, plus a metre-scale grid to read off ROI cut coordinates. A
    subprocess (rather than pv.Plotter().show() in this process) avoids
    nesting VTK's native message loop inside Tk's Tcl callback loop, which on
    Windows can create the window without ever painting it. As a bonus the
    control panel stays usable while the preview window is open.

The merged+filtered cloud is written out as a SINGLE PointCloud2 message
(xyz only, float32, same frame_id as the source) on the same topic, in a new
rosbag2 folder -- readable by view_pointcloud.py, fit_planes.py, and the
MATLAB scripts exactly like a normal bag (they all merge-all-frames anyway).
Voxel downsampling merges points across original frames, so per-frame
structure can't be preserved in general -- this tool bakes the result into
one frame instead of pretending otherwise.

Usage:
    python filter_gui.py [bag_folder]

Venv: rosbags, numpy, scipy, pyvista (tkinter is stdlib). C:\\venvs\\planefit
already has all of these installed; otherwise `pip install -r requirements.txt`
in a fresh venv (scipy is optional -- SOR falls back to a cruder numpy filter
without it, same as view_pointcloud.py).
"""
import argparse
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from rosbags.highlevel import AnyReader
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# matplotlib's mplot3d has no GPU accel and redraws every point on every
# mouse-drag frame -- rotation gets sluggish well before this many points.
DEFAULT_EMBEDDED_CAP = 30_000
# Safety cap on how many points PyVista is asked to draw at once. VTK handles
# far more than matplotlib comfortably, so this is just a backstop for raw
# multi-million-point clouds before any filtering has thinned them; 0 = no cap.
DEFAULT_PYVISTA_CAP = 500_000


# ----------------------------------------------------------------------------
# Filters -- same logic as PointCloudView/view_pointcloud.py, plus roi_crop
# (ported from MATLAB_PointCloudVisualization/ROS2_PointVisualization.m sec 5a)
# ----------------------------------------------------------------------------

def roi_crop(xyz, roi, log):
    """roi = (xmin, xmax, ymin, ymax, zmin, zmax); +-inf allowed."""
    xmin, xmax, ymin, ymax, zmin, zmax = roi
    keep = ((xyz[:, 0] >= xmin) & (xyz[:, 0] <= xmax) &
            (xyz[:, 1] >= ymin) & (xyz[:, 1] <= ymax) &
            (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax))
    out = xyz[keep]
    log(f"ROI crop: {len(xyz)} -> {len(out)} points "
        f"(removed {len(xyz) - len(out)}, {100 * (1 - len(out) / max(len(xyz), 1)):.1f}%)")
    return out


def statistical_outlier_removal(xyz, k, std_ratio, log):
    if len(xyz) <= k:
        return xyz
    if _HAVE_SCIPY:
        tree = cKDTree(xyz)
        dist, _ = tree.query(xyz, k=k + 1, workers=-1)
        mean_d = dist[:, 1:].mean(axis=1)
        thresh = mean_d.mean() + std_ratio * mean_d.std()
        keep = mean_d < thresh
        log(f"SOR (scipy, k={k}, std={std_ratio}): removed {int((~keep).sum())} / {len(xyz)} points")
        return xyz[keep]
    return _voxel_outlier_removal(xyz, min_neighbors=k, std_ratio=std_ratio, log=log)


def _voxel_outlier_removal(xyz, min_neighbors, std_ratio, log):
    span = xyz.max(axis=0) - xyz.min(axis=0)
    size = float(np.median(span)) / 200.0 or 0.05
    keys = np.floor(xyz / size).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    per_point = counts[inv]
    thresh = max(2, per_point.mean() - std_ratio * per_point.std())
    keep = per_point >= thresh
    log(f"SOR (numpy fallback, voxel={size:.3f}m): removed {int((~keep).sum())} / {len(xyz)} points "
        f"(install scipy for true SOR)")
    return xyz[keep]


def cluster_labels(xyz, gap):
    keys = np.floor(xyz / gap).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    index = {tuple(v): i for i, v in enumerate(uniq)}
    parent = np.arange(len(uniq))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    neigh = [(dx, dy, dz)
             for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
             if (dx, dy, dz) != (0, 0, 0)]
    for i, v in enumerate(uniq):
        for dx, dy, dz in neigh:
            j = index.get((v[0] + dx, v[1] + dy, v[2] + dz))
            if j is not None and j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(len(uniq))])
    _, comp = np.unique(roots, return_inverse=True)
    return comp[inv]


def declutter(xyz, gap, min_size, keep_dist, log):
    labels = cluster_labels(xyz, gap)
    counts = np.bincount(labels)
    main = int(counts.argmax())
    n_clusters = len(counts)

    if keep_dist > 0 and _HAVE_SCIPY:
        tree = cKDTree(xyz[labels == main])
        keep = labels == main
        for c in range(n_clusters):
            if c == main:
                continue
            if min_size and counts[c] < min_size:
                continue
            pts = xyz[labels == c]
            if tree.query(pts, k=1)[0].min() <= keep_dist:
                keep |= labels == c
    elif min_size:
        big = np.where(counts >= min_size)[0]
        keep = np.isin(labels, big)
    else:
        keep = labels == main

    log(f"declutter (gap={gap}m): {n_clusters} clusters, kept {int(keep.sum())} / {len(xyz)} points")
    return xyz[keep]


def voxel_downsample(xyz, size, log):
    if size <= 0:
        return xyz
    keys = np.floor(xyz / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    out = xyz[idx]
    log(f"voxel {size}m: {len(xyz)} -> {len(out)} points ({100 * len(out) / max(len(xyz), 1):.1f}%)")
    return out


def run_pipeline(xyz, params, log):
    """Apply ROI -> SOR -> declutter -> voxel, in that order, per params dict."""
    out = xyz
    if params["roi_on"]:
        out = roi_crop(out, params["roi"], log)
    if params["sor_on"]:
        out = statistical_outlier_removal(out, params["sor_k"], params["sor_std"], log)
    if params["declutter_on"]:
        out = declutter(out, params["cluster_gap"], params["min_cluster"], params["cluster_dist"], log)
    if params["voxel"] > 0:
        out = voxel_downsample(out, params["voxel"], log)
    return out


# ----------------------------------------------------------------------------
# rosbag2 I/O
# ----------------------------------------------------------------------------

def read_pointcloud2(msg):
    step = msg.point_step
    n = msg.width * msg.height
    raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * step).reshape(n, step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(n, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def load_bag(bag_path, topic, store_name, log):
    """Read+merge all frames on `topic`. Returns (xyz float32 Nx3, frame_id)."""
    typestore = get_typestore(Stores[store_name])
    frames = []
    frame_id = "map"
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            topics = sorted({c.topic for c in reader.connections})
            raise ValueError(f"Topic {topic!r} not found. Available: {topics}")
        n_msgs = 0
        for connection, timestamp, rawdata in reader.messages(connections=conns):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if n_msgs == 0:
                frame_id = msg.header.frame_id or frame_id
            frames.append(read_pointcloud2(msg))
            n_msgs += 1
            if n_msgs % 200 == 0:
                log(f"  ...read {n_msgs} frames")
    xyz = np.vstack(frames) if frames else np.empty((0, 3), dtype=np.float32)
    log(f"Read {n_msgs} frame(s), {len(xyz)} points, frame_id={frame_id!r}")
    return xyz, frame_id


def show_pointcloud_window(pts, title_note):
    """Blocking PyVista (VTK) window -- same renderer as view_pointcloud.py.

    Draws a bounding-box ruler with metre tick labels on every axis (data is
    already in metres, same frame as the ROI crop fields), so you can read
    off X/Y/Z coordinates straight off the plot to pick ROI cut values."""
    cloud = pv.PolyData(pts)
    cloud["z"] = pts[:, 2]
    p = pv.Plotter()
    p.add_mesh(cloud, scalars="z", cmap="turbo", point_size=2,
               render_points_as_spheres=False, show_scalar_bar=False)
    p.set_background("white")
    p.show_grid(color="black", xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
                font_size=10, fmt="%.1f")
    p.add_axes()
    p.add_text(f"{title_note} -- close this window when done",
               color="black", font_size=10)
    p.show()


def write_bag(out_path, topic, frame_id, xyz, store_name, log):
    """Write `xyz` as a single PointCloud2 message in a new rosbag2 folder."""
    typestore = get_typestore(Stores[store_name])
    PointField = typestore.types["sensor_msgs/msg/PointField"]
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]

    xyz32 = np.ascontiguousarray(xyz, dtype=np.float32)
    data = np.frombuffer(xyz32.tobytes(), dtype=np.uint8)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg = PointCloud2(
        header=Header(stamp=Time(sec=0, nanosec=0), frame_id=frame_id),
        height=1, width=len(xyz32), fields=fields, is_bigendian=False,
        point_step=12, row_step=12 * len(xyz32), data=data, is_dense=True,
    )
    raw = typestore.serialize_cdr(msg, PointCloud2.__msgtype__)

    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists -- pick a new folder/name")
    with Writer(out_path, version=9) as writer:
        conn = writer.add_connection(topic, PointCloud2.__msgtype__, typestore=typestore)
        writer.write(conn, 0, raw)
    log(f"Wrote {len(xyz32)} points -> {out_path}")


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------

class FilterGUI(tk.Tk):
    def __init__(self, initial_bag=None):
        super().__init__()
        self.title("Point Cloud Filter")
        self.geometry("1200x820")

        self.bag_path = Path(initial_bag) if initial_bag else None
        self.raw_xyz = None
        self.frame_id = "map"
        self.filtered_xyz = None
        self.msg_queue = queue.Queue()
        self.busy = False
        self.pv_process = None  # subprocess.Popen of the currently open PyVista window, if any

        self._build_widgets()
        if self.bag_path:
            self.bag_entry_var.set(str(self.bag_path))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_queue)

    # -- layout --------------------------------------------------------

    def _build_widgets(self):
        root = ttk.Frame(self, padding=6)
        root.pack(fill="both", expand=True)

        controls = ttk.Frame(root, width=340)
        controls.pack(side="left", fill="y", padx=(0, 6))
        controls.pack_propagate(False)

        view = ttk.Frame(root)
        view.pack(side="right", fill="both", expand=True)

        # --- bag selection ---
        f = ttk.LabelFrame(controls, text="Bag")
        f.pack(fill="x", pady=4)
        self.bag_entry_var = tk.StringVar()
        ttk.Entry(f, textvariable=self.bag_entry_var).pack(fill="x", padx=4, pady=2)
        row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
        ttk.Button(row, text="Browse...", command=self._browse_bag).pack(side="left")
        ttk.Button(row, text="Load", command=self._load_bag).pack(side="left", padx=4)

        row2 = ttk.Frame(f); row2.pack(fill="x", padx=4, pady=2)
        ttk.Label(row2, text="Topic").pack(side="left")
        self.topic_var = tk.StringVar(value="/cloud_registered")
        ttk.Entry(row2, textvariable=self.topic_var, width=18).pack(side="left", padx=4)
        self.store_var = tk.StringVar(value="ROS2_HUMBLE")
        ttk.Entry(row2, textvariable=self.store_var, width=12).pack(side="left")

        # --- ROI ---
        f = ttk.LabelFrame(controls, text="ROI crop")
        f.pack(fill="x", pady=4)
        self.roi_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="enable", variable=self.roi_on).pack(anchor="w", padx=4)
        self.roi_vars = {}
        for label, default in [("xmin", "-inf"), ("xmax", "inf"),
                                ("ymin", "-inf"), ("ymax", "inf"),
                                ("zmin", "-inf"), ("zmax", "inf")]:
            row = ttk.Frame(f); row.pack(fill="x", padx=4)
            ttk.Label(row, text=label, width=6).pack(side="left")
            v = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=v, width=10).pack(side="left")
            self.roi_vars[label] = v

        # --- SOR ---
        f = ttk.LabelFrame(controls, text="Statistical Outlier Removal")
        f.pack(fill="x", pady=4)
        self.sor_on = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="enable", variable=self.sor_on).pack(anchor="w", padx=4)
        row = ttk.Frame(f); row.pack(fill="x", padx=4)
        ttk.Label(row, text="k", width=6).pack(side="left")
        self.sor_k_var = tk.StringVar(value="16")
        ttk.Entry(row, textvariable=self.sor_k_var, width=10).pack(side="left")
        row = ttk.Frame(f); row.pack(fill="x", padx=4)
        ttk.Label(row, text="std", width=6).pack(side="left")
        self.sor_std_var = tk.StringVar(value="1.5")
        ttk.Entry(row, textvariable=self.sor_std_var, width=10).pack(side="left")
        if not _HAVE_SCIPY:
            ttk.Label(f, text="(scipy not found -- using cruder numpy fallback)",
                      foreground="#a06000").pack(anchor="w", padx=4)

        # --- declutter ---
        f = ttk.LabelFrame(controls, text="Declutter (disconnected islands)")
        f.pack(fill="x", pady=4)
        self.declutter_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="enable", variable=self.declutter_on).pack(anchor="w", padx=4)
        for label, var_name, default in [
            ("cluster-gap (m)", "cluster_gap_var", "0.30"),
            ("min-cluster (0=off)", "min_cluster_var", "0"),
            ("cluster-dist (0=off)", "cluster_dist_var", "0.0"),
        ]:
            row = ttk.Frame(f); row.pack(fill="x", padx=4)
            ttk.Label(row, text=label, width=16).pack(side="left")
            v = tk.StringVar(value=default)
            ttk.Entry(row, textvariable=v, width=10).pack(side="left")
            setattr(self, var_name, v)

        # --- voxel ---
        f = ttk.LabelFrame(controls, text="Voxel downsample")
        f.pack(fill="x", pady=4)
        row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="size (m, 0=off)", width=16).pack(side="left")
        self.voxel_var = tk.StringVar(value="0.0")
        ttk.Entry(row, textvariable=self.voxel_var, width=10).pack(side="left")

        # --- preview density ---
        f = ttk.LabelFrame(controls, text="Preview")
        f.pack(fill="x", pady=4)
        row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="embedded max pts", width=16).pack(side="left")
        self.embedded_cap_var = tk.StringVar(value=str(DEFAULT_EMBEDDED_CAP))
        ttk.Entry(row, textvariable=self.embedded_cap_var, width=10).pack(side="left")
        row = ttk.Frame(f); row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="pyvista max pts\n(0 = no cap)", width=16, justify="left").pack(side="left")
        self.pyvista_cap_var = tk.StringVar(value=str(DEFAULT_PYVISTA_CAP))
        ttk.Entry(row, textvariable=self.pyvista_cap_var, width=10).pack(side="left")

        # --- actions ---
        f = ttk.Frame(controls)
        f.pack(fill="x", pady=8)
        self.preview_btn = ttk.Button(f, text="Apply filters / Preview", command=self._preview)
        self.preview_btn.pack(fill="x", pady=2)
        self.pv_btn = ttk.Button(f, text="Open smooth 3D view (PyVista)...",
                                  command=self._open_pyvista, state="disabled")
        self.pv_btn.pack(fill="x", pady=2)
        self.save_btn = ttk.Button(f, text="Save filtered bag...", command=self._save, state="disabled")
        self.save_btn.pack(fill="x", pady=2)

        self.status_var = tk.StringVar(value="No bag loaded.")
        ttk.Label(controls, textvariable=self.status_var, wraplength=320).pack(fill="x", pady=4)

        self.log_box = scrolledtext.ScrolledText(controls, height=14, state="disabled")
        self.log_box.pack(fill="both", expand=True, pady=4)

        # --- embedded (CPU) 3D preview ---
        self.figure = Figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=view)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, view)
        toolbar.update()

    # -- logging / thread-safe UI updates -------------------------------

    def _log(self, text):
        self.msg_queue.put(("log", text))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self.log_box.configure(state="normal")
                    self.log_box.insert("end", payload + "\n")
                    self.log_box.see("end")
                    self.log_box.configure(state="disabled")
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "bag_loaded":
                    self.raw_xyz, self.frame_id = payload
                    self.status_var.set(f"Loaded {len(self.raw_xyz)} points. Adjust filters and Preview.")
                    self._set_busy(False)
                elif kind == "filtered":
                    self.filtered_xyz = payload
                    self._draw(self.filtered_xyz)
                    self.save_btn.configure(state="normal")
                    self.pv_btn.configure(state="normal")
                    self._set_busy(False)
                elif kind == "saved":
                    self._set_busy(False)
                    messagebox.showinfo("Saved", payload)
                elif kind == "error":
                    self._set_busy(False)
                    messagebox.showerror("Error", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.preview_btn.configure(state=state)
        have_result = self.filtered_xyz is not None
        self.save_btn.configure(state=state if have_result else "disabled")
        self.pv_btn.configure(state=state if have_result else "disabled")

    # -- actions ---------------------------------------------------------

    def _browse_bag(self):
        d = filedialog.askdirectory(title="Select rosbag2 folder (contains metadata.yaml)")
        if d:
            self.bag_entry_var.set(d)

    def _load_bag(self):
        if self.busy:
            return
        bag_str = self.bag_entry_var.get().strip()
        if not bag_str:
            messagebox.showwarning("No bag", "Pick a rosbag2 folder first.")
            return
        bag_path = Path(bag_str)
        if not (bag_path / "metadata.yaml").exists():
            messagebox.showwarning("Not a bag", f"No metadata.yaml in {bag_path}")
            return
        self.bag_path = bag_path
        topic = self.topic_var.get().strip() or "/cloud_registered"
        store = self.store_var.get().strip() or "ROS2_HUMBLE"
        self.filtered_xyz = None
        self.save_btn.configure(state="disabled")
        self._set_busy(True)
        self._log(f"Loading {bag_path} (topic={topic}, store={store})...")

        def worker():
            try:
                xyz, frame_id = load_bag(bag_path, topic, store, self._log)
                self.msg_queue.put(("bag_loaded", (xyz, frame_id)))
            except Exception as exc:
                self.msg_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _read_params(self):
        def f(v, default=None):
            s = v.get().strip()
            if s == "":
                return default
            return float(s)

        roi = tuple(f(self.roi_vars[k]) for k in
                    ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"))
        return {
            "roi_on": self.roi_on.get(),
            "roi": roi,
            "sor_on": self.sor_on.get(),
            "sor_k": int(f(self.sor_k_var, 16)),
            "sor_std": f(self.sor_std_var, 1.5),
            "declutter_on": self.declutter_on.get(),
            "cluster_gap": f(self.cluster_gap_var, 0.30),
            "min_cluster": int(f(self.min_cluster_var, 0)),
            "cluster_dist": f(self.cluster_dist_var, 0.0),
            "voxel": f(self.voxel_var, 0.0),
        }

    def _preview(self):
        if self.busy:
            return
        if self.raw_xyz is None:
            messagebox.showwarning("No cloud", "Load a bag first.")
            return
        try:
            params = self._read_params()
        except ValueError as exc:
            messagebox.showerror("Bad filter value", str(exc))
            return

        self._set_busy(True)
        self._log("--- applying filters ---")

        def worker():
            try:
                out = run_pipeline(self.raw_xyz, params, self._log)
                self.msg_queue.put(("filtered", out))
            except Exception as exc:
                self.msg_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _draw(self, xyz):
        """Embedded matplotlib preview -- fast to glance at right after
        filtering, capped separately from the PyVista window (see
        DEFAULT_EMBEDDED_CAP)."""
        self.ax.clear()
        if len(xyz) == 0:
            self.ax.set_title("No points left after filtering")
            self.canvas.draw()
            return
        try:
            cap = int(self.embedded_cap_var.get().strip())
        except ValueError:
            cap = DEFAULT_EMBEDDED_CAP
        pts = xyz
        if len(pts) > cap:
            idx = np.random.choice(len(pts), cap, replace=False)
            pts = pts[idx]
            note = f" (showing {cap} of {len(xyz)} points -- use PyVista button for full-res)"
        else:
            note = f" ({len(xyz)} points)"
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=pts[:, 2], cmap="turbo")
        self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)"); self.ax.set_zlabel("Z (m)")
        self.ax.set_title("Filtered preview" + note)
        span = pts.max(axis=0) - pts.min(axis=0)
        max_span = max(span.max(), 1e-3)
        mid = (pts.max(axis=0) + pts.min(axis=0)) / 2
        self.ax.set_xlim(mid[0] - max_span / 2, mid[0] + max_span / 2)
        self.ax.set_ylim(mid[1] - max_span / 2, mid[1] + max_span / 2)
        self.ax.set_zlim(mid[2] - max_span / 2, mid[2] + max_span / 2)
        self.canvas.draw()

    def _close_pyvista(self):
        """Terminate the previously opened PyVista subprocess, if it's still
        running. Only one preview window is kept alive at a time -- opening a
        new one closes the old rather than piling up windows."""
        if self.pv_process is not None and self.pv_process.poll() is None:
            self._log("Closing previous preview window...")
            self.pv_process.terminate()
        self.pv_process = None

    def _open_pyvista(self):
        """Open a PyVista (VTK) window with the filtered cloud, in a SEPARATE
        PROCESS (same renderer/args as view_pointcloud.py). Calling
        pv.Plotter().show() in-process from here would nest VTK's native
        message loop inside the Tcl callback that invoked us -- on Windows
        that can create the window but leave it unpainted/invisible instead
        of raising an error. A subprocess sidesteps that entirely and, as a
        bonus, doesn't block the control panel while the window is open."""
        if self.filtered_xyz is None or self.busy:
            return
        self._close_pyvista()
        xyz = self.filtered_xyz
        if len(xyz) == 0:
            messagebox.showwarning("Empty result", "No points left after filtering.")
            return
        try:
            cap = int(self.pyvista_cap_var.get().strip() or 0)
        except ValueError:
            cap = DEFAULT_PYVISTA_CAP
        pts = xyz
        title_note = f"{len(xyz)} points"
        if cap > 0 and len(pts) > cap:
            idx = np.random.choice(len(pts), cap, replace=False)
            pts = pts[idx]
            title_note = f"showing {cap} of {len(xyz)} points"

        fd, npy_path = tempfile.mkstemp(suffix=".npy", prefix="pcfilter_preview_")
        os.close(fd)
        np.save(npy_path, pts.astype(np.float32))
        self._log(f"Opening preview window ({title_note})...")
        try:
            self.pv_process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()),
                 "--show-npy", npy_path, "--title", title_note])
        except Exception as exc:
            messagebox.showerror("Preview failed", f"Could not launch preview window:\n{exc}")

    def _on_close(self):
        self._close_pyvista()
        self.destroy()

    def _save(self):
        if self.busy or self.filtered_xyz is None:
            return
        default_name = (self.bag_path.name + "_filtered") if self.bag_path else "filtered_bag"
        initial_dir = str(self.bag_path.parent) if self.bag_path else "."
        out_str = filedialog.asksaveasfilename(
            title="Save filtered bag as (new folder name)",
            initialdir=initial_dir, initialfile=default_name, defaultextension="")
        if not out_str:
            return
        out_path = Path(out_str)
        topic = self.topic_var.get().strip() or "/cloud_registered"
        store = self.store_var.get().strip() or "ROS2_HUMBLE"

        self._set_busy(True)
        self._log(f"Saving to {out_path}...")

        def worker():
            try:
                write_bag(out_path, topic, self.frame_id, self.filtered_xyz, store, self._log)
                self.msg_queue.put(("saved", f"Wrote {len(self.filtered_xyz)} points to:\n{out_path}"))
            except Exception as exc:
                self.msg_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", nargs="?", default=None, help="rosbag2 folder to preload (optional)")
    # internal: re-exec'd as a subprocess by FilterGUI._open_pyvista to open
    # the preview window outside the Tk process -- not meant to be typed by hand.
    ap.add_argument("--show-npy", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--title", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.show_npy is not None:
        pts = np.load(args.show_npy)
        try:
            show_pointcloud_window(pts, args.title)
        finally:
            args.show_npy.unlink(missing_ok=True)
        return

    app = FilterGUI(initial_bag=args.bag)
    app.mainloop()


if __name__ == "__main__":
    main()
