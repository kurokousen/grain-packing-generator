#!/usr/bin/env python3
"""
Grain Packing / Porous Media Generator - Aplikasi Desktop (GUI)
=================================================================
Aplikasi mandiri (tidak perlu VS Code / terminal khusus) untuk membuat
model porous media 3D (grain packing), menghitung porositas, luas
permukaan, dan estimasi permeabilitas (Kozeny-Carman), lalu mengekspor
volume (TIF/BMP) dan ringkasan (Excel).

Cara menjalankan:
    python app.py
(Windows: bisa dobel-klik app.py kalau Python sudah terpasang & file
terasosiasi ke python.exe. Kalau tidak, buka Command Prompt lalu
jalankan "python app.py" di folder ini.)

Dependensi (install sekali lewat pip):
    pip install numpy scipy pandas openpyxl tifffile imageio
(tkinter sudah termasuk instalasi Python standar di Windows/Mac;
di Linux mungkin perlu: sudo apt install python3-tk)
"""

import os
import sys
import queue
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core import GrainPackingSimulation  # noqa: E402


APP_TITLE = "Grain Packing / Porous Media Generator"


class ScrollableFrame(ttk.Frame):
    """Frame dengan scrollbar vertikal, dipakai untuk panel parameter yang panjang."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)


def labeled_entry(parent, row, label, default="", width=14, tooltip=None):
    ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
    var = tk.StringVar(value=str(default))
    entry = ttk.Entry(parent, textvariable=var, width=width)
    entry.grid(row=row, column=1, sticky="w", padx=6, pady=3)
    return var


def section_label(parent, row, text):
    lbl = ttk.Label(parent, text=text, font=("Segoe UI", 10, "bold"))
    lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(14, 4))
    return lbl


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x720")
        self.minsize(860, 600)

        self.sim_thread = None
        self.cancel_event = threading.Event()
        self.msg_queue = queue.Queue()

        self._build_ui()
        self.after(100, self._poll_queue)

    # ------------------------------------------------------------
    def _build_ui(self):
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ScrollableFrame(paned, width=430)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_params_panel(left.inner)
        self._build_run_panel(right)

    # ------------------------------------------------------------
    def _build_params_panel(self, f):
        row = 0
        ttk.Label(f, text="Parameter Simulasi", font=("Segoe UI", 12, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(6, 10)); row += 1

        # --- Dimensi ---
        section_label(f, row, "Dimensi (mm)"); row += 1
        self.v_box_size = labeled_entry(f, row, "Ukuran wadah total:", 10.0); row += 1
        self.v_core_size = labeled_entry(f, row, "Ukuran core evaluasi:", 6.0); row += 1
        self.v_res = labeled_entry(f, row, "Resolusi voxel (n^3):", 150); row += 1

        # --- Mode simulasi ---
        section_label(f, row, "Mode Simulasi"); row += 1
        self.v_mode = tk.StringVar(value="1")
        ttk.Radiobutton(f, text="Target jumlah grain", variable=self.v_mode, value="1",
                         command=self._toggle_mode).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Target porositas (CORE)", variable=self.v_mode, value="2",
                         command=self._toggle_mode).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        self.v_target_grain = labeled_entry(f, row, "Jumlah grain:", 300); row += 1
        self.v_target_porosity = labeled_entry(f, row, "Target porositas (0-1):", 0.35); row += 1

        # --- Mode posisi ---
        section_label(f, row, "Mode Posisi Grain (hanya utk 'target jumlah grain')"); row += 1
        self.v_position_mode = tk.StringVar(value="1")
        ttk.Radiobutton(f, text="Acak", variable=self.v_position_mode, value="1",
                         command=self._toggle_position).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Manual (paste koordinat X Y Z, mm)", variable=self.v_position_mode, value="2",
                         command=self._toggle_position).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        self.v_force_overlap_check = tk.BooleanVar(value=True)
        self.chk_force_overlap = ttk.Checkbutton(
            f, text="Cek overlap saat mode manual (skip kalau tabrakan)",
            variable=self.v_force_overlap_check)
        self.chk_force_overlap.grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        ttk.Label(f, text="Koordinat manual (1 baris = 1 grain, format: x y z):").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0)); row += 1
        self.txt_manual = tk.Text(f, width=40, height=6, state="disabled")
        self.txt_manual.grid(row=row, column=0, columnspan=2, padx=6, pady=(0, 4)); row += 1

        # --- Bentuk grain ---
        section_label(f, row, "Bentuk Grain"); row += 1
        self.v_shape_mode = tk.StringVar(value="1")
        shape_opts = [
            ("1", "Convex Hull acak (gerigi acak)"),
            ("2", "Ellipsoid acak (rasio & orientasi acak)"),
            ("3", "Bulat sempurna (sphere)"),
            ("4", "Ellipsoid sempurna (rasio tetap)"),
        ]
        for val, text in shape_opts:
            ttk.Radiobutton(f, text=text, variable=self.v_shape_mode, value=val,
                             command=self._toggle_shape).grid(row=row, column=0, columnspan=2, sticky="w", padx=6)
            row += 1

        self.v_n_vertex = labeled_entry(f, row, "Jumlah vertex hull:", 300); row += 1
        self.v_angularity = labeled_entry(f, row, "Angularity (0=bulat,1=tajam):", 0.3); row += 1
        self.v_max_overlap = labeled_entry(f, row, "Overlap maksimum (0-1):", 0.1); row += 1

        # Ellipsoid tetap
        self.lbl_ellipsoid = section_label(f, row, "Ellipsoid Sempurna (Shape Mode 4)"); row += 1
        self.v_ellip_rx = labeled_entry(f, row, "Rasio sumbu X:", 1.0); row += 1
        self.v_ellip_ry = labeled_entry(f, row, "Rasio sumbu Y:", 0.7); row += 1
        self.v_ellip_rz = labeled_entry(f, row, "Rasio sumbu Z:", 0.5); row += 1
        self.v_ellip_orient = tk.StringVar(value="2")
        ttk.Radiobutton(f, text="Orientasi tetap/sejajar", variable=self.v_ellip_orient, value="1").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Orientasi acak tiap grain", variable=self.v_ellip_orient, value="2").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        # --- Sortasi ---
        section_label(f, row, "Sortasi Ukuran Butiran"); row += 1
        self.v_sort_mode = tk.StringVar(value="2")
        ttk.Radiobutton(f, text="Well sorted", variable=self.v_sort_mode, value="1",
                         command=self._toggle_sort).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Moderately sorted", variable=self.v_sort_mode, value="2",
                         command=self._toggle_sort).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Poorly sorted", variable=self.v_sort_mode, value="3",
                         command=self._toggle_sort).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        self.v_r_fixed = labeled_entry(f, row, "Radius grain tetap (mm):", 0.5); row += 1
        self.v_mean_radius = labeled_entry(f, row, "Mean radius grain (mm):", 0.5); row += 1

        # --- Adaptive packing ---
        section_label(f, row, "Adaptive Packing"); row += 1
        self.v_adaptive_on = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Aktifkan adaptive packing", variable=self.v_adaptive_on,
                         command=self._toggle_adaptive).grid(row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        self.v_adaptive_mode = tk.StringVar(value="3")
        ttk.Radiobutton(f, text="Perbesar overlap bertahap", variable=self.v_adaptive_mode, value="1").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Perkecil butiran bertahap", variable=self.v_adaptive_mode, value="2").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="Keduanya (overlap dulu, lalu perkecil)", variable=self.v_adaptive_mode, value="3").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        self.v_reject_count_thresh = labeled_entry(f, row, "Reject count threshold:", 100); row += 1
        self.v_reject_ratio_thresh = labeled_entry(f, row, "Reject ratio threshold:", 0.25); row += 1
        self.v_max_overlap_limit = labeled_entry(f, row, "Batas atas overlap:", 0.4); row += 1
        self.v_overlap_step = labeled_entry(f, row, "Langkah kenaikan overlap:", 0.02); row += 1
        self.v_min_radius_factor = labeled_entry(f, row, "Batas bawah radius (fraksi):", 0.3); row += 1
        self.v_radius_shrink_step = labeled_entry(f, row, "Langkah penyusutan radius:", 0.05); row += 1

        # --- Output ---
        section_label(f, row, "Output"); row += 1
        self.v_out_format = tk.StringVar(value="tif")
        ttk.Radiobutton(f, text="TIF (volume 3D)", variable=self.v_out_format, value="tif").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1
        ttk.Radiobutton(f, text="BMP (per slice)", variable=self.v_out_format, value="bmp").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=6); row += 1

        ttk.Label(f, text="Folder output:").grid(row=row, column=0, sticky="w", padx=6, pady=(6, 3))
        self.v_outdir = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "grain_packing_output"))
        out_row = ttk.Frame(f)
        out_row.grid(row=row, column=1, sticky="w", padx=6, pady=(6, 3))
        ttk.Entry(out_row, textvariable=self.v_outdir, width=24).pack(side="left")
        ttk.Button(out_row, text="...", width=3, command=self._browse_outdir).pack(side="left", padx=(4, 0))
        row += 1

        self._toggle_mode()
        self._toggle_position()
        self._toggle_shape()
        self._toggle_sort()
        self._toggle_adaptive()

    # ------------------------------------------------------------
    def _build_run_panel(self, f):
        top = ttk.Frame(f)
        top.pack(fill="x", padx=10, pady=10)

        self.btn_run = ttk.Button(top, text="▶ Jalankan Simulasi", command=self._on_run)
        self.btn_run.pack(side="left")
        self.btn_stop = ttk.Button(top, text="■ Stop", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))
        self.btn_open_outdir = ttk.Button(top, text="Buka Folder Output", command=self._open_outdir,
                                           state="disabled")
        self.btn_open_outdir.pack(side="left", padx=(8, 0))

        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(f, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10, pady=(0, 4))
        self.status_var = tk.StringVar(value="Siap.")
        ttk.Label(f, textvariable=self.status_var).pack(fill="x", padx=10, anchor="w")

        ttk.Label(f, text="Log:").pack(fill="x", padx=10, pady=(10, 0), anchor="w")
        log_frame = ttk.Frame(f)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self.txt_log = tk.Text(log_frame, wrap="word", state="disabled", bg="#111", fg="#0f0",
                                insertbackground="#0f0", font=("Consolas", 9))
        sb = ttk.Scrollbar(log_frame, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    # ------------------------------------------------------------
    # Toggle visibility helpers (enable/disable, bukan hide, biar layout stabil)
    def _toggle_mode(self):
        is_grain = self.v_mode.get() == "1"
        state_grain = "normal" if is_grain else "disabled"
        state_poro = "disabled" if is_grain else "normal"
        self._set_entry_state(self.v_target_grain, state_grain)
        self._set_entry_state(self.v_target_porosity, state_poro)

    def _toggle_position(self):
        is_manual = self.v_position_mode.get() == "2"
        self.txt_manual.configure(state="normal" if is_manual else "disabled")
        self.chk_force_overlap.configure(state="normal" if is_manual else "disabled")

    def _toggle_shape(self):
        shape = self.v_shape_mode.get()
        angularity_state = "normal" if shape in ("1", "2") else "disabled"
        self._set_entry_state(self.v_angularity, angularity_state)
        ellip_state = "normal" if shape == "4" else "disabled"
        for v in (self.v_ellip_rx, self.v_ellip_ry, self.v_ellip_rz):
            self._set_entry_state(v, ellip_state)

    def _toggle_sort(self):
        well = self.v_sort_mode.get() == "1"
        self._set_entry_state(self.v_r_fixed, "normal" if well else "disabled")
        self._set_entry_state(self.v_mean_radius, "disabled" if well else "normal")

    def _toggle_adaptive(self):
        on = self.v_adaptive_on.get()
        state = "normal" if on else "disabled"
        for v in (self.v_reject_count_thresh, self.v_reject_ratio_thresh,
                  self.v_max_overlap_limit, self.v_overlap_step,
                  self.v_min_radius_factor, self.v_radius_shrink_step):
            self._set_entry_state(v, state)

    @staticmethod
    def _set_entry_state(var, state):
        # var adalah StringVar; kita cari widget entry pasangannya lewat trace tidak praktis,
        # jadi kita simpan referensi entry langsung. Untuk simplisitas, entry tersimpan di var._entry.
        entry = getattr(var, "_entry", None)
        if entry is not None:
            entry.configure(state=state)

    def _browse_outdir(self):
        d = filedialog.askdirectory(initialdir=self.v_outdir.get() or os.getcwd())
        if d:
            self.v_outdir.set(d)

    def _open_outdir(self):
        d = self.v_outdir.get()
        if not os.path.isdir(d):
            messagebox.showwarning(APP_TITLE, "Folder output belum ada.")
            return
        if sys.platform.startswith("win"):
            os.startfile(d)  # noqa
        elif sys.platform == "darwin":
            os.system(f'open "{d}"')
        else:
            os.system(f'xdg-open "{d}"')

    # ------------------------------------------------------------
    def _collect_params(self):
        def f(var):
            return float(var.get())

        def i(var):
            return int(float(var.get()))

        params = {
            "box_size": f(self.v_box_size),
            "core_size": f(self.v_core_size),
            "res": i(self.v_res),
            "mode": i(self.v_mode),
            "shape_mode": i(self.v_shape_mode),
            "n_vertex": i(self.v_n_vertex),
            "angularity": f(self.v_angularity) if self.v_shape_mode.get() in ("1", "2") else 0.0,
            "max_overlap": f(self.v_max_overlap),
            "sort_mode": i(self.v_sort_mode),
            "out_format": self.v_out_format.get(),
            "outdir": self.v_outdir.get().strip(),
            "adaptive_on": bool(self.v_adaptive_on.get()),
        }

        if params["mode"] == 1:
            params["target_grain"] = i(self.v_target_grain)
        else:
            params["target_porosity"] = f(self.v_target_porosity)

        params["position_mode"] = i(self.v_position_mode) if params["mode"] == 1 else 1
        if params["position_mode"] == 2:
            lines = self.txt_manual.get("1.0", "end").strip().splitlines()
            positions = []
            for ln_no, ln in enumerate(lines, start=1):
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.replace(",", " ").split()
                if len(parts) != 3:
                    raise ValueError(f"Baris {ln_no} koordinat manual tidak valid: '{ln}' "
                                      f"(harus 3 angka x y z)")
                positions.append([float(x) for x in parts])
            if not positions:
                raise ValueError("Mode posisi manual dipilih tapi kotak koordinat kosong.")
            params["manual_positions"] = positions
            params["force_overlap_check"] = bool(self.v_force_overlap_check.get())

        if params["shape_mode"] == 4:
            params["ellipsoid_rx"] = f(self.v_ellip_rx)
            params["ellipsoid_ry"] = f(self.v_ellip_ry)
            params["ellipsoid_rz"] = f(self.v_ellip_rz)
            params["ellipsoid_orient_mode"] = i(self.v_ellip_orient)

        if params["sort_mode"] == 1:
            params["r_fixed_sort"] = f(self.v_r_fixed)
        else:
            params["mean_radius"] = f(self.v_mean_radius)

        if params["adaptive_on"]:
            params["adaptive_mode"] = i(self.v_adaptive_mode)
            params["reject_count_thresh"] = i(self.v_reject_count_thresh)
            params["reject_ratio_thresh"] = f(self.v_reject_ratio_thresh)
            if params["adaptive_mode"] in (1, 3):
                params["max_overlap_limit"] = f(self.v_max_overlap_limit)
                params["overlap_step"] = f(self.v_overlap_step)
            if params["adaptive_mode"] in (2, 3):
                params["min_radius_factor"] = f(self.v_min_radius_factor)
                params["radius_shrink_step"] = f(self.v_radius_shrink_step)

        if not params["outdir"]:
            raise ValueError("Folder output belum diisi.")

        return params

    # ------------------------------------------------------------
    def _on_run(self):
        try:
            params = self._collect_params()
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Parameter tidak valid:\n{e}")
            return

        self._clear_log()
        self.btn_run.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_open_outdir.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Menjalankan simulasi...")
        self.cancel_event.clear()

        self.sim_thread = threading.Thread(target=self._run_worker, args=(params,), daemon=True)
        self.sim_thread.start()

    def _on_stop(self):
        self.cancel_event.set()
        self.status_var.set("Membatalkan...")

    def _run_worker(self, params):
        def log(text):
            self.msg_queue.put(("log", text))

        def progress(frac, text=""):
            self.msg_queue.put(("progress", (frac, text)))

        def should_cancel():
            return self.cancel_event.is_set()

        try:
            sim = GrainPackingSimulation(params, log=log, progress=progress, should_cancel=should_cancel)
            result = sim.run()
            self.msg_queue.put(("done", result))
        except Exception as e:
            tb = traceback.format_exc()
            self.msg_queue.put(("error", (str(e), tb)))

    # ------------------------------------------------------------
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    frac, text = payload
                    self.progress_var.set(max(0, min(100, frac * 100)))
                    if text:
                        self.status_var.set(text)
                elif kind == "done":
                    self._append_log(f"\nSelesai. Grain: {payload['grain']}, "
                                      f"Porositas: {payload['porosity']:.4f}, "
                                      f"Permeabilitas: {payload['permeability']:.4e} mm^2")
                    self.status_var.set("Selesai.")
                    self.progress_var.set(100)
                    self.btn_run.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.btn_open_outdir.configure(state="normal")
                elif kind == "error":
                    err, tb = payload
                    self._append_log(f"\n[ERROR] {err}\n{tb}")
                    self.status_var.set("Terjadi error.")
                    self.btn_run.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    messagebox.showerror(APP_TITLE, f"Simulasi gagal:\n{err}")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _append_log(self, text):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", text + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _clear_log(self):
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        self.txt_log.configure(state="disabled")


def _patch_entry_refs():
    """
    Tambahkan hook supaya labeled_entry() menyimpan referensi widget Entry
    di var._entry, dipakai _set_entry_state untuk enable/disable.
    """
    orig_entry_init = ttk.Entry.__init__

    def new_init(self, *args, **kwargs):
        orig_entry_init(self, *args, **kwargs)
        tv = kwargs.get("textvariable")
        if tv is not None:
            tv._entry = self

    ttk.Entry.__init__ = new_init


if __name__ == "__main__":
    _patch_entry_refs()
    app = App()
    app.mainloop()
