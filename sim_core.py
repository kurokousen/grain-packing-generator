"""
sim_core.py
Inti algoritma grain packing / porous media generator.
Direfactor dari script asli (berbasis input()) menjadi class dengan
callback log/progress supaya bisa dipakai dari GUI (thread terpisah)
tanpa mengubah logika fisik/algoritma aslinya.

Perbaikan kecil dibanding script asli (dicatat supaya jelas):
- Grain yang gagal divoxelisasi (nvox > MAX_VOXEL_PER_GRAIN) TIDAK lagi
  dihitung sebagai grain diterima (menghindari "ghost grain" yang
  membuat statistik menyesatkan).
- MAX_OVERLAP di-clip ke [0, 1].
- Ada mekanisme cancel (berhenti di tengah jalan) untuk GUI.
- Semua print/tqdm diganti callback log(...) dan progress(...).
"""

import os
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import tifffile
import imageio.v2 as imageio
import pandas as pd

MAX_VOXEL_PER_GRAIN = 5_000_000
MAX_ATTEMPT = 2_000_000


class SimulationCancelled(Exception):
    pass


class GrainPackingSimulation:
    def __init__(self, params: dict, log=print, progress=lambda frac, text="": None,
                 should_cancel=lambda: False):
        """
        params: dict berisi semua parameter (lihat GUI untuk daftar lengkap key).
        log(text): callback untuk menampilkan pesan ke GUI/console.
        progress(frac, text): callback 0..1 untuk progress bar, + teks status.
        should_cancel(): callback -> True kalau user menekan Stop.
        """
        self.p = params
        self.log = log
        self.progress = progress
        self.should_cancel = should_cancel

        self._validate_and_derive()

        # State simulasi
        self.solid = np.zeros((self.RES, self.RES, self.RES), dtype=bool)
        self.centers = []
        self.radii = []
        self.tree = None

        self.current_overlap = self.MAX_OVERLAP
        self.current_r_factor = 1.0
        self.consecutive_rejects = 0
        self.total_rejects = 0
        self.last_adapt_grain = 0
        self.adapt_log = []
        self.skipped_positions = []
        self.failed_voxelization = 0

        self.grain = 0
        self.attempt = 0

    # ------------------------------------------------------------
    def _validate_and_derive(self):
        p = self.p
        self.BOX_SIZE = float(p["box_size"])
        self.CORE_SIZE = float(p["core_size"])
        self.RES = int(p["res"])

        if self.CORE_SIZE >= self.BOX_SIZE:
            raise ValueError("CORE_SIZE harus lebih kecil dari BOX_SIZE")
        if self.RES < 4:
            raise ValueError("Resolusi voxel terlalu kecil (minimal 4)")

        self.VOXEL = self.BOX_SIZE / self.RES

        self.mode = int(p["mode"])  # 1 target grain, 2 target porositas
        if self.mode == 1:
            self.TARGET_GRAIN = int(p["target_grain"])
            self.TARGET_POROSITY = None
        else:
            self.TARGET_POROSITY = float(p["target_porosity"])
            if not (0.0 <= self.TARGET_POROSITY < 1.0):
                raise ValueError("Target porositas harus antara 0 dan 1")
            self.PORO_TOL = 0.01
            self.TARGET_GRAIN = 10 ** 9

        self.POSITION_MODE = int(p.get("position_mode", 1)) if self.mode == 1 else 1
        self.MANUAL_POSITIONS = None
        self.FORCE_OVERLAP_CHECK = True
        if self.POSITION_MODE == 2:
            positions = p.get("manual_positions", [])
            if len(positions) == 0:
                raise ValueError("Mode posisi manual dipilih tapi tidak ada koordinat diberikan")
            self.MANUAL_POSITIONS = [np.array(pos, dtype=float) for pos in positions]
            self.FORCE_OVERLAP_CHECK = bool(p.get("force_overlap_check", True))

        self.SHAPE_MODE = int(p["shape_mode"])
        if self.SHAPE_MODE in (1, 2):
            self.N_VERTEX = int(p["n_vertex"])
            self.ANGULARITY = float(np.clip(p["angularity"], 0.0, 1.0))
        else:
            self.N_VERTEX = int(p["n_vertex"])
            self.ANGULARITY = 0.0

        self.MAX_OVERLAP = float(np.clip(p["max_overlap"], 0.0, 1.0))

        if self.SHAPE_MODE == 4:
            self.ELLIPSOID_RX = float(p["ellipsoid_rx"])
            self.ELLIPSOID_RY = float(p["ellipsoid_ry"])
            self.ELLIPSOID_RZ = float(p["ellipsoid_rz"])
            self.ELLIPSOID_ORIENT_MODE = int(p["ellipsoid_orient_mode"])
            self.FIXED_ROTATION_MATRIX = np.eye(3) if self.ELLIPSOID_ORIENT_MODE == 1 else None
        else:
            self.ELLIPSOID_RX = self.ELLIPSOID_RY = self.ELLIPSOID_RZ = 1.0
            self.ELLIPSOID_ORIENT_MODE = 2
            self.FIXED_ROTATION_MATRIX = None

        self.SORT_MODE = int(p["sort_mode"])
        if self.SORT_MODE == 1:
            self.R_FIXED_SORT = float(p["r_fixed_sort"])
            self.MEAN_RADIUS = self.R_FIXED_SORT
        else:
            self.MEAN_RADIUS = float(p["mean_radius"])
            self.R_FIXED_SORT = None

        self.SORT_SIGMA = {1: 0.05, 2: 0.35, 3: 0.75}[self.SORT_MODE]

        if self.MEAN_RADIUS < self.VOXEL:
            raise ValueError("Grain lebih kecil dari voxel! Naikkan resolusi atau perbesar grain.")

        self.ADAPTIVE_ON = bool(p.get("adaptive_on", False))
        self.ADAPTIVE_MODE = 0
        if self.ADAPTIVE_ON:
            self.ADAPTIVE_MODE = int(p["adaptive_mode"])
            self.REJECT_COUNT_THRESH = int(p["reject_count_thresh"])
            self.REJECT_RATIO_THRESH = float(p["reject_ratio_thresh"])
            if self.ADAPTIVE_MODE in (1, 3):
                self.MAX_OVERLAP_LIMIT = float(np.clip(p["max_overlap_limit"], 0.0, 1.0))
                self.OVERLAP_STEP = float(p["overlap_step"])
            if self.ADAPTIVE_MODE in (2, 3):
                self.MIN_RADIUS_FACTOR = float(p["min_radius_factor"])
                self.RADIUS_SHRINK_STEP = float(p["radius_shrink_step"])

        self.OUT_FORMAT = p["out_format"]  # "tif" atau "bmp"
        self.OUTDIR = p["outdir"]
        self.SLICE_DIR = os.path.join(self.OUTDIR, "slices")
        os.makedirs(self.OUTDIR, exist_ok=True)
        if self.OUT_FORMAT == "bmp":
            os.makedirs(self.SLICE_DIR, exist_ok=True)

        self.MARGIN = (self.BOX_SIZE - self.CORE_SIZE) / 2
        self.core_min = int(round(self.MARGIN / self.VOXEL))
        self.core_max = int(round((self.BOX_SIZE - self.MARGIN) / self.VOXEL))
        self.core_max = max(self.core_max, self.core_min + 1)
        self.core_max = min(self.core_max, self.RES)

    # ------------------------------------------------------------
    # Helper functions (identik secara logika dengan script asli)
    # ------------------------------------------------------------
    def random_radius(self, r_factor=1.0):
        if self.SORT_MODE == 1:
            r = np.random.normal(self.R_FIXED_SORT, self.R_FIXED_SORT * 0.03)
        else:
            r = np.random.lognormal(np.log(self.MEAN_RADIUS), self.SORT_SIGMA)
        r *= r_factor
        r = max(r, self.VOXEL * 1.5)
        r = min(r, self.MEAN_RADIUS * 2.0)
        return r

    @staticmethod
    def random_rotation_matrix():
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        psi = np.random.uniform(0, 2 * np.pi)
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
        Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                        [0, 1, 0],
                        [-np.sin(phi), 0, np.cos(phi)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                        [np.sin(psi), np.cos(psi), 0],
                        [0, 0, 1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def generate_vertices(n, angularity):
        pts = np.random.normal(size=(n, 3))
        pts /= np.linalg.norm(pts, axis=1)[:, None]
        if angularity > 0:
            noise = np.random.normal(scale=angularity, size=pts.shape)
            pts += noise
        pts /= np.linalg.norm(pts, axis=1)[:, None]
        return pts

    def generate_hull(self, center, r):
        if self.SHAPE_MODE == 3:
            pts = self.generate_vertices(self.N_VERTEX, 0.0) * r
            return ConvexHull(pts + center)

        elif self.SHAPE_MODE == 4:
            pts = self.generate_vertices(self.N_VERTEX, 0.0)
            pts[:, 0] *= r * self.ELLIPSOID_RX
            pts[:, 1] *= r * self.ELLIPSOID_RY
            pts[:, 2] *= r * self.ELLIPSOID_RZ
            R = self.random_rotation_matrix() if self.FIXED_ROTATION_MATRIX is None else self.FIXED_ROTATION_MATRIX
            pts = pts @ R.T
            return ConvexHull(pts + center)

        elif self.SHAPE_MODE == 2:
            pts = self.generate_vertices(self.N_VERTEX, self.ANGULARITY)
            a, b, c = r * np.random.uniform(0.5, 1.6, 3)
            pts[:, 0] *= a
            pts[:, 1] *= b
            pts[:, 2] *= c
            deform = np.random.uniform(0.7, 1.3, size=(self.N_VERTEX, 1))
            pts *= deform
            R = self.random_rotation_matrix()
            pts = pts @ R.T
            return ConvexHull(pts + center)

        else:  # SHAPE_MODE == 1
            pts = self.generate_vertices(self.N_VERTEX, self.ANGULARITY)
            scale = r * (0.6 + 0.8 * np.random.rand(self.N_VERTEX, 1))
            pts *= scale
            return ConvexHull(pts + center)

    def voxelize_hull(self, hull):
        v = hull.points[hull.vertices]
        mn = np.clip(np.floor(v.min(0) / self.VOXEL).astype(int), 0, self.RES - 1)
        mx = np.clip(np.ceil(v.max(0) / self.VOXEL).astype(int), 0, self.RES - 1)

        xs = np.arange(mn[0], mx[0] + 1)
        ys = np.arange(mn[1], mx[1] + 1)
        zs = np.arange(mn[2], mx[2] + 1)
        nx, ny, nz = len(xs), len(ys), len(zs)
        nvox = nx * ny * nz

        if nvox > MAX_VOXEL_PER_GRAIN:
            self.log(f"[WARNING] Grain terlalu besar ({nvox:,} voxel), dilewati (tidak dihitung).")
            return None

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        P = np.column_stack([
            (X.ravel() + 0.5) * self.VOXEL,
            (Y.ravel() + 0.5) * self.VOXEL,
            (Z.ravel() + 0.5) * self.VOXEL
        ])

        eq = hull.equations
        chunk_size = 100000
        inside_parts = []
        for start in range(0, len(P), chunk_size):
            stop = min(start + chunk_size, len(P))
            inside_chunk = np.all(P[start:stop] @ eq[:, :3].T + eq[:, 3] <= 0, axis=1)
            inside_parts.append(inside_chunk)
        inside = np.concatenate(inside_parts)

        return X.ravel()[inside], Y.ravel()[inside], Z.ravel()[inside]

    def porosity_core(self, vol):
        core = vol[self.core_min:self.core_max,
                   self.core_min:self.core_max,
                   self.core_min:self.core_max]
        return 1.0 - np.count_nonzero(core) / core.size

    @staticmethod
    def compute_surface_area(vol, voxel):
        sa = 0
        sa += np.logical_and(vol[1:, :, :], ~vol[:-1, :, :]).sum()
        sa += np.logical_and(vol[:-1, :, :], ~vol[1:, :, :]).sum()
        sa += np.logical_and(vol[:, 1:, :], ~vol[:, :-1, :]).sum()
        sa += np.logical_and(vol[:, :-1, :], ~vol[:, 1:, :]).sum()
        sa += np.logical_and(vol[:, :, 1:], ~vol[:, :, :-1]).sum()
        sa += np.logical_and(vol[:, :, :-1], ~vol[:, :, 1:]).sum()
        return sa * (voxel ** 2)

    def try_adapt(self, grain_accepted):
        if not self.ADAPTIVE_ON:
            return False

        reject_ratio = self.consecutive_rejects / max(grain_accepted, 1)
        cond_count = self.consecutive_rejects >= self.REJECT_COUNT_THRESH
        cond_ratio = reject_ratio >= self.REJECT_RATIO_THRESH
        if not (cond_count or cond_ratio):
            return False

        adapted = False
        if self.ADAPTIVE_MODE in (1, 3):
            if self.current_overlap < self.MAX_OVERLAP_LIMIT:
                old_val = round(self.current_overlap, 4)
                self.current_overlap = min(self.current_overlap + self.OVERLAP_STEP, self.MAX_OVERLAP_LIMIT)
                self.adapt_log.append({"grain_accepted": grain_accepted, "event": "overlap_increase",
                                        "old": old_val, "new": round(self.current_overlap, 4)})
                self.log(f"  [ADAPTIF] Grain={grain_accepted} | Reject={self.consecutive_rejects} "
                         f"(ratio={reject_ratio:.2f}) -> overlap {old_val:.3f} -> {self.current_overlap:.3f}")
                adapted = True
                if self.ADAPTIVE_MODE == 3 and self.current_overlap >= self.MAX_OVERLAP_LIMIT:
                    self.ADAPTIVE_MODE = 2
                    self.log("  [ADAPTIF] Overlap limit tercapai -> beralih ke perkecil butiran")

        if self.ADAPTIVE_MODE == 2:
            min_factor = self.MIN_RADIUS_FACTOR
            if self.current_r_factor > min_factor:
                old_val = round(self.current_r_factor, 4)
                self.current_r_factor = max(self.current_r_factor - self.RADIUS_SHRINK_STEP, min_factor)
                self.adapt_log.append({"grain_accepted": grain_accepted, "event": "radius_shrink",
                                        "old": old_val, "new": round(self.current_r_factor, 4)})
                self.log(f"  [ADAPTIF] Grain={grain_accepted} | Reject={self.consecutive_rejects} "
                         f"(ratio={reject_ratio:.2f}) -> r_factor {old_val:.3f} -> {self.current_r_factor:.3f}")
                adapted = True

        if adapted:
            self.consecutive_rejects = 0
            self.last_adapt_grain = grain_accepted
        return adapted

    def check_overlap_manual(self, center, r):
        if not self.centers:
            return False
        tree = cKDTree(np.array(self.centers))
        idx = tree.query_ball_point(center, r * 2)
        return any(
            np.linalg.norm(center - self.centers[i]) < (r + self.radii[i]) * (1 - self.current_overlap)
            for i in idx
        )

    def _place_grain(self, center, r):
        """Coba voxelisasi & tempatkan grain. Return True kalau berhasil diterima."""
        hull = self.generate_hull(center, r)
        result = self.voxelize_hull(hull)
        if result is None:
            self.failed_voxelization += 1
            return False
        gx, gy, gz = result
        self.solid[gx, gy, gz] = True
        self.centers.append(center)
        self.radii.append(r)
        self.grain += 1
        return True

    # ------------------------------------------------------------
    def run(self):
        p = self.p
        self.log(f"Voxel size: {self.VOXEL:.4f} mm")
        self.log("Simulasi dimulai...\n")

        if self.POSITION_MODE == 2:
            self._run_manual_positions()
        else:
            self._run_random_positions()

        return self._finalize()

    def _run_manual_positions(self):
        total = len(self.MANUAL_POSITIONS)
        for i, center in enumerate(self.MANUAL_POSITIONS):
            if self.should_cancel():
                self.log("Dibatalkan oleh pengguna.")
                break
            self.attempt += 1
            r = self.random_radius(self.current_r_factor)

            if np.any(center - r < 0) or np.any(center + r > self.BOX_SIZE):
                self.log(f"  [SKIP] Grain #{i+1} di {center.tolist()} keluar batas box, dilewati.")
                self.skipped_positions.append({"grain_index": i + 1, "position": center.tolist(),
                                                "reason": "keluar batas box"})
                self.progress((i + 1) / total, f"Grain {i+1}/{total}")
                continue

            rejected = self.FORCE_OVERLAP_CHECK and self.check_overlap_manual(center, r)
            if rejected:
                self.total_rejects += 1
                self.log(f"  [SKIP] Grain #{i+1} di {center.tolist()} overlap, dilewati.")
                self.skipped_positions.append({"grain_index": i + 1, "position": center.tolist(),
                                                "reason": "overlap"})
                self.progress((i + 1) / total, f"Grain {i+1}/{total}")
                continue

            self._place_grain(center, r)
            self.progress((i + 1) / total, f"Grain {i+1}/{total}")

    def _run_random_positions(self):
        target_display = self.TARGET_GRAIN if self.mode == 1 else 100
        while self.grain < self.TARGET_GRAIN and self.attempt < MAX_ATTEMPT:
            if self.should_cancel():
                self.log("Dibatalkan oleh pengguna.")
                break

            self.attempt += 1
            r = self.random_radius(self.current_r_factor)
            center = np.random.uniform(r, self.BOX_SIZE - r, size=3)

            rejected = False
            if self.centers:
                if self.tree is None or self.grain % 20 == 0:
                    self.tree = cKDTree(np.array(self.centers))
                idx = self.tree.query_ball_point(center, r * 2)
                if any(
                    np.linalg.norm(center - self.centers[i]) < (r + self.radii[i]) * (1 - self.current_overlap)
                    for i in idx
                ):
                    rejected = True

            if rejected:
                self.total_rejects += 1
                self.consecutive_rejects += 1
                self.try_adapt(self.grain)
                continue

            self.consecutive_rejects = 0
            placed = self._place_grain(center, r)
            if not placed:
                continue

            if self.mode == 1:
                self.progress(self.grain / self.TARGET_GRAIN, f"Grain {self.grain}/{self.TARGET_GRAIN}")
            else:
                phi_core = self.porosity_core(self.solid)
                denom = (1 - self.TARGET_POROSITY)
                prog = (1 - phi_core) / denom if denom > 0 else 0
                prog = min(max(prog, 0.0), 1.0)
                self.progress(prog, f"Porositas saat ini: {phi_core:.4f} (target {self.TARGET_POROSITY:.4f})")
                if abs(phi_core - self.TARGET_POROSITY) < self.PORO_TOL:
                    break

        if self.attempt >= MAX_ATTEMPT:
            self.log("[PERINGATAN] Batas maksimum percobaan tercapai sebelum target terpenuhi.")

    # ------------------------------------------------------------
    def _finalize(self):
        solid_core = self.solid[self.core_min:self.core_max,
                                 self.core_min:self.core_max,
                                 self.core_min:self.core_max]
        final_phi = self.porosity_core(self.solid)
        surface_area = self.compute_surface_area(solid_core, self.VOXEL)

        V_core = self.CORE_SIZE ** 3
        Sv = surface_area / V_core if V_core > 0 else 0.0

        if Sv > 0 and 0.0 < final_phi < 1.0:
            permeability = (final_phi ** 3) / (5.0 * Sv ** 2 * (1 - final_phi) ** 2)
        else:
            permeability = float("nan")
            self.log("[PERINGATAN] Tidak bisa menghitung permeabilitas secara valid "
                     "(porositas 0/1 atau luas permukaan 0). Nilai diset NaN.")

        shape_names = {1: "Convex Hull acak", 2: "Ellipsoid acak",
                       3: "Bulat sempurna", 4: "Ellipsoid sempurna"}

        self.log(f"\n{'='*50}")
        self.log(f"Mode bentuk       : {shape_names[self.SHAPE_MODE]}")
        self.log(f"Mode posisi       : {'Manual' if self.POSITION_MODE == 2 else 'Acak'}")
        self.log(f"Total grain       : {self.grain}")
        self.log(f"Total attempt     : {self.attempt}")
        self.log(f"Total reject      : {self.total_rejects}")
        self.log(f"Reject ratio      : {self.total_rejects/max(self.attempt,1):.3f}")
        if self.failed_voxelization:
            self.log(f"Gagal voxelisasi  : {self.failed_voxelization} (grain terlalu besar, tidak dihitung)")
        if self.POSITION_MODE == 2:
            self.log(f"Grain dilewati    : {len(self.skipped_positions)}")
        self.log(f"Porositas CORE    : {final_phi:.4f}")
        self.log(f"Surface Area      : {surface_area:.4e} mm^2")
        self.log(f"Permeabilitas     : {permeability:.4e} mm^2")
        if self.ADAPTIVE_ON:
            self.log(f"Final overlap     : {self.current_overlap:.4f}")
            self.log(f"Final r_factor    : {self.current_r_factor:.4f}")
            self.log(f"Jumlah adaptasi   : {len(self.adapt_log)}")
        self.log('=' * 50)

        # ---------------- Output volume ----------------
        solid_uint8 = (solid_core * 255).astype(np.uint8)
        if self.OUT_FORMAT == "tif":
            tifffile.imwrite(os.path.join(self.OUTDIR, "volume_core.tif"), solid_uint8)
        else:
            for z in range(solid_uint8.shape[2]):
                imageio.imwrite(os.path.join(self.SLICE_DIR, f"slice_{z:04d}.bmp"), solid_uint8[:, :, z])

        # ---------------- Output excel ----------------
        radii_arr = np.array(self.radii) if len(self.radii) > 0 else np.array([0.0])
        summary = pd.DataFrame([{
            "ShapeMode": shape_names[self.SHAPE_MODE],
            "PositionMode": "Manual" if self.POSITION_MODE == 2 else "Acak",
            "TotalGrain": self.grain,
            "TotalAttempt": self.attempt,
            "TotalReject": self.total_rejects,
            "RejectRatio": round(self.total_rejects / max(self.attempt, 1), 4),
            "FailedVoxelization": self.failed_voxelization,
            "SkippedManual": len(self.skipped_positions) if self.POSITION_MODE == 2 else 0,
            "Porosity": final_phi,
            "SurfaceArea_mm2": surface_area,
            "Permeability_mm2": permeability,
            "MeanRadius_mm": radii_arr.mean(),
            "StdRadius_mm": radii_arr.std(),
            "FinalOverlap": self.current_overlap,
            "FinalRadiusFactor": self.current_r_factor,
        }])

        xlsx_path = os.path.join(self.OUTDIR, "results.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
            summary.to_excel(w, sheet_name="summary", index=False)
            pd.DataFrame({"Radius_mm": radii_arr}).to_excel(w, sheet_name="radius_distribution", index=False)
            if self.ADAPTIVE_ON and self.adapt_log:
                pd.DataFrame(self.adapt_log).to_excel(w, sheet_name="adapt_log", index=False)
            if self.POSITION_MODE == 2 and self.skipped_positions:
                pd.DataFrame(self.skipped_positions).to_excel(w, sheet_name="skipped_manual", index=False)

        self.log(f"\nOUTPUT SELESAI di folder: '{self.OUTDIR}'")

        return {
            "grain": self.grain,
            "attempt": self.attempt,
            "total_rejects": self.total_rejects,
            "porosity": final_phi,
            "surface_area": surface_area,
            "permeability": permeability,
            "outdir": self.OUTDIR,
            "xlsx_path": xlsx_path,
        }
