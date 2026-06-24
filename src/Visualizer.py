# Visualization and export for a fitted Metamorphosis. Consolidates
# visualize.py and 03_visualize_results.py at the repo root into one class.
# The key scientific diagnostic (book's premise) is the v-vs-z separation:
# v should explain geometric change, z should explain pure intensity change.
import os

import numpy as np
import matplotlib.pyplot as plt

from Metamorphosis import Metamorphosis


class MetamorphosisVisualizer:
    def __init__(self, metamorphosis: Metamorphosis, n_grid: int = 25):
        self.m = metamorphosis
        self.T = metamorphosis.v_traj_x.shape[0]
        self.H, self.W = metamorphosis.a_traj.shape[1:]
        self.n_grid = n_grid
        self.rows_idx = np.linspace(0, self.H - 1, n_grid).round().astype(int)
        self.cols_idx = np.linspace(0, self.W - 1, n_grid).round().astype(int)

    # ---- summary figures --------------------------------------------------

    def plot_trajectory_strip(self, output_path: str):
        fig, axes = plt.subplots(1, self.T + 1, figsize=(2.0 * (self.T + 1), 2.3))
        for t, ax in enumerate(axes):
            ax.imshow(self.m.a_traj[t], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f"t={t}", fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=130, bbox_inches='tight')
        plt.close(fig)

    def plot_matching(self, output_path: str):
        a_final = self.m.a_traj[-1]
        diff = a_final - self.m.a_target
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        axes[0].imshow(self.m.a_target, cmap='gray')
        axes[0].set_title("Target a(1)")
        axes[1].imshow(a_final, cmap='gray')
        axes[1].set_title("Reconstructed a(T)")
        im = axes[2].imshow(diff, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[2].set_title("Residual error")
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=130)
        plt.close(fig)

    def plot_deformation_vs_residual(self, output_path: str):
        v_map = self.m.deformation_magnitude()
        z_signed = self.m.z_traj.sum(axis=0)
        z_abs = self.m.residual_magnitude()

        fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
        axes[0].imshow(self.m.a_traj[0], cmap='gray')
        axes[0].set_title("a(0)")
        axes[1].imshow(self.m.a_target, cmap='gray')
        axes[1].set_title("a(1)")

        im2 = axes[2].imshow(v_map, cmap='inferno')
        axes[2].set_title("Cumulative |v|\n(geometric change)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        vmax_z = np.abs(z_signed).max()
        im3 = axes[3].imshow(z_signed, cmap='RdBu_r', vmin=-vmax_z, vmax=vmax_z)
        axes[3].set_title("Signed cumulative z\n(intensity change)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(z_abs, cmap='inferno')
        axes[4].set_title("Cumulative |z|")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close(fig)

    def plot_vector_field(self, output_path: str, step: int = None):
        step = step or max(1, min(self.H, self.W) // 16)
        vx_net = self.m.v_traj_x.sum(axis=0)
        vy_net = self.m.v_traj_y.sum(axis=0)
        Y, X = np.mgrid[0:self.H, 0:self.W]

        fig, ax = plt.subplots(figsize=(6, 6 * self.H / self.W))
        ax.imshow(self.m.a_traj[0], cmap='gray', vmin=0, vmax=1)
        q = ax.quiver(X[::step, ::step], Y[::step, ::step],
                      vx_net[::step, ::step], -vy_net[::step, ::step],
                      color='red', width=0.004)
        ref = np.percentile(np.sqrt(vx_net ** 2 + vy_net ** 2), 99)
        ax.quiverkey(q, X=0.83, Y=1.04, U=ref, label=f'{ref:.3f} px (net)', labelpos='E',
                     fontproperties={'size': 8})
        ax.set_title("Net deformation field v(x), cumulated over the trajectory")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=130)
        plt.close(fig)

    # ---- frame sequences (for video/GIF montage) --------------------------

    def export_trajectory_frames(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        vmin, vmax = self.m.a_traj.min(), self.m.a_traj.max()
        for t in range(self.T + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.m.a_traj[t], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"a(t={t}/{self.T})")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/frame_{t:03d}.png", dpi=140)
            plt.close(fig)

    def export_deformation_grid_frames(self, output_dir: str):
        # phi(t) = position at time t of each initial grid point, integrated
        # forward via d(phi)/dt = v(t, phi(t)), phi(0) = identity.
        os.makedirs(output_dir, exist_ok=True)
        dt = 1.0 / self.T
        yy, xx = np.meshgrid(np.arange(self.H, dtype=np.float64),
                              np.arange(self.W, dtype=np.float64), indexing='ij')
        phi_x, phi_y = xx.copy(), yy.copy()
        phis_x, phis_y = [phi_x.copy()], [phi_y.copy()]
        for t in range(self.T):
            vx_here = self._bilinear_sample(self.m.v_traj_x[t], phi_x, phi_y)
            vy_here = self._bilinear_sample(self.m.v_traj_y[t], phi_x, phi_y)
            phi_x = phi_x + dt * vx_here
            phi_y = phi_y + dt * vy_here
            phis_x.append(phi_x.copy())
            phis_y.append(phi_y.copy())

        for t in range(self.T + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            for i in self.rows_idx:
                ax.plot(phis_x[t][i, :], phis_y[t][i, :], color='k', linewidth=0.6)
            for j in self.cols_idx:
                ax.plot(phis_x[t][:, j], phis_y[t][:, j], color='k', linewidth=0.6)
            ax.set_xlim(0, self.W - 1)
            ax.set_ylim(self.H - 1, 0)
            ax.set_aspect('equal')
            ax.set_title(f"Pure deformation (grid), t={t}/{self.T}")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/grid_{t:03d}.png", dpi=140)
            plt.close(fig)

    def export_residual_frames(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        vmax = np.abs(self.m.z_traj).max()
        for t in range(self.T):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.m.z_traj[t], cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f"Pure residual (texture), z(t={t})")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/residual_{t:03d}.png", dpi=140)
            plt.close(fig)
        self._save_colorbar(f"{output_dir}/_colorbar.png", 'RdBu_r', -vmax, vmax)

    def export_velocity_frames(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        v_mag_all = np.sqrt(self.m.v_traj_x ** 2 + self.m.v_traj_y ** 2)
        vmax_v = v_mag_all.max()
        Y, X = np.mgrid[0:self.H, 0:self.W]
        xs = X[np.ix_(self.rows_idx, self.cols_idx)]
        ys = Y[np.ix_(self.rows_idx, self.cols_idx)]
        spacing = min((self.H - 1) / (self.n_grid - 1), (self.W - 1) / (self.n_grid - 1))
        scale_v = vmax_v / (spacing * 1.5)

        for t in range(self.T):
            vxs = self.m.v_traj_x[t][np.ix_(self.rows_idx, self.cols_idx)]
            vys = self.m.v_traj_y[t][np.ix_(self.rows_idx, self.cols_idx)]
            mags = np.sqrt(vxs ** 2 + vys ** 2)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.quiver(xs, ys, vxs, vys, mags, cmap='viridis', angles='xy',
                      scale_units='xy', scale=scale_v, clim=(0, vmax_v))
            ax.set_xlim(0, self.W - 1)
            ax.set_ylim(self.H - 1, 0)
            ax.set_aspect('equal')
            ax.set_title(f"Velocity field v, t={t}")
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/velocity_{t:03d}.png", dpi=140)
            plt.close(fig)
        self._save_colorbar(f"{output_dir}/_colorbar.png", 'viridis', 0, vmax_v)

    def export_all(self, output_dir: str):
        self.plot_matching(f"{output_dir}/fig_matching.png")
        self.plot_deformation_vs_residual(f"{output_dir}/fig_v_vs_z.png")
        self.plot_trajectory_strip(f"{output_dir}/fig_trajectory.png")
        self.plot_vector_field(f"{output_dir}/fig_vector_field.png")
        self.export_trajectory_frames(f"{output_dir}/frames")
        self.export_deformation_grid_frames(f"{output_dir}/frames_deformation")
        self.export_residual_frames(f"{output_dir}/frames_residual")
        self.export_velocity_frames(f"{output_dir}/frames_velocity")

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _bilinear_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        H, W = field.shape
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        wx, wy = x - x0, y - y0
        Ia, Ib = field[y0, x0], field[y1, x0]
        Ic, Id = field[y0, x1], field[y1, x1]
        return (Ia * (1 - wx) * (1 - wy) + Ic * wx * (1 - wy)
                + Ib * (1 - wx) * wy + Id * wx * wy)

    @staticmethod
    def _save_colorbar(output_path: str, cmap: str, vmin: float, vmax: float):
        fig, cax = plt.subplots(figsize=(1.5, 6))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        plt.colorbar(sm, cax=cax)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close(fig)
