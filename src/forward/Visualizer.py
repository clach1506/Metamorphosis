# Plotting for a fitted Metamorphosis. The key
# scientific diagnostic (book's premise) is the v-vs-z separation: v should
# explain geometric change, z should explain pure intensity change.
#
# Pure plotting only -- every number these figures need (Dice, RMS, energy
# breakdowns, convergence status) is computed by MetamorphosisMetrics and
# consumed here, not recomputed. Use MetamorphosisMetrics directly for
# summaries/diagnostics without generating any figures.
import os

import numpy as np
import matplotlib.pyplot as plt

from forward.Metamorphosis import Metamorphosis
from forward.Metrics import MetamorphosisMetrics


class MetamorphosisVisualizer:
    def __init__(self, metamorphosis: Metamorphosis, n_grid: int = 25):
        self.m = metamorphosis
        self.metrics = MetamorphosisMetrics(metamorphosis)
        self.T = metamorphosis.T
        self.dt = metamorphosis.dt
        self.H, self.W = metamorphosis.a_traj.shape[1:]
        self.n_grid = n_grid
        self.rows_idx = np.linspace(0, self.H - 1, n_grid).round().astype(int)
        self.cols_idx = np.linspace(0, self.W - 1, n_grid).round().astype(int)

    # ---- text summary (delegates to MetamorphosisMetrics) -------------------

    def summary(self) -> str:
        return self.metrics.summary()

    def export_summary(self, output_path: str):
        self.metrics.export_summary(output_path)

    # ---- summary figures --------------------------------------------------

    def plot_trajectory_strip(self, output_path: str):
        fig, axes = plt.subplots(1, self.T + 1, figsize=(2.0 * (self.T + 1), 2.3))
        for t, ax in enumerate(axes):
            ax.imshow(self.m.a_traj[t], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"t={t}", fontsize=9)
            ax.axis("off")
        self._save(fig, output_path)

    def plot_matching(self, output_path: str):
        # a(T) is anchored to a(1), so compare the target with a(0) transported
        # by v alone: what's left over is exactly what the residual z explains.
        a_pure = self.m.pure_deformation_trajectory()[-1]
        diff = self.m.a_target - a_pure
        explained = self.metrics.deformation_vs_residual()["explained_by_deformation"]
        vmax = max(np.abs(diff).max(), 1e-9)
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(self.m.a_traj[0], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("a(0)")
        axes[1].imshow(a_pure, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("a(0) warped by v only")
        axes[2].imshow(self.m.a_target, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Target a(1)")
        im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[3].set_title(f"a(1) - warped a(0)\n({explained * 100:.1f}% explained by v)")
        plt.colorbar(im, ax=axes[3], fraction=0.046)
        for ax in axes:
            ax.axis("off")
        self._save(fig, output_path)

    def plot_matching_segmentation(self, output_path: str):
        # Same idea as plot_matching: S(T) is anchored to S(1), so show S(0)
        # transported by v alone against the target.
        self._require_segmentation()
        s_pure = self.m.pure_deformation_segmentation_trajectory()[-1]
        seg = self.metrics.segmentation_scores()
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        axes[0].imshow(self.m.s_traj[0], cmap="gray", vmin=0, vmax=1)
        axes[0].set_title(f"S(0)\n(Dice vs S(1)={seg['dice_no_motion']:.3f})")
        axes[1].imshow(s_pure, cmap="gray", vmin=0, vmax=1)
        axes[1].contour(self.m.s_target > 0.5, levels=[0.5], colors="lime", linewidths=1.2)
        axes[1].set_title(f"S(0) warped by v only\n(Dice={seg['dice_pure_deformation']:.3f}, green = S(1))")
        axes[2].imshow(self.m.s_target, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title("Target S(1)")
        for ax in axes:
            ax.axis("off")
        self._save(fig, output_path)

    def plot_residual_on_segmentation(self, output_path: str):
        self._require_segmentation()
        z_cum_abs = self.m.residual_magnitude()
        z_cum_signed = self.m.z_traj.sum(axis=0) * self.dt

        fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

        axes[0].imshow(self.m.a_traj[0], cmap="gray", vmin=0, vmax=1)
        axes[0].contour(self.m.s_traj[0] > 0.5, levels=[0.5], colors="lime", linewidths=1.5)
        axes[0].set_title("a(0)  [green = S(0)]")

        axes[1].imshow(self.m.a_target, cmap="gray", vmin=0, vmax=1)
        axes[1].contour(self.m.s_target > 0.5, levels=[0.5], colors="lime", linewidths=1.5)
        axes[1].set_title("a(1)  [green = S(1)]")

        im2 = axes[2].imshow(z_cum_abs, cmap="inferno", vmin=0)
        axes[2].contour(self.m.s_target > 0.5, levels=[0.5], colors="lime", linewidths=1.5)
        axes[2].set_title("∫|z| dt\n[green = S(1)]")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        vmax_signed = max(np.abs(z_cum_signed).max(), 1e-9)
        im3 = axes[3].imshow(z_cum_signed, cmap="RdBu_r", vmin=-vmax_signed, vmax=vmax_signed)
        axes[3].contour(self.m.s_target > 0.5, levels=[0.5], colors="lime", linewidths=1.5)
        axes[3].set_title("Signed ∫z dt\n[green = S(1)]")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        for ax in axes:
            ax.axis("off")
        self._save(fig, output_path, dpi=140)

    def export_residual_on_segmentation_frames(self, output_dir: str):
        self._require_segmentation()
        os.makedirs(output_dir, exist_ok=True)
        vmax = max(np.abs(self.m.z_traj).max(), 1e-9)
        for t in range(self.T):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.m.z_traj[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.contour(self.m.s_traj[t] > 0.5, levels=[0.5], colors="lime", linewidths=1.5)
            ax.set_title(f"Residual z(t={t})  [green = S(t)]")
            ax.axis("off")
            self._save(fig, f"{output_dir}/residual_seg_{t:03d}.png", dpi=140)
        self._save_colorbar(f"{output_dir}/_colorbar.png", "RdBu_r", -vmax, vmax)

    def export_segmentation_frames(self, output_dir: str):
        self._require_segmentation()
        os.makedirs(output_dir, exist_ok=True)
        for t in range(self.T + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.m.s_traj[t], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"S(t={t}/{self.T})")
            ax.axis("off")
            self._save(fig, f"{output_dir}/frame_{t:03d}.png", dpi=140)

    def export_pure_deformation_segmentation_frames(self, output_dir: str):
        self._require_segmentation()
        os.makedirs(output_dir, exist_ok=True)
        s_pure_traj = self.m.pure_deformation_segmentation_trajectory()
        for t in range(self.T + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(s_pure_traj[t], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"Pure deformation S(t={t}/{self.T})")
            ax.axis("off")
            self._save(fig, f"{output_dir}/frame_{t:03d}.png", dpi=140)

    def plot_deformation_vs_residual(self, output_path: str):
        v_map = self.m.deformation_magnitude()
        z_signed = self.m.z_traj.sum(axis=0) * self.dt
        z_abs = self.m.residual_magnitude()

        fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
        axes[0].imshow(self.m.a_traj[0], cmap="gray")
        axes[0].set_title("a(0)")
        axes[1].imshow(self.m.a_target, cmap="gray")
        axes[1].set_title("a(1)")

        im2 = axes[2].imshow(v_map, cmap="inferno")
        axes[2].set_title("∫|v| dt (px)\n(geometric change)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        vmax_z = max(np.abs(z_signed).max(), 1e-9)
        im3 = axes[3].imshow(z_signed, cmap="RdBu_r", vmin=-vmax_z, vmax=vmax_z)
        axes[3].set_title("Signed ∫z dt\n(intensity change)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(z_abs, cmap="inferno")
        axes[4].set_title("∫|z| dt")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        for ax in axes:
            ax.axis("off")
        self._save(fig, output_path, dpi=140)

    def plot_loss_history(self, output_path: str):
        data = self.metrics.loss_curve_data()
        if len(data["iterations"]) == 0:
            raise ValueError("no history on this Metamorphosis -- nothing to plot")
        fig, ax = plt.subplots(figsize=(8, 4))
        for label in data["labels"]:
            ax.plot(data["columns"][label], label=label)
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.set_ylabel("energy")
        for x in data["level_boundaries"]:
            ax.axvline(x, color="gray", linewidth=0.7, linestyle=":")
        ax.legend()
        ax.set_title("Loss history (dotted lines = pyramid level change)")
        self._save(fig, output_path)

    def plot_convergence(self, output_path: str):
        # Companion to plot_loss_history: shows the iteration-over-iteration
        # relative loss change (log scale) against the convergence tolerance,
        # so a plateaued level reads as "dropped below the dashed line" and a
        # level that hit its iteration cap reads as "still above it when the
        # level ends" -- both drawn straight from MetamorphosisMetrics, no
        # recomputation here.
        data = self.metrics.loss_curve_data()
        report = self.metrics.convergence_report()
        if len(data["iterations"]) == 0:
            raise ValueError("no history on this Metamorphosis -- nothing to plot")

        loss = data["columns"]["loss"]
        levels = data["levels"]
        tol = (self.m.solver_config or {}).get("convergence_tol", 1e-4)
        rel_change = np.empty_like(loss)
        rel_change[0] = np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_change[1:] = np.abs(np.diff(loss)) / np.abs(loss[:-1])

        fig, (ax_loss, ax_rate) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax_loss.plot(loss, color="C0")
        ax_loss.set_yscale("log")
        ax_loss.set_ylabel("loss")
        ax_loss.set_title("Convergence: loss and relative change per iteration")

        ax_rate.plot(rel_change, color="C1", linewidth=0.9)
        ax_rate.axhline(tol, color="gray", linewidth=0.8, linestyle="--", label=f"tol={tol:g}")
        ax_rate.set_yscale("log")
        ax_rate.set_xlabel("iteration")
        ax_rate.set_ylabel("|Δloss| / loss")
        ax_rate.legend(fontsize=8)

        for x in data["level_boundaries"]:
            ax_loss.axvline(x, color="gray", linewidth=0.6, linestyle=":")
            ax_rate.axvline(x, color="gray", linewidth=0.6, linestyle=":")

        for r in report:
            idx = np.flatnonzero(levels == r["level"])
            x_end = idx[-1]
            label = "hit cap" if r["hit_iter_cap"] else "converged" if r["plateaued_at_end"] else "still moving"
            ax_loss.annotate(
                f"L{r['level']}: {label}",
                xy=(x_end, loss[x_end]),
                fontsize=7,
                rotation=90,
                va="bottom",
                ha="right",
            )

        self._save(fig, output_path)

    def plot_displacement_field(self, output_path: str, step: int = None):
        # Net displacement = integral of v(t) dt over the trajectory, i.e.
        # sum_t v(t)*dt -- a position offset (px), NOT a velocity. Matches
        # the `disp` quantity in summary(); without the dt factor this would
        # be T times too large and isn't a meaningful physical quantity.
        step = step or max(1, min(self.H, self.W) // 16)
        dx_net = self.m.v_traj_x.sum(axis=0) * self.dt
        dy_net = self.m.v_traj_y.sum(axis=0) * self.dt
        Y, X = np.mgrid[0 : self.H, 0 : self.W]

        fig, ax = plt.subplots(figsize=(6, 6 * self.H / self.W))
        ax.imshow(self.m.a_traj[0], cmap="gray", vmin=0, vmax=1)
        q = ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            dx_net[::step, ::step],
            -dy_net[::step, ::step],
            color="red",
            width=0.004,
        )
        ref = np.percentile(np.sqrt(dx_net**2 + dy_net**2), 99)
        ax.quiverkey(
            q,
            X=0.83,
            Y=1.04,
            U=ref,
            label=f"{ref:.3f} px (net)",
            labelpos="E",
            fontproperties={"size": 8},
        )
        ax.set_title("Net displacement field (Σ v dt), cumulated over the trajectory")
        ax.axis("off")
        self._save(fig, output_path)

    def plot_velocity_field(self, output_path: str, t: int = None, step: int = None):
        # The actual instantaneous velocity field v(t,x) at a single t --
        # unlike plot_displacement_field this is genuinely v (px/step) at
        # one moment, not a quantity collapsed across the trajectory.
        # (Time-averaging v(t,x) over t would NOT give a "real" velocity
        # distinct from the displacement field: mean_t v(t,x) ==
        # sum_t v(t,x) * dt since dt = 1/T, i.e. the exact same field
        # plot_displacement_field already shows.)
        t = self.T // 2 if t is None else t
        step = step or max(1, min(self.H, self.W) // 16)
        vx = self.m.v_traj_x[t]
        vy = self.m.v_traj_y[t]
        Y, X = np.mgrid[0 : self.H, 0 : self.W]

        fig, ax = plt.subplots(figsize=(6, 6 * self.H / self.W))
        ax.imshow(self.m.a_traj[t], cmap="gray", vmin=0, vmax=1)
        q = ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            vx[::step, ::step],
            -vy[::step, ::step],
            color="cyan",
            width=0.004,
        )
        ref = np.percentile(np.sqrt(vx**2 + vy**2), 99)
        ax.quiverkey(
            q,
            X=0.83,
            Y=1.04,
            U=ref,
            label=f"{ref:.3f} px/step",
            labelpos="E",
            fontproperties={"size": 8},
        )
        ax.set_title(f"Velocity field v(t={t}, x), instantaneous (not cumulated)")
        ax.axis("off")
        self._save(fig, output_path)

    # ---- frame sequences (for video/GIF montage) --------------------------

    def export_trajectory_frames(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        vmin, vmax = self.m.a_traj.min(), self.m.a_traj.max()
        for t in range(self.T + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.m.a_traj[t], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"a(t={t}/{self.T})")
            ax.axis("off")
            self._save(fig, f"{output_dir}/frame_{t:03d}.png", dpi=140)
        self._save_colorbar(f"{output_dir}/_colorbar.png", "gray", vmin, vmax)

    def export_deformation_grid_frames(self, output_dir: str, n_grid: int = None):
        # phi(t) = position at time t of each initial grid point, integrated
        # forward via d(phi)/dt = v(t, phi(t)), phi(0) = identity.
        #
        # Denser than the shared self.n_grid by default: a deformation grid
        # benefits from more lines to show local warping detail, unlike the
        # velocity quiver (export_velocity_frames) where dense arrows just
        # overlap and become unreadable.
        n_grid = n_grid or max(self.n_grid * 2, 50)
        rows_idx = np.linspace(0, self.H - 1, n_grid).round().astype(int)
        cols_idx = np.linspace(0, self.W - 1, n_grid).round().astype(int)

        os.makedirs(output_dir, exist_ok=True)
        yy, xx = np.meshgrid(
            np.arange(self.H, dtype=np.float64),
            np.arange(self.W, dtype=np.float64),
            indexing="ij",
        )
        phi_x, phi_y = xx.copy(), yy.copy()
        phis_x, phis_y = [phi_x.copy()], [phi_y.copy()]
        for t in range(self.T):
            vx_here = self._sample_spline(self.m.v_traj_x[t], phi_x, phi_y)
            vy_here = self._sample_spline(self.m.v_traj_y[t], phi_x, phi_y)
            phi_x = phi_x + self.dt * vx_here
            phi_y = phi_y + self.dt * vy_here
            phis_x.append(phi_x.copy())
            phis_y.append(phi_y.copy())

        for t in range(self.T + 1):
            fig, ax = plt.subplots(figsize=(6, 6))
            for i in rows_idx:
                ax.plot(phis_x[t][i, :], phis_y[t][i, :], color="k", linewidth=0.6)
            for j in cols_idx:
                ax.plot(phis_x[t][:, j], phis_y[t][:, j], color="k", linewidth=0.6)
            ax.set_xlim(0, self.W - 1)
            ax.set_ylim(self.H - 1, 0)
            ax.set_aspect("equal")
            ax.set_title(f"Pure deformation (grid), t={t}/{self.T}")
            ax.axis("off")
            self._save(fig, f"{output_dir}/grid_{t:03d}.png", dpi=140)

    def export_residual_frames(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        vmax = max(np.abs(self.m.z_traj).max(), 1e-9)
        for t in range(self.T):
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(self.m.z_traj[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"Pure residual (texture), z(t={t})")
            ax.axis("off")
            self._save(fig, f"{output_dir}/residual_{t:03d}.png", dpi=140)

        self._save_colorbar(f"{output_dir}/_colorbar.png", "RdBu_r", -vmax, vmax)

    def export_velocity_frames(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        v_mag_all = np.sqrt(self.m.v_traj_x**2 + self.m.v_traj_y**2)
        vmax_v = v_mag_all.max()
        Y, X = np.mgrid[0 : self.H, 0 : self.W]
        xs = X[np.ix_(self.rows_idx, self.cols_idx)]
        ys = Y[np.ix_(self.rows_idx, self.cols_idx)]
        spacing = min(
            (self.H - 1) / (self.n_grid - 1), (self.W - 1) / (self.n_grid - 1)
        )
        scale_v = vmax_v / (spacing * 1.5)

        for t in range(self.T):
            vxs = self.m.v_traj_x[t][np.ix_(self.rows_idx, self.cols_idx)]
            vys = self.m.v_traj_y[t][np.ix_(self.rows_idx, self.cols_idx)]
            mags = np.sqrt(vxs**2 + vys**2)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.quiver(
                xs,
                ys,
                vxs,
                vys,
                mags,
                cmap="viridis",
                angles="xy",
                scale_units="xy",
                scale=scale_v,
                clim=(0, vmax_v),
            )
            ax.set_xlim(0, self.W - 1)
            ax.set_ylim(self.H - 1, 0)
            ax.set_aspect("equal")
            ax.set_title(f"Velocity field v, t={t}")
            ax.axis("off")
            self._save(fig, f"{output_dir}/velocity_{t:03d}.png", dpi=140)

        self._save_colorbar(f"{output_dir}/_colorbar.png", "viridis", 0, vmax_v)

    def export_all(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.export_summary(f"{output_dir}/summary.txt")
        self.plot_matching(f"{output_dir}/fig_matching.png")
        self.plot_deformation_vs_residual(f"{output_dir}/fig_v_vs_z.png")
        self.plot_trajectory_strip(f"{output_dir}/fig_trajectory.png")
        self.plot_displacement_field(f"{output_dir}/fig_displacement_field.png")
        self.plot_velocity_field(f"{output_dir}/fig_velocity_field.png")
        if self.m.history is not None and len(self.m.history):
            self.plot_loss_history(f"{output_dir}/fig_loss.png")
            self.plot_convergence(f"{output_dir}/fig_convergence.png")
        if self.m.s_traj is not None:
            self.plot_matching_segmentation(
                f"{output_dir}/fig_matching_segmentation.png"
            )
            self.plot_residual_on_segmentation(
                f"{output_dir}/fig_residual_on_segmentation.png"
            )
            self.export_segmentation_frames(f"{output_dir}/frames_segmentation")
            self.export_pure_deformation_segmentation_frames(
                f"{output_dir}/frames_segmentation_pure"
            )
            self.export_residual_on_segmentation_frames(
                f"{output_dir}/frames_residual_segmentation"
            )
        self.export_trajectory_frames(f"{output_dir}/frames")
        self.export_deformation_grid_frames(f"{output_dir}/frames_deformation")
        self.export_residual_frames(f"{output_dir}/frames_residual")
        self.export_velocity_frames(f"{output_dir}/frames_velocity")

    # ---- helpers -----------------------------------------------------------

    def _require_segmentation(self):
        if self.m.s_traj is None:
            raise ValueError(
                "no segmentation data on this Metamorphosis "
                "(fit with path_s0/path_s1, or load a directory that has s_traj.npy)"
            )

    @staticmethod
    def _sample_spline(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Cubic spline sampling for the grid-deformation integration -- no
        # gradients flow through this (visualization only), so unlike the
        # warp used during optimization (SemiLagrangianWarp, which must
        # stay differentiable) we're free to use scipy here.
        from scipy.ndimage import map_coordinates

        return map_coordinates(field, [y, x], order=3, mode="nearest")

    @staticmethod
    def _save_colorbar(output_path: str, cmap: str, vmin: float, vmax: float):
        fig, cax = plt.subplots(figsize=(1.5, 6))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        plt.colorbar(sm, cax=cax)
        MetamorphosisVisualizer._save(fig, output_path, dpi=140)

    @staticmethod
    def _save(fig, output_path: str, dpi: int = 130, bbox_inches: str = "tight"):
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
