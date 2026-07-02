# Visualization for a whole MetamorphosisSeries. Builds on
# MetamorphosisVisualizer by feeding it the series' stitched trajectory
# (MetamorphosisSeries.as_metamorphosis()), so every single-leg plot --
# trajectory strip, v-vs-z, loss history, velocity/displacement fields,
# segmentation dice -- works unchanged across the whole chain. Only the
# views that genuinely need the leg structure (one panel per ORIGINAL
# frame rather than per substep, and a per-leg energy breakdown) are
# added here.
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from forward.MetamorphosisSeries import MetamorphosisSeries
from forward.Visualizer import MetamorphosisVisualizer


class MetamorphosisSeriesVisualizer:
    def __init__(self, series: MetamorphosisSeries, n_grid: int = 25):
        self.s = series
        self.viz = MetamorphosisVisualizer(series.as_metamorphosis(), n_grid=n_grid)
        self.T = series.legs[0].v_traj_x.shape[
            0
        ]  # steps per leg, shared across the series

    def summary(self) -> str:
        totals = self.s.total_energy()
        lines = [
            "Metamorphosis series summary",
            "=============================",
            f"{len(self.s.legs)} legs, {len(self.s.image_paths)} frames",
            f"first frame: {self.s.image_paths[0]}",
            f"last frame:  {self.s.image_paths[-1]}",
            "",
            "Total energy (sum over all legs, the quantity the series minimizes):",
        ]
        lines += [f"  {k:10s} = {v:,.4f}" for k, v in totals.items()]
        lines += ["", self.viz.summary()]
        return "\n".join(lines)

    def export_summary(self, output_path: str):
        with open(output_path, "w") as f:
            f.write(self.summary() + "\n")

    # ---- series-only views --------------------------------------------------

    def plot_input_frame_strip(self, output_path: str, max_frames: int = 12):
        # One panel per ORIGINAL input frame (leg boundary), not per
        # substep -- plot_trajectory_strip on the stitched object would
        # otherwise need len(legs)*T+1 panels.
        full = self.s.full_a_traj()
        n_frames = len(self.s.image_paths)
        boundary_idx = [n * self.T for n in range(n_frames)]
        keep = sorted(
            set(
                np.linspace(0, n_frames - 1, min(max_frames, n_frames))
                .round()
                .astype(int)
            )
        )

        fig, axes = plt.subplots(1, len(keep), figsize=(2.0 * len(keep), 2.3))
        axes = np.atleast_1d(axes)
        for ax, frame_n in zip(axes, keep):
            ax.imshow(full[boundary_idx[frame_n]], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"frame {frame_n}", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def plot_per_leg_energy(self, output_path: str):
        cols = (
            ("E_kinetic", "E_data", "E_seg")
            if self.s.legs[0].s_traj is not None
            else ("E_kinetic", "E_data")
        )
        col_idx = {"E_kinetic": 2, "E_data": 3, "E_seg": 4}
        values = {c: [leg.history[-1][col_idx[c]] for leg in self.s.legs] for c in cols}

        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(self.s.legs)), 4))
        x = np.arange(len(self.s.legs))
        width = 0.8 / len(cols)
        for i, c in enumerate(cols):
            ax.bar(x + i * width, values[c], width=width, label=c)
        ax.set_xlabel("leg")
        ax.set_ylabel("energy")
        ax.set_yscale("log")
        ax.set_xticks(x + width * (len(cols) - 1) / 2)
        ax.set_xticklabels([str(n) for n in x])
        ax.legend()
        ax.set_title("Per-leg final energy")
        plt.tight_layout()
        plt.savefig(output_path, dpi=130)
        plt.close(fig)

    def plot_loss_curves_per_leg(self, output_path: str):
        # Small-multiples grid: one subplot per leg, each showing its own
        # convergence curve with pyramid-level boundaries.  The stitched
        # plot_loss_history (N_legs × iters) is too compressed to diagnose
        # per-leg convergence; this view shows each leg independently so
        # you can spot outliers or stalled legs at a glance.
        n_legs = len(self.s.legs)
        has_seg = self.s.legs[0].s_traj is not None
        labels = ("loss", "E_kinetic", "E_data", "E_seg") if has_seg else ("loss", "E_kinetic", "E_data")

        ncols = min(n_legs, 6)
        nrows = math.ceil(n_legs / ncols)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.5 * ncols, 2.8 * nrows),
            squeeze=False,
        )

        for n, leg in enumerate(self.s.legs):
            ax = axes[n // ncols][n % ncols]
            if leg.history is None or len(leg.history) == 0:
                ax.set_visible(False)
                continue
            h = np.array(leg.history, dtype=float)
            levels = h[:, 0]
            for col, label in zip(h[:, 1 : 1 + len(labels)].T, labels):
                ax.plot(col, label=label, linewidth=0.9)
            ax.set_yscale("log")
            for x in np.flatnonzero(np.diff(levels)) + 1:
                ax.axvline(x, color="gray", linewidth=0.6, linestyle=":")
            ax.set_title(f"Leg {n}", fontsize=9)
            ax.tick_params(labelsize=7)
            if n == 0:
                ax.legend(fontsize=7)

        for n in range(n_legs, nrows * ncols):
            axes[n // ncols][n % ncols].set_visible(False)

        fig.suptitle(
            "Per-leg loss convergence  (dotted = pyramid level change)",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def plot_energy_across_series(self, output_path: str):
        # Final energy of each leg as a function of leg index — answers
        # "which transitions were geometrically expensive vs. easy?"
        # Separate subplot per component because their magnitudes differ
        # by orders of magnitude (E_kinetic >> E_data, E_seg).
        has_seg = self.s.legs[0].s_traj is not None
        cols = ("E_kinetic", "E_data", "E_seg") if has_seg else ("E_kinetic", "E_data")
        col_idx = {"E_kinetic": 2, "E_data": 3, "E_seg": 4}

        x = np.arange(len(self.s.legs))
        values = {c: [] for c in cols}
        for leg in self.s.legs:
            if leg.history is None or len(leg.history) == 0:
                for c in cols:
                    values[c].append(float("nan"))
                continue
            row = leg.history[-1]
            for c in cols:
                idx = col_idx[c]
                values[c].append(float(row[idx]) if idx < len(row) else float("nan"))

        fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4), sharey=False)
        axes = np.atleast_1d(axes)

        for ax, c in zip(axes, cols):
            ax.plot(x, values[c], marker="o", linewidth=1.5, markersize=4)
            ax.set_xticks(x)
            ax.set_xticklabels([str(n) for n in x], fontsize=8)
            ax.set_title(c, fontsize=10)
            ax.set_xlabel("leg index")
            ax.set_ylabel("final energy")
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(
            "Final energy per leg — which transitions were hard?",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ---- delegate everything else to MetamorphosisVisualizer on the stitched series

    def export_all(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.export_summary(f"{output_dir}/summary.txt")
        self.plot_input_frame_strip(f"{output_dir}/fig_input_frames.png")
        self.plot_per_leg_energy(f"{output_dir}/fig_per_leg_energy.png")
        self.plot_loss_curves_per_leg(f"{output_dir}/fig_loss_per_leg.png")
        self.plot_energy_across_series(f"{output_dir}/fig_energy_across_series.png")
        self.viz.export_all(output_dir)
