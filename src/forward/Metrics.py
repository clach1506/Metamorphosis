# Quantitative diagnostics for a fitted Metamorphosis: numeric summaries,
# convergence tests, and structured loss-curve data. No matplotlib here --
# Visualizer.py is the plotting layer and consumes these numbers/arrays
# instead of recomputing them.
import numpy as np

from forward.Metamorphosis import Metamorphosis


class MetamorphosisMetrics:
    def __init__(self, metamorphosis: Metamorphosis):
        self.m = metamorphosis
        self.T = metamorphosis.T
        self.dt = metamorphosis.dt
        self.H, self.W = metamorphosis.a_traj.shape[1:]

    # ---- scalar indicators --------------------------------------------------

    @staticmethod
    def dice_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
        pred_bin = pred > threshold
        target_bin = target > threshold
        intersection = np.logical_and(pred_bin, target_bin).sum()
        denom = pred_bin.sum() + target_bin.sum()
        return 2.0 * intersection / denom if denom else float("nan")

    def energy_breakdown(self) -> dict:
        # Final energy at the finest level.
        if self.m.history is None or len(self.m.history) == 0:
            return {}
        row = self.m.history[-1]
        level, loss, e_kinetic, e_data = row[:4]
        e_seg = row[4] if len(row) > 4 else None
        n_elems = self.T * self.H * self.W
        out = {
            "loss": float(loss),
            "E_kinetic": float(e_kinetic),
            "E_data": float(e_data),
            "rms_data": float((e_data / n_elems) ** 0.5),
        }
        if e_seg is not None:
            out["E_seg"] = float(e_seg)
            out["rms_seg"] = float((e_seg / n_elems) ** 0.5)
        return out

    def deformation_vs_residual(self) -> dict:
        # How much of the a(0) -> a(1) intensity change the deformation alone
        # explains: a(0) transported by v only (no residual) vs the target.
        # explained = 1 - rms(pure - a1) / rms(a0 - a1); 1 = pure geometry,
        # 0 = the warp doesn't help at all, i.e. everything is residual.
        a0, a1 = self.m.a_traj[0], self.m.a_target
        a_pure = self.m.pure_deformation_trajectory()[-1]
        rms_before = float(np.sqrt(np.mean((a0 - a1) ** 2)))
        rms_pure = float(np.sqrt(np.mean((a_pure - a1) ** 2)))
        return {
            "rms_a0_vs_a1": rms_before,
            "rms_pure_deformation_vs_a1": rms_pure,
            "explained_by_deformation": 1.0 - rms_pure / rms_before if rms_before else float("nan"),
            "sum_v": float(self.m.deformation_magnitude().sum()),
            "sum_z": float(self.m.residual_magnitude().sum()),
        }

    def displacement_stats(self) -> dict:
        disp = np.sqrt(
            (self.m.v_traj_x.sum(0) * self.dt) ** 2
            + (self.m.v_traj_y.sum(0) * self.dt) ** 2
        )
        return {
            "max_net_displacement": float(disp.max()),
            "mean_net_displacement": float(disp.mean()),
            "max_residual_step": float(np.abs(self.m.z_traj).max()),
        }

    def segmentation_scores(self) -> dict:
        if self.m.s_traj is None:
            return None
        # S(T) itself is anchored to S(1) (Dice trivially 1), so the useful
        # number is S(0) transported by v alone, compared to the no-motion
        # baseline Dice(S(0), S(1)).
        return {
            "dice_no_motion": self.dice_score(self.m.s_traj[0], self.m.s_target),
            "dice_pure_deformation": self.dice_score(
                self.m.pure_deformation_segmentation_trajectory()[-1], self.m.s_target
            ),
        }

    # ---- loss curve data (no plotting) --------------------------------------

    def loss_curve_data(self) -> dict:
        # Structured arrays a plotting layer can consume directly -- same
        # (level, *cols) layout as Metamorphosis.history / solver.history.
        if self.m.history is None or len(self.m.history) == 0:
            return {"iterations": np.array([]), "levels": np.array([]), "columns": {}}
        history = np.asarray(self.m.history, dtype=float)
        has_seg = history.shape[1] > 4
        labels = ("loss", "E_kinetic", "E_data", "E_seg") if has_seg else ("loss", "E_kinetic", "E_data")
        levels = history[:, 0]
        columns = {label: col for label, col in zip(labels, history[:, 1 : 1 + len(labels)].T)}
        level_boundaries = (np.flatnonzero(np.diff(levels)) + 1).tolist()
        return {
            "iterations": np.arange(len(history)),
            "levels": levels,
            "columns": columns,
            "labels": labels,
            "level_boundaries": level_boundaries,
        }

    # ---- convergence diagnostics ---------------------------------------------

    def convergence_report(self) -> list:
        # One entry per pyramid level: did it converge before its iteration
        # cap, or run out the clock? `level_iters`/tol/patience come from
        # solver_config; unknown for older saved runs, in which case
        # hit_iter_cap/iter_cap are None and only the tail-plateau heuristic
        # is reported.
        data = self.loss_curve_data()
        if len(data["iterations"]) == 0:
            return []
        loss = data["columns"]["loss"]
        levels = data["levels"]
        cfg = self.m.solver_config or {}
        level_iters = cfg.get("level_iters")
        tol = cfg.get("convergence_tol", 1e-4)
        patience = cfg.get("convergence_patience", 30)

        report = []
        for level in np.unique(levels):
            idx = np.flatnonzero(levels == level)
            level_loss = loss[idx]
            n_iters = len(level_loss)
            initial_loss, final_loss = float(level_loss[0]), float(level_loss[-1])
            rel_decrease = (
                (initial_loss - final_loss) / initial_loss if initial_loss else float("nan")
            )

            window = level_loss[-min(patience, n_iters) :]
            tail_rel_change = (
                (window[0] - window[-1]) / window[0] if window[0] else 0.0
            )
            plateaued = tail_rel_change < tol

            iter_cap = None
            hit_iter_cap = None
            if level_iters is not None and int(level) < len(level_iters):
                iter_cap = int(level_iters[int(level)])
                hit_iter_cap = n_iters >= iter_cap

            report.append(
                {
                    "level": int(level),
                    "n_iters": n_iters,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                    "relative_decrease": rel_decrease,
                    "tail_relative_change": float(tail_rel_change),
                    "plateaued_at_end": bool(plateaued),
                    "iter_cap": iter_cap,
                    "hit_iter_cap": hit_iter_cap,
                }
            )
        return report

    def converged(self) -> bool:
        # Whole-fit verdict: did the last (finest) level settle, rather than
        # get cut off still improving?
        report = self.convergence_report()
        if not report:
            return False
        last = report[-1]
        return last["plateaued_at_end"] and last["hit_iter_cap"] is not True

    # ---- text summary --------------------------------------------------------

    def summary(self) -> str:
        m = self.m
        lines = [
            "Metamorphosis summary",
            "======================",
            f"a(0): {m.path_a0 or '(unknown -- not recorded)'}",
            f"a(1): {m.path_a1 or '(unknown -- not recorded)'}",
            f"resolution solved at: {self.H}x{self.W}, T={self.T} steps",
        ]
        native_shape = (m.solver_config or {}).get("native_shape")
        if native_shape and tuple(native_shape) != (self.H, self.W):
            lines.append(
                f"  NB: input images are {native_shape[0]}x{native_shape[1]} natively -- "
                f"pyramid_scales={(m.solver_config or {}).get('pyramid_scales')} stopped "
                f"short of 1.0, so this run (and every figure/frame below) is at "
                f"{self.H}x{self.W}, not native resolution."
            )
        if m.solver_config:
            seg_str = (
                f"  lambda_seg={m.solver_config.get('lambda_seg')}"
                if m.s_traj is not None
                else ""
            )
            lines.append(
                f"lambda_data={m.solver_config.get('lambda_data')}{seg_str}  "
                f"kernel_sigma_frac={m.solver_config.get('kernel_sigma_frac')}"
            )
        lines.append("")

        eb = self.energy_breakdown()
        if eb:
            lines += [
                "Final energy (finest level):",
                f"  loss         = {eb['loss']:,.2f}",
                f"  E_kinetic    = {eb['E_kinetic']:,.2f}",
                f"  E_data       = {eb['E_data']:,.4f}  (rms {eb['rms_data']:.5f}, "
                f"~{eb['rms_data'] * 255:.2f}/255 gray levels)",
            ]
            if "E_seg" in eb:
                lines.append(
                    f"  E_seg        = {eb['E_seg']:,.4f}  (rms {eb['rms_seg']:.5f})"
                )
            lines.append("")

        dvr = self.deformation_vs_residual()
        disp = self.displacement_stats()
        lines += [
            "Geometric deformation vs. residual:",
            f"  rms(a0 - a1)                     = {dvr['rms_a0_vs_a1']:.5f}",
            f"  rms(a0 warped by v only - a1)    = {dvr['rms_pure_deformation_vs_a1']:.5f}",
            f"  -> explained by deformation      = {dvr['explained_by_deformation'] * 100:5.1f}% "
            f"(the rest is residual z)",
            f"  integral |v| dt summed = {dvr['sum_v']:,.2f} px   integral |z| dt summed = {dvr['sum_z']:,.2f}",
            f"  max net pixel displacement: {disp['max_net_displacement']:.4f}px  "
            f"(mean {disp['mean_net_displacement']:.5f}px)",
            f"  max |z| (single step):      {disp['max_residual_step']:.4f}",
        ]

        seg = self.segmentation_scores()
        if seg:
            lines += [
                "",
                "Segmentation (S(T) itself is anchored to S(1), so not reported):",
                f"  Dice S(0) vs S(1) (no motion)            = {seg['dice_no_motion']:.4f}",
                f"  Dice S(0) warped by v only vs S(1)       = {seg['dice_pure_deformation']:.4f}",
            ]

        conv = self.convergence_report()
        if conv:
            lines += ["", "Convergence per level:"]
            for r in conv:
                cap_str = (
                    f"  ({'hit iter cap -- may not be fully converged' if r['hit_iter_cap'] else 'stopped early, converged'} "
                    f"of {r['iter_cap']})"
                    if r["iter_cap"] is not None
                    else ""
                )
                lines.append(
                    f"  level {r['level']:>2d}: {r['n_iters']:4d} iters, "
                    f"loss {r['initial_loss']:,.3g} -> {r['final_loss']:,.3g} "
                    f"({r['relative_decrease'] * 100:5.1f}% decrease), "
                    f"tail plateaued={r['plateaued_at_end']}{cap_str}"
                )
        return "\n".join(lines)

    def export_summary(self, output_path: str):
        with open(output_path, "w") as f:
            f.write(self.summary() + "\n")
