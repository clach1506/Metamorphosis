# The metamorphosis object itself (book section 13.4.3): a fitted image
# trajectory a(t) together with the velocity field v(t) that transports it,
# and the implicit residual z(t) it leaves behind. This is the object the
# rest of the project (visualization, export) should consume.
#
# Optionally also carries a segmentation mask trajectory s_traj, fitted with
# the SAME v (see Solver.py/Energy.py's lambda_seg channel) -- None if no
# segmentation was used.

import json
import os

import numpy as np
import torch

from core.Image import Image
from forward.Solver import MetamorphosisSolver
from core.Warp import SemiLagrangianWarp


class Metamorphosis:
    def __init__(
        self,
        a_traj: np.ndarray,
        v_traj_x: np.ndarray,
        v_traj_y: np.ndarray,
        z_traj: np.ndarray,
        a_target: np.ndarray = None,
        history: np.ndarray = None,
        path_a0: str = None,
        path_a1: str = None,
        solver_config: dict = None,
        s_traj: np.ndarray = None,
        s_target: np.ndarray = None,
        path_s0: str = None,
        path_s1: str = None,
    ):
        self.a_traj = a_traj  # (T+1, H, W) -- at whatever resolution the solver ran at,
        self.v_traj_x = (
            v_traj_x  # (T, H, W)      not necessarily the input images' native size
        )
        self.v_traj_y = v_traj_y  # (see solver_config["pyramid_scales"])
        self.z_traj = z_traj  # (T, H, W), implicit residual
        # The true target a1: equal to a_traj[-1] for the exact/collocation
        # scheme (a(T) is anchored to it), but a genuine reconstruction error
        # may exist for other schemes (e.g. shooting), so keep it distinct.
        self.a_target = a_traj[-1] if a_target is None else a_target
        self.history = (
            history  # (n_iters, 4 or 5): (level, loss, E_kinetic, E_data[, E_seg])
        )
        self.path_a0 = path_a0
        self.path_a1 = path_a1
        self.solver_config = solver_config or {}

        self.s_traj = s_traj  # (T+1, H, W) segmentation mask trajectory, or None
        self.s_target = (
            s_traj[-1] if (s_traj is not None and s_target is None) else s_target
        )
        self.path_s0 = path_s0
        self.path_s1 = path_s1

    @classmethod
    def fit(
        cls,
        path_a0: str,
        path_a1: str,
        path_s0: str = None,
        path_s1: str = None,
        verbose: bool = True,
        warm_start_velocity=None,
        **solver_kwargs,
    ) -> "Metamorphosis":
        a0 = torch.tensor(Image(path_a0).array, dtype=torch.float32)
        a1 = torch.tensor(Image(path_a1).array, dtype=torch.float32)
        use_seg = path_s0 is not None and path_s1 is not None
        s0 = (
            torch.tensor(Image(path_s0).array, dtype=torch.float32) if use_seg else None
        )
        s1 = (
            torch.tensor(Image(path_s1).array, dtype=torch.float32) if use_seg else None
        )

        solver = MetamorphosisSolver(**solver_kwargs)
        velocity, trajectory, warp, mask_trajectory = solver.fit(
            a0, a1, s0, s1, verbose=verbose, warm_start_velocity=warm_start_velocity
        )

        with torch.no_grad():
            a = trajectory.full()
            v_traj_x, v_traj_y, z_traj = [], [], []
            for t in range(solver.T):
                vx, vy = velocity.velocity_at(t)
                warped_next = warp(a[t + 1], solver.dt * vx, solver.dt * vy)
                v_traj_x.append(vx.clone())
                v_traj_y.append(vy.clone())
                z_traj.append(((warped_next - a[t]) / solver.dt).clone())

            a_traj = a.cpu().numpy()
            v_traj_x = torch.stack(v_traj_x).cpu().numpy()
            v_traj_y = torch.stack(v_traj_y).cpu().numpy()
            z_traj = torch.stack(z_traj).cpu().numpy()
            s_traj = mask_trajectory.full().cpu().numpy() if use_seg else None
            s_target = mask_trajectory.a1.cpu().numpy() if use_seg else None

        solver_config = {
            "T": solver.T,
            "lambda_data": solver.lambda_data,
            "lambda_seg": solver.lambda_seg,
            "kernel_sigma_frac": solver.kernel_sigma_frac,
            "pyramid_scales": list(solver.pyramid_scales),
            "native_shape": list(a0.shape),
        }
        result = cls(
            a_traj,
            v_traj_x,
            v_traj_y,
            z_traj,
            a_target=trajectory.a1.cpu().numpy(),
            history=np.array(solver.history),
            path_a0=path_a0,
            path_a1=path_a1,
            solver_config=solver_config,
            s_traj=s_traj,
            s_target=s_target,
            path_s0=path_s0,
            path_s1=path_s1,
        )
        # Transient: used by MetamorphosisSeries for warm-start chaining only.
        # Not saved, not part of the data model.
        result._velocity = velocity
        return result

    @classmethod
    def load(cls, directory: str) -> "Metamorphosis":
        a_traj = np.load(f"{directory}/a_traj.npy")
        v_traj_x = np.load(f"{directory}/v_traj_x.npy")
        v_traj_y = np.load(f"{directory}/v_traj_y.npy")
        z_traj = np.load(f"{directory}/z_traj.npy")
        a_target = np.load(f"{directory}/a1.npy")
        history_path = f"{directory}/history.npy"
        history = np.load(history_path) if os.path.exists(history_path) else None
        meta_path = f"{directory}/meta.json"
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)

        s_traj_path = f"{directory}/s_traj.npy"
        s_traj = np.load(s_traj_path) if os.path.exists(s_traj_path) else None
        s_target_path = f"{directory}/s1.npy"
        s_target = (
            np.load(s_target_path)
            if (s_traj is not None and os.path.exists(s_target_path))
            else None
        )

        return cls(
            a_traj,
            v_traj_x,
            v_traj_y,
            z_traj,
            a_target=a_target,
            history=history,
            path_a0=meta.get("path_a0"),
            path_a1=meta.get("path_a1"),
            solver_config=meta.get("solver_config"),
            s_traj=s_traj,
            s_target=s_target,
            path_s0=meta.get("path_s0"),
            path_s1=meta.get("path_s1"),
        )

    def pure_deformation_segmentation_trajectory(self) -> np.ndarray:
        # Re-transports S(0) through the SAME fitted v(t), but by pure
        # advection (no implicit residual at all) -- unlike s_traj, whose
        # mismatch with the warp is only penalized by lambda_seg, not
        # forced to zero, so it's free to drift off the geometric flow to
        # hit s1 exactly. Comparing this against s_traj/s_target answers
        # "how much of the segmentation's change does geometry alone
        # explain?", the same v-vs-z question the book asks of intensity.
        #
        # Forward semi-Lagrangian step, derived from the same approximation
        # the energy enforces -- channel[t+1](y) = channel[t](y - dt*v(t,y)) --
        # i.e. warp(channel[t], -dt*vx, -dt*vy) evaluated at y.
        if self.s_traj is None:
            raise ValueError(
                "no segmentation data on this Metamorphosis "
                "(fit with path_s0/path_s1, or load a directory that has s_traj.npy)"
            )
        T, H, W = self.v_traj_x.shape
        # For a stitched series, solver_config["T"] holds the per-leg step count while T
        # is N_legs*T_per_leg.  Using T directly would give 1/N_legs of the intended
        # displacement per step.  For a single-pair run solver_config["T"] == T, so
        # behaviour is unchanged.
        T_per_step = (self.solver_config or {}).get("T") or T
        dt = 1.0 / T_per_step
        warp = SemiLagrangianWarp(H, W)
        s = torch.tensor(self.s_traj[0], dtype=torch.float32)
        frames = [s]
        with torch.no_grad():
            for t in range(T):
                vx = torch.tensor(self.v_traj_x[t], dtype=torch.float32)
                vy = torch.tensor(self.v_traj_y[t], dtype=torch.float32)
                s = warp(s, -dt * vx, -dt * vy)
                frames.append(s)
        return torch.stack(frames).numpy()

    def deformation_magnitude(self) -> np.ndarray:
        # Cumulative |v| over time: how much geometric deformation happened where.
        return np.sqrt(self.v_traj_x**2 + self.v_traj_y**2).sum(axis=0)

    def residual_magnitude(self) -> np.ndarray:
        # Cumulative |z| over time: how much pure intensity change happened where.
        return np.abs(self.z_traj).sum(axis=0)

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        np.save(f"{output_dir}/a_traj.npy", self.a_traj)
        np.save(f"{output_dir}/v_traj_x.npy", self.v_traj_x)
        np.save(f"{output_dir}/v_traj_y.npy", self.v_traj_y)
        np.save(f"{output_dir}/z_traj.npy", self.z_traj)
        np.save(f"{output_dir}/a0.npy", self.a_traj[0])
        np.save(f"{output_dir}/a1.npy", self.a_target)
        if self.history is not None:
            np.save(f"{output_dir}/history.npy", self.history)
        if self.s_traj is not None:
            np.save(f"{output_dir}/s_traj.npy", self.s_traj)
            np.save(f"{output_dir}/s0.npy", self.s_traj[0])
            np.save(f"{output_dir}/s1.npy", self.s_target)
        meta = {
            "path_a0": self.path_a0,
            "path_a1": self.path_a1,
            "solver_config": self.solver_config,
            "path_s0": self.path_s0,
            "path_s1": self.path_s1,
        }
        with open(f"{output_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)
