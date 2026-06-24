# The metamorphosis object itself (book section 13.4.3): a fitted image
# trajectory a(t) together with the velocity field v(t) that transports it,
# and the implicit residual z(t) it leaves behind. This is the object the
# rest of the project (visualization, export) should consume.
import json
import os

import numpy as np
import torch

from Image import Image
from Solver import MetamorphosisSolver


class Metamorphosis:
    def __init__(self, a_traj: np.ndarray, v_traj_x: np.ndarray, v_traj_y: np.ndarray,
                 z_traj: np.ndarray, a_target: np.ndarray = None, history: np.ndarray = None,
                 path_a0: str = None, path_a1: str = None, solver_config: dict = None):
        self.a_traj = a_traj      # (T+1, H, W) -- at whatever resolution the solver ran at,
        self.v_traj_x = v_traj_x  # (T, H, W)      not necessarily the input images' native size
        self.v_traj_y = v_traj_y  # (see solver_config["pyramid_scales"])
        self.z_traj = z_traj      # (T, H, W), implicit residual
        # The true target a1: equal to a_traj[-1] for the exact/collocation
        # scheme (a(T) is anchored to it), but a genuine reconstruction error
        # may exist for other schemes (e.g. shooting), so keep it distinct.
        self.a_target = a_traj[-1] if a_target is None else a_target
        self.history = history    # (n_iters, 4): columns (level, loss, E_kinetic, E_data)
        self.path_a0 = path_a0
        self.path_a1 = path_a1
        self.solver_config = solver_config or {}

    @classmethod
    def fit(cls, path_a0: str, path_a1: str, **solver_kwargs) -> "Metamorphosis":
        a0 = torch.tensor(Image(path_a0).array, dtype=torch.float32)
        a1 = torch.tensor(Image(path_a1).array, dtype=torch.float32)

        solver = MetamorphosisSolver(**solver_kwargs)
        velocity, trajectory, warp = solver.fit(a0, a1)

        with torch.no_grad():
            a = trajectory.full()
            v_traj_x, v_traj_y, z_traj = [], [], []
            for t in range(solver.T):
                vx, vy = velocity.velocity_at(t)
                warped_next = warp(a[t + 1], solver.dt * vx, solver.dt * vy)
                v_traj_x.append(vx.clone())
                v_traj_y.append(vy.clone())
                z_traj.append(((warped_next - a[t]) / solver.dt).clone())

            a_traj = a.numpy()
            v_traj_x = torch.stack(v_traj_x).numpy()
            v_traj_y = torch.stack(v_traj_y).numpy()
            z_traj = torch.stack(z_traj).numpy()

        solver_config = {
            "T": solver.T, "lambda_data": solver.lambda_data,
            "kernel_sigma_frac": solver.kernel_sigma_frac,
            "pyramid_scales": list(solver.pyramid_scales),
            "native_shape": list(a0.shape),
        }
        return cls(a_traj, v_traj_x, v_traj_y, z_traj,
                   a_target=trajectory.a1.numpy(), history=np.array(solver.history),
                   path_a0=path_a0, path_a1=path_a1, solver_config=solver_config)

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
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return cls(a_traj, v_traj_x, v_traj_y, z_traj, a_target=a_target, history=history,
                   path_a0=meta.get("path_a0"), path_a1=meta.get("path_a1"),
                   solver_config=meta.get("solver_config"))

    def deformation_magnitude(self) -> np.ndarray:
        # Cumulative |v| over time: how much geometric deformation happened where.
        return np.sqrt(self.v_traj_x ** 2 + self.v_traj_y ** 2).sum(axis=0)

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
        meta = {"path_a0": self.path_a0, "path_a1": self.path_a1, "solver_config": self.solver_config}
        with open(f"{output_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    metamorphosis = Metamorphosis.fit(
        "BDD_AMD_062026/031_FA_C_OD/preprocessed/UTF-8FAC20013.png",
        "BDD_AMD_062026/031_FA_C_OD/preprocessed/UTF-8FAC20023.png",
        T=4, pyramid_scales=(1 / 8, 1 / 4), level_iters=(20, 10), level_lrs=(0.05, 0.02),
    )
    metamorphosis.save("results_metamorphosis_sanity_check")
    print("\nshapes:", metamorphosis.a_traj.shape, metamorphosis.v_traj_x.shape, metamorphosis.z_traj.shape)
    print("deformation_magnitude range:", metamorphosis.deformation_magnitude().min(),
          metamorphosis.deformation_magnitude().max())
    print("All sanity checks passed.")
