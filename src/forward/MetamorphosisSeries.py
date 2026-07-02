# A whole longitudinal series I_1..I_N, metamorphosed leg by leg
# (leg n: I_n -> I_{n+1}) and minimized jointly as
#
#   sum_{n=1}^{N} E_n(v^n, z^n)
#
# i.e. the sum of N per-leg book-formula (13.13) energies (see Energy.py),
# one per consecutive pair. The energy has no term coupling different legs,
# so the sum is minimized exactly by minimizing each leg independently --
# this class is an orchestration layer around the existing single-pair
# Metamorphosis.fit, not a new numerical scheme.
import json
import os

import numpy as np

from forward.Metamorphosis import Metamorphosis


class MetamorphosisSeries:
    def __init__(
        self,
        legs: list,
        image_paths: list,
        mask_paths: list = None,
        solver_config: dict = None,
    ):
        self.legs = legs  # [Metamorphosis], one per consecutive pair, len N
        self.image_paths = image_paths  # [str], len N+1
        self.mask_paths = mask_paths  # [str] or None, len N+1
        self.solver_config = solver_config or {}

    @classmethod
    def fit(
        cls,
        image_paths: list,
        mask_paths: list = None,
        verbose: bool = True,
        warm_start: bool = False,
        **solver_kwargs,
    ) -> "MetamorphosisSeries":
        if len(image_paths) < 2:
            raise ValueError(
                f"need at least 2 images to form a series, got {len(image_paths)}"
            )
        use_seg = mask_paths is not None
        if use_seg and len(mask_paths) != len(image_paths):
            raise ValueError(
                f"mask_paths ({len(mask_paths)}) must match image_paths ({len(image_paths)})"
            )

        n_legs = len(image_paths) - 1
        legs = []
        prev_velocity = None
        for n in range(n_legs):
            if verbose:
                ws_str = "  [warm start]" if (warm_start and prev_velocity is not None) else ""
                print(
                    f"\n=== Leg {n + 1}/{n_legs}: "
                    f"{os.path.basename(image_paths[n])} -> {os.path.basename(image_paths[n + 1])}"
                    f"{ws_str} ==="
                )
            leg = Metamorphosis.fit(
                image_paths[n],
                image_paths[n + 1],
                path_s0=mask_paths[n] if use_seg else None,
                path_s1=mask_paths[n + 1] if use_seg else None,
                verbose=verbose,
                warm_start_velocity=prev_velocity if warm_start else None,
                **solver_kwargs,
            )
            if warm_start:
                prev_velocity = getattr(leg, "_velocity", None)
            legs.append(leg)

        solver_config = legs[0].solver_config
        return cls(legs, image_paths, mask_paths, solver_config)

    @classmethod
    def load(cls, directory: str) -> "MetamorphosisSeries":
        with open(f"{directory}/meta.json") as f:
            meta = json.load(f)
        n_legs = meta["n_legs"]
        legs = [Metamorphosis.load(f"{directory}/leg_{n:03d}") for n in range(n_legs)]
        return cls(
            legs, meta["image_paths"], meta.get("mask_paths"), meta.get("solver_config")
        )

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for n, leg in enumerate(self.legs):
            leg.save(f"{output_dir}/leg_{n:03d}")
        meta = {
            "n_legs": len(self.legs),
            "image_paths": self.image_paths,
            "mask_paths": self.mask_paths,
            "solver_config": self.solver_config,
        }
        with open(f"{output_dir}/meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ---- stitching: each leg's a(T) is hard-anchored to the next leg's
    # a(0) (the collocation scheme reconstructs the target exactly), so the
    # shared boundary frame is dropped from every leg but the last.

    def full_a_traj(self) -> np.ndarray:
        return np.concatenate(
            [leg.a_traj[:-1] for leg in self.legs[:-1]] + [self.legs[-1].a_traj], axis=0
        )

    def full_v_traj(self):
        v_x = np.concatenate([leg.v_traj_x for leg in self.legs], axis=0)
        v_y = np.concatenate([leg.v_traj_y for leg in self.legs], axis=0)
        return v_x, v_y

    def full_z_traj(self) -> np.ndarray:
        return np.concatenate([leg.z_traj for leg in self.legs], axis=0)

    def full_s_traj(self) -> np.ndarray:
        if self.legs[0].s_traj is None:
            raise ValueError(
                "no segmentation data on this series (fit with mask_paths)"
            )
        return np.concatenate(
            [leg.s_traj[:-1] for leg in self.legs[:-1]] + [self.legs[-1].s_traj], axis=0
        )

    # ---- aggregate energies: sum_{n=1}^{N} E_n, the quantity the series as
    # a whole minimizes (each term taken from that leg's finest-level history).

    def total_energy(self) -> dict:
        cols = ("loss", "E_kinetic", "E_data", "E_seg")
        totals = {c: 0.0 for c in cols}
        any_seg = False
        for leg in self.legs:
            if leg.history is None or len(leg.history) == 0:
                continue
            row = leg.history[-1]
            for i, c in enumerate(cols):
                if i + 1 < len(row):
                    totals[c] += float(row[i + 1])
                    if c == "E_seg":
                        any_seg = True
        if not any_seg:
            totals.pop("E_seg")
        return totals

    def deformation_magnitude(self) -> np.ndarray:
        return sum(leg.deformation_magnitude() for leg in self.legs)

    def residual_magnitude(self) -> np.ndarray:
        return sum(leg.residual_magnitude() for leg in self.legs)

    def as_metamorphosis(self) -> Metamorphosis:
        # Repackages the stitched series as a single Metamorphosis object so
        # every existing MetamorphosisVisualizer plot (trajectory strip,
        # v-vs-z, loss history, velocity fields, segmentation dice, ...)
        # works unchanged across the whole chain instead of just one leg.
        use_seg = self.legs[0].s_traj is not None
        v_x, v_y = self.full_v_traj()
        return Metamorphosis(
            self.full_a_traj(),
            v_x,
            v_y,
            self.full_z_traj(),
            a_target=self.legs[-1].a_target,
            history=self._stitched_history(),
            path_a0=self.image_paths[0],
            path_a1=self.image_paths[-1],
            solver_config={**self.solver_config, "n_legs": len(self.legs)},
            s_traj=self.full_s_traj() if use_seg else None,
            s_target=self.legs[-1].s_target if use_seg else None,
            path_s0=self.mask_paths[0] if use_seg else None,
            path_s1=self.mask_paths[-1] if use_seg else None,
        )

    def _stitched_history(self) -> np.ndarray:
        # Overwrites each row's "level" column with the leg index, so the
        # dotted boundary lines in plot_loss_history mark leg transitions
        # instead of (no-longer-relevant, leg-local) pyramid levels.
        rows = []
        for n, leg in enumerate(self.legs):
            h = np.array(leg.history, dtype=float).copy()
            h[:, 0] = n
            rows.append(h)
        return np.concatenate(rows, axis=0)
