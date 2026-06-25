# Multi-resolution solver for the exact discrete metamorphosis energy : collocation + semi-Lagrangian transport +
# v = K^(1/2) w.

# Optionally takes a second "segmentation" channel (s0_full/s1_full) sharing
# the SAME velocity field as the image channel -- see Energy.py. This is
# purely additive: pass nothing and it behaves exactly as before.
#

import os
import torch

from Kernel import GaussianKernel
from VelocityField import VelocityField
from ImageTrajectory import ImageTrajectory
from Warp import SemiLagrangianWarp
from Energy import MetamorphosisEnergy
from Pyramid import ResolutionPyramid
from Convergence import ConvergenceTracker


class MetamorphosisSolver:
    def __init__(self, T=10, lambda_data=20.0, lambda_seg=0.0, kernel_sigma_frac=0.03,
                 pyramid_scales=(1 / 8, 1 / 4, 1 / 2, 1.0),
                 level_iters=(400, 300, 250, 300),
                 level_lrs=(0.01, 0.008, 0.005, 0.005),
                 convergence_tol=1e-4, convergence_patience=30, convergence_min_iters=30,
                 device="cpu"):
        self.T = T
        self.dt = 1.0 / T
        self.lambda_data = lambda_data
        self.lambda_seg = lambda_seg  # 0.0 = segmentation channel disabled
        self.kernel_sigma_frac = kernel_sigma_frac
        self.pyramid_scales = pyramid_scales
        self.level_iters = level_iters  # per-level iteration cap; may stop earlier on convergence
        self.level_lrs = level_lrs
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.convergence_min_iters = convergence_min_iters
        self.device = torch.device(device)
        if self.device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
            raise RuntimeError(
                "device='mps' needs PYTORCH_ENABLE_MPS_FALLBACK=1 set in the shell "
                "BEFORE starting this process (e.g. `PYTORCH_ENABLE_MPS_FALLBACK=1 "
                "python ...` or `... jupyter lab`) -- torch reads this flag once at "
                "import time, so setting it from within Python after `import torch` "
                "has already run (as it has by the time this code executes) won't work."
            )
        self.history = []

    def fit(self, a0_full: torch.Tensor, a1_full: torch.Tensor,
            s0_full: torch.Tensor = None, s1_full: torch.Tensor = None, verbose: bool = True):
        use_seg = s0_full is not None and s1_full is not None
        a0_full, a1_full = a0_full.to(self.device), a1_full.to(self.device)
        if use_seg:
            s0_full, s1_full = s0_full.to(self.device), s1_full.to(self.device)
        H0, W0 = a0_full.shape
        pyramid = ResolutionPyramid(H0, W0, self.pyramid_scales, self.level_iters, self.level_lrs)
        energy_fn = MetamorphosisEnergy(self.dt, self.lambda_data, self.lambda_seg)

        velocity = trajectory = mask_trajectory = warp = None
        self.history = []

        for level, ((h, w), n_iter, lr) in enumerate(pyramid):
            a0 = ResolutionPyramid.resize_image(a0_full, (h, w))
            a1 = ResolutionPyramid.resize_image(a1_full, (h, w))
            kernel = GaussianKernel.at_scale_fraction(self.kernel_sigma_frac, h, w)
            warp = SemiLagrangianWarp(h, w, device=self.device)
            if use_seg:
                s0 = ResolutionPyramid.resize_image(s0_full, (h, w))
                s1 = ResolutionPyramid.resize_image(s1_full, (h, w))

            if velocity is None:
                velocity = VelocityField.zeros(self.T, h, w, kernel, device=self.device)
                trajectory = ImageTrajectory.linear_init(a0, a1, self.T)
                if use_seg:
                    mask_trajectory = ImageTrajectory.linear_init(s0, s1, self.T)
            else:
                velocity = velocity.upsampled((h, w), kernel)
                trajectory = trajectory.upsampled(a0, a1, (h, w))
                if use_seg:
                    mask_trajectory = mask_trajectory.upsampled(s0, s1, (h, w))

            params = velocity.parameters() + trajectory.parameters()
            if use_seg:
                params = params + mask_trajectory.parameters()
            optimizer = torch.optim.Adam(params, lr=lr)

            if verbose:
                print(f"\n--- Level {level + 1}/{len(pyramid)} ({h}x{w}, "
                      f"sigma_K={kernel.sigma:.2f}, lr={lr}, max {n_iter} iters) ---")
            tracker = ConvergenceTracker(self.convergence_tol, self.convergence_patience, self.convergence_min_iters)
            n_data_elems = self.T * h * w
            for it in range(n_iter):
                optimizer.zero_grad()
                loss, e_kinetic, e_data, e_seg = energy_fn.compute(velocity, trajectory, warp, mask_trajectory)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                e_data_val = e_data.item()
                e_seg_val = e_seg.item() if e_seg is not None else None
                row = (level, loss_val, e_kinetic.item(), e_data_val)
                self.history.append(row + (e_seg_val,) if use_seg else row)
                # RMS mismatch per pixel, in the same [0,1] scale as the images
                # (and masks) -- unlike the raw sums above, this is comparable
                # across runs/resolutions and tells you how accurate the fit is.
                rms_data = (e_data_val / n_data_elems) ** 0.5
                seg_str = f"  E_seg={e_seg_val:12.6f}" if use_seg else ""
                if verbose and (it % 100 == 0 or it == n_iter - 1):
                    print(f"  iter {it:4d}  loss={loss_val:12.4f}  "
                          f"E_kin={e_kinetic.item():10.4f}  E_data={e_data_val:12.6f}{seg_str}  "
                          f"rms={rms_data:.5f} (~{rms_data * 255:.2f}/255)")
                if tracker.step(loss_val):
                    if verbose:
                        print(f"  converged at iter {it} (no improvement for {self.convergence_patience} iters)  "
                              f"rms={rms_data:.5f} (~{rms_data * 255:.2f}/255)")
                    break

        return velocity, trajectory, warp, mask_trajectory
