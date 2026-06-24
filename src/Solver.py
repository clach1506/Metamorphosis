# Multi-resolution solver for the exact discrete metamorphosis energy (book
# section 13.4.3, formula 13.13): collocation + semi-Lagrangian transport +
# v = K^(1/2) w. Consolidates solver.py and 02b_metamorphosis_solver_exact.py
# at the repo root, which were two copies of this same scheme.
import torch

from Kernel import GaussianKernel
from VelocityField import VelocityField
from ImageTrajectory import ImageTrajectory
from Warp import SemiLagrangianWarp
from Energy import MetamorphosisEnergy
from Pyramid import ResolutionPyramid


class MetamorphosisSolver:
    def __init__(self, T=10, lambda_data=20.0, kernel_sigma_frac=0.03,
                 pyramid_scales=(1 / 8, 1 / 4, 1 / 2, 1.0),
                 level_iters=(400, 300, 250, 300),
                 level_lrs=(0.01, 0.008, 0.005, 0.005)):
        self.T = T
        self.dt = 1.0 / T
        self.lambda_data = lambda_data
        self.kernel_sigma_frac = kernel_sigma_frac
        self.pyramid_scales = pyramid_scales
        self.level_iters = level_iters
        self.level_lrs = level_lrs
        self.history = []

    def fit(self, a0_full: torch.Tensor, a1_full: torch.Tensor, verbose: bool = True):
        H0, W0 = a0_full.shape
        pyramid = ResolutionPyramid(H0, W0, self.pyramid_scales, self.level_iters, self.level_lrs)
        energy_fn = MetamorphosisEnergy(self.dt, self.lambda_data)

        velocity = trajectory = warp = None
        self.history = []

        for level, ((h, w), n_iter, lr) in enumerate(pyramid):
            a0 = ResolutionPyramid.resize_image(a0_full, (h, w))
            a1 = ResolutionPyramid.resize_image(a1_full, (h, w))
            kernel = GaussianKernel.at_scale_fraction(self.kernel_sigma_frac, h, w)
            warp = SemiLagrangianWarp(h, w)

            if velocity is None:
                velocity = VelocityField.zeros(self.T, h, w, kernel)
                trajectory = ImageTrajectory.linear_init(a0, a1, self.T)
            else:
                velocity = velocity.upsampled((h, w), kernel)
                trajectory = trajectory.upsampled(a0, a1, (h, w))

            optimizer = torch.optim.Adam(velocity.parameters() + trajectory.parameters(), lr=lr)

            if verbose:
                print(f"\n--- Level {level + 1}/{len(pyramid)} ({h}x{w}, "
                      f"sigma_K={kernel.sigma:.2f}, lr={lr}, {n_iter} iters) ---")
            for it in range(n_iter):
                optimizer.zero_grad()
                loss, e_kinetic, e_data = energy_fn.compute(velocity, trajectory, warp)
                loss.backward()
                optimizer.step()
                self.history.append((level, loss.item(), e_kinetic.item(), e_data.item()))
                if verbose and (it % 100 == 0 or it == n_iter - 1):
                    print(f"  iter {it:4d}  loss={loss.item():12.4f}  "
                          f"E_kin={e_kinetic.item():10.4f}  E_data={e_data.item():12.6f}")

        return velocity, trajectory, warp
