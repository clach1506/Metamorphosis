# Simplified ("shooting") metamorphosis scheme, originally 02_metamorphosis_solver.py
# at the repo root. Kept in its own file, separate from the exact scheme in
# solver.py, since it's a deliberate approximation rather than book-faithful:
#
# - SHOOTING, not collocation: a(1)..a(T) are produced by forward simulation
#   from a(0), instead of being free variables optimized jointly with v.
# - EXPLICIT residual z(t,x), optimized directly, instead of an implicit
#   mismatch term.
# - EULERIAN transport da/dt = z - grad(a)^T v (a first-order linearization),
#   instead of true semi-Lagrangian warping.
# - v = K w, not K^(1/2) w (consistent with the explicit z absorbing what the
#   exact scheme's data term would otherwise capture).
import torch
import torch.nn.functional as Fnn

from Kernel import GaussianKernel
from Pyramid import ResolutionPyramid
from Convergence import ConvergenceTracker


class SimplifiedMetamorphosisSolver:
    def __init__(self, T=10, sigma2=0.05, lambda_data=2000.0, kernel_sigma_frac=0.03,
                 pyramid_scales=(1 / 8, 1 / 4, 1 / 2, 1.0),
                 level_iters=(400, 300, 150, 80),
                 level_lrs=(0.05, 0.03, 0.02, 0.01),
                 convergence_tol=1e-4, convergence_patience=30, convergence_min_iters=30):
        self.T = T
        self.dt = 1.0 / T
        self.sigma2 = sigma2
        self.lambda_data = lambda_data
        self.kernel_sigma_frac = kernel_sigma_frac
        self.pyramid_scales = pyramid_scales
        self.level_iters = level_iters  # per-level iteration cap; may stop earlier on convergence
        self.level_lrs = level_lrs
        self.convergence_tol = convergence_tol
        self.convergence_patience = convergence_patience
        self.convergence_min_iters = convergence_min_iters
        self.history = []

    @staticmethod
    def _image_gradient(a: torch.Tensor):
        img = a.unsqueeze(0).unsqueeze(0)
        gx = Fnn.pad(img, (1, 1, 0, 0), mode='replicate')
        gx = (gx[:, :, :, 2:] - gx[:, :, :, :-2]) / 2.0
        gy = Fnn.pad(img, (0, 0, 1, 1), mode='replicate')
        gy = (gy[:, :, 2:, :] - gy[:, :, :-2, :]) / 2.0
        return gx.squeeze(0).squeeze(0), gy.squeeze(0).squeeze(0)

    def fit(self, a0_full: torch.Tensor, a1_full: torch.Tensor, verbose: bool = True):
        H0, W0 = a0_full.shape
        pyramid = ResolutionPyramid(H0, W0, self.pyramid_scales, self.level_iters, self.level_lrs)

        w_vx = w_vy = z_raw = None
        self.history = []

        for level, ((h, w), n_iter, lr) in enumerate(pyramid):
            a0 = ResolutionPyramid.resize_image(a0_full, (h, w))
            a1 = ResolutionPyramid.resize_image(a1_full, (h, w))
            kernel = GaussianKernel.at_scale_fraction(self.kernel_sigma_frac, h, w)

            if w_vx is None:
                w_vx = torch.zeros(self.T, h, w, requires_grad=True)
                w_vy = torch.zeros(self.T, h, w, requires_grad=True)
                z_raw = torch.zeros(self.T, h, w, requires_grad=True)
            else:
                w_vx = ResolutionPyramid.upsample_stack(w_vx.detach(), (h, w)).requires_grad_(True)
                w_vy = ResolutionPyramid.upsample_stack(w_vy.detach(), (h, w)).requires_grad_(True)
                z_raw = ResolutionPyramid.upsample_stack(z_raw.detach(), (h, w)).requires_grad_(True)

            optimizer = torch.optim.Adam([w_vx, w_vy, z_raw], lr=lr)

            def simulate_and_energy():
                a = a0.clone()
                energy_v = 0.0
                energy_z = 0.0
                for t in range(self.T):
                    vx = kernel.smooth(w_vx[t])
                    vy = kernel.smooth(w_vy[t])
                    z = z_raw[t]
                    gx, gy = self._image_gradient(a)
                    a = a + self.dt * (z - (gx * vx + gy * vy))
                    energy_v = energy_v + self.dt * (vx ** 2 + vy ** 2).sum()
                    energy_z = energy_z + self.dt * (z ** 2).sum()
                return a, energy_v, energy_z

            if verbose:
                print(f"\n--- Level {level + 1}/{len(pyramid)} ({h}x{w}, "
                      f"sigma_K={kernel.sigma:.2f}, lr={lr}, max {n_iter} iters) ---")
            tracker = ConvergenceTracker(self.convergence_tol, self.convergence_patience, self.convergence_min_iters)
            n_pixels = h * w
            for it in range(n_iter):
                optimizer.zero_grad()
                a_final, energy_v, energy_z = simulate_and_energy()
                data_term = ((a_final - a1) ** 2).sum()
                loss = energy_v + (1.0 / self.sigma2) * energy_z + self.lambda_data * data_term
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                data_term_val = data_term.item()
                self.history.append((level, loss_val, energy_v.item(), energy_z.item(), data_term_val))
                # RMS mismatch per pixel between the reconstructed a(T) and the
                # real target a1, in [0,1] intensity scale -- the interpretable
                # "is this accurate" number, as opposed to the raw sums above.
                rms_data = (data_term_val / n_pixels) ** 0.5
                if verbose and (it % 100 == 0 or it == n_iter - 1):
                    print(f"  iter {it:4d}  loss={loss_val:12.4f}  E_v={energy_v.item():10.4f}  "
                          f"E_z={energy_z.item():10.4f}  data_term={data_term_val:10.5f}  "
                          f"rms={rms_data:.5f} (~{rms_data * 255:.2f}/255)")
                if tracker.step(loss_val):
                    if verbose:
                        print(f"  converged at iter {it} (no improvement for {self.convergence_patience} iters)  "
                              f"rms={rms_data:.5f} (~{rms_data * 255:.2f}/255)")
                    break

        return w_vx, w_vy, z_raw, a0, a1
