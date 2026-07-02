# The regularization kernel K of the RKHS V : a Gaussian
# convolution turning a raw control field w into a spatially smooth velocity.
# K^(1/2) is the Gaussian kernel of half the variance

import math
import torch
import torch.nn.functional as Fnn


class GaussianKernel:
    def __init__(self, sigma: float):
        self.sigma = sigma

    @classmethod
    def at_scale_fraction(
        cls, frac: float, height: int, width: int, min_sigma: float = 2.0
    ) -> "GaussianKernel":
        # Keeps sigma a fixed fraction of the image size across pyramid levels.
        return cls(max(min_sigma, frac * max(height, width)))

    def sqrt_smooth(self, field: torch.Tensor) -> torch.Tensor:
        # v = K^(1/2) * w, so that ||v||_V^2 == ||w||_2^2 (book section 13.4.3)
        return self._separable_conv(field, self.sigma / math.sqrt(2.0))

    @staticmethod
    def _gaussian_kernel_1d(
        sigma: float, ksize: int = None, device=None
    ) -> torch.Tensor:
        if ksize is None:
            ksize = int(6 * sigma) | 1  # odd
        ax = torch.arange(ksize, dtype=torch.float32, device=device) - ksize // 2
        g = torch.exp(-(ax**2) / (2 * sigma**2))
        return g / g.sum()

    def _separable_conv(self, field: torch.Tensor, sigma: float) -> torch.Tensor:
        kernel_1d = self._gaussian_kernel_1d(sigma, device=field.device)
        pad = kernel_1d.shape[0] // 2
        kx = kernel_1d.view(1, 1, 1, -1)
        ky = kernel_1d.view(1, 1, -1, 1)

        f = field.unsqueeze(0).unsqueeze(0)
        f = Fnn.pad(f, (pad, pad, 0, 0), mode="reflect")
        f = Fnn.conv2d(f, kx)
        f = Fnn.pad(f, (0, 0, pad, pad), mode="reflect")
        f = Fnn.conv2d(f, ky)
        return f.squeeze(0).squeeze(0)
