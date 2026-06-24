# The control field w(t,x) and the velocity it generates, v(t) = K^(1/2) w(t)
# (book section 13.4.3). Optimizing in w-space rather than directly over v
# makes the kinetic term sum|w|^2 equal to the RKHS norm ||v||_V^2 by
# construction, with no need to invert K to evaluate the norm.
import torch

from Kernel import GaussianKernel
from Pyramid import ResolutionPyramid


class VelocityField:
    def __init__(self, w_x: torch.Tensor, w_y: torch.Tensor, kernel: GaussianKernel):
        self.w_x = w_x
        self.w_y = w_y
        self.kernel = kernel

    @classmethod
    def zeros(cls, T: int, height: int, width: int, kernel: GaussianKernel) -> "VelocityField":
        w_x = torch.zeros(T, height, width, requires_grad=True)
        w_y = torch.zeros(T, height, width, requires_grad=True)
        return cls(w_x, w_y, kernel)

    def upsampled(self, size, kernel: GaussianKernel) -> "VelocityField":
        w_x = ResolutionPyramid.upsample_stack(self.w_x.detach(), size).requires_grad_(True)
        w_y = ResolutionPyramid.upsample_stack(self.w_y.detach(), size).requires_grad_(True)
        return VelocityField(w_x, w_y, kernel)

    def velocity_at(self, t: int):
        # v(t) = K^(1/2) w(t)
        return self.kernel.sqrt_smooth(self.w_x[t]), self.kernel.sqrt_smooth(self.w_y[t])

    def kinetic_energy(self) -> torch.Tensor:
        # ||w||_2^2 == ||v||_V^2 -- the point of parameterizing by w, not v.
        return (self.w_x ** 2 + self.w_y ** 2).sum()

    def parameters(self):
        return [self.w_x, self.w_y]
