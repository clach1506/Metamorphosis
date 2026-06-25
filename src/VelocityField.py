# The control field w(t,x) and the velocity it generates, v(t) = K^(1/2) w(t)
# Optimizing in w-space rather than directly over v
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
    def zeros(cls, T: int, height: int, width: int, kernel: GaussianKernel, device=None) -> "VelocityField":
        w_x = torch.zeros(T, height, width, device=device, requires_grad=True)
        w_y = torch.zeros(T, height, width, device=device, requires_grad=True)
        return cls(w_x, w_y, kernel)

    def upsampled(self, size, kernel: GaussianKernel) -> "VelocityField":
        # w_x/w_y are pixel-unit displacements (Warp adds them straight to
        # pixel coordinates), so spatial resizing alone isn't enough: the
        # same raw pixel value is a smaller *relative* displacement on a
        # bigger grid. Rescale by the resolution ratio so the deformation's
        # magnitude survives the level change, matching standard pyramidal
        # optical-flow upsampling.
        old_h, old_w = self.w_x.shape[-2:]
        new_h, new_w = size
        w_x = ResolutionPyramid.upsample_stack(self.w_x.detach(), size) * (new_w / old_w)
        w_y = ResolutionPyramid.upsample_stack(self.w_y.detach(), size) * (new_h / old_h)
        return VelocityField(w_x.requires_grad_(True), w_y.requires_grad_(True), kernel)

    def velocity_at(self, t: int):
        # v(t) = K^(1/2) w(t)
        return self.kernel.sqrt_smooth(self.w_x[t]), self.kernel.sqrt_smooth(self.w_y[t])

    def kinetic_energy(self) -> torch.Tensor:
        # ||w||_2^2 == ||v||_V^2 -- the point of parameterizing by w, not v.
        return (self.w_x ** 2 + self.w_y ** 2).sum()

    def parameters(self):
        return [self.w_x, self.w_y]
