# The collocation variables of the discrete metamorphosis
# intermediate images a(1)..a(T-1) are free variables optimized
# jointly with the velocity field, not the output of a forward simulation
# from a(0) ("shooting"). Endpoints a(0) and a(T) stay fixed to the input
# and target images.
import torch

from core.Pyramid import ResolutionPyramid


class ImageTrajectory:
    def __init__(self, a0: torch.Tensor, a1: torch.Tensor, a_inner: torch.Tensor):
        self.a0 = a0
        self.a1 = a1
        self.a_inner = a_inner  # (T-1, H, W): free variables a(1)..a(T-1)

    @classmethod
    def linear_init(
        cls, a0: torch.Tensor, a1: torch.Tensor, T: int
    ) -> "ImageTrajectory":
        # Neutral starting point for the optimization: linear interpolation
        # in intensity between the endpoints.
        taus = torch.linspace(1.0 / T, (T - 1) / T, T - 1, device=a0.device).view(
            -1, 1, 1
        )
        a_inner = (
            (a0.unsqueeze(0) + taus * (a1 - a0).unsqueeze(0))
            .clone()
            .requires_grad_(True)
        )
        return cls(a0, a1, a_inner)

    def upsampled(self, a0: torch.Tensor, a1: torch.Tensor, size) -> "ImageTrajectory":
        a_inner = ResolutionPyramid.upsample_stack(
            self.a_inner.detach(), size
        ).requires_grad_(True)
        return ImageTrajectory(a0, a1, a_inner)

    def full(self) -> torch.Tensor:
        # a(0)..a(T), shape (T+1, H, W)
        return torch.cat(
            [self.a0.unsqueeze(0), self.a_inner, self.a1.unsqueeze(0)], dim=0
        )

    def parameters(self):
        return [self.a_inner]
