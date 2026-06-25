# Semi-Lagrangian transport (book section 13.4.3): evaluates an image at the
# points displaced by the velocity field, a(t+1, x + dt*v(t,x)), via true
# bilinear interpolation -- the scheme the book singles out as stable for
# large displacements, unlike a first-order (gradient * velocity) linearization.
import torch
import torch.nn.functional as Fnn


class SemiLagrangianWarp:
    def __init__(self, height: int, width: int, device=None):
        self.height = height
        self.width = width
        self.ys, self.xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width, dtype=torch.float32, device=device),
            indexing='ij',
        )

    def __call__(self, image: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        # Samples `image` at (x + dx(x), y + dy(x)) by bilinear interpolation.
        norm_x = 2.0 * (self.xs + dx) / (self.width - 1) - 1.0
        norm_y = 2.0 * (self.ys + dy) / (self.height - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
        # Border padding is unsupported on MPS -- clamping the grid to
        # [-1, 1] up front is equivalent (no sample then lands out of
        # bounds), so any padding mode works; use 'zeros', which MPS supports.
        grid = grid.clamp(-1.0, 1.0)
        out = Fnn.grid_sample(image.unsqueeze(0).unsqueeze(0), grid,
                               mode='bilinear', padding_mode='zeros', align_corners=True)
        return out.squeeze(0).squeeze(0)
