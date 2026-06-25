# Coarse-to-fine resolution schedule. This is a numerical-optimization
# choice independent of the energy formula: it speeds up convergence
# without changing what is being minimized (discretized identically at every level, just at a different grid size).

import torch
import torch.nn.functional as Fnn


class ResolutionPyramid:
    def __init__(self, base_height: int, base_width: int, scales, iters, lrs, min_size: int = 8):
        self.sizes = [(max(min_size, round(base_height * s)), max(min_size, round(base_width * s)))
                      for s in scales]
        self.iters = iters
        self.lrs = lrs

    def __iter__(self):
        return iter(zip(self.sizes, self.iters, self.lrs))

    def __len__(self):
        return len(self.sizes)

    @staticmethod
    def resize_image(image: torch.Tensor, size) -> torch.Tensor:
        # Anti-aliased resize, used to move a0/a1 across pyramid levels.
        if tuple(image.shape) == tuple(size):
            return image
        x = image.unsqueeze(0).unsqueeze(0)
        x = Fnn.interpolate(x, size=size, mode='bilinear', align_corners=False, antialias=True)
        return x.squeeze(0).squeeze(0)

    @staticmethod
    def upsample_stack(stack: torch.Tensor, size) -> torch.Tensor:
        # Re-samples a (N, H, W) stack of fields/trajectory states between levels.
        f = stack.unsqueeze(1)
        f = Fnn.interpolate(f, size=size, mode='bilinear', align_corners=False)
        return f.squeeze(1)
