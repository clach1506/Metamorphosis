import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from core.Warp import SemiLagrangianWarp
from forward.Metamorphosis import Metamorphosis
from forward.MetamorphosisSeries import MetamorphosisSeries


# ponytail: one-file fixture layout; expand only when a second patient is needed
_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "065_LE_A_OD")


def _image(name: str) -> str:
    return os.path.join(_FIXTURE_DIR, "images", f"{name}.png")


def _seg(name: str) -> str:
    return os.path.join(_FIXTURE_DIR, "segmentations", f"{name}.png")


def _dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    # ponytail: inline stdlib-only dice; Visualizer._dice_score is private
    p = pred > threshold
    t = target > threshold
    denom = p.sum() + t.sum()
    return 2.0 * np.logical_and(p, t).sum() / denom if denom else float("nan")


def _max_seg_warp_mismatch(m: Metamorphosis) -> float:
    # ponytail: max |warp(S[t+1], dt*v) - S[t]|; with a dominant lambda_seg
    # the mask is transported by deformation alone, so this should be small.
    if m.s_traj is None:
        return float("nan")
    T = m.v_traj_x.shape[0]
    H, W = m.v_traj_x.shape[1:]
    dt = 1.0 / ((m.solver_config or {}).get("T") or T)
    warp = SemiLagrangianWarp(H, W)
    s = torch.tensor(m.s_traj, dtype=torch.float32)
    vx = torch.tensor(m.v_traj_x, dtype=torch.float32)
    vy = torch.tensor(m.v_traj_y, dtype=torch.float32)
    max_mismatch = 0.0
    with torch.no_grad():
        for t in range(T):
            warped_next = warp(s[t + 1], dt * vx[t], dt * vy[t])
            mismatch = (warped_next - s[t]).abs().max().item()
            max_mismatch = max(max_mismatch, mismatch)
    return max_mismatch


# ponytail: coarse single-scale config keeps the suite under a minute while still
# exercising the real optimizer on real 811x843 retinal images.
_FAST_KWARGS = dict(
    T=4,
    lambda_data=200.0,
    kernel_sigma_frac=0.03,
    pyramid_scales=(0.25,),
    level_iters=(20,),
    level_lrs=(0.01,),
    convergence_tol=1e-4,
    convergence_patience=5,
    convergence_min_iters=5,
    device="cpu",
)


class TestRealMetamorphosis(unittest.TestCase):
    def test_loss_decreases_on_real_pair(self):
        """The optimizer must do better than the linear initialization."""
        m = Metamorphosis.fit(
            _image("frame_00"), _image("frame_04"), verbose=False, **_FAST_KWARGS
        )
        self.assertIsNotNone(m.history)
        self.assertGreater(len(m.history), 1)
        losses = m.history[:, 1]
        self.assertLess(losses[-1], losses[0])

    def test_segmentation_guidance_with_zero_mask_residual(self):
        """A dominant lambda_seg should transport the mask by deformation alone."""
        kwargs = {**_FAST_KWARGS, "lambda_data": 10.0, "lambda_seg": 5000.0}
        m = Metamorphosis.fit(
            _image("frame_00"),
            _image("frame_04"),
            path_s0=_seg("frame_00"),
            path_s1=_seg("frame_04"),
            verbose=False,
            **kwargs,
        )
        # ponytail: 0.3 is the practical floor for binary masks at quarter-res;
        # bilinear interpolation at moving boundaries creates fractional pixels.
        self.assertLess(_max_seg_warp_mismatch(m), 0.3)


class TestRealSeries(unittest.TestCase):
    def test_series_stitches_three_real_frames_with_zero_mask_residual(self):
        """First/middle/last frames form two legs with negligible mask residual."""
        images = [_image("frame_00"), _image("frame_04"), _image("frame_08")]
        masks = [_seg("frame_00"), _seg("frame_04"), _seg("frame_08")]
        kwargs = {**_FAST_KWARGS, "lambda_seg": 5000.0}
        series = MetamorphosisSeries.fit(images, masks, verbose=False, **kwargs)
        self.assertEqual(len(series.legs), 2)
        self.assertEqual(series.full_a_traj().shape[0], 2 * _FAST_KWARGS["T"] + 1)
        for leg in series.legs:
            # ponytail: 0.3 is the practical floor for binary masks at quarter-res.
            self.assertLess(_max_seg_warp_mismatch(leg), 0.3)


if __name__ == "__main__":
    unittest.main()
