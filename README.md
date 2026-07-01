# Metamorphosis

Image registration between two 2D images `a(0)` and `a(1)` using the
**metamorphosis** model: the change between the two images is split into a
*geometric deformation* `v(t)` (a diffeomorphic warp) and a *residual*
`z(t)` (pure intensity/texture change that no warp can explain — e.g.
necrosis, lesion growth). Built against longitudinal retinal angiography
data, where both effects genuinely occur together.

## Background

The discrete energy minimized (from L. Younes Shapes and Diffeomorphisms book section 13.4.3) is

```
E = sum_{t=1}^{T} |w(t,x)|^2
    + lambda * dt^-2 * sum_{t=1}^{T} |a(t+1, x + dt*v(t,x)) - a(t,x)|^2
```

- `w(t,x)` is a raw control field; the velocity is `v(t) = K^(1/2) w(t)`
  for a Gaussian kernel `K`, chosen so the first (kinetic) term equals
  `||w||_2^2` directly — no need to invert `K` to evaluate the RKHS norm.
- The second term is the data term: each image `a(t)` should match the
  *next* image `a(t+1)` warped back along the flow, evaluated by true
  bilinear (semi-Lagrangian) resampling, not a linearized approximation.
- `a(1)..a(T-1)` are free variables solved jointly with `v` (collocation),
  not produced by forward-simulating from `a(0)` (shooting).
- Optimized coarse-to-fine over an image pyramid for speed.

`src/SimplifiedSolver.py` implements a deliberately approximate variant of
this scheme (explicit residual field, shooting, Eulerian transport) kept
only for comparison — see its module docstring for the exact differences.

### Optional segmentation channel

If you have a binary lesion/structure mask for each of `a0`/`a1`, pass
`path_s0`/`path_s1` to `Metamorphosis.fit()` and a weight `lambda_seg`. The
mask trajectory `S(t)` is fit with the exact same structure (collocation +
the same shared `v`) as the image trajectory, adding a second data term:

```
+ lambda_seg * dt^-2 * sum_{t=1}^{T} |S(t+1, x + dt*v(t,x)) - S(t,x)|^2
```

Since the mask boundary is a much cleaner geometric signal than diffuse
intensity differences, this acts as a ground-truth geometric regularizer on
`v` — it can pull out genuine deformation that the image term alone leaves
near zero. Purely additive: omit `path_s0`/`path_s1` and nothing changes.

### Whole series (more than 2 images)

A longitudinal series has more than one timepoint (`BDD_AMD_062026/031_FA_C_OD/`
alone has 22). `MetamorphosisSeries` chains the pairwise model above across
every consecutive pair ("leg") `I_n -> I_{n+1}` in the series and minimizes
the sum of their energies:

```
sum_{n=1}^{N} E_n(v^n, z^n)
```

where `E_n` is exactly the single-pair energy above, evaluated on leg `n`.
There is no term coupling different legs, so this sum is minimized exactly
by minimizing each leg independently — `MetamorphosisSeries` is an
orchestration layer around the existing `Metamorphosis.fit`, not a new
numerical scheme. `MetamorphosisSeriesVisualizer` stitches the per-leg
trajectories into one continuous sequence and reuses every
`MetamorphosisVisualizer` plot on it.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```python
import sys
sys.path.insert(0, "src")

from Metamorphosis import Metamorphosis
from Visualizer import MetamorphosisVisualizer

metamorphosis = Metamorphosis.fit("a0.png", "a1.png")   # runs MetamorphosisSolver
metamorphosis.save("results/run1")                       # a_traj/v_traj_x/v_traj_y/z_traj/a0/a1

viz = MetamorphosisVisualizer(metamorphosis)
viz.plot_deformation_vs_residual("results/run1/fig_v_vs_z.png")
viz.export_all("results/run1")                            # every figure + frame sequence

# Re-load a previous run without re-solving:
metamorphosis = Metamorphosis.load("results/run1")
```

With a binary mask available for each input image, add `lambda_seg` and the
mask paths to get the segmentation-regularized fit described below:

```python
metamorphosis = Metamorphosis.fit(
    "a0.png", "a1.png",
    path_s0="s0.png", path_s1="s1.png", lambda_seg=500.0,
)
viz = MetamorphosisVisualizer(metamorphosis)
viz.plot_matching_segmentation("results/run1/fig_matching_segmentation.png")
```

For a whole series of more than 2 images (chained pairwise, masks optional,
same as above):

```python
from MetamorphosisSeries import MetamorphosisSeries
from SeriesVisualizer import MetamorphosisSeriesVisualizer

series = MetamorphosisSeries.fit(
    ["frame_00.png", "frame_01.png", "frame_02.png", "frame_03.png"],   # N+1 frames -> N legs
    mask_paths=["mask_00.png", "mask_01.png", "mask_02.png", "mask_03.png"],
    lambda_seg=500.0,
)
series.save("results/series1")                            # one subdir (leg_000/, leg_001/, ...) per leg

viz = MetamorphosisSeriesVisualizer(series)
print(viz.summary())                                       # total energy across all legs + per-leg detail
viz.export_all("results/series1")                           # series-level figures + every per-leg figure

# Re-load a previous run without re-solving:
series = MetamorphosisSeries.load("results/series1")
```

See [Metamorphosis.ipynb] for a full walkthrough that exercises every piece below individually before running the solvers.

## Module reference (`src/`)

| File | Class | Role |
|---|---|---|
| `Image.py` | `Image`, `ImageOperators` | load/save a grayscale image as a float32 `[0,1]` array; resize, Gaussian smoothing, gradient |
| `Kernel.py` | `GaussianKernel` | the RKHS kernel `K` and `K^(1/2)` (Fig. 13.1) |
| `VelocityField.py` | `VelocityField` | control `w(t)` and `v(t) = K^(1/2) w(t)`, plus the kinetic energy |
| `ImageTrajectory.py` | `ImageTrajectory` | the free collocation states `a(1)..a(T-1)` -- generic over any single-channel field, reused for the segmentation mask trajectory `S(t)` too |
| `Warp.py` | `SemiLagrangianWarp` | bilinear transport `a(t+1, x + dt*v(t,x))` |
| `Energy.py` | `MetamorphosisEnergy` | formula (13.13): kinetic + data term, plus an optional `lambda_seg`-weighted segmentation term sharing the same `v` |
| `Pyramid.py` | `ResolutionPyramid` | coarse-to-fine size schedule, resize/upsample utilities |
| `Solver.py` | `MetamorphosisSolver` | the exact/collocation solver tying the above together; takes optional `s0_full`/`s1_full` masks |
| `SimplifiedSolver.py` | `SimplifiedMetamorphosisSolver` | the approximate shooting scheme, kept separate |
| `Metamorphosis.py` | `Metamorphosis` | facade: `.fit()` / `.load()` / `.save()`, `deformation_magnitude()`, `residual_magnitude()`; carries `s_traj`/`s_target` when fit with `path_s0`/`path_s1` |
| `Visualizer.py` | `MetamorphosisVisualizer` | matching/separation/trajectory/loss plots, frame-sequence export, text `summary()` (incl. Dice score when segmentation is used) |
| `MetamorphosisSeries.py` | `MetamorphosisSeries` | chains `Metamorphosis.fit` across every consecutive pair in a series; stitches per-leg trajectories (`full_a_traj()`/`full_v_traj()`/`full_z_traj()`/`full_s_traj()`), sums per-leg energies (`total_energy()`), `.save()`/`.load()` (one subdir per leg) |
| `SeriesVisualizer.py` | `MetamorphosisSeriesVisualizer` | wraps a `MetamorphosisSeries` as a single stitched `Metamorphosis` and reuses every `MetamorphosisVisualizer` plot on it; adds a per-original-frame strip and a per-leg energy breakdown |

All modules use flat same-directory imports (e.g. `from Kernel import
GaussianKernel`), so callers need `src/` on `sys.path`, not a package import.

## Data

`BDD_AMD_062026/` holds one subfolder per patient/eye (e.g.
`031_FA_C_OD/`), each with a `preprocessed/` directory of grayscale PNG
frames at different timepoints — these are the `a0`/`a1` inputs. This
folder is gitignored (not distributed via version control); point
`Metamorphosis.fit()` at any two frames from the same series.

## Results

`results/`, `results_collocation/`, `results_exact/` and
`exported_frames/` hold output from previous runs (`*.npy` trajectories
and figures) — also gitignored, fully regenerable via `Metamorphosis.fit`
+ `.save()` / `MetamorphosisVisualizer.export_all()`. Any of them can be
reloaded directly with `Metamorphosis.load(directory)`.
