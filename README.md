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

See [Metamorphosis.ipynb] for a full walkthrough that exercises every piece below individually before running the solvers.

## Module reference (`src/`)

| File | Class | Role |
|---|---|---|
| `Image.py` | `Image`, `ImageOperators` | load/save a grayscale image as a float32 `[0,1]` array; resize, Gaussian smoothing, gradient |
| `Kernel.py` | `GaussianKernel` | the RKHS kernel `K` and `K^(1/2)` (Fig. 13.1) |
| `VelocityField.py` | `VelocityField` | control `w(t)` and `v(t) = K^(1/2) w(t)`, plus the kinetic energy |
| `ImageTrajectory.py` | `ImageTrajectory` | the free collocation states `a(1)..a(T-1)` |
| `Warp.py` | `SemiLagrangianWarp` | bilinear transport `a(t+1, x + dt*v(t,x))` |
| `Energy.py` | `MetamorphosisEnergy` | formula (13.13): kinetic + data term |
| `Pyramid.py` | `ResolutionPyramid` | coarse-to-fine size schedule, resize/upsample utilities |
| `Solver.py` | `MetamorphosisSolver` | the exact/collocation solver tying the above together |
| `SimplifiedSolver.py` | `SimplifiedMetamorphosisSolver` | the approximate shooting scheme, kept separate |
| `Metamorphosis.py` | `Metamorphosis` | facade: `.fit()` / `.load()` / `.save()`, `deformation_magnitude()`, `residual_magnitude()` |
| `Visualizer.py` | `MetamorphosisVisualizer` | matching/separation/trajectory plots and frame-sequence export |

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
