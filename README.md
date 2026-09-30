# Metamorphosis

Image registration between two 2D images `a(0)` and `a(1)` using the
**metamorphosis** model: the change between the two images is split into a
*geometric deformation* `v(t)` (a diffeomorphic warp) and a *residual*
`z(t)` (pure intensity/texture change that no warp can explain — e.g.
necrosis, lesion growth). Built against longitudinal retinal angiography
data, where both effects genuinely occur together.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

### Command line

```bash
# image only
python run_metamorphosis.py a0.png a1.png -o results/run1

# image + segmentation masks (same velocity field for both channels)
python run_metamorphosis.py a0.png a1.png --s0 mask0.png --s1 mask1.png -o results/run1
```

This fits, saves the result, prints the summary and exports every figure
and frame sequence to the output directory. Useful options:
`--preset fast|full` (half resolution vs native, default `fast`), `--T`,
`--lambda-data`, `--lambda-seg` (defaults to `--lambda-data`),
`--kernel-sigma-frac`, `--device cpu|cuda|mps`, `--no-figures`.
Run `python run_metamorphosis.py -h` for the full list.

### Python

```python
import sys
sys.path.insert(0, "src")

from forward.Metamorphosis import Metamorphosis
from forward.Visualizer import MetamorphosisVisualizer

m = Metamorphosis.fit("a0.png", "a1.png")                              # image only
m = Metamorphosis.fit("a0.png", "a1.png", "mask0.png", "mask1.png")   # + segmentation

m.save("results/run1")
viz = MetamorphosisVisualizer(m)
print(viz.summary())
viz.export_all("results/run1")

m = Metamorphosis.load("results/run1")   # reload without re-solving
```

Any keyword argument of `MetamorphosisSolver` can be passed to `fit()`
(`T`, `lambda_data`, `lambda_seg`, `kernel_sigma_frac`, `pyramid_scales`,
`level_iters`, `level_lrs`, `convergence_*`, `device`). For images already
in memory use `Metamorphosis.from_arrays(a0, a1, s0=None, s1=None, ...)`.

`device="mps"` needs `PYTORCH_ENABLE_MPS_FALLBACK=1` set in the shell
before Python starts.

## Background

The discrete energy minimized (L. Younes, *Shapes and Diffeomorphisms*,
section 13.4.3) is

```
E = sum_{t=1}^{T} |w(t,x)|^2
    + lambda_data * dt^-2 * sum_{t=1}^{T} |a(t+1, x + dt*v(t,x)) - a(t,x)|^2
    + lambda_seg  * dt^-2 * sum_{t=1}^{T} |S(t+1, x + dt*v(t,x)) - S(t,x)|^2   (optional)
```

- `w(t,x)` is a raw control field; the velocity is `v(t) = K^(1/2) w(t)`
  for a Gaussian kernel `K`, chosen so the kinetic term equals `||w||_2^2`
  directly — no need to invert `K` to evaluate the RKHS norm.
- The data term makes each image `a(t)` match the *next* image `a(t+1)`
  warped back along the flow, evaluated by true (bicubic) semi-Lagrangian
  resampling, not a linearized approximation.
- `a(1)..a(T-1)` are free variables solved jointly with `v` (collocation),
  not produced by forward-simulating from `a(0)` (shooting).
- Optimized coarse-to-fine over an image pyramid for speed.

### Optional segmentation channel

With a binary lesion/structure mask for each of `a0`/`a1`, a mask
trajectory `S(t)` is fitted with the same structure and the **same** `v`
(third term above). The mask boundary is a much cleaner geometric signal
than diffuse intensity differences, so it pulls out genuine deformation
that the image term alone can leave near zero. Passing the masks is all it
takes: `lambda_seg` defaults to `lambda_data`. Omit them and nothing changes.

### Reading the results

`a(T)` (and `S(T)`) are anchored to the target by construction, so
comparing them to `a(1)`/`S(1)` says nothing. The meaningful diagnostics
transport `a(0)` (or `S(0)`) through `v` **alone**, with no residual
(`Metamorphosis.pure_deformation_trajectory()`):

- *explained by deformation* — `1 - rms(warped a0 - a1) / rms(a0 - a1)`:
  the share of the intensity change geometry accounts for; the rest is `z`.
- *Dice of the warped `S(0)` vs `S(1)`*, next to the no-motion baseline
  `Dice(S(0), S(1))`.
- `E_data`/rms, and per pyramid level whether the optimization converged
  or hit its iteration cap.

## Module reference (`src/`)

| File | Class | Role |
|---|---|---|
| `core/Image.py` | `Image` | load a grayscale image (8- or 16-bit) as a float32 `[0,1]` array |
| `core/Kernel.py` | `GaussianKernel` | the RKHS kernel `K` and `K^(1/2)` |
| `core/VelocityField.py` | `VelocityField` | control `w(t)` and `v(t) = K^(1/2) w(t)`, plus the kinetic energy |
| `core/ImageTrajectory.py` | `ImageTrajectory` | the free collocation states `a(1)..a(T-1)`, also used for `S(t)` |
| `core/Warp.py` | `SemiLagrangianWarp` | transport `a(t+1, x + dt*v(t,x))` |
| `core/Pyramid.py` | `ResolutionPyramid` | coarse-to-fine size schedule, resize/upsample utilities |
| `forward/Energy.py` | `MetamorphosisEnergy` | the energy above |
| `forward/Convergence.py` | `make_convergence_tracker` | patience-based early stopping per pyramid level |
| `forward/Solver.py` | `MetamorphosisSolver` | the multi-resolution Adam loop tying the above together |
| `forward/Metamorphosis.py` | `Metamorphosis` | facade: `.fit()` / `.from_arrays()` / `.save()` / `.load()`, pure-deformation transport |
| `forward/Metrics.py` | `MetamorphosisMetrics` | numbers only: energies, deformation-vs-residual, Dice, convergence report, summary text |
| `forward/Visualizer.py` | `MetamorphosisVisualizer` | figures and frame sequences; takes every number from `MetamorphosisMetrics` |

Callers need `src/` on `sys.path` and import by subdirectory, e.g.
`from forward.Metamorphosis import Metamorphosis`.

## Data

`BDD_AMD_062026/` holds one subfolder per patient/eye (e.g.
`031_FA_C_OD/`), each with a `preprocessed/` directory of grayscale PNG
frames at different timepoints — these are the `a0`/`a1` inputs. This
folder is gitignored, as are `results/` and `notebook_outputs*/` (all
regenerable, and reloadable with `Metamorphosis.load(directory)`).
