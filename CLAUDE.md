# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Image registration between two 2D images `a(0)` and `a(1)` using the
**metamorphosis** model (L. Younes, *Shapes and Diffeomorphisms*, section
13.4.3): the change between the two images is split into a *geometric
deformation* `v(t)` (a diffeomorphic warp) and a *residual* `z(t)` (pure
intensity/texture change no warp can explain — e.g. necrosis, lesion
growth). Built against longitudinal retinal angiography data, where both
effects genuinely occur together.

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Fit a pair end to end (fit + save + summary + all figures); masks optional
python run_metamorphosis.py a0.png a1.png [--s0 m0.png --s1 m1.png] -o results/run1
```

There is currently no test suite (the old synthetic tests and benchmark
were removed). To sanity-check a change, fit a small in-memory pair via
`Metamorphosis.from_arrays` with a single coarse pyramid level.

There is no linter/formatter configured in this repo.

`device="mps"` requires `PYTORCH_ENABLE_MPS_FALLBACK=1` set in the shell
*before* the process starts (torch reads it once at import time) —
`Solver.py` raises immediately if this precondition isn't met.

## Architecture

### Package layout and import convention

Code lives under `src/`, split into `src/core/` (generic image-processing
primitives, no metamorphosis-specific logic) and `src/forward/` (the
metamorphosis model, solver, and diagnostics). There is no `src/__init__.py`
and the package isn't installed, so **callers must put `src/` itself on
`sys.path`** and import by subdirectory: `from core.Kernel import
GaussianKernel`, `from forward.Metamorphosis import Metamorphosis`. Every
module in `src/` follows this convention — if you see an import written as
`from src.core... import ...` inside a file under `src/`, that's
inconsistent with the rest of the codebase and will only resolve when the
*repo root* (not `src/`) happens to be on `sys.path` too.

### The numerical model (energy → solver)

The discrete energy being minimized, generalized to an optional second
channel:

```
E = sum_t |w(t,x)|^2                                             (kinetic)
    + lambda_data * dt^-2 * sum_t |a(t+1, x+dt*v(t,x)) - a(t,x)|^2   (data term)
    + lambda_seg  * dt^-2 * sum_t |S(t+1, x+dt*v(t,x)) - S(t,x)|^2   (optional, same v)
```

Reading `src/core/` in this order mirrors how the pieces compose:

- `VelocityField` — holds the raw control field `w(t)`; `v(t) = K^(1/2)
  w(t)` via `GaussianKernel.sqrt_smooth`, chosen so `kinetic_energy() ==
  ||w||_2^2` directly (no need to invert `K`).
- `ImageTrajectory` — the free collocation variables `a(1)..a(T-1)`.
  Endpoints `a(0)`/`a(T)` are fixed inputs; the interior frames are
  optimized jointly with `v`, not produced by forward-simulating from
  `a(0)` ("shooting"). Generic over any single-channel field, so the same
  class also holds the segmentation mask trajectory `S(t)`.
- `SemiLagrangianWarp` — the transport operator: `warp(image, dx, dy)`
  bicubic-samples `image` at `(x+dx, y+dy)`. The data term's pull
  relation is `a(t,x) ≈ warp(a(t+1), dt*v(t,x))(x)`.
- `ResolutionPyramid` — coarse-to-fine size schedule; a pure optimization
  speedup, doesn't change what's minimized (same discretization at every
  scale).

`forward/Energy.py`'s `MetamorphosisEnergy.compute()` combines these into
the loss above; `forward/Solver.py`'s `MetamorphosisSolver.fit()` runs the
per-pyramid-level Adam loop, using `forward/Convergence.py`'s
`make_convergence_tracker` to early-stop a level once loss stops improving
(patience-based) rather than always burning the full iteration cap.

Segmentation is fully additive: passing `s0_full`/`s1_full` (or
`path_s0`/`path_s1` at the `Metamorphosis.fit()` level) fits a second
`ImageTrajectory` for the mask channel, sharing the exact same `v` —
`Energy.py`'s `mismatch_energy()` is just called twice. `lambda_seg=None`
(default) resolves to `lambda_data` when masks are given and 0 otherwise;
the solver raises on inconsistent combinations (one mask only, masks with
`lambda_seg<=0`, `lambda_seg>0` without masks, shape mismatch). The value
actually used is recorded in `solver_config["lambda_seg"]`. Omit the mask
args and nothing changes.

### `Metamorphosis`: the facade / result object

`forward/Metamorphosis.py` is what the rest of the codebase consumes.
`Metamorphosis.fit(path_a0, path_a1, ...)` loads images from disk and
delegates to `Metamorphosis.from_arrays(a0, a1, ...)`, which runs the
solver and packages the result: `a_traj`, `v_traj_x`/`v_traj_y`, `z_traj`
(the implicit residual, back-derived from the data-term mismatch),
`history` (per-iteration `(level, loss, E_kinetic, E_data[, E_seg])`), and
`solver_config` (records enough of the solver's settings — including
`level_iters`/`convergence_tol`/`convergence_patience` — for post-hoc
convergence diagnostics). `from_arrays()` exists separately so callers that
already have exact float32 tensors (e.g. synthetic or preprocessed arrays)
can skip the lossy 8-bit PNG round trip.
`pure_deformation_trajectory(channel0)` transports a(0) (or S(0)) through
`v` alone with no residual — the basis of every meaningful fit diagnostic,
since a(T)/S(T) themselves are anchored to the target. `.save()`/`.load()` round-trip
through `.npy` files plus a `meta.json`.

### Diagnostics: Metrics (numbers) vs Visualizer (plots)

These are deliberately split, and neither should duplicate the other's
work:

- `forward/Metrics.py` (`MetamorphosisMetrics`) — pure computation, no
  matplotlib: energy breakdown, deformation-vs-residual (share of
  `rms(a0-a1)` removed by the pure-deformation warp), pure-deformation Dice,
  `loss_curve_data()` (structured arrays for a plotting layer to consume),
  and `convergence_report()` — per pyramid level, distinguishes "converged
  early" from "hit the iteration cap" using the `level_iters`/tol/patience
  recorded in `solver_config`.
- `forward/Visualizer.py` (`MetamorphosisVisualizer`) — matplotlib only.
  Holds a `self.metrics = MetamorphosisMetrics(...)` and delegates every
  number a figure needs (e.g. Dice for a plot title) to it rather than
  recomputing.

Note: `a(T)` (and `S(T)` when segmentation is used) is *anchored* to the
target by the collocation scheme, so matching error there is trivially
~0 — it says nothing about fit quality. Real accuracy signal comes from
`E_data`/rms and the pure-deformation comparisons above.

### What's not here

There is currently no multi-image/longitudinal-series abstraction —
`Metamorphosis` only fits a single pairwise `a0 → a1`. (A prior
`MetamorphosisSeries`/`SeriesVisualizer` that chained legs across a series
was removed, as were `Benchmark.py` and the synthetic demo scripts; don't
assume they still exist.)

`BDD_AMD_062026/` (real patient data) and the various `results*/` output
directories are gitignored and not present in a fresh checkout.
