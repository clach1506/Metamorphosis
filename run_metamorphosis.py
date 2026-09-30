# Fit a metamorphosis between two images, save it, print the summary and
# export every figure. Masks are optional: pass --s0/--s1 to add the
# segmentation channel (sharing the same velocity field).
#
# Usage:
#   python run_metamorphosis.py a0.png a1.png -o results/run1
#   python run_metamorphosis.py a0.png a1.png --s0 m0.png --s1 m1.png -o results/run1
#   python run_metamorphosis.py a0.png a1.png --preset full --lambda-data 200
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from forward.Metamorphosis import Metamorphosis
from forward.Visualizer import MetamorphosisVisualizer

PRESETS = {
    # Stops the pyramid at half resolution: quick on CPU for ~768px images.
    "fast": dict(
        pyramid_scales=(1 / 8, 1 / 4, 1 / 2),
        level_iters=(200, 150, 100),
        level_lrs=(0.01, 0.008, 0.005),
    ),
    # Solver defaults: full native resolution.
    "full": dict(),
}


def main():
    parser = argparse.ArgumentParser(description="Fit a metamorphosis a0 -> a1 (optionally with masks).")
    parser.add_argument("a0", help="source image")
    parser.add_argument("a1", help="target image")
    parser.add_argument("--s0", help="mask for a0 (optional, needs --s1)")
    parser.add_argument("--s1", help="mask for a1 (optional, needs --s0)")
    parser.add_argument("-o", "--output-dir", default="results/run")
    parser.add_argument("--preset", choices=PRESETS, default="fast")
    parser.add_argument("--T", type=int, default=8, help="number of time steps")
    parser.add_argument("--lambda-data", type=float, default=100.0)
    parser.add_argument(
        "--lambda-seg", type=float, default=None,
        help="weight of the mask channel (default: same as --lambda-data)",
    )
    parser.add_argument("--kernel-sigma-frac", type=float, default=0.02)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    parser.add_argument("--no-figures", action="store_true", help="skip export_all()")
    args = parser.parse_args()
    if (args.s0 is None) != (args.s1 is None):
        parser.error("--s0 and --s1 must be given together")

    m = Metamorphosis.fit(
        args.a0,
        args.a1,
        path_s0=args.s0,
        path_s1=args.s1,
        T=args.T,
        lambda_data=args.lambda_data,
        lambda_seg=args.lambda_seg,
        kernel_sigma_frac=args.kernel_sigma_frac,
        device=args.device,
        **PRESETS[args.preset],
    )
    m.save(args.output_dir)

    viz = MetamorphosisVisualizer(m)
    print("\n" + viz.summary())
    if not args.no_figures:
        viz.export_all(args.output_dir)
    print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()
