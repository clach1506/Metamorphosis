# Early-stopping criterion for a per-level optimization loop: stop once the
# loss hasn't improved by more than `tol` (relative) for `patience`
# consecutive iterations, instead of always burning the full iteration cap.
#
# `min_iters` guards against a false positive at the very start: the
# zero-deformation initialization (w=0) has zero kinetic cost, so the
# un-trained iteration-0 loss can look like a (misleading) early optimum
# that the optimizer must temporarily move away from before real progress
# shows up. No stop is allowed before `min_iters` steps have run.


def make_convergence_tracker(
    tol: float = 1e-4, patience: int = 30, min_iters: int = 30
):
    best_loss = float("inf")
    stale_iters = 0
    iters_seen = 0

    def step(loss: float) -> bool:
        nonlocal best_loss, stale_iters, iters_seen
        iters_seen += 1
        if iters_seen <= min_iters:
            return False  # warmup: too early to trust as a convergence baseline
        if loss < best_loss * (1.0 - tol):
            best_loss = loss
            stale_iters = 0
        else:
            stale_iters += 1
        return stale_iters >= patience

    return step
