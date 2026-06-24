# Early-stopping criterion for a per-level optimization loop: stop once the
# loss hasn't improved by more than `tol` (relative) for `patience`
# consecutive iterations, instead of always burning the full iteration cap.
#
# `min_iters` guards against a false positive at the very start: the
# zero-deformation initialization (w=0) has zero kinetic cost, so the
# un-trained iteration-0 loss can look like a (misleading) early optimum
# that the optimizer must temporarily move away from before real progress
# shows up. No stop is allowed before `min_iters` steps have run.
class ConvergenceTracker:
    def __init__(self, tol: float = 1e-4, patience: int = 30, min_iters: int = 30):
        self.tol = tol
        self.patience = patience
        self.min_iters = min_iters
        self.best_loss = float("inf")
        self.stale_iters = 0
        self.iters_seen = 0

    def step(self, loss: float) -> bool:
        self.iters_seen += 1
        if self.iters_seen <= self.min_iters:
            return False  # warmup: too early to trust as a convergence baseline
        if loss < self.best_loss * (1.0 - self.tol):
            self.best_loss = loss
            self.stale_iters = 0
        else:
            self.stale_iters += 1
        return self.stale_iters >= self.patience
