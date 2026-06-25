# The discrete metamorphosis energy, book formula (13.13), generalized to an
# optional second "segmentation" channel sharing the SAME velocity field:
#
#   E = sum_{t=1}^{T} |w(t,x)|^2
#       + lambda_a   * dt^-2 * sum_{t=1}^{T} |a(t+1, x+dt*v(t,x)) - a(t,x)|^2
#       + lambda_seg * dt^-2 * sum_{t=1}^{T} |S(t+1, x+dt*v(t,x)) - S(t,x)|^2


import torch

from VelocityField import VelocityField
from ImageTrajectory import ImageTrajectory
from Warp import SemiLagrangianWarp


class MetamorphosisEnergy:
    def __init__(self, dt: float, lambda_data: float, lambda_seg: float = 0.0):
        self.dt = dt
        self.data_weight = lambda_data * dt ** -2
        self.seg_weight = lambda_seg * dt ** -2

    @staticmethod
    def _mismatch_energy(velocity: VelocityField, trajectory: ImageTrajectory,
                          warp: SemiLagrangianWarp, dt: float) -> torch.Tensor:
        channel = trajectory.full()
        T = velocity.w_x.shape[0]
        energy = 0.0
        for t in range(T):
            vx, vy = velocity.velocity_at(t)
            warped_next = warp(channel[t + 1], dt * vx, dt * vy)
            energy = energy + ((warped_next - channel[t]) ** 2).sum()
        return energy

    def compute(self, velocity: VelocityField, trajectory: ImageTrajectory, warp: SemiLagrangianWarp,
                mask_trajectory: ImageTrajectory = None):
        energy_kinetic = velocity.kinetic_energy()
        energy_data = self._mismatch_energy(velocity, trajectory, warp, self.dt)
        loss = energy_kinetic + self.data_weight * energy_data

        energy_seg = None
        if mask_trajectory is not None:
            energy_seg = self._mismatch_energy(velocity, mask_trajectory, warp, self.dt)
            loss = loss + self.seg_weight * energy_seg

        return loss, energy_kinetic, energy_data, energy_seg
