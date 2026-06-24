# The discrete metamorphosis energy, book formula (13.13):
#
#   E = sum_{t=1}^{T} |w(t,x)|^2
#       + lambda * dt^-2 * sum_{t=1}^{T} |a(t+1, x + dt*v(t,x)) - a(t,x)|^2
#
# The first term is the kinetic energy of the deformation (RKHS norm of v,
# computed as ||w||_2^2 since v = K^(1/2) w). The second is the data term:
# mismatch between each image and the next one, transported back by the flow.
import torch

from VelocityField import VelocityField
from ImageTrajectory import ImageTrajectory
from Warp import SemiLagrangianWarp


class MetamorphosisEnergy:
    def __init__(self, dt: float, lambda_data: float):
        self.dt = dt
        self.data_weight = lambda_data * dt ** -2

    def compute(self, velocity: VelocityField, trajectory: ImageTrajectory, warp: SemiLagrangianWarp):
        a = trajectory.full()
        T = velocity.w_x.shape[0]
        energy_kinetic = velocity.kinetic_energy()
        energy_data = 0.0
        for t in range(T):
            vx, vy = velocity.velocity_at(t)
            warped_next = warp(a[t + 1], self.dt * vx, self.dt * vy)
            mismatch = warped_next - a[t]
            energy_data = energy_data + (mismatch ** 2).sum()
        loss = energy_kinetic + self.data_weight * energy_data
        return loss, energy_kinetic, energy_data
