import numpy as np
from physics.rigidbody import RigidBody
from physics.force_generators import ForceGenerator
from dataclasses import dataclass


@dataclass
class BuoyancyFactors:
    max_depth: float
    volume: float
    water_height: float
    center_of_buoyancy: np.ndarray
    liquid_density: float = 1000.0


class Buoyancy(ForceGenerator):
    def __init__(self, buoyancy_factors: BuoyancyFactors):
        self.buoyancy_factors = buoyancy_factors

    def update_force(self, body: RigidBody, dt: float) -> None:
        factors = self.buoyancy_factors
        body_position = body.position[1]
        depth = body_position - factors.water_height

        ratio = np.interp(depth, [-factors.max_depth, factors.max_depth], [1.0, 0.0])

        force = np.array(
            [0.0, factors.liquid_density * factors.volume * 9.81 * ratio, 0.0]
        )
        body.add_force_to_body_point(force, factors.center_of_buoyancy)
