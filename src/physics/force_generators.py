import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from .rigidbody import RigidBody
from .transform import Transform4x4
from .quaternions import Quaternion


class ForceGenerator(ABC):
    @abstractmethod
    def update_force(self, body: RigidBody, dt: float) -> None: ...


class Gravity(ForceGenerator):
    def __init__(self):
        self.gravity = np.array([0, -9.81, 0], dtype="float")

    def update_force(self, body: RigidBody, dt: float) -> None:
        if np.isclose(body.inverse_mass, 0.0):
            return
        body.add_force(self.gravity * body.mass)


class Spring(ForceGenerator):
    def __init__(
        self,
        other: RigidBody,
        my_connection_point: np.ndarray,
        other_connection_point: np.ndarray,
        spring_constant: float,
        natural_length: float,
    ):
        self._other = other
        self._my_pt = my_connection_point
        self._other_pt = other_connection_point
        self._k = spring_constant
        self._L = natural_length

    def update_force(self, body: RigidBody, dt: float) -> None:
        world_p1 = Transform4x4(body.transform_matrix).local_to_world(self._my_pt)
        world_p2 = Transform4x4(self._other.transform_matrix).local_to_world(
            self._other_pt
        )

        displacement = world_p1 - world_p2
        distance = np.linalg.norm(displacement)
        if np.isclose(distance, 0.0):
            return

        direction = displacement / distance
        force = -self._k * (distance - self._L) * direction
        body.add_force_to_point(force, world_p1)


def make_spring_pair(
    body_a: RigidBody,
    body_b: RigidBody,
    local_p1: np.ndarray,
    local_p2: np.ndarray,
    spring_constant: float,
    natural_length: float,
) -> tuple[Spring, Spring]:
    gen_a = Spring(body_b, local_p1, local_p2, spring_constant, natural_length)
    gen_b = Spring(body_a, local_p2, local_p1, spring_constant, natural_length)
    return gen_a, gen_b

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


@dataclass
class AeroSurface:
    relative_position: np.ndarray
    aero_tensor: np.ndarray
    wind_velocity: np.ndarray
    orientation: Quaternion = field(
        default_factory=lambda: Quaternion(1.0, 0.0, 0.0, 0.0)
    )


class Aero(ForceGenerator):
    def __init__(self, aero_surface: AeroSurface):
        self.surface = aero_surface

    def update_force(self, body: RigidBody, dt: float) -> None:
        self._update_force_from_tensor(body, self.surface.aero_tensor)

    def _update_force_from_tensor(self, body: RigidBody, aero_tensor: np.ndarray):
        sum_velocity = body.velocity + self.surface.wind_velocity

        body_transform = Transform4x4(body.transform_matrix)
        body_velocity = body_transform.world_to_local_dir(sum_velocity)

        surface_rotation = Transform4x4.from_rotation_and_translation(
            self.surface.orientation.to_rotation_matrix(), np.zeros(3)
        )
        surface_velocity = surface_rotation.world_to_local_dir(body_velocity)
        surface_force = aero_tensor @ surface_velocity

        body_force = surface_rotation.local_to_world_dir(surface_force)
        world_force = body_transform.local_to_world_dir(body_force)
        body.add_force_to_body_point(world_force, self.surface.relative_position)


class AeroControl(Aero):
    def __init__(
        self, base_surface: AeroSurface, min_tensor: np.ndarray, max_tensor: np.ndarray
    ):
        super().__init__(base_surface)

        self.min_tensor = min_tensor
        self.max_tensor = max_tensor
        self.control_setting = 0.0

    def set_control(self, value: float) -> None:
        self.control_setting = max(-1.0, min(1.0, value))

    def get_tensor(self) -> np.ndarray:
        if self.control_setting <= 0.0:
            t = self.control_setting + 1.0
            return (1.0 - t) * self.min_tensor + self.surface.aero_tensor * t
        t = self.control_setting
        return (1.0 - t) * self.surface.aero_tensor + self.max_tensor * t

    def update_force(self, body: RigidBody, dt: float) -> None:
        current_tensor = self.get_tensor()
        self._update_force_from_tensor(body, current_tensor)

