import numpy as np
from abc import ABC, abstractmethod
from .rigidbody import RigidBody
from .transform import Transform4x4


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
