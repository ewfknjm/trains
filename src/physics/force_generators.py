import numpy as np
from abc import ABC, abstractmethod
from rigidbody import RigidBody
from transform import Transform4x4


class ForceGenerator(ABC):
    @abstractmethod
    def update_force(self, body: RigidBody, dt: float) -> None: ...


class Gravity(ForceGenerator):
    def __init__(self):
        self.gravity = np.array([0, -9.81, 0], dtype="float")

    def update_force(self, body: RigidBody, dt: float) -> None:
        if np.isclose(body.inverse_mass, 0.0):
            raise ValueError("Mass is not finite")

        body.add_force(self.gravity * body.mass)


class Spring(ForceGenerator):
    def __init__(
        self,
        other: RigidBody,
        local_p1: np.ndarray,
        local_p2: np.ndarray,
        spring_constant: float,
        natural_length: float,
    ):
        self.local_connection_point: np.ndarray = local_p1
        self.local_other_connection_point: np.ndarray = local_p2
        self.other: RigidBody = other
        self.spring_constant: float = spring_constant
        self.natural_length: float = natural_length

    def update_force(self, body: RigidBody, dt: float) -> None:
        other = self.other

        transform_matrix_Body = Transform4x4(body.transform_matrix)
        transform_matrix_Other = Transform4x4(other.transform_matrix)

        world_p1 = transform_matrix_Body.local_to_world(self.local_connection_point)
        world_p2 = transform_matrix_Other.local_to_world(
            self.local_other_connection_point
        )

        displacement = world_p1 - world_p2  # P2 -> P1
        distance = np.linalg.norm(displacement)

        if np.isclose(distance, 0.0):
            return

        direction = displacement / distance
        magnitude = -self.spring_constant * (distance - self.natural_length)
        force = magnitude * direction

        body.add_force(force)
