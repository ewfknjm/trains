import numpy as np
from physics.force_generators import ForceGenerator
from physics.rigidbody import RigidBody
from physics.transform import Transform4x4
from dataclasses import dataclass, field
from physics.quaternions import Quaternion


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
