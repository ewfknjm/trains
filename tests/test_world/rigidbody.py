import numpy as np
from quaternions import Quaternion
from matrix import Transform4x4


class RigidBody:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._position = np.array([x, y, z], dtype="float")
        self.velocity = np.zeros(3, dtype="float")
        self.acceleration = np.zeros(3, dtype="float")
        self.rotation = np.zeros(3, dtype="float")
        self._mass = 1.0
        self.inverse_mass: float = 1.0
        self.linearDamping: float = 1.0
        self._orientation = Quaternion(1, 0, 0, 0)
        self._transform_matrix = np.eye(
            4
        )  # derive from orientation and position once per frame
        self._inertia_tensor: np.ndarray = np.zeros((3, 3), dtype="float")
        self.inver_inertia_tensor: np.ndarray = np.zeros((3, 3), dtype="float")
        self._inver_inertia_tensor_world: np.ndarray = np.zeros((3, 3), dtype="float")
        self.force_accum: np.ndarray = np.zeros(3, dtype="float")
        self.torque_accum: np.ndarray = np.zeros(3, dtype="float")
        self._dirty = True

    @property
    def inertia_tensor(self) -> np.ndarray:
        return self._inertia_tensor

    @inertia_tensor.setter
    def inertia_tensor(self, inertia_tensor: np.ndarray):
        self._inertia_tensor = inertia_tensor
        self.inver_inertia_tensor = np.linalg.inv(inertia_tensor)
        self._dirty = True

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, pos: np.ndarray):
        self._position = pos
        self._dirty = True

    @property
    def orientation(self) -> Quaternion:
        return self._orientation

    @orientation.setter
    def orientation(self, orientation: Quaternion):
        self._orientation = orientation
        self._dirty = True

    @property
    def transform_matrix(self) -> np.ndarray:
        if self._dirty:
            self.rebuild_transform()
        return self._transform_matrix

    def rebuild_transform(self):
        self._transform_matrix = np.eye(4)
        self._transform_matrix[:3, :3] = self._orientation.to_rotation_matrix()
        self._transform_matrix[:3, 3] = self.position
        self._recalculate_inver_inertia_tensor_world()
        self._dirty = False

    def _recalculate_inver_inertia_tensor_world(self):
        M = self._transform_matrix[:3, :3]
        self._inver_inertia_tensor_world = M @ self.inver_inertia_tensor @ M.T

    @property
    def mass(self):
        return self._mass

    @mass.setter
    def mass(self, value: float):
        if value <= 0:
            raise ValueError()
        self._mass = value
        self.inverse_mass = 1.0 / value

    def add_force(self, force: np.ndarray):
        self.force_accum += force

    def add_torque(self, torque: np.ndarray):
        self.torque_accum += torque

    def clear_accum(self):
        self.force_accum[:] = 0.0
        self.torque_accum[:] = 0.0

    def add_force_to_point(self, force: np.ndarray, point: np.ndarray):
        com_to_point = point - self.position
        self.add_force(force)
        self.add_torque(np.cross(com_to_point, force))

    def add_force_to_body_point(self, force: np.ndarray, point: np.ndarray):
        transform_matrix = Transform4x4(self.transform_matrix)
        pt = transform_matrix.local_to_world(point)
        self.add_force_to_point(force, pt)

    def integrate(self, dt: float):
        self.acceleration += self.force_accum * self.inverse_mass

        self.velocity += self.acceleration * dt
        self.velocity *= self.linearDamping**dt

        self.position = self.position + self.velocity * dt

        I_world = self._inver_inertia_tensor_world
        gyroscopic = np.cross(self.rotation, np.linalg.solve(I_world, self.rotation))
        angular_accel = I_world @ (self.torque_accum - gyroscopic)

        self.rotation += angular_accel * dt
        self.rotation *= self.linearDamping**dt

        self._orientation.add_scaled_vector(self.rotation, dt)
        self._dirty = True

        self.clear_accum()
