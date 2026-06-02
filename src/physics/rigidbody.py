import numpy as np
from .quaternions import Quaternion
from .transform import Transform4x4


class RigidBody:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._position = np.array([x, y, z], dtype="float")
        self.velocity = np.zeros(3, dtype="float")
        self.acceleration = np.zeros(3, dtype="float")
        self.rotation = np.zeros(3, dtype="float")

        self._mass = 1.0
        self.inverse_mass: float = 1.0
        self.linear_damping: float = 1.0
        self.angular_damping: float = 1.0

        self._orientation = Quaternion(1, 0, 0, 0)
        self._transform_matrix = np.eye(
            4
        )  # derive from orientation and position once per frame

        self._inertia_tensor: np.ndarray = np.zeros((3, 3), dtype="float")
        self._inertia_tensor_world: np.ndarray = np.zeros((3, 3), dtype=float)
        self.inverse_inertia_tensor: np.ndarray = np.zeros((3, 3), dtype="float")
        self._inverse_inertia_tensor_world: np.ndarray = np.zeros((3, 3), dtype="float")

        self.force_accum: np.ndarray = np.zeros(3, dtype="float")
        self.torque_accum: np.ndarray = np.zeros(3, dtype="float")

        self._dirty = True

    @property
    def inertia_tensor(self) -> np.ndarray:
        return self._inertia_tensor

    @inertia_tensor.setter
    def inertia_tensor(self, inertia_tensor: np.ndarray):
        self._inertia_tensor = inertia_tensor
        self.inverse_inertia_tensor = np.linalg.inv(inertia_tensor)
        self._dirty = True

    @property
    def inertia_tensor_world(self) -> np.ndarray:
        if self._dirty:
            self._rebuild_derived_data()
        return self._inertia_tensor_world

    @property
    def inverse_inertia_tensor_world(self) -> np.ndarray:
        if self._dirty:
            self._rebuild_derived_data()
        return self._inverse_inertia_tensor_world

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
            self._rebuild_derived_data()
        return self._transform_matrix

    def _rebuild_derived_data(self):
        self._transform_matrix = np.eye(4)
        M = self._orientation.to_rotation_matrix()
        self._transform_matrix[:3, :3] = M
        self._transform_matrix[:3, 3] = self._position

        self._inertia_tensor_world = M @ self._inertia_tensor @ M.T
        self._inverse_inertia_tensor_world = M @ self.inverse_inertia_tensor @ M.T

        self._dirty = False

    def mark_dirty(self):
        self._dirty = True

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
