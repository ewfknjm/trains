import numpy as np
from .quaternions import Quaternion


class Transform4x4:
    def __init__(self, matrix: np.ndarray):
        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got shape {matrix.shape}")
        self.matrix = np.array(matrix, dtype=float)

    def __repr__(self) -> str:
        return f"Transform4x4(\n{self.matrix}\n)"

    @classmethod
    def from_rotation_and_translation(
        cls, rotation_3x3: np.ndarray, pos: np.ndarray
    ) -> "Transform4x4":
        mat = np.eye(4)
        mat[:3, :3] = rotation_3x3
        mat[:3, 3] = np.ravel(pos)
        return cls(mat)

    @classmethod
    def from_quaternion(cls, q: Quaternion, pos: np.ndarray) -> "Transform4x4":
        return cls.from_rotation_and_translation(q.to_rotation_matrix(), pos)

    def inverse(self) -> "Transform4x4":
        """Fast inverse. Only valid when the rotation component is orthonormal."""
        rot_t = self.matrix[:3, :3].T
        pos = self.matrix[:3, 3]
        inv_mat = np.eye(4)
        inv_mat[:3, :3] = rot_t
        inv_mat[:3, 3] = -rot_t @ pos
        return Transform4x4(inv_mat)

    def __matmul__(self, other: "Transform4x4") -> "Transform4x4":
        return Transform4x4(self.matrix @ other.matrix)

    def local_to_world(self, pos: np.ndarray) -> np.ndarray:
        pos_4d = np.append(np.ravel(pos), 1.0)
        return (self.matrix @ pos_4d)[:3]

    def world_to_local(self, pos: np.ndarray) -> np.ndarray:
        return self.inverse().local_to_world(pos)

    def local_to_world_dir(self, direction: np.ndarray) -> np.ndarray:
        dir_4d = np.append(np.ravel(direction), 0.0)
        return (self.matrix @ dir_4d)[:3]

    def world_to_local_dir(self, direction: np.ndarray) -> np.ndarray:
        return self.inverse().local_to_world_dir(direction)
