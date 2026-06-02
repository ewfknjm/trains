from scipy.spatial.transform import Rotation as R
import numpy as np


class Quaternion:
    def __init__(self, r, i, j, k):
        self._r = R.from_quat([i, j, k, r])

    def to_rotation_matrix(self) -> np.ndarray:
        return self._r.as_matrix()

    def __imul__(self, other: "Quaternion") -> "Quaternion":
        self._r = self._r * other._r
        return self

    def add_scaled_vector(self, vector: np.ndarray, scale: float):
        qx, qy, qz, qw = self._r.as_quat()
        wx, wy, wz = np.asarray(vector, dtype=float) * scale

        dw = -(wx * qx + wy * qy + wz * qz)
        dx = wx * qw + wy * qz - wz * qy
        dy = wy * qw + wz * qx - wx * qz
        dz = wz * qw + wx * qy - wy * qx

        self._r = R.from_quat(
            [
                qx + 0.5 * dx,
                qy + 0.5 * dy,
                qz + 0.5 * dz,
                qw + 0.5 * dw,
            ]
        )
