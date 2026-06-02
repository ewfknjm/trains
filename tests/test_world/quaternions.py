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
        q = Quaternion(0, *vector * scale)
        q._r = q._r * self._r
        wxyz = q._r.as_quat()
        current = self._r.as_quat()
        current += wxyz[[3, 0, 1, 2]] * 0.5

        self._r = R.from_quat(current[[1, 2, 3, 0]])
