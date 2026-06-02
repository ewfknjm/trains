import numpy as np


class Particle:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.position = np.array([x, y, z], dtype="float")
        self.velocity = np.zeros(3, dtype="float")
        self.acceleration = np.zeros(3, dtype="float")
        self._mass = 1.0
        self.inverse_mass: float = 1.0
        self.force_accum = np.zeros(3, dtype="float")

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

    def integrate(self, dt: float):
        if self.inverse_mass <= 0:
            return

        self.position += self.velocity * dt
        resulting_acceleration = self.acceleration + (
            self.force_accum * self.inverse_mass
        )
        self.velocity += resulting_acceleration * dt

        self.clear_accumulator()

    def clear_accumulator(self):
        self.force_accum.fill(0.0)
