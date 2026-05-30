import numpy as np
from typing import Protocol


class Particle:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.position = np.array([x, y, z], dtype="float")
        self.velocity = np.zeros(3, dtype="float")
        self.acceleration = np.zeros(3, dtype="float")
        self.inverse_mass: float = 0.0
        self.force_accum = np.zeros(3, dtype="float")

    def set_mass(self, mass: float):
        if mass <= 0:
            self.inverse_mass = 0.0

        else:
            self.inverse_mass = 1.0 / mass

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


class ForceGenerator(Protocol):
    def update_force(self, particle: Particle): ...


class Gravity:
    def __init__(self, gravity=np.array([0.0, -5, 0.0], dtype="float")):
        self.gravity = gravity

    def update_force(self, particle: Particle):
        if particle.inverse_mass <= 0.0:
            return

        particle.add_force(self.gravity / particle.inverse_mass)


class Drag:
    def __init__(self, k1: float = 0.1, k2: float = 0.01):
        self.k1 = k1
        self.k2 = k2

    def update_force(self, particle: Particle):
        velocity = particle.velocity
        speed = np.linalg.norm(velocity)
        if speed == 0:
            return

        drag_magnitude = (self.k1 * speed) + (self.k2 * speed * speed)
        drag_force = -drag_magnitude * (velocity / speed)
        particle.add_force(drag_force)


class Spring:
    def __init__(self, other: np.ndarray, k: float = 0.0, a: float = 0.0):
        self.k = k
        self.a = a
        self.other = other

    def update_force(self, particle: Particle):
        displacement = particle.position - self.other
        distance = np.linalg.norm(displacement)

        if distance == 0:
            return

        direction = displacement / distance
        force_magnitude = -self.k * (distance - self.a)
        force_vector = direction * force_magnitude

        particle.add_force(force_vector)


class ParticleForceRegistry:
    def __init__(self):
        self.registrations = []

    def add(self, particle: Particle, force_generator: ForceGenerator):
        self.registrations.append((particle, force_generator))

    def remove(self, particle: Particle, force_generator: ForceGenerator):
        if (particle, force_generator) in self.registrations:
            self.registrations.remove((particle, force_generator))

    def clear(self):
        self.registrations.clear()

    def update_forces(self, dt: float):
        for particle, force_generator in self.registrations:
            force_generator.update_force(particle)

        integrated_particles = set()
        for particle, _ in self.registrations:
            if particle not in integrated_particles:
                particle.integrate(dt)
                integrated_particles.add(particle)
