import math
import random
from typing import Protocol
import numpy as np
from particles import Particle


class ForceGenerator(Protocol):
    def update_force(self, particle: Particle, dt: float): ...


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


class Repulsion:
    def __init__(
        self,
        other_pos: np.ndarray,
        max_force: float,
        inner_radius: float,
        outer_radius: float,
    ):
        self.other_pos = other_pos
        self.max_force = max_force
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius

    def update_force(self, particle: Particle):
        displacement = particle.position - self.other_pos
        distance = np.linalg.norm(displacement) - (self.inner_radius * 2)

        if distance >= self.outer_radius or distance == 0:
            return

        direction = displacement / distance

        if distance <= self.inner_radius:
            falloff = 1.0
        else:
            falloff_range = self.outer_radius - self.inner_radius
            distance_into_falloff = distance - self.inner_radius
            falloff = 1.0 - (distance_into_falloff / falloff_range)
            falloff = falloff**2

        force_vector = direction * (self.max_force * falloff)
        particle.add_force(force_vector)


class WanderForce:
    def __init__(self, magnitude: float = 10.0, frequency: float = 1.0):
        self.magnitude = magnitude
        self.frequency = frequency
        self.seed_x = random.uniform(0, 1000)
        self.seed_y = random.uniform(0, 1000)
        self.seed_z = random.uniform(0, 1000)
        self.time_accum = 0.0

    def update_force(self, particle: Particle):
        self.time_accum += 0.016

        fx = (
            math.sin((self.time_accum * self.frequency) + self.seed_x) * self.magnitude
        )  # not my idea
        fy = (
            math.cos((self.time_accum * self.frequency) + self.seed_y) * self.magnitude
        )  # increasing the frequency decreases the wavelength of the graphs
        fz = (
            math.sin((self.time_accum * self.frequency) + self.seed_z) * self.magnitude
        )  # making the force more 'jittery'

        force = (
            np.array([fx, fy, fz], dtype="float") * self.magnitude
        )  # the seed shifts each function by a random amount, so the graphs hardly overlap and produce the same output
        particle.add_force(
            force
        )  # magnitude just amps up the force, because sin and cos [-1,1]


class Attractor:
    def __init__(
        self, other: Particle, modulusE: float, trail_length: float, time_buffer: float
    ):
        self.other = other
        self.modulus_e = modulusE
        self.trail_length = trail_length
        self.time_buffer = time_buffer  # time buffer is dt + time skipped

    def calculateTension(
        self, bullet_position: np.ndarray, target_position: np.ndarray
    ):
        displacement = bullet_position - target_position
        distance = np.linalg.norm(displacement)

        if distance == 0 or distance <= self.trail_length:
            return np.zeros(3)

        direction = displacement / distance

        tension_magnitude = -self.modulus_e * (distance - self.trail_length)
        tension = tension_magnitude * direction
        return tension

    def update_force(self, particle: Particle):
        bullet_position = particle.position.copy()

        bullet_position += (
            (particle.velocity * self.time_buffer)
            + 0.5 * particle.acceleration * self.time_buffer** 2
        )  # predicted future position based on current velocity

        target_position = self.other.position.copy()

        target_position += (
            self.other.velocity * self.time_buffer
            + 0.5 * self.other.acceleration * self.time_buffer**2
        )  # predicted future position based on current velocity, consider both bullet and target for accelerations

        particle.add_force(self.calculateTension(bullet_position, target_position))


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
            force_generator.update_force(particle, dt)
