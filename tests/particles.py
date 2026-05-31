import numpy as np
from typing import Protocol
from dataclasses import dataclass


class Particle:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.position = np.array([x, y, z], dtype="float")
        self.velocity = np.zeros(3, dtype="float")
        self.acceleration = np.zeros(3, dtype="float")
        self._mass = 0.0
        self.inverse_mass: float = 0.0
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


class ParticleContact:
    def __init__(
        self,
        particle_a: Particle,
        particle_b: Particle,
        restitution: float,
        contact_normal: np.ndarray,
        penetration_depth: float,
    ):
        if restitution < 0:
            raise ValueError("restitution must be above 0")
        norm = np.linalg.norm(contact_normal)
        if norm == 0:
            raise ValueError("contact_normal cannot be a zero vector")
        self.particle_a = particle_a
        self.particle_b = particle_b
        self.restitution = restitution
        self.contact_normal = contact_normal / norm
        self.penetration_depth = penetration_depth

    @staticmethod
    def calculate_closing_velocity(
        particle_a_velocity: np.ndarray,
        particle_b_velocity: np.ndarray,
        contact_normal: np.ndarray,
    ):
        return np.dot((particle_a_velocity - particle_b_velocity), contact_normal)

    def resolve_impulse(self, dt: float):
        contact_normal = self.contact_normal
        particle_a = self.particle_a
        particle_b = self.particle_b
        closing_velocity = self.calculate_closing_velocity(
            particle_a.velocity, particle_b.velocity, contact_normal
        )

        if closing_velocity > 0:
            return

        opening_velocity = -self.restitution * closing_velocity

        rel_accel_induced_vel = 0.0
        rel_accel_induced_vel = particle_a.acceleration
        rel_accel_induced_vel -= particle_b.acceleration
        rel_accel_induced_vel = np.dot(rel_accel_induced_vel, contact_normal) * dt

        if rel_accel_induced_vel < 0.0:
            opening_velocity = max(
                0.0, opening_velocity + self.restitution * rel_accel_induced_vel
            )

        delta_velocity = opening_velocity - closing_velocity

        total_inverse_sum = particle_a.inverse_mass + particle_b.inverse_mass
        if total_inverse_sum == 0:
            return

        impulse = (delta_velocity / total_inverse_sum) * contact_normal

        particle_a.velocity += particle_a.inverse_mass * impulse
        particle_b.velocity += particle_b.inverse_mass * -impulse

    def resolve_interpenetration(self):
        depth = self.penetration_depth
        if depth <= 0:
            return
        particle_a = self.particle_a
        particle_b = self.particle_b

        total_inverse_sum = particle_a.inverse_mass + particle_b.inverse_mass

        if total_inverse_sum <= 0:
            return

        particle_a_movement = (
            depth
            * self.contact_normal
            * (1.0 / total_inverse_sum)
            * particle_a.inverse_mass
        )
        particle_b_movement = (
            depth
            * self.contact_normal
            * (1.0 / total_inverse_sum)
            * particle_b.inverse_mass
            * -1.0
        )

        particle_a.position += particle_a_movement
        particle_b.position += particle_b_movement


class ContactGenerator(Protocol):
    def add_contact(self) -> ParticleContact | None: ...


class SphereContactGenerator:
    def __init__(
        self,
        particle_a: Particle,
        particle_b: Particle,
        radius_a: float,
        radius_b: float,
        restitution: float,
    ):
        self.particle_a = particle_a
        self.particle_b = particle_b
        self.radius_a = radius_a
        self.radius_b = radius_b
        self.restitution = restitution

    def add_contact(self) -> ParticleContact | None:
        diff = self.particle_a.position - self.particle_b.position
        distance = np.linalg.norm(diff)
        combined_radius = self.radius_a + self.radius_b

        if distance >= combined_radius:
            return None  # no contact, nothing to add

        contact_normal = diff / distance
        penetration_depth = combined_radius - distance

        return ParticleContact(
            self.particle_a,
            self.particle_b,
            self.restitution,
            contact_normal,
            float(penetration_depth),
        )


def calculate_current_length(
    particle_a_position: np.ndarray, particle_b_position: np.ndarray
):
    return np.linalg.norm(particle_a_position - particle_b_position)


@dataclass
class CableContactGenerator:
    particle_a: Particle
    particle_b: Particle
    restitution: float
    max_length: float

    def add_contact(self) -> ParticleContact | None:
        a = self.particle_a
        b = self.particle_b

        length = calculate_current_length(a.position, b.position)
        if length <= self.max_length:
            return

        penetration = length - self.max_length
        rel_vec = b.position - a.position
        contact_normal = rel_vec / length

        return ParticleContact(
            a, b, self.restitution, contact_normal, float(penetration)
        )


@dataclass
class RodContactGenerator:
    particle_a: Particle
    particle_b: Particle
    max_length: float

    def add_contact(self) -> ParticleContact | None:
        a = self.particle_a
        b = self.particle_b

        length = calculate_current_length(a.position, b.position)
        if length == self.max_length:
            return None

        rel_vec = b.position - a.position
        contact_normal = rel_vec / length

        if length > self.max_length:
            penetration = length - self.max_length
        else:
            contact_normal *= -1.0
            penetration = self.max_length - length

        return ParticleContact(a, b, 0.0, contact_normal, float(penetration))


class ParticleContactResolver:
    def __init__(self, iterations: int):
        self.iterations = iterations

    def resolve_contacts(self, contacts: list[ParticleContact], dt: float):
        iterations_used = 0
        while iterations_used < self.iterations:
            min_velocity = 0.0
            chosen = None
            for contact in contacts:
                cv = ParticleContact.calculate_closing_velocity(
                    contact.particle_a.velocity,
                    contact.particle_b.velocity,
                    contact.contact_normal,
                )
                if cv < min_velocity:
                    min_velocity = cv
                    chosen = contact

            if chosen is None:
                break

            chosen.resolve_impulse(dt)
            chosen.resolve_interpenetration()
            iterations_used += 1
