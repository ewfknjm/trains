from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from .rigidbody import RigidBody
from typing import Optional

# consider duff et al.

# ----------------------- helper functions ----------------------------------------- #


def get_contact_basis(normal: np.ndarray) -> np.ndarray:
    nx = normal[0]
    ny = normal[1]
    nz = normal[2]

    if abs(nx) <= abs(ny):
        tx = 0.0
        ty = nz
        tz = -ny
    else:
        tx = -nz
        ty = 0.0
        tz = nx

    t_len = math.sqrt(tx * tx + ty * ty + tz * tz)
    tx /= t_len
    ty /= t_len
    tz /= t_len

    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx

    return np.array([[nx, tx, bx], [ny, ty, by], [nz, tz, bz]], dtype=float)


def angular_delta_velocity(
    body: RigidBody, relative_position: np.ndarray, contact_normal: np.ndarray
) -> float:
    rxn = np.cross(relative_position, contact_normal)
    delta_omega = body.inverse_inertia_tensor_world @ rxn
    return float(delta_omega @ rxn)


# -------------------- end of helper functions -------------------------------------- #


@dataclass
class Contact:
    body_a: RigidBody
    body_b: Optional[RigidBody]
    contact_point: np.ndarray
    contact_normal: np.ndarray
    contact_penetration: float
    collision_restitution: float
    coefficient_of_friction: float

    def calculate_velocity_change_per_unit_impulse(self) -> float:
        body_a = self.body_a
        body_b = self.body_b

        contact_point = self.contact_point
        contact_normal = self.contact_normal
        a_relative_contact_position = contact_point - body_a.position
        # only along contact normal, assumed no friction
        delta_velocity = body_a.inverse_mass
        delta_velocity += angular_delta_velocity(
            body_a, a_relative_contact_position, contact_normal
        )

        if not body_b:
            return delta_velocity

        b_relative_contact_position = contact_point - body_b.position
        delta_velocity += body_b.inverse_mass + angular_delta_velocity(
            body_b, b_relative_contact_position, contact_normal
        )

        return delta_velocity

    def calculate_closing_velocity_world(self: Contact) -> np.ndarray:
        body_a = self.body_a
        body_b = self.body_b

        a_relative_contact_position = self.contact_point - body_a.position
        a_angular_velocity = np.cross(body_a.omega, a_relative_contact_position)
        closing_velocity = a_angular_velocity + body_a.velocity

        if not body_b:
            return closing_velocity

        b_relative_contact_position = self.contact_point - body_b.position
        b_angular_velocity = np.cross(body_b.omega, b_relative_contact_position)
        closing_velocity -= b_angular_velocity + body_b.velocity

        return closing_velocity

    def calculate_closing_velocity_contact(self: Contact) -> float:
        # transform = self.get_contact_basis(self.contact_normal)
        # contact_velocity = transform.T @ closing_velocity_world
        # return contact_velocity[0]
        closing_velocity_world = self.calculate_closing_velocity_world()
        return float(closing_velocity_world @ self.contact_normal)

    def calculate_desired_velocity_change(self: Contact) -> float:
        closing_contact_velocity = self.calculate_closing_velocity_contact()
        return float(-closing_contact_velocity * (1 + self.collision_restitution))

    def calculate_impulse(self: Contact) -> float:
        delta_velocity = self.calculate_velocity_change_per_unit_impulse()
        desired_delta_velocity = self.calculate_desired_velocity_change()
        if math.isclose(delta_velocity, 0.0):
            return 0.0

        impulse_x = desired_delta_velocity / delta_velocity
        return float(self.contact_normal @ impulse_x)

    def _calculate_omega_change(
        self, body: RigidBody, contact_point: np.ndarray, impulse: float
    ) -> float:
        relative_contact_position = contact_point - body.position
        impulsive_torque = np.cross(impulse, relative_contact_position)
        omega_change = body.inverse_inertia_tensor_world @ impulsive_torque
        return float(omega_change)

    def apply_impulse(self):
        impulse = self.calculate_impulse()

        body_a = self.body_a
        a_velocity_change = impulse * body_a.inverse_mass
        a_omega_change = self._calculate_omega_change(
            body_a, self.contact_point, impulse
        )

        body_a.velocity += a_velocity_change
        body_a.omega += a_omega_change

        body_b = self.body_b
        if not body_b:
            return

        b_velocity_change = -impulse * body_b.inverse_mass
        b_omega_change = self._calculate_omega_change(
            body_b, self.contact_point, -impulse
        )

        body_b.velocity += b_velocity_change
        body_b.omega += b_omega_change


@dataclass
class ContactData:
    max_contacts: int
    contacts: list[Contact] = field(default_factory=list)

    @property
    def contact_count(self) -> int:
        return len(self.contacts)

    @property
    def is_full(self) -> bool:
        return len(self.contacts) >= self.max_contacts

    def add_contact(self, contact: Contact) -> bool:
        if self.is_full:
            return False
        self.contacts.append(contact)
        return True

    def clear(self):
        self.contacts.clear()
