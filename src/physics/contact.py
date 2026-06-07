import numpy as np
from dataclasses import dataclass, field
from .rigidbody import RigidBody
from typing import Optional

# consider duff et al.


@dataclass
class Contact:
    body_a: RigidBody
    body_b: Optional[RigidBody]
    contact_point: np.ndarray
    contact_normal: np.ndarray
    contact_penetration: float
    collision_restitution: float
    coefficient_of_friction: float


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

    t_len = (tx * tx + ty * ty + tz * tz) ** 0.5
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
    delta = np.cross(relative_position, contact_normal)
    delta = body.inverse_inertia_tensor_world @ delta
    delta = np.cross(delta, relative_position)
    return float(delta @ contact_normal)


def calculate_delta_velocity_local(
    body_a: RigidBody, body_b: Optional[RigidBody], contact_info: Contact
) -> float:
    contact_point = contact_info.contact_point
    contact_normal = contact_info.contact_normal
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


def velocity_change_per_unit_impulse(
    body_a: RigidBody, body_b: Optional[RigidBody], contact_info: Contact
) -> np.ndarray:
    a_relative_contact_position = contact_info.contact_point - body_a.position
    a_angular_velocity = np.cross(body_a.rotation, a_relative_contact_position)
    closing_velocity = a_angular_velocity + body_a.velocity

    if not body_b:
        return closing_velocity

    b_relative_contact_position = contact_info.contact_point - body_b.position
    b_angular_velocity = np.cross(body_b.rotation, b_relative_contact_position)
    closing_velocity -= b_angular_velocity + body_b.velocity

    return closing_velocity


def calculate_closing_velocity_contact(
    closing_velocity_world: np.ndarray, contact_info: Contact
) -> float:
    transform = get_contact_basis(contact_info.contact_normal)
    contact_velocity = transform.T @ closing_velocity_world
    return contact_velocity[0]
