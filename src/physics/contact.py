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
