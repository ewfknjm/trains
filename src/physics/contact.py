import numpy as np
import math
from dataclasses import dataclass
from .rigidbody import RigidBody
from typing import Optional


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
    contacts: list[Contact]
    contacts_left: int
    _contact_count: int

    @property
    def contact_count(self):
        return self._contact_count

    @contact_count.setter
    def contact_count(self, value: int):
        if value < 0:
            raise ValueError("Negative number of contacts")

        added = value - self._contact_count
        if added < 0:
            raise ValueError("Cannot decrease contact count directly this way")

        new_contact_left = self.contacts_left - added
        if new_contact_left < 0:
            raise ValueError("contact_count exceeds contact_left")

        self._contact_count = value
        self.contacts_left = new_contact_left

    def add_contact(self, contact: Contact) -> bool:
        if self.contacts_left <= 0:
            return False

        self.contacts.append(contact)
        self.contact_count += 1
        return True

    @staticmethod
    def _make_orthonormal_basis(normal: np.ndarray) -> np.ndarray:
        pass
