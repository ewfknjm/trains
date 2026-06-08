from dataclasses import dataclass
from typing import Optional

import numpy as np
from .contact import Contact, ContactData
from .rigidbody import RigidBody
import math

ANGULAR_LIMIT_CONSTANT = 0.2


@dataclass
class PreparedContact:
    contact: Contact
    relative_a: np.ndarray
    rxn_a: np.ndarray
    k_a: float

    relative_b: Optional[np.ndarray]
    rxn_b: Optional[np.ndarray]
    k_b: float

    @property
    def K(self) -> float:
        return self.k_a + self.k_b

    def contact_velocity(self) -> float:
        c = self.contact
        closing_velocity = c.body_a.velocity + np.cross(c.body_a.omega, self.relative_a)
        if c.body_b and self.relative_b is not None:
            closing_velocity -= c.body_b.velocity + np.cross(
                c.body_b.omega, self.relative_b
            )

        return float(closing_velocity @ c.contact_normal)

    def apply_velocity_change(self, impulse: float):
        c = self.contact
        impulse_vector = impulse * c.contact_normal

        c.body_a.velocity += impulse_vector * c.body_a.inverse_mass
        c.body_a.omega += c.body_a.inverse_inertia_tensor_world @ np.cross(
            self.relative_a, impulse_vector
        )

        if c.body_b and self.relative_b is not None:
            c.body_b.velocity -= impulse_vector * c.body_b.inverse_mass
            c.body_b.omega -= c.body_b.inverse_inertia_tensor_world @ np.cross(
                self.relative_b, impulse_vector
            )

    def apply_position_change(self, depth: float):
        c = self.contact
        n = c.contact_normal

        delta_pos_a, delta_pos_b = np.zeros(3), np.zeros(3)
        delta_rot_a, delta_rot_b = np.zeros(3), np.zeros(3)

        if depth <= 0 or math.isclose(self.K, 0.0, abs_tol=1e-8):
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        K_angular_a = self.k_a - c.body_a.inverse_mass
        lin_a = depth * c.body_a.inverse_mass / self.K
        ang_a = depth * K_angular_a / self.K

        limit = ANGULAR_LIMIT_CONSTANT * np.linalg.norm(self.relative_a)
        if abs(ang_a) > limit:
            total = lin_a + ang_a
            ang_a = limit if ang_a >= 0 else -limit
            lin_a = total - ang_a

        delta_pos_a = lin_a * n
        c.body_a.position += delta_pos_a

        if not math.isclose(K_angular_a, 0.0) and np.linalg.norm(self.rxn_a) > 1e-8:
            inertia_rxn_a = c.body_a.inverse_inertia_tensor_world @ self.rxn_a
            p_a = ang_a / max(float(self.rxn_a @ inertia_rxn_a), 1e-8)
            delta_rot_a = inertia_rxn_a * float(p_a)
            c.body_a.orientation.add_scaled_vector(inertia_rxn_a, float(p_a))

        c.body_a.mark_dirty()

        delta_pos_b = np.zeros(3)
        if c.body_b and self.relative_b is not None and self.rxn_b is not None:
            K_angular_b = self.k_b - c.body_b.inverse_mass
            lin_b = depth * c.body_b.inverse_mass / self.K
            ang_b = depth * K_angular_b / self.K

            limit = ANGULAR_LIMIT_CONSTANT * np.linalg.norm(self.relative_b)
            if abs(ang_b) > limit:
                total = lin_b + ang_b
                ang_b = limit if ang_b >= 0 else -limit
                lin_b = total - ang_b

            delta_pos_b = -lin_b * n
            c.body_b.position += delta_pos_b

            if not math.isclose(K_angular_b, 0.0) and np.linalg.norm(self.rxn_b) > 1e-8:
                inertia_rxn_b = c.body_b.inverse_inertia_tensor_world @ self.rxn_b
                p_b = ang_b / max(float(self.rxn_b @ inertia_rxn_b), 1e-8)
                delta_rot_b = inertia_rxn_b * float(-p_b)
                c.body_b.orientation.add_scaled_vector(inertia_rxn_b, float(-p_b))

            c.body_b.mark_dirty()

        return delta_rot_a, delta_rot_b, delta_pos_a, delta_pos_b


def _inertia_contribution(
    body: RigidBody, r: np.ndarray, contact_normal: np.ndarray
) -> tuple[float, np.ndarray]:
    rxn = np.cross(r, contact_normal)
    K = float(body.inverse_mass + rxn @ (body.inverse_inertia_tensor_world @ rxn))
    return K, rxn


def prepare_contacts(contacts: list[Contact]) -> list[PreparedContact]:
    prepared = []
    for c in contacts:
        n = c.contact_normal
        r_a = c.contact_point - c.body_a.position
        K_a, rxn_a = _inertia_contribution(c.body_a, r_a, n)

        r_b, rxn_b, K_b = None, None, 0.0
        if c.body_b:
            r_b = c.contact_point - c.body_b.position
            K_b, rxn_b = _inertia_contribution(c.body_b, r_b, n)

        prepared.append(PreparedContact(c, r_a, rxn_a, K_a, r_b, rxn_b, K_b))
    return prepared


class ContactResolver:
    def __init__(self, velocity_iterations: int, position_iterations: int):
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    def resolve(self, data: ContactData):
        if not data.contacts:
            return

        prepared = prepare_contacts(data.contacts)
        self._resolve_penetrations(prepared)
        self._resolve_velocities(prepared)

    def _resolve_penetrations(self, prepared_contacts: list[PreparedContact]):
        for _ in range(self.position_iterations):
            worst_penetration = 0
            worst_contact: Optional[PreparedContact] = None

            for contact in prepared_contacts:
                if contact.contact.contact_penetration > worst_penetration:
                    worst_penetration = contact.contact.contact_penetration
                    worst_contact = contact

            if worst_contact is None:
                break

            c = worst_contact.contact
            worst_contact.contact.contact_penetration = 0.0

            delta_rot_a, delta_rot_b, delta_pos_a, delta_pos_b = (
                worst_contact.apply_position_change(worst_penetration)
            )

            for other in prepared_contacts:
                if other is worst_contact:
                    continue

                oc = other.contact
                n = oc.contact_normal

                if oc.body_a is c.body_a:
                    displacement = delta_pos_a + np.cross(delta_rot_a, other.relative_a)
                    oc.contact_penetration -= float(displacement @ n)
                elif oc.body_a is c.body_b:
                    displacement = delta_pos_b + np.cross(delta_rot_b, other.relative_a)
                    oc.contact_penetration -= float(displacement @ n)

                if oc.body_b is c.body_a and other.relative_b is not None:
                    displacement = delta_pos_a + np.cross(delta_rot_a, other.relative_b)
                    oc.contact_penetration += float(displacement @ n)
                elif oc.body_b is c.body_b and other.relative_b is not None:
                    displacement = delta_pos_b + np.cross(delta_rot_b, other.relative_b)
                    oc.contact_penetration += float(displacement @ n)

    def _resolve_velocities(self, prepared_list: list[PreparedContact]):
        VELOCITY_EPSILON = 0.25

        for _ in range(self.velocity_iterations):
            worst_contact_velocity = 0.0
            worst_contact: Optional[PreparedContact] = None

            for pc in prepared_list:
                current_contact_velocity = pc.contact_velocity()
                if current_contact_velocity < worst_contact_velocity:
                    worst_contact_velocity = current_contact_velocity
                    worst_contact = pc

            if worst_contact is None:
                break

            K = worst_contact.K
            if math.isclose(K, 0.0):
                continue

            e = worst_contact.contact.collision_restitution

            if abs(worst_contact_velocity) < VELOCITY_EPSILON:
                e = 0.0

            j = -(1.0 + e) * worst_contact_velocity / K
            worst_contact.apply_velocity_change(j)
