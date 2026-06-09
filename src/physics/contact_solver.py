from dataclasses import dataclass
from typing import Optional
import numpy as np
from .contact import Contact, ContactData, get_contact_basis
from .rigidbody import RigidBody
import math

ANGULAR_LIMIT_CONSTANT = 0.05
STATIC_FRICTION_RATIO = 1.2


@dataclass
class PreparedContact:
    contact: Contact
    relative_a: np.ndarray
    rxn_a: np.ndarray
    k_a: float

    relative_b: Optional[np.ndarray]
    rxn_b: Optional[np.ndarray]
    k_b: float

    contact_basis: Optional[np.ndarray]
    impulse_matrix: np.ndarray
    inverse_mass_matrix: np.ndarray
    angular_a: np.ndarray
    angular_b: Optional[np.ndarray]
    rxn_a_norm: float
    rxn_b_norm: float

    @property
    def K(self) -> float:
        return self.k_a + self.k_b

    def contact_velocity(self) -> np.ndarray:
        c = self.contact
        closing_velocity = c.body_a.velocity + np.cross(c.body_a.omega, self.relative_a)
        if c.body_b and self.relative_b is not None:
            closing_velocity -= c.body_b.velocity + np.cross(
                c.body_b.omega, self.relative_b
            )

        if self.contact_basis is None:
            raise ValueError("prepare first")

        return self.contact_basis.T @ closing_velocity

    def apply_velocity_change(self, impulse_vector: np.ndarray):
        c = self.contact

        c.body_a.velocity += impulse_vector * c.body_a.inverse_mass
        c.body_a.omega += self.angular_a @ impulse_vector

        if c.body_b and self.relative_b is not None and self.angular_b is not None:
            c.body_b.velocity -= impulse_vector * c.body_b.inverse_mass
            c.body_b.omega -= self.angular_b @ impulse_vector

    def apply_position_change(self, depth: float):
        PENETRATION_SLOP = 0.01
        depth -= PENETRATION_SLOP

        c = self.contact
        n = c.contact_normal

        delta_pos_a, delta_pos_b = np.zeros(3), np.zeros(3)
        delta_rot_a, delta_rot_b = np.zeros(3), np.zeros(3)

        if depth <= 0 or math.isclose(self.K, 0.0, abs_tol=1e-8):
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        K_angular_a = self.k_a - c.body_a.inverse_mass
        lin_a = depth * c.body_a.inverse_mass / self.K
        ang_a = depth * K_angular_a / self.K

        _ra = self.relative_a
        limit = ANGULAR_LIMIT_CONSTANT * math.sqrt(
            _ra[0] * _ra[0] + _ra[1] * _ra[1] + _ra[2] * _ra[2]
        )
        if abs(ang_a) > limit:
            total = lin_a + ang_a
            ang_a = limit if ang_a >= 0 else -limit
            lin_a = total - ang_a

        delta_pos_a = lin_a * n
        c.body_a.position += delta_pos_a

        if not math.isclose(K_angular_a, 0.0, abs_tol=1e-8) and self.rxn_a_norm > 1e-8:
            inertia_rxn_a = c.body_a.inverse_inertia_tensor_world @ self.rxn_a
            p_a = ang_a / max(float(self.rxn_a @ inertia_rxn_a), 1e-8)
            delta_rot_a = inertia_rxn_a * float(p_a)
            c.body_a.orientation.add_scaled_vector(inertia_rxn_a, float(p_a))

        c.body_a.mark_dirty()

        if c.body_b and self.relative_b is not None and self.rxn_b is not None:
            K_angular_b = self.k_b - c.body_b.inverse_mass
            lin_b = depth * c.body_b.inverse_mass / self.K
            ang_b = depth * K_angular_b / self.K

            _rb = self.relative_b
            limit = ANGULAR_LIMIT_CONSTANT * math.sqrt(
                _rb[0] * _rb[0] + _rb[1] * _rb[1] + _rb[2] * _rb[2]
            )
            if abs(ang_b) > limit:
                total = lin_b + ang_b
                ang_b = limit if ang_b >= 0 else -limit
                lin_b = total - ang_b

            delta_pos_b = -lin_b * n
            c.body_b.position += delta_pos_b

            if (
                not math.isclose(K_angular_b, 0.0, abs_tol=1e-8)
                and self.rxn_b_norm > 1e-8
            ):
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


def skew(v: np.ndarray) -> np.ndarray:
    vx, vy, vz = v
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ]
    )


def prepare_contacts(contacts: list[Contact]) -> list[PreparedContact]:
    prepared = []
    for c in contacts:
        n = c.contact_normal
        r_a = c.contact_point - c.body_a.position
        K_a, rxn_a = _inertia_contribution(c.body_a, r_a, n)
        rxn_a_norm = math.sqrt(float(rxn_a @ rxn_a))
        skewed_a = skew(r_a)
        I_inv_a = c.body_a.inverse_inertia_tensor_world
        Z = -skewed_a @ I_inv_a @ skewed_a
        angular_a = I_inv_a @ skewed_a
        inverse_mass = c.body_a.inverse_mass

        r_b, rxn_b, K_b = None, None, 0.0
        angular_b: Optional[np.ndarray] = None
        rxn_b_norm = 0.0
        if c.body_b:
            r_b = c.contact_point - c.body_b.position
            K_b, rxn_b = _inertia_contribution(c.body_b, r_b, n)
            rxn_b_norm = math.sqrt(float(rxn_b @ rxn_b))
            skewed_b = skew(r_b)
            I_inv_b = c.body_b.inverse_inertia_tensor_world
            Z += -skewed_b @ I_inv_b @ skewed_b
            angular_b = I_inv_b @ skewed_b
            inverse_mass += c.body_b.inverse_mass

        contact_basis = get_contact_basis(n)
        Y = contact_basis.T @ Z @ contact_basis
        Y += np.eye(3) * inverse_mass
        try:
            impulse_matrix = np.linalg.inv(Y)
        except np.linalg.LinAlgError:
            impulse_matrix = np.zeros(
                (
                    3,
                    3,
                )
            )

        prepared.append(
            PreparedContact(
                c,
                r_a,
                rxn_a,
                K_a,
                r_b,
                rxn_b,
                K_b,
                contact_basis,
                impulse_matrix,
                Y,
                angular_a,
                angular_b,
                rxn_a_norm,
                rxn_b_norm,
            )
        )
    return prepared


class ContactResolver:
    def __init__(self, velocity_iterations: int, position_iterations: int):
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    def resolve(self, data: ContactData, dt: float = 1 / 60):
        if not data.contacts:
            return

        for contact in data.contacts:
            if contact.body_a.is_sleeping:
                if contact.body_b is not None and not contact.body_b.is_sleeping:
                    contact.body_a.set_awake()
            if contact.body_b is not None and contact.body_b.is_sleeping:
                if not contact.body_a.is_sleeping:
                    contact.body_b.set_awake()

        prepared = prepare_contacts(data.contacts)
        self._resolve_velocities(prepared, dt)
        self._resolve_penetrations(prepared)

    def _resolve_penetrations(self, prepared_contacts: list[PreparedContact]):
        for _ in range(self.position_iterations):
            worst_penetration = 0
            worst_contact: Optional[PreparedContact] = None

            for contact in prepared_contacts:
                if contact.contact.contact_penetration > worst_penetration:
                    worst_penetration = contact.contact.contact_penetration
                    worst_contact = contact

            if worst_contact is None or worst_penetration <= 0.01:
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

    def _resolve_velocities(self, prepared_list: list[PreparedContact], dt: float):
        cv_cache = [pc.contact_velocity() for pc in prepared_list]
        body_contacts: dict[int, list[int]] = {}
        for i, pc in enumerate(prepared_list):
            body_contacts.setdefault(id(pc.contact.body_a), []).append(i)
            if pc.contact.body_b is not None:
                body_contacts.setdefault(id(pc.contact.body_b), []).append(i)

        for _ in range(self.velocity_iterations):
            worst_contact_velocity = 0.0
            worst_idx = -1

            for i, cv in enumerate(cv_cache):
                if cv[0] < worst_contact_velocity:
                    worst_contact_velocity = cv[0]
                    worst_idx = i

            if worst_idx == -1:
                break

            worst_contact = prepared_list[worst_idx]
            worst_cv = cv_cache[worst_idx]

            K = worst_contact.K
            if K < 1e-8:
                continue

            c = worst_contact.contact
            e = c.collision_restitution
            mu = c.coefficient_of_friction
            STATIC_FRICTION = mu * STATIC_FRICTION_RATIO
            DYNAMIC_FRICTION = mu

            a_acceleration = c.body_a.last_frame_acceleration
            b_acceleration = (
                c.body_b.last_frame_acceleration if c.body_b else np.zeros(3)
            )

            if worst_contact.contact_basis is None:
                raise ValueError("prepare contacts first")

            accel_velocity = (a_acceleration - b_acceleration) * dt
            accel_contact = worst_contact.contact_basis.T @ accel_velocity
            accel_build_up = accel_contact[0]

            bounce_velocity = -e * worst_contact_velocity
            if accel_build_up < 0.0:
                bounce_velocity += e * accel_build_up
                bounce_velocity = 0.0 if bounce_velocity < 0.0 else bounce_velocity

            desired_velocity_change = bounce_velocity - worst_contact_velocity
            target_velocity = np.array(
                [desired_velocity_change, -worst_cv[1], -worst_cv[2]]
            )

            impulse_contact = worst_contact.impulse_matrix @ target_velocity
            planar_impulse = math.hypot(impulse_contact[1], impulse_contact[2])

            if planar_impulse > impulse_contact[0] * STATIC_FRICTION:
                if planar_impulse > 1e-8:
                    impulse_contact[1] /= planar_impulse
                    impulse_contact[2] /= planar_impulse

                delta_velocity = worst_contact.inverse_mass_matrix[0, :]
                denominator = (
                    delta_velocity[0]
                    + delta_velocity[1] * DYNAMIC_FRICTION * impulse_contact[1]
                    + delta_velocity[2] * DYNAMIC_FRICTION * impulse_contact[2]
                )

                if denominator <= 1e-8:
                    continue

                impulse_contact[0] = desired_velocity_change / denominator
                impulse_contact[1] *= DYNAMIC_FRICTION * impulse_contact[0]
                impulse_contact[2] *= DYNAMIC_FRICTION * impulse_contact[0]

            impulse_world = worst_contact.contact_basis @ impulse_contact
            worst_contact.apply_velocity_change(impulse_world)

            body_a, body_b = c.body_a, c.body_b
            to_update: set[int] = set(body_contacts.get(id(body_a), []))
            if body_b is not None:
                to_update.update(body_contacts.get(id(body_b), []))
            for i in to_update:
                cv_cache[i] = prepared_list[i].contact_velocity()
