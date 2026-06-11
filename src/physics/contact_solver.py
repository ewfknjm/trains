from dataclasses import dataclass
from typing import Optional
import numpy as np
from .contact import Contact, ContactManifold, ContactData, get_contact_basis
import math

ANGULAR_LIMIT_CONSTANT = 0.05


@dataclass
class PreparedContact:
    manifold: ContactManifold
    contact: Contact
    relative_a: np.ndarray
    rxn_a: np.ndarray
    k_a: float

    relative_b: Optional[np.ndarray]
    rxn_b: Optional[np.ndarray]
    k_b: float

    contact_basis: Optional[np.ndarray]
    inverse_mass_matrix: np.ndarray
    angular_a: np.ndarray
    angular_b: Optional[np.ndarray]
    rxn_a_norm: float
    rxn_b_norm: float

    @property
    def K(self) -> float:
        return self.k_a + self.k_b

    def contact_velocity(self) -> np.ndarray:
        m = self.manifold
        closing_velocity = m.body_a.velocity + np.cross(m.body_a.omega, self.relative_a)
        if m.body_b and self.relative_b is not None:
            closing_velocity -= m.body_b.velocity + np.cross(
                m.body_b.omega, self.relative_b
            )

        if self.contact_basis is None:
            raise ValueError("prepare first")

        return self.contact_basis.T @ closing_velocity

    def apply_velocity_change(self, impulse_vector: np.ndarray):
        m = self.manifold

        m.body_a.velocity += impulse_vector * m.body_a.inverse_mass
        m.body_a.omega += self.angular_a @ impulse_vector

        if m.body_b and self.relative_b is not None and self.angular_b is not None:
            m.body_b.velocity -= impulse_vector * m.body_b.inverse_mass
            m.body_b.omega -= self.angular_b @ impulse_vector

    def apply_position_change(self, depth: float):
        PENETRATION_SLOP = 0.01
        depth -= PENETRATION_SLOP

        c = self.contact
        m = self.manifold
        n = c.world_normal

        delta_pos_a, delta_pos_b = np.zeros(3), np.zeros(3)
        delta_rot_a, delta_rot_b = np.zeros(3), np.zeros(3)

        if depth <= 0 or math.isclose(self.K, 0.0, abs_tol=1e-8):
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

        K_angular_a = self.k_a - m.body_a.inverse_mass
        lin_a = depth * m.body_a.inverse_mass / self.K
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
        m.body_a.position += delta_pos_a

        if not math.isclose(K_angular_a, 0.0, abs_tol=1e-8) and self.rxn_a_norm > 1e-8:
            inertia_rxn_a = m.body_a.inverse_inertia_tensor_world @ self.rxn_a
            p_a = ang_a / max(float(self.rxn_a @ inertia_rxn_a), 1e-8)
            delta_rot_a = inertia_rxn_a * float(p_a)
            m.body_a.orientation.add_scaled_vector(inertia_rxn_a, float(p_a))

        m.body_a.mark_dirty()

        if m.body_b and self.relative_b is not None and self.rxn_b is not None:
            K_angular_b = self.k_b - m.body_b.inverse_mass
            lin_b = depth * m.body_b.inverse_mass / self.K
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
            m.body_b.position += delta_pos_b

            if (
                not math.isclose(K_angular_b, 0.0, abs_tol=1e-8)
                and self.rxn_b_norm > 1e-8
            ):
                inertia_rxn_b = m.body_b.inverse_inertia_tensor_world @ self.rxn_b
                p_b = ang_b / max(float(self.rxn_b @ inertia_rxn_b), 1e-8)
                delta_rot_b = inertia_rxn_b * float(-p_b)
                m.body_b.orientation.add_scaled_vector(inertia_rxn_b, float(-p_b))

            m.body_b.mark_dirty()

        return delta_rot_a, delta_rot_b, delta_pos_a, delta_pos_b


def skew(v: np.ndarray) -> np.ndarray:
    vx, vy, vz = v
    return np.array(
        [
            [0.0, -vz, vy],
            [vz, 0.0, -vx],
            [-vy, vx, 0.0],
        ]
    )


def prepare_contacts(data: ContactData) -> list[PreparedContact]:
    prepared = []
    for manifold in data.manifolds.values():
        for c in manifold.contacts.values():
            n = c.world_normal
            r_a = c.world_point - manifold.body_a.position
            rxn_a = np.cross(r_a, n)
            rxn_a_norm = np.linalg.norm(rxn_a)
            skewed_a = skew(r_a)
            I_inv_a = manifold.body_a.inverse_inertia_tensor_world
            Z = -skewed_a @ I_inv_a @ skewed_a
            angular_a = I_inv_a @ skewed_a

            contact_basis = get_contact_basis(n)
            r_b, rxn_b, X = None, None, np.zeros((3, 3))
            angular_b: Optional[np.ndarray] = None
            rxn_b_norm = 0.0
            if manifold.body_b:
                r_b = c.world_point - manifold.body_b.position
                rxn_b = np.cross(r_b, n)
                rxn_b_norm = np.linalg.norm(rxn_b)
                skewed_b = skew(r_b)
                I_inv_b = manifold.body_b.inverse_inertia_tensor_world
                X = -skewed_b @ I_inv_b @ skewed_b
                angular_b = I_inv_b @ skewed_b
                X = (contact_basis.T @ X @ contact_basis) + (
                    np.eye(3) * manifold.body_b.inverse_mass
                )

            Z = (contact_basis.T @ Z @ contact_basis) + (
                np.eye(3) * manifold.body_a.inverse_mass
            )
            Y = Z + X

            prepared.append(
                PreparedContact(
                    manifold,
                    c,
                    r_a,
                    rxn_a,
                    Z[0, 0],
                    r_b,
                    rxn_b,
                    X[0, 0],
                    contact_basis,
                    Y,
                    angular_a,
                    angular_b,
                    float(rxn_a_norm),
                    float(rxn_b_norm),
                )
            )
    return prepared


class ContactResolver:
    def __init__(self, velocity_iterations: int, position_iterations: int):
        self.velocity_iterations = velocity_iterations
        self.position_iterations = position_iterations

    def resolve(self, data: ContactData, dt: float = 1 / 60):
        if data.contact_count == 0:
            return

        for manifold in data.manifolds.values():
            if manifold.body_a.is_sleeping:
                if manifold.body_b is not None and not manifold.body_b.is_sleeping:
                    manifold.body_a.set_awake()
            if manifold.body_b is not None and manifold.body_b.is_sleeping:
                if not manifold.body_a.is_sleeping:
                    manifold.body_b.set_awake()

        prepared = prepare_contacts(data)
        self._resolve_velocities(prepared, dt)
        self._resolve_penetrations(prepared)

    def _resolve_penetrations(self, prepared_contacts: list[PreparedContact]):
        for _ in range(self.position_iterations):
            worst_penetration = 0
            worst_contact: Optional[PreparedContact] = None

            for contact in prepared_contacts:
                if contact.contact.penetration > worst_penetration:
                    worst_penetration = contact.contact.penetration
                    worst_contact = contact

            if worst_contact is None or worst_penetration <= 0.01:
                break

            m = worst_contact.manifold
            worst_contact.contact.penetration = 0.0

            delta_rot_a, delta_rot_b, delta_pos_a, delta_pos_b = (
                worst_contact.apply_position_change(worst_penetration)
            )

            for other in prepared_contacts:
                if other is worst_contact:
                    continue

                oc = other.contact
                om = other.manifold
                n = oc.world_normal

                if om.body_a is m.body_a:
                    displacement = delta_pos_a + np.cross(delta_rot_a, other.relative_a)
                    oc.penetration -= float(displacement @ n)
                elif om.body_a is m.body_b:
                    displacement = delta_pos_b + np.cross(delta_rot_b, other.relative_a)
                    oc.penetration -= float(displacement @ n)

                if om.body_b is m.body_a and other.relative_b is not None:
                    displacement = delta_pos_a + np.cross(delta_rot_a, other.relative_b)
                    oc.penetration += float(displacement @ n)
                elif om.body_b is m.body_b and other.relative_b is not None:
                    displacement = delta_pos_b + np.cross(delta_rot_b, other.relative_b)
                    oc.penetration += float(displacement @ n)

    def _resolve_velocities(self, prepared_list: list[PreparedContact], dt: float):
        WARM_FACTOR = 0.95

        for pc in prepared_list:
            c = pc.contact
            impulse_contact = np.array(
                [
                    c.normal_impulse * WARM_FACTOR,
                    c.tangent_impulse_1 * WARM_FACTOR,
                    c.tangent_impulse_2 * WARM_FACTOR,
                ]
            )
            c.normal_impulse *= WARM_FACTOR
            c.tangent_impulse_1 *= WARM_FACTOR
            c.tangent_impulse_2 *= WARM_FACTOR

            impulse_world = pc.contact_basis @ impulse_contact
            pc.apply_velocity_change(impulse_world)

        for _ in range(self.velocity_iterations):
            for pc in prepared_list:
                if pc.contact_basis is None:
                    raise ValueError("prepare contacts first")

                c = pc.contact
                m = pc.manifold
                r0, r1, r2 = pc.relative_a
                w0, w1, w2 = m.body_a.omega

                cv = [
                    (w1 * r2 - w2 * r1) + m.body_a.velocity[0],
                    (w2 * r0 - w0 * r2) + m.body_a.velocity[1],
                    (w0 * r1 - w1 * r0) + m.body_a.velocity[2],
                ]

                if m.body_b and pc.relative_b is not None:
                    r0, r1, r2 = pc.relative_b
                    w0, w1, w2 = m.body_b.omega

                    cv[0] -= (w1 * r2 - w2 * r1) + m.body_b.velocity[0]
                    cv[1] -= (w2 * r0 - w0 * r2) + m.body_b.velocity[1]
                    cv[2] -= (w0 * r1 - w1 * r0) + m.body_b.velocity[2]

                P = pc.contact_basis
                cv_n = P[0, 0] * cv[0] + P[1, 0] * cv[1] + P[2, 0] * cv[2]
                cv_1 = P[0, 1] * cv[0] + P[1, 1] * cv[1] + P[2, 1] * cv[2]
                cv_2 = P[0, 2] * cv[0] + P[1, 2] * cv[1] + P[2, 2] * cv[2]

                Y = pc.inverse_mass_matrix
                impulse_1 = -cv_1 / Y[1, 1]
                old_impulse = c.tangent_impulse_1
                max_friction = m.friction * c.normal_impulse
                c.tangent_impulse_1 = max(
                    -max_friction, min(max_friction, old_impulse + impulse_1)
                )
                impulse_1 = c.tangent_impulse_1 - old_impulse

                impulse_2 = -cv_2 / Y[2, 2]
                old_impulse = c.tangent_impulse_2
                c.tangent_impulse_2 = max(
                    -max_friction, min(max_friction, old_impulse + impulse_2)
                )
                impulse_2 = c.tangent_impulse_2 - old_impulse

                friction_contact = np.array([0.0, impulse_1, impulse_2])
                pc.apply_velocity_change(P @ friction_contact)

                friction_induced = Y[0, 1] * impulse_1 + Y[0, 2] * impulse_2
                cv_n += friction_induced

                bounce = 0.0
                if cv_n < -1.0:
                    bounce = -m.restitution * cv_n

                impulse_n = (-cv_n + bounce) / Y[0, 0]
                old_impulse = c.normal_impulse
                c.normal_impulse = max(0.0, old_impulse + impulse_n)
                impulse_n = c.normal_impulse - old_impulse

                normal_contact = np.array([impulse_n, 0.0, 0.0])
                pc.apply_velocity_change(P @ normal_contact)
