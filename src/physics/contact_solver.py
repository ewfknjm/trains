import numpy as np
from typing import Optional
from .contact import Contact
from .rigidbody import RigidBody
import math

# replace rotation with omega, use github to check


# probably not suitable class
class ContactResolver:
    def __init__(self, iterations: int):
        self.iterations = iterations

    @staticmethod
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

    @staticmethod
    def angular_delta_velocity(
        body: RigidBody, relative_position: np.ndarray, contact_normal: np.ndarray
    ) -> float:
        rxn = np.cross(relative_position, contact_normal)
        delta_omega = body.inverse_inertia_tensor_world @ rxn
        return float(delta_omega @ rxn)

    @staticmethod
    def calculate_velocity_change_per_unit_impulse(
        body_a: RigidBody, body_b: Optional[RigidBody], contact_info: Contact
    ) -> float:
        contact_point = contact_info.contact_point
        contact_normal = contact_info.contact_normal
        a_relative_contact_position = contact_point - body_a.position
        # only along contact normal, assumed no friction
        delta_velocity = body_a.inverse_mass
        delta_velocity += ContactResolver.angular_delta_velocity(
            body_a, a_relative_contact_position, contact_normal
        )

        if not body_b:
            return delta_velocity

        b_relative_contact_position = contact_point - body_b.position
        delta_velocity += body_b.inverse_mass + ContactResolver.angular_delta_velocity(
            body_b, b_relative_contact_position, contact_normal
        )

        return delta_velocity

    @staticmethod
    def calculate_closing_velocity_world(
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

    @staticmethod
    def calculate_closing_velocity_contact(
        closing_velocity_world: np.ndarray, contact_info: Contact
    ) -> float:
        # transform = ContactResolver.get_contact_basis(contact_info.contact_normal)
        # contact_velocity = transform.T @ closing_velocity_world
        # return contact_velocity[0]
        return float(closing_velocity_world @ contact_info.contact_normal)

    @staticmethod
    def calculate_desired_velocity_change(
        restitution: float, closing_contact_velocity: float
    ):
        return -closing_contact_velocity * (1 + restitution)
