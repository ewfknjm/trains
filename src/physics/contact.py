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


# -------------------- end of helper functions -------------------------------------- #


@dataclass(frozen=True)
class FeatureID:
    reference_face_index: int | None = None
    incident_vertex_index: int | None = None
    axis_one: int | None = None
    axis_two: int | None = None


@dataclass
class ClipVertex:
    world_position: np.ndarray
    feature_id: FeatureID


@dataclass
class Contact:
    local_point_a: np.ndarray
    local_point_b: Optional[np.ndarray]
    world_point: np.ndarray
    world_normal: np.ndarray
    penetration: float
    feature_id: FeatureID | None

    normal_impulse: float = 0.0
    tangent_impulse_1: float = 0.0
    tangent_impulse_2: float = 0.0


@dataclass
class ContactManifold:
    body_a: RigidBody
    body_b: Optional[RigidBody]
    restitution: float
    friction: float
    contacts: dict[FeatureID | None, Contact] = field(default_factory=dict)

    def add_contact(self, contact: Contact):
        if contact.feature_id in self.contacts:
            old_contact = self.contacts[contact.feature_id]
            contact.normal_impulse = old_contact.normal_impulse
            contact.tangent_impulse_1 = old_contact.tangent_impulse_1
            contact.tangent_impulse_2 = old_contact.tangent_impulse_2
        self.contacts[contact.feature_id] = contact


@dataclass
class ContactData:
    max_contacts: int
    manifolds: dict[tuple[int, int], ContactManifold] = field(default_factory=dict)

    @property
    def contact_count(self) -> int:
        return sum(len(m.contacts) for m in self.manifolds.values())

    @property
    def is_full(self) -> bool:
        return self.contact_count >= self.max_contacts

    def get_manifold(
        self,
        body_a: RigidBody,
        body_b: Optional[RigidBody],
        restitution: float,
        friction: float,
    ) -> ContactManifold:
        a_id = id(body_a)
        b_id = id(body_b) if body_b else 0
        pair = (a_id, b_id)
        if pair not in self.manifolds:
            self.manifolds[pair] = ContactManifold(
                body_a, body_b, restitution, friction
            )
        else:
            self.manifolds[pair].restitution = restitution
            self.manifolds[pair].friction = friction
        return self.manifolds[pair]

    def merge_from(self, old: "ContactData") -> None:
        for pair_key, manifold in self.manifolds.items():
            old_manifold = old.manifolds.get(pair_key)
            if old_manifold is None:
                continue
            for fid, contact in manifold.contacts.items():
                old_contact = old_manifold.contacts.get(fid)
                if old_contact is not None:
                    contact.normal_impulse = old_contact.normal_impulse
                    contact.tangent_impulse_1 = old_contact.tangent_impulse_1
                    contact.tangent_impulse_2 = old_contact.tangent_impulse_2

    def clear(self):
        self.manifolds.clear()
