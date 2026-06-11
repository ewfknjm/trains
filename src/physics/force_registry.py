from .rigidbody import RigidBody
from .force_generators import ForceGenerator
from dataclasses import dataclass
from typing import NamedTuple


@dataclass
class ForceRegistrant:
    body: RigidBody
    force_generator: ForceGenerator


class _RegistrationKey(NamedTuple):
    body_id: int
    generator_id: int


class ForceRegistry:
    def __init__(self):
        self._registrations: list[ForceRegistrant] = []
        self._key_set: set[_RegistrationKey] = set() # AI project-wide audit revealed possibility of duplicated pairs

    def _key(self, body: RigidBody, generator: ForceGenerator) -> _RegistrationKey:
        return _RegistrationKey(id(body), id(generator))

    def register_force_generator(
        self, body: RigidBody, force_generator: ForceGenerator
    ):
        key = self._key(body, force_generator)
        if key in self._key_set:
            raise ValueError("force generator is already registered")

        self._registrations.append(ForceRegistrant(body, force_generator))
        self._key_set.add(key)

    def deregister(self, body: RigidBody, force_generator: ForceGenerator) -> None:
        key = self._key(body, force_generator)
        if key not in self._key_set:
            raise ValueError("force generator is not registered")

        self._registrations = [
            r
            for r in self._registrations
            if not (r.body is body and r.force_generator is force_generator)
        ]
        self._key_set.discard(key)

    def deregister_all_generators(self, body: RigidBody) -> None:
        to_remove = [
            self._key(body, r.force_generator)
            for r in self._registrations
            if r.body is body
        ]
        self._registrations = [r for r in self._registrations if r.body is not body]

        for key in to_remove:
            self._key_set.discard(key)

    def clear_registry(self):
        self._registrations.clear()
        self._key_set.clear()

    def update_force(self, dt: float):
        for registrant in self._registrations:
            registrant.force_generator.update_force(registrant.body, dt)
