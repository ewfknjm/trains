from rigidbody import RigidBody
from force_generators import ForceGenerator
from dataclasses import dataclass


@dataclass
class ForceRegistrant:
    body: RigidBody
    force_generator: ForceGenerator


class ForceRegistry:
    def __init__(self):
        self.registrations = []

    def register_force_generator(
        self, body: RigidBody, force_generator: ForceGenerator
    ):
        if force_generator is not None:
            self.registrations.append(ForceRegistrant(body, force_generator))

    def clear_registry(self):
        self.registrations.clear()

    def update_force(self, dt: float):
        for registrant in self.registrations:
            registrant.force_generator.update_force(registrant.body, dt)
