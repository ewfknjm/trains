from rb_force_gen import ForceGenerator
from rb_force_reg import ForceRegistry
from rigidbody import RigidBody


class World:
    def __init__(self):
        self._rb_registrations: list[RigidBody] = []
        self._force_registry: ForceRegistry = ForceRegistry()

    def add_rigid_body(self, body: RigidBody):
        if body in self._rb_registrations:
            raise ValueError("RigidBody is already registered")
        self._rb_registrations.append(body)

    def add_force_generators(self, body: RigidBody, force_generator: ForceGenerator):
        self._force_registry.register_force_generator(body, force_generator)

    def start_frame(self):
        for rigidbody in self._rb_registrations:
            rigidbody.clear_accum()

    def integrate(self, dt: float):
        for rigidbody in self._rb_registrations:
            rigidbody.integrate(dt)

    def run_physics(self, dt: float):
        self.start_frame()
        self._force_registry.update_force(dt)
        self.integrate(dt)
