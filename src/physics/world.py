from .contact import ContactData
from .contact_solver import ContactResolver
from .force_generators import ForceGenerator
from .force_registry import ForceRegistry
from .rigidbody import RigidBody
from .integrator import EulerIntegrator


class World:
    def __init__(self):
        self._rb_registrations: list[RigidBody] = []
        self._force_registry: ForceRegistry = ForceRegistry()
        self._integrator: EulerIntegrator = EulerIntegrator()
        self._contact_data: ContactData
        self._resolver: ContactResolver

    def add_rigid_body(self, body: RigidBody):
        if body in self._rb_registrations:
            raise ValueError("RigidBody is already registered")
        self._rb_registrations.append(body)

    def add_force_generators(self, body: RigidBody, force_generator: ForceGenerator):
        self._force_registry.register_force_generator(body, force_generator)

    def start_frame(self):
        for rigid_body in self._rb_registrations:
            rigid_body.clear_accum()

    def integrate(self, dt: float) -> None:
        for rigid_body in self._rb_registrations:
            self._integrator.integrate(rigid_body, dt)

    def run_physics(self, dt: float):
        self.start_frame()
        self._force_registry.update_force(dt)
        self.integrate(dt)
        self._contact_data.clear()
        self._resolver.resolve(self._contact_data)
