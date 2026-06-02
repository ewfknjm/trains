import numpy as np
from force_generators import ForceGenerator
from force_registry import ForceRegistry
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
            rigidbody.acceleration = rigidbody.force_accum * rigidbody.inverse_mass

            rigidbody.velocity += rigidbody.acceleration * dt
            rigidbody.velocity *= rigidbody.linear_damping**dt

            rigidbody.position += rigidbody.velocity * dt

            I_world = rigidbody._inverse_inertia_tensor_world
            gyroscopic = np.cross(
                rigidbody.rotation, np.linalg.solve(I_world, rigidbody.rotation)
            )
            angular_accel = I_world @ (rigidbody.torque_accum - gyroscopic)

            rigidbody.rotation += angular_accel * dt
            rigidbody.rotation *= rigidbody.linear_damping**dt

            rigidbody._orientation.add_scaled_vector(rigidbody.rotation, dt)
            rigidbody._dirty = True

            rigidbody.clear_accum()

    def run_physics(self, dt: float):
        self.start_frame()
        self._force_registry.update_force(dt)
        self.integrate(dt)
