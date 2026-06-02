import numpy as np
from .rigidbody import RigidBody


class EulerIntegrator:
    @staticmethod
    def integrate(body: RigidBody, dt: float) -> None:
        body.acceleration = body.force_accum * body.inverse_mass
        body.velocity += body.acceleration * dt
        body.velocity *= body.linear_damping**dt
        body.position += body.velocity * dt

        inv_I_world = body.inverse_inertia_tensor_world
        I_world = body.inertia_tensor_world
        gyroscopic = np.cross(body.rotation, I_world @ body.rotation)
        angular_accel = inv_I_world @ (body.torque_accum - gyroscopic)

        body.rotation += angular_accel * dt
        body.rotation *= body.angular_damping**dt
        body.orientation.add_scaled_vector(body.rotation, dt)

        body.mark_dirty()
