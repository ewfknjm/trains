import numpy as np
from .rigidbody import RigidBody


class EulerIntegrator:
    @staticmethod
    def integrate(body: RigidBody, dt: float) -> None:
        body.acceleration = body.force_accum * body.inverse_mass
        body.velocity += body.acceleration * dt
        body.velocity *= body.linear_damping**dt
        body.position += body.velocity * dt

        R = body.transform_matrix[:3, :3]
        angular_velocity_world = body.rotation
        angular_velocity_local = R.T @ angular_velocity_world

        I_local = body.inertia_tensor
        inverse_I_local = body.inverse_inertia_tensor

        angular_momentum_local = I_local @ angular_velocity_local
        gyroscopic_torque_local = np.cross(
            angular_velocity_local, angular_momentum_local
        )

        torque_local = R.T @ body.torque_accum
        angular_acceleration_local = inverse_I_local @ (
            torque_local - gyroscopic_torque_local
        )

        angular_acceleration_world = R @ angular_acceleration_local

        body.rotation += angular_acceleration_world * dt
        body.rotation *= body.angular_damping**dt
        body.orientation.add_scaled_vector(body.rotation, dt)

        body.mark_dirty()
