import numpy as np
from .rigidbody import RigidBody

SLEEP_BIAS = 0.1


class EulerIntegrator:
    @staticmethod
    def integrate(body: RigidBody, dt: float) -> None:
        if body.is_sleeping:
            return

        body.last_frame_acceleration = body.acceleration.copy()
        body.acceleration = body.force_accum * body.inverse_mass
        body.velocity += body.acceleration * dt
        body.velocity *= body.linear_damping**dt
        body.position += body.velocity * dt

        R = body.transform_matrix[:3, :3]
        angular_velocity_world = body.omega
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

        body.omega += angular_acceleration_world * dt
        body.omega *= body.angular_damping**dt
        body.orientation.add_scaled_vector(body.omega, dt)

        body.mark_dirty()

        if body.can_sleep:
            current_motion = float(
                np.dot(body.velocity, body.velocity) + np.dot(body.omega, body.omega)
            )
            bias = SLEEP_BIAS**dt
            body.motion = bias * body.motion + (1.0 - bias) * current_motion

            if body.motion < body.sleep_motion_threshold:
                body.set_sleeping()
