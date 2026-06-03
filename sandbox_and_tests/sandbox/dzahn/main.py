import numpy as np
from ursina import Ursina, Entity, Vec3, color, time
from ursina import EditorCamera

from physics.world import World
from physics.rigidbody import RigidBody
from physics.quaternions import Quaternion

world = World()

body = RigidBody()
body.mass = 1.0

I1, I2, I3 = 1.0, 2.0, 4.0
body.inertia_tensor = np.diag([I1, I2, I3])

body.rotation = np.array([0.0, 1.0, 0.0])
body.orientation = Quaternion(0.991, 0.0, 0.0, 0.131)

world.add_rigid_body(body)

app = Ursina()
EditorCamera()

box = Entity(model="cube", scale=Vec3(3, 0.3, 1.5), color=color.cyan)

marker = Entity(
    model="sphere",
    scale=Vec3(0.25, 0.25, 0.25),
    color=color.red,
    position=Vec3(1.4, 0.2, 0.6),
    parent=box,
)


def rotation_matrix_to_euler_ursina(R: np.ndarray):
    """Convert a 3x3 rotation matrix to Ursina-compatible euler angles (degrees)."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_matrix(R)
    angles = r.as_euler("xyz", degrees=True)
    return Vec3(float(angles[0]), float(angles[1]), float(angles[2]))


def update():
    dt = time.dt  # type: ignore
    if dt <= 0:
        return

    world.run_physics(dt)

    R_mat = body.transform_matrix[:3, :3]
    box.rotation = rotation_matrix_to_euler_ursina(R_mat)


app.run()  # type: ignore
