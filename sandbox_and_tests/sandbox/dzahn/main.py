# Need better integrator, return in future

import numpy as np
from ursina import Ursina, Entity, Vec3, color, time, Text
from ursina import EditorCamera

from physics.world import World
from physics.rigidbody import RigidBody

world = World()

body = RigidBody()
body.mass = 1.0

I1, I2, I3 = 1.0, 2.0, 4.0
body.inertia_tensor = np.diag([I1, I2, I3])

body.rotation = np.array([0.01, 1.0, 0.0])

world.add_rigid_body(body)

app = Ursina()
EditorCamera()

box = Entity(model="cube", scale=Vec3(3, 0.3, 1.5), color=color.cyan)

AXIS_LENGTH = 1.5
AXIS_THICKNESS = 0.06

axis_x = Entity(
    model="cube",
    color=color.red,
    scale=Vec3(AXIS_LENGTH, AXIS_THICKNESS, AXIS_THICKNESS),
    position=Vec3(AXIS_LENGTH / 2, 0, 0),
    parent=box,
)
axis_y = Entity(
    model="cube",
    color=color.green,
    scale=Vec3(AXIS_THICKNESS, AXIS_LENGTH, AXIS_THICKNESS),
    position=Vec3(0, AXIS_LENGTH / 2, 0),
    parent=box,
)
axis_z = Entity(
    model="cube",
    color=color.blue,
    scale=Vec3(AXIS_THICKNESS, AXIS_THICKNESS, AXIS_LENGTH),
    position=Vec3(0, 0, AXIS_LENGTH / 2),
    parent=box,
)

ang_vel_arrow = Entity(
    model="cube",
    color=color.yellow,
    scale=Vec3(0.08, 0.08, 0.08),  # updated every frame
)

GRID_SIZE = 10
GRID_STEP = 1
for i in range(-GRID_SIZE, GRID_SIZE + 1, GRID_STEP):
    Entity(
        model="cube",
        color=color.dark_gray,
        scale=Vec3(GRID_SIZE * 2, 0.01, 0.01),
        position=Vec3(0, -1.0, i),
    )
    Entity(
        model="cube",
        color=color.dark_gray,
        scale=Vec3(0.01, 0.01, GRID_SIZE * 2),
        position=Vec3(i, -1.0, 0),
    )

hud = Text(
    text="",
    position=(-0.85, 0.47),
    scale=0.7,
    color=color.white,
    background=True,
)


def rotation_matrix_to_euler_ursina(R: np.ndarray):
    """Convert a 3x3 rotation matrix to Ursina-compatible euler angles (degrees)."""
    from scipy.spatial.transform import Rotation

    r = Rotation.from_matrix(R)
    angles = r.as_euler("xyz", degrees=True)
    return Vec3(float(angles[0]), float(angles[1]), float(angles[2]))


def direction_to_euler(direction: np.ndarray):
    """Return Ursina euler angles so that the local +Z axis points along 'direction'."""
    from scipy.spatial.transform import Rotation

    direction = direction / (np.linalg.norm(direction) + 1e-9)
    z_axis = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z_axis, direction)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(z_axis, direction)

    if cross_norm < 1e-9:
        # parallel or anti-parallel
        if dot > 0:
            return Vec3(0, 0, 0)
        else:
            return Vec3(180, 0, 0)

    axis = cross / cross_norm
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    r = Rotation.from_rotvec(axis * angle)
    angles = r.as_euler("xyz", degrees=True)
    return Vec3(float(angles[0]), float(angles[1]), float(angles[2]))


def update():
    dt = time.dt  # type: ignore
    if dt <= 0:
        return

    SUB_STEPS = 10
    sub_dt = dt / SUB_STEPS
    for _ in range(SUB_STEPS):
        world.run_physics(sub_dt)

    R_mat = body.transform_matrix[:3, :3]
    box.rotation = rotation_matrix_to_euler_ursina(R_mat)
    box.position = Vec3(*body.position)
    omega = body.rotation  # world-space angular velocity vector
    omega_mag = np.linalg.norm(omega)
    if omega_mag > 1e-4:
        ang_vel_arrow.enabled = True
        ang_vel_arrow.position = box.position  # starts at body origin
        ang_vel_arrow.scale = Vec3(0.08, 0.08, max(omega_mag, 0.1))
        ang_vel_arrow.rotation = direction_to_euler(omega)
    else:
        ang_vel_arrow.enabled = False

    pos = body.position
    vel = body.velocity
    q = body.orientation
    euler = box.rotation

    hud.text = (
        f"[Position]  x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}\n"
        f"[Velocity]  x={vel[0]:+.3f}  y={vel[1]:+.3f}  z={vel[2]:+.3f}\n"
        f"[AngVel]    x={omega[0]:+.3f}  y={omega[1]:+.3f}  z={omega[2]:+.3f}  |w|={omega_mag:.3f}\n"
        f"[Quaternion]  w={q.w:+.3f}  x={q.x:+.3f}  y={q.y:+.3f}  z={q.z:+.3f}\n"
        f"[Euler(deg)]  x={euler[0]:+.1f}  y={euler[1]:+.1f}  z={euler[2]:+.1f}\n"
        f"\n"
        f"Axis arrows — Red:X  Green:Y  Blue:Z\n"
        f"Yellow arrow = angular velocity (world)"
    )


app.run()  # type: ignore
