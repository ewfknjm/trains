import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import numpy as np
from ursina import Ursina, Entity, Vec3, color, time
from ursina import EditorCamera

from physics.world import World
from physics.rigidbody import RigidBody

# ── Physics setup ──────────────────────────────────────────────────────────
world = World()

body = RigidBody()
body.mass = 1.0

# Asymmetric inertia tensor: I1 < I2 < I3 (three distinct principal moments)
# Rotation near the intermediate axis (I2) is unstable → Dzhanibekov effect
I1, I2, I3 = 1.0, 2.0, 4.0
body.inertia_tensor = np.diag([I1, I2, I3])

# Initial angular velocity slightly off the intermediate (Y) axis
body.rotation = np.array([0.05, 1.0, 0.0])

world.add_rigid_body(body)

# ── Ursina setup ──────────────────────────────────────────────────────────
app = Ursina()
EditorCamera()

# Represent the body as a flat box (like a book/racket) to make the flip visible
box = Entity(model="cube", scale=Vec3(3, 0.3, 1.5), color=color.cyan)

# A small red dot to mark one face so you can clearly see flips
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
    # Ursina uses XYZ extrinsic euler in degrees
    angles = r.as_euler("xyz", degrees=True)
    return Vec3(float(angles[0]), float(angles[1]), float(angles[2]))


def update():
    dt = time.dt  # type: ignore
    if dt <= 0:
        return

    world.run_physics(dt)

    # Sync orientation from physics to Ursina entity
    R_mat = body.transform_matrix[:3, :3]
    box.rotation = rotation_matrix_to_euler_ursina(R_mat)


app.run()  # type: ignore
