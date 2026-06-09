"""
wrecking_ball.py  –  Wrecking-ball demolition demo
===================================================
Physics features on display:
  • Spring force generator   – pendulum cable connecting anchor to ball
  • Gravity force generator  – acts on the ball + every box
  • Sphere ↔ Box collision   – ball smashes the tower
  • Box ↔ Box collision      – blocks cascade into each other
  • Box ↔ Plane collision    – debris bounces off the concrete floor
  • PhysicsMaterial mixing   – RUBBER ball / WOOD blocks / CONCRETE floor
  • Full rigid-body angular dynamics – blocks tumble realistically on impact
  • BSH broad-phase + Sutherland-Hodgman narrow-phase collision pipeline
"""

import math
import numpy as np
from ursina import color, Entity, Vec3, Mesh

from debug.debug_renderer import DebugRenderer
from physics.rigidbody import RigidBody
from physics.force_generators import Spring
from physics.material import Materials

# ── World / renderer ──────────────────────────────────────────────────────────
renderer = DebugRenderer.make(
    velocity_iters=20,
    position_iters=20,
    max_contacts=512,
    floor_y=0.0,
)
renderer.add_floor_plane(y=0.0)

# ── Anchor (ghost body — position reference only, never added to the world) ───
#
# Spring.update_force only reads _anchor_body.transform_matrix to locate the
# far end of the spring, so the body doesn't have to be registered in the
# World.  It will never be integrated or collision-tested; it simply stays put.
ANCHOR = np.array([0.0, 8.0, 0.0])
_anchor_body = RigidBody(float(ANCHOR[0]), float(ANCHOR[1]), float(ANCHOR[2]))
_anchor_body.inertia_tensor = np.eye(3)  # keeps transform_matrix valid

# Visual: overhead crane beam + pivot knob
Entity(
    model="cube", scale=Vec3(14, 0.18, 0.18), position=Vec3(0, 8.1, 0), color=color.gray
)
Entity(
    model="cube",
    scale=Vec3(0.28, 0.28, 0.28),
    position=Vec3(*ANCHOR),
    color=color.dark_gray,
)

# ── Wrecking ball (spring pendulum) ───────────────────────────────────────────
BALL_RADIUS = 0.65
NATURAL_LEN = 6.0
SPRING_K = 80.0

# Start the ball at exactly natural-length distance from the anchor,
# displaced 50° from vertical.  Spring PE = 0 at t = 0, so the swing
# is driven purely by gravitational potential energy.
theta = math.radians(50)
START = [
    ANCHOR[0] + NATURAL_LEN * math.sin(theta),  # ≈ +4.6
    ANCHOR[1] - NATURAL_LEN * math.cos(theta),  # ≈ +4.1
    0.0,
]

ball_body, ball_shape = renderer.add_sphere(
    position=START,
    radius=BALL_RADIUS,
    mass=5.0,
    body_color=color.orange,
)
ball_shape.material = Materials.RUBBER  # bouncy + high friction
ball_body.can_sleep = False  # pendulum swings forever; turning-point velocity must not trigger sleep

renderer._world.add_force_generators(
    ball_body,
    Spring(
        other=_anchor_body,
        my_connection_point=np.zeros(3),
        other_connection_point=np.zeros(3),
        spring_constant=SPRING_K,
        natural_length=NATURAL_LEN,
    ),
)


# Live cable visual — Ursina calls Entity.update() every frame automatically,
# so this stays in sync without touching the module-level update function.
class _CableEntity(Entity):
    def __init__(self):
        super().__init__(
            model=Mesh(
                vertices=[Vec3(*ANCHOR), Vec3(*ball_body.position)],
                mode="line",
            ),
            color=color.white,
        )

    def update(self):
        self.model.vertices = [Vec3(*ANCHOR), Vec3(*ball_body.position)]
        self.model.generate()


_CableEntity()

# ── Box tower ─────────────────────────────────────────────────────────────────
# Geometry: tower centre at x ≈ -3.8.  At that x the pendulum arc is at
# y ≈ 3.1, so the ball strikes between rows 2 and 3 – maximum chaos.
TOWER_X = -3.8
BOX_HS = 0.5  # half-size; full side = 1 m
ROWS, COLS = 5, 1
ROW_COLORS = [color.red, color.yellow, color.green, color.cyan]

for row in range(ROWS):
    for col in range(COLS):
        x = TOWER_X + (col - 1) * (BOX_HS * 2 + 0.04)
        y = BOX_HS + row * (BOX_HS * 2 + 0.04)
        _, shape = renderer.add_box(
            position=[x, y, 0.0],
            half_size=[BOX_HS, BOX_HS, BOX_HS],
            mass=1.0,
            body_color=ROW_COLORS[row % len(ROW_COLORS)],
        )
        shape.material = Materials.WOOD

# ── Gravity (applied to all bodies currently in renderer._records) ────────────
# The anchor is not in _records, so it is unaffected.
# Gravity.update_force short-circuits on inverse_mass == 0 regardless.
renderer.add_gravity()
# ── Go ────────────────────────────────────────────────────────────────────────
# 8 sub-steps → dt ≈ 0.002 s, well inside the spring stability limit
# (T/10 ≈ 0.14 s for k=80, m=4).
renderer.run(sub_steps=8)
