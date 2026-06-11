import math
import numpy as np
from ursina import color, Entity, Vec3, Mesh

from debug.debug_renderer import DebugRenderer
from physics.rigidbody import RigidBody
from physics.force_generators import Spring
from physics.material import Materials

renderer = DebugRenderer.make(
    velocity_iters=20,
    position_iters=20,
    max_contacts=512,
    floor_y=0.0,
)
renderer.add_floor_plane(y=0.0)

ANCHOR = np.array([0.0, 8.0, 0.0])
_anchor_body = RigidBody(float(ANCHOR[0]), float(ANCHOR[1]), float(ANCHOR[2]))
_anchor_body.inertia_tensor = np.eye(3)

Entity(
    model="cube", scale=Vec3(14, 0.18, 0.18), position=Vec3(0, 8.1, 0), color=color.gray
)
Entity(
    model="cube",
    scale=Vec3(0.28, 0.28, 0.28),
    position=Vec3(*ANCHOR),
    color=color.dark_gray,
)

BALL_RADIUS = 0.65
NATURAL_LEN = 6.0
SPRING_K = 80.0

theta = math.radians(50)
START = [
    ANCHOR[0] + NATURAL_LEN * math.sin(theta),
    ANCHOR[1] - NATURAL_LEN * math.cos(theta),
    0.0,
]

ball_body, ball_shape = renderer.add_sphere(
    position=START,
    radius=BALL_RADIUS,
    mass=6.0,
    body_color=color.orange,
)
ball_shape.material = Materials.RUBBER
ball_body.can_sleep = False

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

TOWER_X = -3.8
BOX_HS = 0.5
ROWS, COLS = 7, 3
ROW_COLORS = [color.red, color.yellow, color.green, color.cyan]

for row in range(ROWS):
    for col in range(COLS):
        x = TOWER_X + (col - 1) * (BOX_HS * 2 + 0.04)
        y = BOX_HS + row * (BOX_HS * 2 + 0.04)
        _, shape = renderer.add_box(
            position=[x, y, 0.0],
            half_size=[BOX_HS, BOX_HS, BOX_HS],
            mass=0.1,
            body_color=ROW_COLORS[row % len(ROW_COLORS)],
        )
        shape.material = Materials.RUBBER


renderer.add_gravity()
renderer.run(sub_steps=8)
