import numpy as np
from ursina import invoke
from debug.debug_renderer import DebugRenderer

renderer = DebugRenderer.make(floor_y=0.0)
renderer.add_floor_plane(y=0.0)

# stack of boxes — no decisions to make about planes, cameras, or HUDs
for i in range(2):
    renderer.add_box(
        position=[0, 1.1 + (i * 5) * 1.1, 0],
        half_size=[0.5, 0.5, 0.5],
    )

renderer.add_gravity()
renderer.run(sub_steps=4)
