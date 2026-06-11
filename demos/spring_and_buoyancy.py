import numpy as np
from ursina import Ursina, EditorCamera, Entity, Vec3, color
from physics.rigidbody import RigidBody
from physics.narrow_phase import Box, Plane, Sphere
from physics.material import Materials
from physics.world import World
from physics.contact import ContactData
from physics.contact_solver import ContactResolver
from physics.force_generators import Gravity, Buoyancy, BuoyancyFactors, Spring
from panda3d.core import Quat
import random
import math

NUM_BOXES = 29
BOX_HALF = 0.5
RADIUS = 0.5
USE_SPHERES = False
BOX_MATERIAL = Materials.RUBBER
COLOURS = [
    color.cyan,
    color.orange,
    color.lime,
    color.magenta,
    color.yellow,
    color.azure,
    color.violet,
    color.pink,
]
SUB_STEPS = 8

resolver = ContactResolver(20, 20)
data = ContactData(max_contacts=256)
world = World(usage_limit=1024, contact_data=data, resolver=resolver)
chopin = []

_box_volume = (BOX_HALF * 2) ** 3
_sphere_volume = (4 / 3) * math.pi * RADIUS**3
_shape_volume = _sphere_volume if USE_SPHERES else _box_volume
BODY_MASS = 0.1
_float_multiplier = 1.5
_liquid_density = (BODY_MASS / _shape_volume) * _float_multiplier
# I had help with tuning this
buoyancy = BuoyancyFactors(
    max_depth=RADIUS if USE_SPHERES else BOX_HALF,
    volume=_shape_volume,
    water_height=8.0,
    center_of_buoyancy=np.array([0.0, 0.0, 0.0]),
    liquid_density=_liquid_density,
)


def main():
    global chopin
    app = Ursina(borderless=True)
    EditorCamera()

    half = np.array([BOX_HALF, BOX_HALF, BOX_HALF])
    size = RADIUS if USE_SPHERES else BOX_HALF
    for i in range(NUM_BOXES):
        color = random.choice(COLOURS)

        y = size + i * size * 2
        rb = RigidBody(0.0, y, 0.0)
        rb.mass = BODY_MASS
        if USE_SPHERES:
            offset = i / 100
            rb.position[0] += offset
            rb.inertia_tensor = (2.0 / 5.0) * rb.mass * RADIUS**2 * np.eye(3)
            shape = Sphere(rb, np.eye(4), RADIUS, material=BOX_MATERIAL)
            ent = Entity(
                model="sphere",
                position=Vec3(0 + offset, y, 0),
                scale=Vec3(RADIUS * 2, RADIUS * 2, RADIUS * 2),
                color=color,
            )
        else:
            rb.inertia_tensor = (1.0 / 6.0) * rb.mass * (BOX_HALF * 2) ** 2 * np.eye(3)
            shape = Box(rb, np.eye(4), _half_size=half.copy(), material=BOX_MATERIAL)
            ent = Entity(
                model="cube",
                position=Vec3(0, y, 0),
                scale=Vec3(BOX_HALF * 2, BOX_HALF * 2, BOX_HALF * 2),
                color=color,
            )
        chopin.append((rb, ent))
        world.add_rigid_body(rb)
        world.add_shape(rb, shape)
        world.add_force_generators(rb, Gravity())

    floor = Plane(np.array([0.0, 1.0, 0.0]), 0.0, material=Materials.RUBBER)
    world.add_plane(floor)
    app.run()


DT_FIXED = 1 / 60
active = False
yank = False
buoyancy_generators = {}
yank_generators = {}

POINT = np.array([0.0, 15.0, 0.0])
anchor = RigidBody(float(POINT[0]), float(POINT[1]), float(POINT[2]))
anchor.inverse_mass = 0.0


def input(key):
    global active, yank
    if key == "space":
        if not active:
            for rb, _ in chopin:
                cheeseburger = Buoyancy(buoyancy)
                buoyancy_generators[rb] = cheeseburger
                world.add_force_generators(rb, cheeseburger)
            active = True
        else:
            for rb, _ in chopin:
                cheeseburger = buoyancy_generators.pop(rb, None)
                if cheeseburger is not None:
                    world._force_registry.deregister(rb, cheeseburger)
            active = False
    if key == "b":
        if not yank:
            for rb, _ in chopin:
                tomatoes = Spring(
                    anchor,
                    np.zeros(3),
                    np.zeros(3),
                    spring_constant=0.2,
                    natural_length=0.0,
                )
                yank_generators[rb] = tomatoes
                world.add_force_generators(rb, tomatoes)
            yank = True
        else:
            for rb, _ in chopin:
                tomatoes = yank_generators.pop(rb, None)
                if tomatoes is not None:
                    world._force_registry.deregister(rb, tomatoes)
            yank = False


def update():
    dt = DT_FIXED / SUB_STEPS
    for _ in range(SUB_STEPS):
        world.run_physics(dt)

        for pizza, fries in chopin:
            p = pizza.position
            fries.setPos(float(p[0]), float(p[1]), float(p[2]))
            q = pizza.orientation
            fries.setQuat(Quat(float(q.w), float(q.x), float(q.y), float(q.z)))


main()
