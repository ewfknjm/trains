from typing import Optional
import numpy as np
from ursina import Ursina, Entity, camera, Vec3, distance, time, destroy
from particles import Particle, ParticleForceRegistry, Repulsion, Drag

app = Ursina()

ground = Entity(model="plane", texture="white_cube", texture_scale=(100, 100))
camera.position = (0, 0, -80)

physics = ParticleForceRegistry()
a: Optional[Particle] = None
b: Optional[Particle] = None
sphere_a: Optional[Entity] = None
sphere_b: Optional[Entity] = None


def clear_ball():
    global sphere_a, sphere_b
    if sphere_a is not None:
        destroy(sphere_a)
    if sphere_b is not None:
        destroy(sphere_b)
    physics.clear()


def spawn_balls():
    global sphere_a, sphere_b
    global a, b
    sphere_a = Entity(model="sphere", scale=Vec3(2, 2, 2), position=Vec3(-10, 0, 0))
    sphere_b = Entity(model="sphere", scale=Vec3(2, 2, 2), position=Vec3(10, 0, 0))
    a = Particle(-10.0, 0.0, 0.0)
    b = Particle(10.0, 0.0, 0.0)
    a.set_mass(1.0)
    b.set_mass(1.0)
    a.velocity[0] = 10.0
    b.velocity[0] = -10.0


def input(key):
    if key == "space":
        global a, b
        clear_ball()
        spawn_balls()
        if a is not None and b is not None:
            physics.add(a, Drag())
            physics.add(b, Drag())


outer_radius = 10.0
inner_radius = 1.0
max_repulsion = 100.0


def update():
    global a, b
    if a is None or b is None:
        return
    physics.clear()

    repulsive_a = Repulsion(b.position, max_repulsion, inner_radius, outer_radius)
    repulsive_b = Repulsion(a.position, max_repulsion, inner_radius, outer_radius)

    physics.add(a, repulsive_a)
    physics.add(b, repulsive_b)
    physics.update_forces(time.dt)  # type: ignore

    global sphere_a, sphere_b
    if sphere_a is None or sphere_b is None:
        return
    sphere_a.position = Vec3(*a.position)
    sphere_b.position = Vec3(*b.position)


app.run()  # type: ignore
