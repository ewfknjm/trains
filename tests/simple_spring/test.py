# pyright: basic, reportUndefinedVariable=false, reportWildcardImportFromLibrary=false
import numpy as np
from particles import ParticleForceRegistry, Spring, Gravity, Drag, Particle
from ursina import Ursina, Entity, camera, time, Vec3

app = Ursina()
ground = Entity(model="plane", texture="white_cube", texture_scale=(10, 10))
sphere = Entity(model="sphere", position=Vec3(0.0, 0.0, 0.0))
camera.position = (0, 0, -100)

orchestrator = ParticleForceRegistry()
fixed_point = [0.0, 0.0, 0.0]

object = []


def spawn_bouncy_sphere():
    p = Particle(0.0, 0.0, 0.0)
    p.set_mass(1.0)
    orchestrator.add(p, Gravity())
    orchestrator.add(p, Spring(np.array(fixed_point), k=2.0, a=4.0))
    orchestrator.add(p, Drag())
    return p


physics_sphere = None


def input(key):
    global physics_sphere
    if key == "space":
        physics_sphere = spawn_bouncy_sphere()


def update():
    dt = time.dt  # type: ignore
    orchestrator.update_forces(dt)

    if physics_sphere:
        sphere.position = physics_sphere.position


app.run()  # type: ignore
