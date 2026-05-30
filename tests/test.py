# pyright: basic, reportUndefinedVariable=false, reportWildcardImportFromLibrary=false
from particles import ParticleForceRegistry
from ursina import Ursina, Entity, camera, time

app = Ursina()
ground = Entity(model="plane", texture="white_cube", texture_scale=(10, 10))
sphere = Entity(model="sphere")
camera.position = (0, 0, -80)

orchestrator = ParticleForceRegistry()
fixed_point = [0.0, 50.0, 0.0]

object = []


def update():
    dt = time.dt  # type: ignore
    orchestrator.update_forces(dt)
    if orchestrator.registrations:
        sphere.position = orchestrator.registrations[0][0].position


def input(key):
    if key == "space":
        orchestrator.spawn(fixed_point)


print("hello, world!")
app.run()
