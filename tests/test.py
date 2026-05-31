from typing import Optional
import numpy as np
from particles import Particle, ParticleForceRegistry, Attractor, WanderForce, Repulsion
from ursina import Ursina, Entity, camera, color, time, destroy, Vec3

app = Ursina()

physics = ParticleForceRegistry()
camera.position = (0, 0, -80)

bullet_model: Optional[Entity] = None
target_model: Optional[Entity] = None
bullet: Optional[Particle] = None
target: Optional[Particle] = None
attractor_switch = False


def spawn_bullet_target():
    global bullet_model, target_model, attractor_switch, bullet, target
    bullet = Particle(-10.0, 0.0, 0.0)  # add position for bullet
    target = Particle(10.0, 0.0, 0.0)  # add position for target
    bullet_model = Entity(model="sphere")
    target_model = Entity(model="sphere", color=color.red)

    bullet.set_mass(1.0)
    target.set_mass(1.0)
    bullet.velocity = np.array([5.0, 0.0, 0.0], dtype="float")
    target.velocity = np.array([0.0, 0.0, 0.0], dtype="float")

    attractor_switch = True


def remove_bullet_target():
    physics.clear()

    if bullet_model is not None:
        destroy(bullet_model)
    if target_model is not None:
        destroy(target_model)


def input(key):
    if key == "space":
        remove_bullet_target()
        spawn_bullet_target()


time_buffer = 1.5


def update():
    global time_buffer, bullet, target, bullet_model, target_model

    if (
        time_buffer is None
        or bullet is None
        or bullet_model is None
        or target is None
        or target_model is None
    ):
        return

    physics.clear()
    dt = time.dt  # type: ignore
    time_skip = dt + time_buffer

    if attractor_switch:
        physics.add(bullet, Attractor(target, 3.5, 0.2, time_skip))
        physics.add(target, WanderForce())
        physics.add(target, Repulsion(bullet.position, 5.0, 2.0, 6.0))

    physics.update_forces(dt)

    target_model.position = Vec3(*target.position)
    bullet_model.position = Vec3(*bullet.position)


app.run()  # type: ignore
