import numpy as np
from ursina import Ursina, Entity, Vec3, color, time, Mesh
from ursina import EditorCamera
from particles import Particle
from particles_force import Gravity
from particle_contact import CableContactGenerator, RodContactGenerator
from particle_world import ParticleWorld

# ── Scene setup ───────────────────────────────────────────────────────────────
world = ParticleWorld()

p0 = Particle(0.0, 0.0, 0.0)  # anchor
p1 = Particle(0.0, -3.0, 0.0)
p2 = Particle(0.0, -5.0, 0.0)
p3 = Particle(0.0, -8.0, 0.0)

p0.inverse_mass = 0.0
for p in (p1, p2, p3):
    p.mass = 1.0

for p in (p0, p1, p2, p3):
    world.add_particle(p)

gravity = Gravity(np.array([0.0, -9.81, 0.0]))
for p in (p1, p2, p3):
    world.force_registry.add(p, gravity)

world.add_contact(CableContactGenerator(p0, p1, restitution=0.4, max_length=3.0))
world.add_contact(RodContactGenerator(p1, p2, max_length=3.5))
world.add_contact(CableContactGenerator(p2, p3, restitution=0.4, max_length=3.0))

# ── Ursina ────────────────────────────────────────────────────────────────────
app = Ursina()
EditorCamera()

# One sphere per particle
PARTICLES = [p0, p1, p2, p3]
COLORS = [color.red, color.cyan, color.yellow, color.green]

spheres = [
    Entity(model="sphere", scale=Vec3(2, 2, 2), color=c, position=Vec3(*p.position))
    for p, c in zip(PARTICLES, COLORS)
]

# Lines connecting neighbours: p0-p1, p1-p2, p2-p3
connections = [(0, 1), (1, 2), (2, 3)]
line_meshes = [
    Mesh(
        vertices=[Vec3(*PARTICLES[a].position), Vec3(*PARTICLES[b].position)],
        mode="line",
    )
    for a, b in connections
]
lines = [Entity(model=mesh, color=color.white) for mesh in line_meshes]


def update():
    dt = time.dt  # type: ignore
    world.run_physics(dt)

    for sphere, p in zip(spheres, PARTICLES):
        sphere.position = Vec3(*p.position)

    for mesh, (a, b) in zip(line_meshes, connections):
        mesh.vertices = [Vec3(*PARTICLES[a].position), Vec3(*PARTICLES[b].position)]
        mesh.generate()


app.run()  # type: ignore
