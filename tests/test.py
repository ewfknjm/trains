import numpy as np
from particles import (
    Particle,
    ParticleContactResolver,
    SphereContactGenerator,
    CableContactGenerator,
)
from ursina import Ursina, Entity, Vec3, camera, color, time

app = Ursina()
camera.position = (0, 0, -80)

p1 = Particle(0, 0, 0)
p2 = Particle(5, 0, 0)
p1.mass = 1.0
p2.mass = 1.5

contact_generator = SphereContactGenerator(p1, p2, 1.0, 1.0, restitution=0.8)
cable_generator = CableContactGenerator(p1, p2, 0.9, 3)
resolver = ParticleContactResolver(iterations=10)

s1 = Entity(
    model="sphere", color=color.red, scale=Vec3(2, 2, 2), position=Vec3(*p1.position)
)
s2 = Entity(
    model="sphere", color=color.blue, scale=Vec3(2, 2, 2), position=Vec3(*p2.position)
)


def input(key):  # type: ignore
    if key == "space":
        p1.velocity = np.array([1.0, 0.0, 0.0], dtype="float")
        p2.velocity = np.array([-1.5, 0.0, 0.0], dtype="float")


def update():  # type: ignore
    dt = time.dt  # type: ignore

    contact = contact_generator.add_contact()
    cable = cable_generator.add_contact()
    contacts = [contact] if contact is not None else []
    if cable is not None:
        contacts.append(cable)

    resolver.resolve_contacts(contacts, dt)
    p1.integrate(dt)
    p2.integrate(dt)

    s1.position = Vec3(*p1.position)
    s2.position = Vec3(*p2.position)


app.run()  # type: ignore

app = Ursina()
camera.position = (0, 0, -80)

p1 = Particle(0, 0, 0)
p2 = Particle(5, 0, 0)
p1.mass = 1.0
p2.mass = 1.0

generator = SphereContactGenerator(p1, p2, 1.0, 1.0, restitution=0.8)
resolver = ParticleContactResolver(iterations=10)

s1 = Entity(model="sphere", color=color.red, position=Vec3(*p1.position))
s2 = Entity(model="sphere", color=color.blue, position=Vec3(*p2.position))


def input(key):
    if key == "space":
        p1.velocity = np.array([1.0, 0.0, 0.0])
        p2.velocity = np.array([-1.0, 0.0, 0.1])


def update():
    dt = time.dt  # type: ignore

    contact = generator.add_contact()
    contacts = [contact] if contact is not None else []
    resolver.resolve_contacts(contacts, dt)
    p1.integrate(dt)
    p2.integrate(dt)

    s1.position = Vec3(*p1.position)
    s2.position = Vec3(*p2.position)


app.run()  # type: ignore

app = Ursina()
camera.position = (0, 0, -80)

p1 = Particle(0, 0, 0)
p2 = Particle(5, 0, 0)
p1.mass = 1.0
p2.mass = 1.0

generator = SphereContactGenerator(p1, p2, 1.0, 1.0, restitution=0.8)
resolver = ParticleContactResolver(iterations=10)

s1 = Entity(model="sphere", color=color.red, position=Vec3(*p1.position))
s2 = Entity(model="sphere", color=color.blue, position=Vec3(*p2.position))


def input(key):
    if key == "space":
        p1.velocity = np.array([1.0, 0.0, 0.0])
        p2.velocity = np.array([-1.0, 0.0, 0.1])


def update():
    dt = time.dt  # type: ignore

    contact = generator.add_contact()
    contacts = [contact] if contact is not None else []
    resolver.resolve_contacts(contacts, dt)
    p1.integrate(dt)
    p2.integrate(dt)

    s1.position = Vec3(*p1.position)
    s2.position = Vec3(*p2.position)


app.run()  # type: ignore
