from ursina import Ursina, Entity, camera, time, color, destroy
from tests.fireworks.particles import FireworkSystem

app = Ursina()

ground = Entity(model="plane", texture="white_cube", texture_scale=(100, 100))
camera.position = (0, 5, -90)
camera.rotation_x = -10

physics = FireworkSystem()
physics.init_rules()

visuals = {}


def update():
    dt = time.dt
    physics.update_system(dt)

    for fw in physics.active_fireworks:
        if fw not in visuals:
            visuals[fw] = Entity(model="sphere", color=color.random_color())

    for fw in list(visuals.keys()):
        if fw not in physics.active_fireworks:
            destroy(visuals[fw])
            del visuals[fw]
        else:
            visuals[fw].position = fw.position[:]


def input(key):
    if key == "space":
        while len(physics.active_fireworks) < 100:
            physics.spawn(0)


app.run()
