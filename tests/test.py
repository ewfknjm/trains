from ursina import *
from particles import *

app = Ursina()

ground = Entity(model = 'plane', scale = (100, 1, 100), texture = 'white_cube', texture_scale = (100, 100))
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
            scale = 0.5 if fw.type == 0 else (0.3 if fw.type == 1 else 0.1)
            visuals[fw] = Entity(model = 'sphere', scale = scale, color = color.random_color())

    for fw in list(visuals.keys()):
        if fw not in physics.active_fireworks:
            destroy(visuals[fw])
            del visuals[fw]
        else:
            visuals[fw].position = fw.position[:] 

def input(key):
    if key == 'space':
        while len(physics.active_fireworks) < 100:
            physics.spawn(0)

app.run()