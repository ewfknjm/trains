import numpy as np
import random


class Payload:
    def __init__(self, firework_type: int, count: int):
        self.type = firework_type
        self.count = count


class Particle:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, -9.81, 0.0], dtype=float)
        self.damping = 0.98
        self.inverse_mass = 1.0

    def integrate(self, dt: float):
        if self.inverse_mass <= 0:
            return

        self.position += self.velocity * dt
        self.velocity += self.acceleration * dt
        self.velocity *= self.damping**dt


class Firework(Particle):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super().__init__(x, y, z)
        self.age = 0.0
        self.type = 0

    def update(self, dt: float) -> bool:
        super().integrate(dt)
        self.age -= dt
        return self.age <= 0.0


class FireworkRule:
    def __init__(
        self,
        rule_type: int,
        min_age: float,
        max_age: float,
        min_velocity: np.ndarray,
        max_velocity: np.ndarray,
        payloads: Payload,
        damping: float,
    ):
        self.type = rule_type
        self.min_age = min_age
        self.max_age = max_age
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.payloads = payloads
        self.damping = damping

    def create(self, parent_firework: Firework = None) -> Firework:
        new_fw = Firework()
        new_fw.type = self.type

        new_fw.age = random.uniform(self.min_age, self.max_age)

        if parent_firework is not None:
            new_fw.position = parent_firework.position.copy()
        else:
            new_fw.position = np.array([0.0, 0.0, 0.0], dtype=float)

        velocityX = random.uniform(self.min_velocity[0], self.max_velocity[0])
        velocityY = random.uniform(self.min_velocity[1], self.max_velocity[1])
        velocityZ = random.uniform(self.min_velocity[2], self.max_velocity[2])
        new_fw.velocity = np.array([velocityX, velocityY, velocityZ], dtype=float)

        return new_fw


class FireworkSystem:
    def __init__(self):
        self.rules = {}
        self.active_fireworks = []

    def init_rules(self):
        self.rules[0] = FireworkRule(
            rule_type=0,
            min_age=0.5,
            max_age=0.9,
            min_velocity=np.array([-5, 25, -5], dtype=float),
            max_velocity=np.array([5, 28, 5], dtype=float),
            payloads=Payload(1, 2),
            damping=0.6,
        )
        self.rules[1] = FireworkRule(
            rule_type=1,
            min_age=1.0,
            max_age=1.5,
            min_velocity=np.array([-2, 15, -2], dtype=float),
            max_velocity=np.array([3, 12, 3], dtype=float),
            payloads=Payload(2, 4),
            damping=0.8,
        )
        self.rules[2] = FireworkRule(
            rule_type=2,
            min_age=0.5,
            max_age=1.0,
            min_velocity=np.array([-10, -5, -10], dtype=float),
            max_velocity=np.array([2, 3, 5], dtype=float),
            payloads=Payload(3, 6),
            damping=0.3,
        )

    def spawn(self, firework_type: int, parent: Firework):
        if firework_type in self.rules:
            new_firework = self.rules[firework_type].create(parent)
            self.active_fireworks.append(new_firework)

    def update_system(self, dt: float):
        for fw in reversed(self.active_fireworks):
            fw_died = fw.update(dt)
            if fw_died:
                rule = self.rules.get(fw.type)
                if rule is not None:
                    payload = rule.payloads
                    for _ in range(payload.count):
                        self.spawn(payload.type, fw)
                self.active_fireworks.remove(fw)

