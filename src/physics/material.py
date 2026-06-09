from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsMaterial:
    restitution: float
    friction: float

    def mix(self, other: PhysicsMaterial) -> tuple[float, float]:
        restitution = min(self.restitution, other.restitution)
        friction = math.sqrt(self.friction * other.friction)
        return restitution, friction


class Materials:
    DEFAULT = PhysicsMaterial(restitution=0.4, friction=0.5)
    STEEL = PhysicsMaterial(restitution=0.2, friction=0.3)
    RUBBER = PhysicsMaterial(restitution=0.7, friction=0.8)
    WOOD = PhysicsMaterial(restitution=0.3, friction=0.5)
    CONCRETE = PhysicsMaterial(restitution=0.1, friction=0.7)
