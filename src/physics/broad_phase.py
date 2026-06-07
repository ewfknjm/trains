from __future__ import annotations
from typing import Protocol
from .rigidbody import RigidBody
from dataclasses import dataclass


@dataclass
class CandidatePair:
    our_body: RigidBody
    other_body: RigidBody


class BroadPhase(Protocol):
    def get_candidate_pairs(self) -> list[CandidatePair]: ...
