from __future__ import annotations
import numpy as np
from .rigidbody import RigidBody
from .BSH import BoundingSphere, BSHTree
from .broad_phase import CandidatePair
from dataclasses import dataclass, field


@dataclass
class BSPPlane:
    position: np.ndarray
    direction: np.ndarray


@dataclass
class BSPLeaf:
    active_bodies: list[RigidBody] = field(default_factory=list)
    active_volumes: list[BoundingSphere] = field(default_factory=list)


@dataclass
class BSPNode:
    plane: BSPPlane
    front: BSPNode | BSPLeaf
    back: BSPNode | BSPLeaf


class BSPTree:
    def __init__(self) -> None:
        self.root: BSPNode | BSPLeaf | None = None
        self._body_leaves: dict[int, list[BSPLeaf]] = {}

    def insert(self, body: RigidBody, volume: BoundingSphere) -> None:
        if self.root is None:
            raise RuntimeError("BSPTree must be built before insertion")

        nodes_to_process = [self.root]
        while nodes_to_process:
            current = nodes_to_process.pop()

            if isinstance(current, BSPLeaf):
                current.active_bodies.append(body)
                current.active_volumes.append(volume)
                self._body_leaves.setdefault(id(body), []).append(current)
                continue

            distance = (
                volume.center - current.plane.position
            ) @ current.plane.direction

            if distance > volume.radius:
                nodes_to_process.append(current.front)
            elif distance < -volume.radius:
                nodes_to_process.append(current.back)
            else:
                nodes_to_process.append(current.front)
                nodes_to_process.append(current.back)

    def remove(self, body: RigidBody) -> None:
        for leaf in self._body_leaves.pop(id(body), []):
            try:
                index = leaf.active_bodies.index(body)
                leaf.active_bodies.pop(index)
                leaf.active_volumes.pop(index)
            except ValueError:
                pass

    def build(self, planes: list[BSPPlane], max_depth: int) -> None:
        count = 0
        self.root = self._build_recursive(planes, len(planes), count, max_depth)

    def _build_recursive(
        self, planes: list[BSPPlane], plane_list_size: int, count: int, depth_left: int
    ) -> BSPLeaf | BSPNode:
        if count == plane_list_size or depth_left <= 0:
            return BSPLeaf()

        plane = planes[count]
        front_branch = self._build_recursive(
            planes, plane_list_size, count + 1, depth_left - 1
        )
        back_branch = self._build_recursive(
            planes, plane_list_size, count + 1, depth_left - 1
        )
        return BSPNode(plane, front_branch, back_branch)

    def clear_leaves(self) -> None:
        if self.root is None:
            return

        stack = [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, BSPLeaf):
                node.active_bodies.clear()
                node.active_volumes.clear()
            else:
                stack.append(node.front)
                stack.append(node.back)
        self._body_leaves.clear()

    def _collect_leaves(self) -> list[BSPLeaf]:
        if self.root is None:
            return []

        stack = [self.root]
        leaves: list[BSPLeaf] = []
        while stack:
            node = stack.pop()
            if isinstance(node, BSPLeaf):
                leaves.append(node)
                continue
            stack.append(node.front)
            stack.append(node.back)

        return leaves

    def get_candidate_pairs_hybrid(self, usage_per_leaf: int) -> list[CandidatePair]:
        if self.root is None:
            return []

        leaves = self._collect_leaves()
        all_pairs: list[CandidatePair] = []
        seen: set[tuple[int, int]] = set()

        for leaf in leaves:
            if len(leaf.active_bodies) < 2:
                continue
            bsh = BSHTree(usage_per_leaf)

            for body, volume in zip(leaf.active_bodies, leaf.active_volumes):
                bsh.insert(body, volume)

            local_pairs = bsh.get_candidate_pairs()
            for pair in local_pairs:
                a = id(pair.our_body)
                b = id(pair.other_body)
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                all_pairs.append(pair)

        return all_pairs
