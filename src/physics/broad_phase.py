from __future__ import annotations
import numpy as np
from typing import Protocol
from .rigidbody import RigidBody
from dataclasses import dataclass, field


@dataclass
class CandidatePair:
    our_body: RigidBody
    other_body: RigidBody


class BroadPhase(Protocol):
    def get_candidate_pairs(self) -> list[CandidatePair]: ...


class BoundingSphere:
    def __init__(self, radius: float, center: np.ndarray) -> None:
        self._radius: float = radius
        self._center: np.ndarray = center

    @property
    def center(self):
        return self._center

    @property
    def radius(self):
        return self._radius

    def overlap(self, other: BoundingSphere) -> bool:
        displacement = self._center - other._center
        distance = np.linalg.norm(displacement)
        radius_sum = self._radius + other._radius

        return bool(distance <= radius_sum)

    def final_radius_from(self, other: BoundingSphere) -> float:
        center_difference = np.linalg.norm(self._center - other._center)
        if abs(self._radius - other._radius) >= center_difference:
            return max(self._radius, other._radius)

        return float((self._radius + other._radius + center_difference) * 0.5)

    @staticmethod
    def create_bounding_sphere(
        our: BoundingSphere, other: BoundingSphere
    ) -> BoundingSphere:
        center_vector = other._center - our._center
        center_distance = np.linalg.norm(center_vector)
        radius_difference = abs(our._radius - other._radius)

        if center_distance < 1e-6:
            return BoundingSphere(max(our._radius, other._radius), our._center)

        if radius_difference >= center_distance:  # enclosed, r1 + d <= r2 rearrangement
            radius = max(our._radius, other._radius)
            center = our._center if our._radius > other._radius else other._center

            return BoundingSphere(radius, center)

        radius = (our._radius + other._radius + center_distance) * 0.5
        scale = (radius - our._radius) / center_distance
        center = our._center + scale * center_vector

        return BoundingSphere(float(radius), center)


@dataclass
class BSHNode:
    volume: BoundingSphere
    body: RigidBody | None = None
    left: BSHNode | None = None
    right: BSHNode | None = None
    parent: BSHNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class BSHTree(BroadPhase):
    def __init__(self, usage_limit: int):
        self.root: BSHNode | None = None
        self._usage_limit = usage_limit

    def get_candidate_pairs(self) -> list[CandidatePair]:
        pairs: list[CandidatePair] = []
        if self.root and self.root.left is not None and self.root.right is not None:
            usage = [self._usage_limit]
            self._get_overlaps_with(self.root.left, self.root.right, pairs, usage)

        return pairs

    def _get_overlaps_with(
        self,
        left: BSHNode,
        right: BSHNode,
        pairs: list[CandidatePair],
        usage: list[int],
    ):
        if not left.volume.overlap(right.volume):
            return
        if usage[0] <= 0:
            return

        usage[0] -= 1

        if left.body is not None and right.body is not None:
            pairs.append(CandidatePair(left.body, right.body))
            return

        if right.body is not None:
            if left.left is not None and left.right is not None:
                self._get_overlaps_with(left.left, right, pairs, usage)
                self._get_overlaps_with(left.right, right, pairs, usage)
            return

        if left.body is not None:
            if right.left is not None and right.right is not None:
                self._get_overlaps_with(right.left, left, pairs, usage)
                self._get_overlaps_with(right.right, left, pairs, usage)
            return

        if (
            left.left is not None
            and left.right is not None
            and right.left is not None
            and right.right is not None
        ):
            if left.volume.radius > right.volume.radius:
                self._get_overlaps_with(left.left, right, pairs, usage)
                self._get_overlaps_with(left.right, right, pairs, usage)
            else:
                self._get_overlaps_with(right.left, left, pairs, usage)
                self._get_overlaps_with(right.right, left, pairs, usage)

    def insert(self, body: RigidBody, volume: BoundingSphere):
        new_leaf = BSHNode(volume=volume, body=body)

        if self.root is None:
            self.root = new_leaf
            return

        current = self.root
        while current.body is None:
            if current.left is not None and current.right is not None:
                left_expansion = (
                    current.left.volume.final_radius_from(volume)
                    - current.left.volume.radius
                )
                right_expansion = (
                    current.right.volume.final_radius_from(volume)
                    - current.right.volume.radius
                )

                current = (
                    current.left if left_expansion < right_expansion else current.right
                )
            else:
                break

        old_leaf = current
        parent = old_leaf.parent

        new_volume = BoundingSphere.create_bounding_sphere(old_leaf.volume, volume)
        new_internal = BSHNode(volume=new_volume)

        new_internal.left = old_leaf
        new_internal.right = new_leaf
        old_leaf.parent = new_internal
        new_leaf.parent = new_internal
        new_internal.parent = parent

        if parent is not None:
            if parent.left is old_leaf:
                parent.left = new_internal
            else:
                parent.right = new_internal
        else:
            self.root = new_internal

        self._refit_spheres(new_internal.parent)

    def remove(self, removed_leaf: BSHNode):
        if not self.root:
            return
        if self.root == removed_leaf:
            self.root = None
            return

        parent = removed_leaf.parent
        if parent is None:
            return

        grandparent = parent.parent
        sibling = parent.right if parent.left is removed_leaf else parent.left

        if sibling is not None:
            sibling.parent = grandparent

        if grandparent is not None:
            if grandparent.left is parent:
                grandparent.left = sibling
            else:
                grandparent.right = sibling

            self._refit_spheres(grandparent)
        else:
            self.root = sibling

        removed_leaf.parent = None
        parent.left = None
        parent.right = None
        parent.parent = None

    def _refit_spheres(self, node: BSHNode | None):
        current = node
        while current is not None:
            if current.left is not None and current.right is not None:
                current.volume = BoundingSphere.create_bounding_sphere(
                    current.left.volume, current.right.volume
                )
            current = current.parent


@dataclass
class Plane:
    position: np.ndarray
    direction: np.ndarray


@dataclass
class BSPLeaf:
    active_bodies: list[RigidBody] = field(default_factory=list)
    active_volumes: list[BoundingSphere] = field(default_factory=list)


@dataclass
class BSPNode:
    plane: Plane
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

    def build(self, planes: list[Plane], max_depth: int) -> None:
        count = 0
        self.root = self._build_recursive(planes, len(planes), count, max_depth)

    def _build_recursive(
        self, planes: list[Plane], plane_list_size: int, count: int, depth_left: int
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
