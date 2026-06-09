from __future__ import annotations
from .broad_phase import BroadPhase, CandidatePair
from .rigidbody import RigidBody
from dataclasses import dataclass
import numpy as np


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
        self._body_to_leaf: dict[int, BSHNode] = {}

    def clear(self) -> None:
        self.root = None
        self._body_to_leaf.clear()

    def get_candidate_pairs(self) -> list[CandidatePair]:
        pairs: list[CandidatePair] = []
        if self.root is not None:
            usage = [self._usage_limit]
            self._get_all_overlaps(self.root, pairs, usage)
        return pairs

    def _get_all_overlaps(
        self,
        node: BSHNode,
        pairs: list[CandidatePair],
        usage: list[int],
    ) -> None:
        """Collect all overlapping pairs by checking each internal node's children
        against each other *and* recursing into each subtree for intra-subtree pairs."""
        if node.body is not None or node.left is None or node.right is None:
            return
        # Cross-pairs between the two child subtrees
        self._get_overlaps_with(node.left, node.right, pairs, usage)
        # Intra-subtree pairs on each side
        self._get_all_overlaps(node.left, pairs, usage)
        self._get_all_overlaps(node.right, pairs, usage)

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
        self._body_to_leaf[id(body)] = new_leaf

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
        if removed_leaf.body is not None:
            self._body_to_leaf.pop(id(removed_leaf.body), None)

        if not self.root:
            return
        if self.root is removed_leaf:
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

    def update_body(self, body: RigidBody, new_volume: BoundingSphere) -> None:
        leaf = self._body_to_leaf.get(id(body))
        if leaf is not None:
            self.remove(leaf)
        self.insert(body, new_volume)

    def contains(self, body: RigidBody) -> bool:
        return id(body) in self._body_to_leaf
