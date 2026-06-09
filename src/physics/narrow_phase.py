from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
import itertools
from typing import Optional

from .material import PhysicsMaterial, Materials
from physics.BSH import BoundingSphere
from .rigidbody import RigidBody
from .contact import ContactData, Contact


class Shape(ABC):
    @abstractmethod
    def collide_with(
        self, other: Shape, data: ContactData, restitution: float, friction: float
    ) -> None: ...

    @abstractmethod
    def collide_with_sphere(
        self, sphere: Sphere, data: ContactData, restitution: float, friction: float
    ) -> None: ...

    @abstractmethod
    def collide_with_box(
        self, box: Box, data: ContactData, restitution: float, friction: float
    ) -> None: ...

    @abstractmethod
    def collide_with_plane(
        self, plane: Plane, data: ContactData, restitution: float, friction: float
    ) -> None: ...


@dataclass
class Primitive(Shape):
    body: RigidBody
    offset_matrix: np.ndarray
    _: KW_ONLY
    material: PhysicsMaterial = field(default_factory=lambda: Materials.DEFAULT)

    def get_transform(self) -> np.ndarray:
        return self.body.transform_matrix @ self.offset_matrix

    def get_axis(self, index: int) -> np.ndarray:
        return self.get_transform()[:3, index]

    @abstractmethod
    def bounding_sphere(self) -> BoundingSphere: ...


@dataclass
class Sphere(Primitive):
    radius: float

    def collide_with(
        self, other: Shape, data: ContactData, restitution: float, friction: float
    ) -> None:
        other.collide_with_sphere(self, data, restitution, friction)

    def collide_with_sphere(
        self, sphere: Sphere, data: ContactData, restitution: float, friction: float
    ) -> None:
        if data.is_full:
            return

        position_one = self.get_axis(3)
        position_two = sphere.get_axis(3)

        vector = position_one - position_two
        magnitude = np.linalg.norm(vector)
        radii_sum = self.radius + sphere.radius

        if magnitude > radii_sum:
            return

        epsilon = 1e-6
        if magnitude < epsilon:
            contact_normal = np.array([1.0, 0, 0])
            penetration = radii_sum
            contact_point = position_one
        else:
            contact_normal = vector / magnitude
            penetration = radii_sum - magnitude
            contact_point = position_two + contact_normal * sphere.radius

        current = Contact(
            self.body,
            sphere.body,
            contact_point,
            contact_normal,
            float(penetration),
            restitution,
            friction,
        )
        data.add_contact(current)

    def collide_with_box(
        self, box: Box, data: ContactData, restitution: float, friction: float
    ) -> None:
        box.collide_with_sphere(self, data, restitution, friction)

    def collide_with_plane(
        self, plane: Plane, data: ContactData, restitution: float, friction: float
    ) -> None:
        if data.is_full:
            return

        sphere_position_vector = self.get_axis(3)
        distance = (
            (plane.normal @ sphere_position_vector) - self.radius - plane.scalar_offset
        )

        if distance >= 0:  # no penetration yet
            return

        surface_of_half_space = sphere_position_vector - (
            plane.normal * (distance + self.radius)
        )
        current = Contact(
            self.body,
            None,
            surface_of_half_space,
            plane.normal,
            float(-distance),
            restitution,
            friction,
        )
        data.add_contact(current)

    def bounding_sphere(self) -> BoundingSphere:
        return BoundingSphere(radius=self.radius, center=self.get_axis(3).copy())


@dataclass
class Plane(Shape):
    normal: np.ndarray
    scalar_offset: float
    material: PhysicsMaterial = field(default_factory=lambda: Materials.DEFAULT)

    def collide_with(
        self, other: Shape, data: ContactData, restitution: float, friction: float
    ) -> None:
        other.collide_with_plane(self, data, restitution, friction)

    def collide_with_sphere(
        self, sphere: Sphere, data: ContactData, restitution: float, friction: float
    ) -> None:
        sphere.collide_with_plane(self, data, restitution, friction)

    def collide_with_box(
        self, box: Box, data: ContactData, restitution: float, friction: float
    ) -> None:
        box.collide_with_plane(self, data, restitution, friction)

    def collide_with_plane(
        self, plane: Plane, data: ContactData, restitution: float, friction: float
    ) -> None:
        pass


BOX_SIGNS = np.array(list(itertools.product([-1, 1], repeat=3)))


@dataclass
class Box(Primitive):
    _half_size: Optional[np.ndarray] = field(default=None, repr=False)
    _vertices: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    @property
    def half_size(self):
        return self._half_size

    @half_size.setter
    def half_size(self, array: np.ndarray):
        self._half_size = array
        self._vertices = None

    @property
    def vertices(self):
        if self._half_size is None:
            raise ValueError("must set half_size first")
        if self._vertices is None:
            self._vertices = BOX_SIGNS * self._half_size
        return self._vertices

    def collide_with(
        self, other: Shape, data: ContactData, restitution: float, friction: float
    ) -> None:
        other.collide_with_box(self, data, restitution, friction)

    def collide_with_sphere(
        self, sphere: Sphere, data: ContactData, restitution: float, friction: float
    ) -> None:
        if self.half_size is None:
            raise ValueError("box is not defined with half_size")

        world_sphere_center = sphere.get_axis(3)
        transform = self.get_transform()
        rotation = transform[:3, :3]
        position = transform[:3, 3]

        box_sphere_center = rotation.T @ (world_sphere_center - position)

        if np.any(np.abs(box_sphere_center) - sphere.radius > self.half_size):
            return

        closest_pt = np.clip(box_sphere_center, -self.half_size, self.half_size)
        distance = np.linalg.norm(closest_pt - box_sphere_center)
        if distance > sphere.radius:
            return

        world_closest_pt = rotation @ closest_pt + position

        direction = world_closest_pt - world_sphere_center
        magnitude = np.linalg.norm(direction)
        if np.isclose(magnitude, 0.0):
            distances_to_face = self.half_size - np.abs(box_sphere_center)
            min_axis = int(np.argmin(distances_to_face))

            local_normal = np.zeros(3)
            local_normal[min_axis] = 1.0 if box_sphere_center[min_axis] > 0 else -1.0

            normal = rotation @ local_normal
            penetration = float(sphere.radius + distances_to_face[min_axis])
        else:
            normal = direction / magnitude
            penetration = float(sphere.radius - distance)

        current = Contact(
            self.body,
            sphere.body,
            world_closest_pt,
            normal,
            penetration,
            restitution,
            friction,
        )
        data.add_contact(current)

    def collide_with_box(
        self, box: Box, data: ContactData, restitution: float, friction: float
    ) -> None:
        if self.half_size is None or box.half_size is None:
            return

        center_difference_vector = box.get_axis(3) - self.get_axis(3)
        box_one_transform = self.get_transform()
        box_two_transform = box.get_transform()
        one_axes = box_one_transform[:3, :3].T
        two_axes = box_two_transform[:3, :3].T

        change_of_basis = one_axes @ two_axes.T
        abs_change_of_basis = np.abs(change_of_basis)

        center_projection_one = one_axes @ center_difference_vector
        penetration_one = (
            self.half_size
            + (abs_change_of_basis @ box.half_size)
            - np.abs(center_projection_one)
        )

        if np.any(penetration_one <= 0):
            return

        center_projection_two = two_axes @ center_difference_vector
        penetration_two = (
            box.half_size
            + (abs_change_of_basis.T @ self.half_size)
            - np.abs(center_projection_two)
        )

        if np.any(penetration_two <= 0):
            return

        face_penetrations = np.concatenate([penetration_one, penetration_two])
        best_single_axis_index = int(np.argmin(face_penetrations))
        best_single_axis_pen = face_penetrations[best_single_axis_index]

        edge_axes = np.cross(
            one_axes[:, np.newaxis, :], two_axes[np.newaxis, :, :]
        ).reshape(9, 3)

        norms = np.linalg.norm(edge_axes, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-6

        normalized_edge_axes = None
        center_projection_edge = None  # pyright

        if np.any(valid):
            safe_norm = np.where(norms > 1e-6, norms, 1.0)
            normalized_edge_axes = edge_axes / safe_norm

            one_projection_edge = (
                np.abs(normalized_edge_axes @ one_axes.T) @ self.half_size
            )
            two_projection_edge = (
                np.abs(normalized_edge_axes @ two_axes.T) @ box.half_size
            )

            center_projection_edge = normalized_edge_axes @ center_difference_vector
            edge_penetrations = (
                one_projection_edge
                + two_projection_edge
                - np.abs(center_projection_edge)
            )

            if np.any((edge_penetrations <= 0) & valid):
                return

            edge_mask = np.where(valid, edge_penetrations, np.inf)
            best_edge_index = int(np.argmin(edge_mask))
            best_edge_pen = edge_mask[best_edge_index]

            is_barely_better = best_edge_pen > best_single_axis_pen - 1e-3
            is_nearly_parallel = norms[best_edge_index, 0] < 1e-3

            if is_barely_better or is_nearly_parallel:
                best_index = best_single_axis_index
                penetration = best_single_axis_pen
            else:
                best_index = best_edge_index + 6
                penetration = best_edge_pen
        else:
            best_index = best_single_axis_index
            penetration = best_single_axis_pen

        if best_index < 6:
            face_axes = np.vstack((one_axes, two_axes))
            normal = face_axes[best_index]
            best_center_projection = np.concatenate(
                [center_projection_one, center_projection_two]
            )[best_index]
        else:
            assert normalized_edge_axes is not None
            assert center_projection_edge is not None
            normal = normalized_edge_axes[best_index - 6]
            best_center_projection = center_projection_edge[best_index - 6]

        if best_center_projection > 0:
            normal = -normal

        if best_index < 6:
            if best_index < 3:
                reference_box = self
                incident_box = box
                reference_axis_index = best_index
                reference_outwards_normal = -normal
            else:
                reference_box = self
                incident_box = box
                reference_axis_index = best_index - 3
                reference_outwards_normal = normal

            incident_face = Sutherland_Hodgman.get_incident_face_vertices(
                incident_box, reference_outwards_normal
            )
            Sutherland_Hodgman.clip_and_generate_contacts(
                reference_box,
                incident_box,
                incident_face,
                reference_axis_index,
                reference_outwards_normal,
                normal,
                data,
                restitution,
                friction,
            )
            return

        cross_index = best_index - 6
        axis_one_index = cross_index // 3
        axis_two_index = cross_index % 3

        dir_one = one_axes[axis_one_index]
        dir_two = two_axes[axis_two_index]

        local_pt_one = np.copysign(self.half_size, one_axes @ -normal)
        local_pt_two = np.copysign(box.half_size, two_axes @ normal)

        local_pt_one[axis_one_index] = 0
        local_pt_two[axis_two_index] = 0

        world_pt_one = self.get_axis(3) + (one_axes.T @ local_pt_one)
        world_pt_two = box.get_axis(3) + (two_axes.T @ local_pt_two)

        center_vector = world_pt_one - world_pt_two

        dot_one_two = dir_one @ dir_two
        dot_one = center_vector @ dir_one
        dot_two = center_vector @ dir_two

        denominator = 1.0 - (dot_one_two * dot_one_two)

        if abs(denominator) < 1e-6:
            s = 0.0
            t = dot_two
        else:
            s = (dot_one_two * dot_two - dot_one) / denominator
            t = (dot_two - dot_one * dot_one_two) / denominator

        one_size = self.half_size[axis_one_index]
        two_size = box.half_size[axis_two_index]

        s_one = np.clip(s, -one_size, one_size)
        t_two = np.clip(t, -two_size, two_size)

        closest_one = world_pt_one + s_one * dir_one
        closest_two = world_pt_two + t_two * dir_two

        contact_point = (closest_one + closest_two) * 0.5
        current = Contact(
            self.body,
            box.body,
            contact_point,
            normal,
            penetration,
            restitution,
            friction,
        )
        data.add_contact(current)

    def collide_with_plane(
        self, plane: Plane, data: ContactData, restitution: float, friction: float
    ) -> None:
        if data.is_full:
            return

        if self.half_size is None:
            raise ValueError("box is not defined with half_size")

        transform = self.get_transform()
        axis_0 = transform[:3, 0]
        axis_1 = transform[:3, 1]
        axis_2 = transform[:3, 2]
        projected_radius = (
            self.half_size[0] * abs(plane.normal @ axis_0)
            + self.half_size[1] * abs(plane.normal @ axis_1)
            + self.half_size[2] * abs(plane.normal @ axis_2)
        )

        box_center = self.get_axis(3)
        box_distance = (box_center @ plane.normal) - plane.scalar_offset

        if box_distance >= projected_radius:
            return

        transform = self.get_transform()
        vertices = self.vertices
        ones = np.ones((len(vertices), 1))
        local_position_4d = np.hstack([vertices, ones])
        world_positions = (transform @ local_position_4d.T).T[:, :3]
        vertex_distance = world_positions @ plane.normal
        mask = vertex_distance <= plane.scalar_offset
        penetrations = plane.scalar_offset - vertex_distance[mask]
        valid_positions = world_positions[mask]
        normal = plane.normal

        for world_position, penetration in zip(valid_positions, penetrations):
            current = Contact(
                self.body,
                None,
                world_position,
                normal,
                penetration,
                restitution,
                friction,
            )
            if not data.add_contact(current):
                break

    def bounding_sphere(self) -> BoundingSphere:
        if self._half_size is None:
            raise ValueError("half_size not set")
        radius = float(np.linalg.norm(self._half_size))
        return BoundingSphere(radius=radius, center=self.get_axis(3).copy())


@dataclass
class Sutherland_Hodgman:
    @staticmethod
    def get_incident_face_vertices(box: Box, normal: np.ndarray) -> np.ndarray:
        transform = box.get_transform()
        axes = transform[:3, :3].T

        projected_normal = axes @ normal
        incident_index = int(np.argmax(np.abs(projected_normal)))

        direction = -np.sign(projected_normal[incident_index])
        if direction == 0:
            direction = 1.0

        face_signs = BOX_SIGNS[BOX_SIGNS[:, incident_index] == direction]
        face_signs = face_signs[[0, 1, 3, 2]]
        local_vertices = face_signs * box.half_size

        num_vertices = local_vertices.shape[0]
        ones = np.ones((num_vertices, 1))
        local_homogenous = np.hstack((local_vertices, ones))

        world_vertices = (transform @ local_homogenous.T).T[:, :3]
        return world_vertices

    @staticmethod
    def clip_and_generate_contacts(
        reference_box: Box,
        incident_box: Box,
        incident_face: np.ndarray,
        reference_axis_index: int,
        reference_outwards_normal: np.ndarray,
        final_contact_normal: np.ndarray,
        data: ContactData,
        restitution: float,
        friction: float,
    ):
        if data.is_full:
            return

        if reference_box.half_size is None:
            raise ValueError("reference_box is not defined with half_size")
        current_polygon = incident_face
        reference_transform = reference_box.get_transform()
        reference_box_center = reference_transform[:3, 3]
        rotation_matrix = reference_transform[:3, :3]

        for i in range(3):
            if i == reference_axis_index:
                continue

            axis = rotation_matrix[:3, i]
            half_size = reference_box.half_size[i]

            offset_position = (reference_box_center @ axis) + half_size
            current_polygon = Sutherland_Hodgman._clip_polygon_against_plane(
                current_polygon, axis, offset_position
            )

            offset_position = -(reference_box_center @ axis) + half_size
            current_polygon = Sutherland_Hodgman._clip_polygon_against_plane(
                current_polygon, -axis, offset_position
            )

            if len(current_polygon) == 0:
                break

        reference_face_offset = (
            reference_box_center @ reference_outwards_normal
        ) + reference_box.half_size[reference_axis_index]
        distance = (current_polygon @ reference_outwards_normal) - reference_face_offset
        mask = distance <= 0.0

        valid_contacts = current_polygon[mask]
        valid_distances = distance[mask]
        for contact, dist in zip(valid_contacts, valid_distances):
            current = Contact(
                reference_box.body,
                incident_box.body,
                contact,
                final_contact_normal,
                float(-dist),
                restitution,
                friction,
            )
            if not data.add_contact(current):
                break

    @staticmethod
    def _clip_polygon_against_plane(
        polygon: np.ndarray, plane_normal: np.ndarray, plane_offset: float
    ) -> np.ndarray:
        if len(polygon) == 0:
            return polygon

        distances = polygon @ plane_normal - plane_offset
        mask = distances <= 0

        if mask.all():
            return polygon
        if not mask.any():
            return np.empty((0, 3))

        N = len(polygon)
        next_index = np.arange(N)
        next_index = np.roll(next_index, -1, axis=0)

        shifted_mask = mask[next_index]
        crossings = mask ^ shifted_mask

        start_distances = distances[crossings]
        end_distances = distances[next_index]

        shifted_polygon = polygon[next_index]
        t = start_distances / (start_distances - end_distances)
        start = polygon[crossings]
        end = shifted_polygon[crossings]
        intersections = start + t[:, np.newaxis] * (end - start)

        output = np.zeros((N, 2, 3))
        valid = np.zeros((N, 2), dtype=bool)

        output[crossings, 0] = intersections
        valid[crossings, 0] = True

        output[shifted_mask, 1] = shifted_polygon[shifted_mask]
        valid[shifted_mask, 1] = True

        clipped_polygon = output[valid]
        return clipped_polygon
