import numpy as np
from dataclasses import dataclass, field
import itertools
from typing import Optional
from .rigidbody import RigidBody
from .transform import Transform4x4


@dataclass
class Contact:
    contact_point: np.ndarray
    contact_normal: np.ndarray
    contact_penetration: float
    collision_restitution: float
    coefficient_of_friction: float


@dataclass
class ContactData:
    contacts: list[Contact]
    contacts_left: int
    _contact_count: int

    @property
    def contact_count(self):
        return self._contact_count

    @contact_count.setter
    def contact_count(self, value: int):
        if value < 0:
            raise ValueError("Negative number of contacts")

        added = value - self._contact_count
        if added < 0:
            raise ValueError("Cannot decrease contact count directly this way")

        new_contact_left = self.contacts_left - added
        if new_contact_left < 0:
            raise ValueError("contact_count exceeds contact_left")

        self._contact_count = value
        self.contacts_left = new_contact_left


@dataclass
class Primitive:
    body: RigidBody
    offset_matrix: np.ndarray

    def get_transform(self) -> np.ndarray:
        return self.body.transform_matrix @ self.offset_matrix

    def get_axis(self, index: int) -> np.ndarray:
        return self.get_transform()[:3, index]


@dataclass
class Sphere(Primitive):
    radius: float


@dataclass
class Plane:
    normal: np.ndarray
    scalar_offset: float


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


class CollisionDetector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def sphere_with_sphere(
        sphere_one: Sphere,
        sphere_two: Sphere,
        data: ContactData,
        collision_restitution: float,
        coefficient_of_friction: float,
    ):
        if data.contacts_left <= 0:
            return

        position_one = sphere_one.get_axis(3)
        position_two = sphere_two.get_axis(3)

        vector = position_one - position_two
        magnitude = np.linalg.norm(vector)
        radii_sum = sphere_one.radius + sphere_two.radius

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
            contact_point = position_two + contact_normal * sphere_two.radius

        current = Contact(
            contact_point,
            contact_normal,
            float(penetration),
            collision_restitution,
            coefficient_of_friction,
        )
        data.contacts.append(current)
        data.contact_count += 1

    @staticmethod
    def sphere_and_half_space(
        sphere: Sphere,
        plane: Plane,
        data: ContactData,
        collision_restitution: float,
        coefficient_of_friction: float,
    ):
        if data.contacts_left <= 0:
            return

        sphere_position_vector = sphere.get_axis(3)
        distance = (
            (plane.normal @ sphere_position_vector)
            - sphere.radius
            - plane.scalar_offset
        )

        if distance >= 0:  # no penetration yet
            return

        surface_of_half_space = sphere_position_vector - (
            plane.normal * (distance + sphere.radius)
        )
        data.contacts.append(
            Contact(
                surface_of_half_space,
                plane.normal,
                float(-distance),
                collision_restitution,
                coefficient_of_friction,
            )
        )
        data.contact_count += 1

    @staticmethod
    def box_and_half_space(
        box: Box, plane: Plane, data: ContactData, restitution: float, friction: float
    ):
        if data.contacts_left <= 0:
            return

        if box.half_size is None:
            raise ValueError("box is not defined with half_size")

        projected_radius = (
            box.half_size[0] * abs(plane.normal @ box.get_axis(0))
            + box.half_size[1] * abs(plane.normal @ box.get_axis(1))
            + box.half_size[2] * abs(plane.normal @ box.get_axis(2))
        )

        box_center = box.get_axis(3)
        box_distance = (box_center @ plane.normal) - plane.scalar_offset

        if box_distance >= projected_radius:
            return

        transform = box.get_transform()
        contacts_used = 0
        for vertex in box.vertices:
            world_position_4 = transform @ np.append(vertex, 1.0)
            world_position = world_position_4[:3]

            vertex_distance = world_position @ plane.normal
            if vertex_distance > plane.scalar_offset:
                continue

            penetration = plane.scalar_offset - vertex_distance
            normal = plane.normal
            contact_point = world_position + (plane.normal * penetration)
            contacts_used += 1
            data.contacts.append(
                Contact(
                    contact_point, normal, float(penetration), restitution, friction
                )
            )
            if contacts_used == data.contacts_left:
                break

        data.contact_count += contacts_used

    @staticmethod
    def box_and_sphere(
        box: Box, sphere: Sphere, data: ContactData, restitution: float, friction: float
    ):
        if box.half_size is None:
            raise ValueError("box is not defined with half_size")

        world_sphere_center = sphere.get_axis(3)
        box_transform = Transform4x4(box.get_transform())
        box_sphere_center = box_transform.world_to_local(world_sphere_center)

        if np.any(np.abs(box_sphere_center) - sphere.radius > box.half_size):
            return

        closest_pt = np.clip(box_sphere_center, -box.half_size, box.half_size)
        distance = np.linalg.norm(closest_pt - box_sphere_center)
        if distance > sphere.radius:
            return

        world_closest_pt = box_transform.local_to_world(closest_pt)

        direction = world_closest_pt - world_sphere_center
        magnitude = np.linalg.norm(direction)
        if np.isclose(magnitude, 0.0):
            distances_to_face = box.half_size - np.abs(box_sphere_center)
            min_axis = int(np.argmin(distances_to_face))

            local_normal = np.zeros(3)
            local_normal[min_axis] = 1.0 if box_sphere_center[min_axis] > 0 else -1.0

            normal = box.get_transform()[:3, :3] @ local_normal
            penetration = float(sphere.radius + distances_to_face[min_axis])
        else:
            normal = direction / magnitude
            penetration = float(sphere.radius - distance)

        data.contacts.append(
            Contact(world_closest_pt, normal, penetration, restitution, friction)
        )
        data.contact_count += 1

    @staticmethod
    def box_and_box(
        box_one: Box,
        box_two: Box,
        data: ContactData,
        restitution: float,
        friction: float,
    ):
        if box_one.half_size is None or box_two.half_size is None:
            return

        center_difference_vector = box_two.get_axis(3) - box_one.get_axis(3)
        box_one_transform = box_one.get_transform()
        box_two_transform = box_two.get_transform()
        one_axes = box_one_transform[:3, :3].T
        two_axes = box_two_transform[:3, :3].T
        face_axes = np.vstack([one_axes, two_axes])
        cross_grid = np.cross(one_axes[:, np.newaxis, :], two_axes[np.newaxis, :, :])
        edge_axes = cross_grid.reshape(9, 3)
        all_axes = np.vstack([face_axes, edge_axes])

        norms = np.linalg.norm(all_axes, axis=1, keepdims=True)
        valid = norms[:, 0] > 1e-6
        safe_norm = np.where(norms > 1e-6, norms, 1.0)
        normalized_axes = all_axes / safe_norm

        one_projection = box_one.half_size @ np.abs(one_axes @ normalized_axes.T)
        two_projection = box_two.half_size @ np.abs(two_axes @ normalized_axes.T)

        center_projection = normalized_axes @ center_difference_vector
        penetrations = one_projection + two_projection - np.abs(center_projection)

        if np.any((penetrations <= 0) & valid):
            return None

        mask = np.where(valid, penetrations, np.inf)
        best_index = int(np.argmin(mask))
        penetration = penetrations[best_index]
        best_single_axis_index = int(np.argmin(mask[:6]))

        is_edge_axis = best_index >= 6
        is_barely_better = mask[best_index] > mask[best_single_axis_index] - 1e-3
        is_nearly_parallel = norms[best_index, 0] < 1e-3

        if is_edge_axis and (is_barely_better or is_nearly_parallel):
            best_index = best_single_axis_index

        normal = normalized_axes[best_index]
        if center_projection[best_index] > 0:
            normal = -normal

        if best_index < 6:
            if best_index < 3:
                reference_box = box_one
                incident_box = box_two
                reference_axis_index = best_index
                reference_outwards_normal = -normal
            else:
                reference_box = box_two
                incident_box = box_one
                reference_axis_index = best_index - 3
                reference_outwards_normal = normal

            incident_face = Sutherland_Hodgman.get_incident_face_vertices(
                incident_box, reference_outwards_normal
            )
            Sutherland_Hodgman.clip_and_generate_contacts(
                reference_box,
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

        local_pt_one = np.copysign(box_one.half_size, one_axes @ -normal)
        local_pt_two = np.copysign(box_two.half_size, two_axes @ normal)

        local_pt_one[axis_one_index] = 0
        local_pt_two[axis_two_index] = 0

        world_pt_one = box_one.get_axis(3) + (one_axes.T @ local_pt_one)
        world_pt_two = box_two.get_axis(3) + (two_axes.T @ local_pt_two)

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

        one_size = box_one.half_size[axis_one_index]
        two_size = box_two.half_size[axis_two_index]

        s_one = np.clip(s, -one_size, one_size)
        t_two = np.clip(t, -two_size, two_size)

        closest_one = world_pt_one + s_one * dir_one
        closest_two = world_pt_two + t_two * dir_two

        contact_point = (closest_one + closest_two) * 0.5
        data.contacts.append(
            Contact(contact_point, normal, penetration, restitution, friction)
        )
        data.contact_count += 1


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
        incident_face: np.ndarray,
        reference_axis_index: int,
        reference_outwards_normal: np.ndarray,
        final_contact_normal: np.ndarray,
        data: ContactData,
        restitution: float,
        friction: float,
    ):
        if data.contacts_left <= 0:
            return

        if reference_box.half_size is None:
            raise ValueError("reference_box is not defined with half_size")
        current_polygon = incident_face
        reference_box_center = reference_box.get_axis(3)
        rotation_matrix = reference_box.get_transform()[:3, :3]

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
            if data.contacts_left <= 0:
                break
            data.contacts.append(
                Contact(
                    contact, final_contact_normal, float(-dist), restitution, friction
                )
            )
            data.contact_count += 1

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

        shifted_mask = np.roll(mask, -1, axis=0)
        crossings = mask ^ shifted_mask

        start_distances = distances[crossings]
        end_distances = np.roll(distances, -1, axis=0)[crossings]

        shifted_polygon = np.roll(polygon, -1, axis=0)[crossings]
        t = start_distances / (start_distances - end_distances)
        start = polygon[crossings]
        end = shifted_polygon[crossings]
        intersections = start + t[:, np.newaxis] * (end - start)

        N = len(polygon)
        output = np.zeros((N, 2, 3))
        valid = np.zeros((N, 2), dtype=bool)

        output[crossings, 0] = intersections
        valid[crossings, 0] = True

        output[shifted_mask, 1] = shifted_polygon[shifted_mask]
        valid[shifted_mask, 1] = True

        clipped_polygon = output[valid]
        return clipped_polygon
