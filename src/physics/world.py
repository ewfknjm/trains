from .contact import ContactData
from .contact_solver import ContactResolver
from .force_generators import ForceGenerator
from .force_registry import ForceRegistry
from .rigidbody import RigidBody
from .integrator import EulerIntegrator
from .narrow_phase import Primitive, Plane
from .broad_phase import CandidatePair
from .BSH import BSHTree, BoundingSphere

# Some AI assisted help for the ID sections, but not outright generations, I've highlighted some lines where it's a gray area between knowing what to do and syntax

class World:
    def __init__(
        self, usage_limit: int, contact_data: ContactData, resolver: ContactResolver
    ):
        self._rb_registrations: list[RigidBody] = []
        self._static_planes: list[Plane] = []
        self._shapes: dict[int, list[Primitive]] = {}
        self._force_registry: ForceRegistry = ForceRegistry()
        self._integrator: EulerIntegrator = EulerIntegrator()
        self._broad_phase: BSHTree = BSHTree(usage_limit)
        self._contact_data: ContactData = contact_data
        self._resolver: ContactResolver = resolver

    def _volume_for(self, body: RigidBody) -> BoundingSphere:
        # I had to refer quite a bit for this section
        shapes = self._shapes.get(id(body))
        if not shapes:
            raise ValueError(f"Body {id(body)} has no registered shapes")

        result = shapes[0].bounding_sphere()
        for shape in shapes[1:]:
            result = BoundingSphere.create_bounding_sphere(
                result, shape.bounding_sphere()
            )
        return result

    def add_shape(self, body: RigidBody, shape: Primitive) -> None:
        self._shapes.setdefault(id(body), []).append(shape)

    def get_shapes(self, body: RigidBody) -> list[Primitive]:
        return list(self._shapes.get(id(body), []))

    def add_plane(self, plane: Plane) -> None:
        self._static_planes.append(plane)

    def add_rigid_body(self, body: RigidBody):
        if body in self._rb_registrations:
            raise ValueError("RigidBody is already registered")
        self._rb_registrations.append(body)

    def remove_rigid_body(self, body: RigidBody) -> None:
        self._rb_registrations.remove(body)
        self._shapes.pop(id(body), None)
        self._force_registry.deregister_all_generators(body)
        if self._broad_phase.contains(body): # *!*
            leaf = self._broad_phase._body_to_leaf[id(body)]
            self._broad_phase.remove(leaf)

    def add_force_generators(self, body: RigidBody, force_generator: ForceGenerator):
        self._force_registry.register_force_generator(body, force_generator)

    def _broad_phase_update(self) -> None:
        self._broad_phase.clear()
        for body in self._rb_registrations:
            if id(body) not in self._shapes:
                continue
            volume = self._volume_for(body)
            self._broad_phase.insert(body, volume)

    def _narrow_phase_pass(self, pairs: list[CandidatePair]) -> None:
        for pair in pairs:
            if pair.our_body.is_sleeping and pair.other_body.is_sleeping:
                continue
            shapes_a = self._shapes.get(id(pair.our_body), [])
            shapes_b = self._shapes.get(id(pair.other_body), [])
            for sa in shapes_a:
                for sb in shapes_b:
                    e, mu = sa.material.mix(sb.material)
                    sa.collide_with(sb, self._contact_data, e, mu)

    def _static_narrow_phase_pass(self) -> None:
        for body in self._rb_registrations:
            if body.is_sleeping:
                continue
            body_shapes = self._shapes.get(id(body), []) # *!*
            for shape in body_shapes:
                for plane in self._static_planes:
                    e, mu = shape.material.mix(plane.material)
                    shape.collide_with_plane(plane, self._contact_data, e, mu)

    def start_frame(self):
        for rigid_body in self._rb_registrations:
            rigid_body.clear_accum()

    def integrate(self, dt: float) -> None:
        for rigid_body in self._rb_registrations:
            self._integrator.integrate(rigid_body, dt)

    def run_physics(self, dt: float):
        self.start_frame()
        self._force_registry.update_force(dt)
        self.integrate(dt)
        self._broad_phase_update()
        pairs = self._broad_phase.get_candidate_pairs()

        new_data = ContactData(max_contacts=self._contact_data.max_contacts)
        old_data = self._contact_data
        self._contact_data = new_data

        self._narrow_phase_pass(pairs)
        self._static_narrow_phase_pass()

        new_data.merge_from(old_data)
        self._resolver.resolve(new_data, dt)
