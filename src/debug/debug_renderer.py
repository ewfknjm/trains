from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from ursina import Ursina, Entity, Vec3, color, time, Text, held_keys
from ursina import EditorCamera
from ursina.color import Color
from panda3d.core import Quat as PandaQuat

from ..physics.world import World
from ..physics.rigidbody import RigidBody
from ..physics.contact import ContactData
from ..physics.contact_solver import ContactResolver
from ..physics.narrow_phase import Primitive, Sphere, Box


# ── colour palette ──────────────────────────────────────────────────────────

COLOURS = [
    color.cyan,
    color.orange,
    color.lime,
    color.magenta,
    color.yellow,
    color.azure,
    color.violet,
    color.pink,
]


# ── shape → Ursina entity factory ───────────────────────────────────────────


def _make_entity(shape: Primitive) -> Entity:
    if isinstance(shape, Sphere):
        return Entity(model="sphere", scale=Vec3(shape.radius * 2) * 1)  # diameter
    elif isinstance(shape, Box):
        if shape.half_size is None:
            raise ValueError("Box has no half_size set")
        hs = shape.half_size
        return Entity(
            model="cube",
            scale=Vec3(float(hs[0] * 2), float(hs[1] * 2), float(hs[2] * 2)),
        )
    else:
        # fallback: small marker sphere
        return Entity(model="sphere", scale=Vec3(0.15, 0.15, 0.15))


# ── registration record ──────────────────────────────────────────────────────


@dataclass
class _BodyRecord:
    body: RigidBody
    shape: Primitive
    entity: Entity


# ── main class ───────────────────────────────────────────────────────────────


class DebugRenderer:
    """
    Minimal Ursina wrapper. You register bodies; it draws them and syncs every frame.

    Usage:
        renderer = DebugRenderer.make()          # world + renderer with sensible defaults
        body, box = renderer.add_box([0,5,0], half_size=[0.5,0.5,0.5])
        renderer.run()
    """

    def __init__(self, world: World, floor_y: float = 0.0, show_hud: bool = True):
        self._app = Ursina()
        EditorCamera()

        self._world = world
        self._records: list[_BodyRecord] = []
        self._colour_index = 0
        self._paused = False
        self._show_hud = show_hud

        self._setup_floor(floor_y)
        self._setup_axes()

        self._hud = (
            Text(
                text="",
                position=(-0.85, 0.47),
                scale=0.55,
                color=color.white,
                background=True,
            )
            if show_hud
            else None
        )

        self._pause_hint = Text(
            text="[SPACE] pause   [R] reset cam",
            position=(-0.85, -0.47),
            scale=0.5,
            color=color.light_gray,
        )

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def make(
        cls,
        velocity_iters: int = 10,
        position_iters: int = 10,
        max_contacts: int = 256,
        floor_y: float = 0.0,
    ) -> "DebugRenderer":
        contact_data = ContactData(max_contacts=max_contacts)
        resolver = ContactResolver(
            velocity_iterations=velocity_iters,
            position_iterations=position_iters,
            dt=1 / 60,
        )
        world = World(
            usage_limit=1024,
            contact_data=contact_data,
            resolver=resolver,
        )
        return cls(world, floor_y=floor_y)

    # ── registration ─────────────────────────────────────────────────────────

    def register(
        self,
        body: RigidBody,
        shape: Primitive,
        body_color: Color | None = None,
    ) -> Entity:
        if body_color is None:
            body_color = COLOURS[self._colour_index % len(COLOURS)]
            self._colour_index += 1

        entity = _make_entity(shape)
        entity.color = body_color

        self._records.append(_BodyRecord(body, shape, entity))
        self._world.add_rigid_body(body)
        self._world.add_shape(body, shape)
        return entity

    # ── convenience builders ─────────────────────────────────────────────────

    def add_box(
        self,
        position: list | np.ndarray,
        half_size: list | np.ndarray,
        mass: float = 1.0,
        body_color: Color | None = None,
    ) -> tuple[RigidBody, Box]:
        from ..physics.rigidbody import RigidBody
        from ..physics.narrow_phase import Box
        import numpy as np

        pos = np.asarray(position, dtype=float)
        hs = np.asarray(half_size, dtype=float)

        body = RigidBody(float(pos[0]), float(pos[1]), float(pos[2]))
        body.mass = mass
        # solid box inertia tensor
        body.inertia_tensor = (
            (mass / 12.0)
            * np.diag(
                [
                    hs[1] ** 2 + hs[2] ** 2,
                    hs[0] ** 2 + hs[2] ** 2,
                    hs[0] ** 2 + hs[1] ** 2,
                ]
            )
            * 4
        )  # 4 = (2hs)^2 factor
        body.linear_damping = 0.995
        body.angular_damping = 0.99

        box = Box(body=body, offset_matrix=np.eye(4))
        box.half_size = hs

        self.register(body, box, body_color)
        return body, box

    def add_sphere(
        self,
        position: list | np.ndarray,
        radius: float = 0.5,
        mass: float = 1.0,
        body_color: Color | None = None,
    ) -> tuple[RigidBody, Sphere]:
        from ..physics.rigidbody import RigidBody
        from ..physics.narrow_phase import Sphere
        import numpy as np

        pos = np.asarray(position, dtype=float)

        body = RigidBody(float(pos[0]), float(pos[1]), float(pos[2]))
        body.mass = mass
        body.inertia_tensor = (2 / 5) * mass * radius**2 * np.eye(3)
        body.linear_damping = 0.999
        body.angular_damping = 0.998

        sphere = Sphere(body=body, offset_matrix=np.eye(4), radius=radius)

        self.register(body, sphere, body_color)
        return body, sphere

    def add_gravity(self) -> None:
        from ..physics.force_generators import Gravity

        g = Gravity()
        for rec in self._records:
            self._world.add_force_generators(rec.body, g)

    def add_floor_plane(self, y: float = 0.0) -> None:
        from ..physics.narrow_phase import Plane
        from ..physics.material import Materials
        import numpy as np

        plane = Plane(
            normal=np.array([0.0, 1.0, 0.0]),
            scalar_offset=y,
            material=Materials.CONCRETE,
        )
        self._world.add_plane(plane)

    # ── internal scene setup ─────────────────────────────────────────────────

    def _setup_floor(self, y: float) -> None:
        Entity(
            model="plane",
            scale=Vec3(40, 1, 40),
            position=Vec3(0, y, 0),
            color=color.dark_gray,
        )

    def _setup_axes(self) -> None:
        T, L = 0.03, 5.0
        for axis, col, scale, pos in [
            ("x", color.red, Vec3(L, T, T), Vec3(L / 2, 0, 0)),
            ("y", color.green, Vec3(T, L, T), Vec3(0, L / 2, 0)),
            ("z", color.blue, Vec3(T, T, L), Vec3(0, 0, L / 2)),
        ]:
            Entity(model="cube", color=col, scale=scale, position=pos)

    # ── sync and loop ─────────────────────────────────────────────────────────

    def _sync_all(self) -> None:
        for rec in self._records:
            p = rec.body.position
            rec.entity.setPos(float(p[0]), float(p[1]), float(p[2]))
            q = rec.body.orientation
            rec.entity.setQuat(
                PandaQuat(float(q.w), float(q.x), float(q.y), float(q.z))
            )

    def _build_hud(self) -> str:
        lines = [f"{'PAUSED' if self._paused else 'running':^30}"]
        for i, rec in enumerate(self._records[:6]):  # cap at 6 to avoid overflow
            b = rec.body
            p, v = b.position, b.velocity
            lines.append(
                f"[{i}] pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}) "
                f"vel=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) "
                f"{'SLEEP' if b.is_sleeping else ''}"
            )
        return "\n".join(lines)

    def run(self, dt_fixed: float = 1 / 60, sub_steps: int = 1) -> None:
        renderer = self

        def input(key):
            if key == "space":
                renderer._paused = not renderer._paused

        def update():
            if not renderer._paused:
                sub_dt = dt_fixed / sub_steps
                for _ in range(sub_steps):
                    renderer._world.run_physics(sub_dt)

            renderer._sync_all()

            if renderer._hud and renderer._show_hud:
                renderer._hud.text = renderer._build_hud()

        # Ursina picks up the module-level `update` function by name.
        # We need to inject ours into the calling module's globals.
        import sys

        caller_frame = sys._getframe(1)
        caller_frame.f_globals["update"] = update

        self._app.run()
