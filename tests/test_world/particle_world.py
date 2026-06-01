from particles_force import ParticleForceRegistry
from particle_contact import (
    ParticleContactRegistry,
    ParticleContactResolver,
    ContactGenerator,
)
from particles import Particle


class ParticleWorld:
    def __init__(self):
        self.particle_registrations: list[Particle] = []
        self.force_registry = ParticleForceRegistry()
        self.contact_registrations = ParticleContactRegistry()
        self.max_contacts: int = 18

    def add_particle(self, particle: Particle):
        if particle in self.particle_registrations:
            raise ValueError("Particle is already registered in the world")
        self.particle_registrations.append(particle)

    def add_contact(self, contact_generator: ContactGenerator):
        self.contact_registrations.add_contact_generator(contact_generator)

    def start_frame(self):
        for particle in self.particle_registrations:
            particle.clear_accumulator()

    def integrate(self, dt: float):
        for particle in self.particle_registrations:
            particle.integrate(dt)

    def run_physics(self, dt: float):
        self.force_registry.update_forces(dt)
        self.integrate(dt)

        contact_register = self.contact_registrations.contact_register
        resolver = ParticleContactResolver(len(contact_register) * 2)

        resolver.resolve_contacts(contact_register, dt)
