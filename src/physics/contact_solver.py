import numpy as np
from .contact import Contact
from .rigidbody import RigidBody
import math

# replace rotation with omega, use github to check


class ContactResolver:
    def __init__(self, iterations: int):
        self.iterations = iterations
