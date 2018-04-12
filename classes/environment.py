
import numpy as np
from .boid import Boid
from .obstacles import Wall


class Environment2D:
    """Environment that contains the population of boids, goals and obstacles."""

    def __init__(self, boundary):
        self._population = []

        xmin, xmax, ymin, ymax = boundary
        self.obstacles = [Wall((xmin, 0), (1, 0), ndim=2),
                          Wall((xmax, 0), (-1, 0), ndim=2),
                          Wall((0, ymin), (0, 1), ndim=2),
                          Wall((0, ymax), (0, -1), ndim=2)]

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, population):
        for boid in population:
            if boid.position.shape != (2,):
                raise ValueError('space dimention of the boid does not match the environment')

        self._population = population

    def update(self, dt):
        """
        Update the state of environment for one time step dt, during which the
        boids move.
        """
        for boid in self.population:
            boid.observe(self)
            boid.decide()
        # Hold off moving agents until all have made decision.
        # This ensures synchronous update.
        for boid in self.population:
            boid.move(dt)
