
import numpy as np
from .boid import Boid
from .obstacles import Obstacle, Wall, Sphere
from .goals import Goal


class Environment2D:
    """Environment that contains the population of boids, goals and obstacles."""

    def __init__(self, boundary):
        self.population = []
        self.goals = []

        xmin, xmax, ymin, ymax = boundary
        self.obstacles = [Wall((xmin, 0), (1, 0), ndim=2),
                          Wall((xmax, 0), (-1, 0), ndim=2),
                          Wall((0, ymin), (0, 1), ndim=2),
                          Wall((0, ymax), (0, -1), ndim=2)]

    def add_agent(self, agent):
        if not isinstance(agent, Boid):
            raise ValueError('agent must be an instance of Boid')

        if agent.ndim != 2:
            raise ValueError('position space of agent must be 2D')
        self.population.append(agent)

    def add_goal(self, goal):
        if not isinstance(goal, Goal):
            raise ValueError('goal must be an instance of Goal')
        if goal.ndim != 2:
            raise ValueError('position space of goal must be 2D')
        self.goals.append(goal)

    def add_obstacle(self, obstacle):
        if not isinstance(obstacle, Obstacle):
            raise ValueError('obstacle must be an instance of Obstacle')
        if obstacle.ndim != 2:
            raise ValueError('position space of obstacle must be 2D')
        self.obstacles.append(obstacle)

    def update(self, dt):
        """
        Update the state of environment for one time step dt, during which the
        boids move.
        """
        for boid in self.population:
            boid.observe(self)
            boid.decide(self.goals)
        # Hold off moving agents until all have made decision.
        # This ensures synchronous update.
        for boid in self.population:
            boid.move(dt)
