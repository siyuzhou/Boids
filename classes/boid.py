import numpy as np


class Boid:
    """Boid agent"""

    def __init__(self, position, velocity, vision=None, comfort_zone=None,
                 speed_cap=None, acceleration_cap=None, ndim=None):
        self._ndim = ndim if ndim else 3

        self._position = np.zeros(self._ndim)
        self._velocity = np.zeros(self._ndim)
        self._acceleration = np.zeros(self._ndim)

        self.position = position
        self.velocity = velocity

        self.vision = float(vision) if vision else np.inf
        self.comfort_zone = float(comfort_zone) if comfort_zone else 0.

        # Max speed the boid can achieve.
        self.speed_cap = float(speed_cap) if speed_cap else None
        self.acceleration_cap = float(acceleration_cap) if acceleration_cap else None

        self.neighbors = []
        self.obstacles = []

    def __repr__(self):
        return 'Boid at position {} with velocity {}'.format(self.position, self.velocity)

    @classmethod
    def random(cls, max_x, max_v, vision=None, comfort_zone=None,
               speed_cap=None, acceleration_cap=None, ndim=3):
        position = np.random.uniform(-max_x, max_x, ndim)
        velocity = np.random.uniform(-max_v, max_v, ndim)

        return cls(position, velocity, vision, comfort_zone, speed_cap, acceleration_cap, ndim)

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position[:] = position[:]

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity[:] = velocity[:]

    def distance(self, other):
        """Distance from the other boid."""
        return np.linalg.norm(self.position - other.position)

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision

    def is_comfortable_with(self, other):
        """Whether the boid feels too close with the other."""
        return self.distance(other) > self.comfort_zone

    def observe(self, environment):
        """Observe the population and take note of neighbors."""
        self.neighbors = [other for other in environment.population
                          if self.can_see(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = environment.obstacles

    def _rule1(self):
        """Boids try to fly towards the center of neighbors."""
        if not self.neighbors:
            return np.zeros(self._ndim)

        center = np.zeros(self._ndim)
        for boid in self.neighbors:
            center += boid.position
        center /= len(self.neighbors)

        return center - self.position

    def _rule2(self):
        """Boids try to keep a small distance away from other objects."""
        repel = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            if not self.is_comfortable_with(neighbor):
                displacement = self.position - neighbor.position
                norm_displacement = np.linalg.norm(displacement)

                # Divergence protection.
                if norm_displacement < 0.01:
                    norm_displacement = 0.01

                repel += displacement / norm_displacement / norm_displacement
                # No averaging taken place.
                # When two neighbors are in the same position, a stronger urge
                # to move away is assumed, despite that distancing itself from
                # one neighbor automatically eludes the other.
        return repel

    def _rule3(self):
        """Boids try to match velocity with near boids."""
        if not self.neighbors:
            return np.zeros(self._ndim)

        avg_velocity = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity /= len(self.neighbors)

        return avg_velocity - self.velocity

    def _avoid_obstacles(self):
        """Boids try to avoid obstacles."""
        # Linear repulsive force model.
        proximity = 5  # Max distance at which the boid starts to react.
        repel = np.zeros(self._ndim)
        for obstacle in self.obstacles:
            distance = obstacle.distance(self.position)
            if distance > proximity:
                continue
            repel += distance ** (-2) * obstacle.direction(self.position)

        return repel

    def _steer_to_goal(self, goal):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not goal:
            return self.velocity / np.linalg.norm(self.velocity)

        # The urge to chase the goal is stronger when farther.
        return goal.position - self.position

    def decide(self, goals):
        """Make decision for acceleration."""
        c1 = 0.08
        c2 = 1
        c3 = 0.2
        c4 = 0.1
        g = 0.05

        goal_steering = np.zeros(self.ndim)
        squared_norm = 0

        for goal in goals:
            goal_steering += self._steer_to_goal(goal) * goal.priority
            squared_norm += goal.priority ** 2

        goal_steering /= np.sqrt(squared_norm)

        self._acceleration = (c1 * self._rule1() +
                              c2 * self._rule2() +
                              c3 * self._rule3() +
                              c4 * self._avoid_obstacles() +
                              g * goal_steering)

    def _regularize(self):
        if self.speed_cap:
            speed = np.linalg.norm(self._velocity)
            if speed > self.speed_cap:
                self._velocity = self._velocity / speed * self.speed_cap

        if self.acceleration_cap:
            acceleration = np.linalg.norm(self._acceleration)
            if acceleration > self.acceleration_cap:
                self._acceleration = self._acceleration / acceleration * self.acceleration_cap

    def move(self, dt):
        self._velocity += self._acceleration * dt
        # Velocity cap
        self._regularize()

        self._position += self._velocity * dt
