import numpy as np


class Boid:
    """Boid agent"""

    def __init__(self, position, velocity, vision=None, comfort_zone=None, speed_cap=None, ndim=None):
        self._ndim = ndim if ndim else 3

        self.position = position
        self.velocity = velocity
        self._acceleration = np.zeros(ndim)

        self.vision = float(vision) if vision else np.inf
        self.comfort_zone = float(comfort_zone) if comfort_zone else 0.

        # Max speed the boid can achieve.
        self.speed_cap = float(speed_cap) if speed_cap else np.inf

        self.neighbors = []
        self.obstacles = []

    def __repr__(self):
        return 'Boid at position {} with velocity {}'.format(self.position, self.velocity)

    @classmethod
    def random(cls, max_x, max_v, vision=None, comfort_zone=None, speed_cap=None, ndim=None):
        position = np.random.uniform(0, max_x, ndim)
        velocity = np.random.uniform(-max_v, max_v, ndim)

        return cls(position, velocity, vision, comfort_zone, speed_cap, ndim)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = np.array(position, dtype=float)
        if self._position.shape != (self._ndim,):
            raise ValueError('position must be of shape ({},)'.format(self._ndim))

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = np.array(velocity, dtype=float)
        if self._velocity.shape != (self._ndim,):
            raise ValueError('velocity must be of shape ({},)'.format(self._ndim))

    def distance(self, other):
        """Distance from the other boid."""
        return np.linalg.norm(self.position - other.position)

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision

    def is_comfortable_with(self, other):
        """Whether the boid feel too close with the other."""
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
                repel += self.position - neighbor.position
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

    def _rule4(self):
        """Boids try to avoid obstacles."""
        # Linear repulsive force model.
        proximity = 3  # Max distance at which the boid starts to react.
        repel = np.zeros(self._ndim)
        for obstacle in self.obstacles:
            distance = obstacle.distance(self.position)
            if distance > proximity:
                continue
            repel += (proximity - distance) ** 2 * obstacle.direction(self.position)

        return repel

    def _goal(self):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction.
        return self.velocity / np.linalg.norm(self.velocity)

    def decide(self):
        """Make decision for acceleration."""
        c1 = 0.08
        c2 = 1
        c3 = 0.3
        c4 = 0.1
        g = 0.2
        self._acceleration = (c1 * self._rule1() +
                              c2 * self._rule2() +
                              c3 * self._rule3() +
                              c4 * self._rule4() +
                              g * self._goal())

    def _speed_cap(self):
        speed = np.linalg.norm(self._velocity)
        if speed > self.speed_cap:
            self._velocity = self._velocity / speed * self.speed_cap

    def move(self, dt):
        self._velocity += self._acceleration * dt
        # Velocity cap
        self._speed_cap()

        self._position += self._velocity * dt
