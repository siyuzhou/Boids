import numpy as np


class Boid:
    """Boid agent"""

    def __init__(self, ndim=None, vision=None, comfort=None,
                 max_speed=None, max_acceleration=None):
        """
        Create a boid with essential attributes.
        `ndim`: dimension of the space it resides in.
        `vision`: the visual range.
        `anticipation`: range of anticipation for its own motion.
        `comfort`: distance the agent wants to keep from other objects.
        `max_speed`: max speed the agent can achieve.
        `max_acceleratoin`: max acceleration the agent can achieve. 
        """
        self._ndim = ndim if ndim else 3

        self.vision = float(vision) if vision else np.inf
        self.comfort = float(comfort) if comfort else 0.

        # Max speed the boid can achieve.
        self.max_speed = float(max_speed) if max_speed else None
        self.max_acceleration = float(max_acceleration) if max_acceleration else None

        self.neighbors = []
        self.obstacles = []

    def initialize(self, position, velocity):
        """Initialize agent's spactial state."""
        self._position = np.zeros(self._ndim)
        self._velocity = np.zeros(self._ndim)
        self._acceleration = np.zeros(self._ndim)

        self.position = position
        self.velocity = velocity

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
        """Distance from the other objects."""
        if isinstance(other, Boid):
            return np.linalg.norm(self.position - other.position)
        # If other is not a boid, let other tell the distance.
        return other.distance(self.position)

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision

    def observe(self, environment):
        """Observe the population and take note of neighbors."""
        self.neighbors = [other for other in environment.population
                          if self.can_see(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = [obstacle for obstacle in environment.obstacles
                          if self.can_see(obstacle)]

    def _cohesion(self):
        """Boids try to fly towards the center of neighbors."""
        if not self.neighbors:
            return np.zeros(self._ndim)

        center = np.zeros(self._ndim)
        for boid in self.neighbors:
            center += boid.position
        center /= len(self.neighbors)

        return center - self.position

    def _seperation(self):
        """Boids try to keep a small distance away from other objects."""
        repel = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            distance = self.distance(neighbor)
            if distance < self.comfort:
                # Divergence protection.
                if distance < 0.01:
                    distance = 0.01

                repel += (self.position - neighbor.position) / distance / distance
                # No averaging taken place.
                # When two neighbors are in the same position, a stronger urge
                # to move away is assumed, despite that distancing itself from
                # one neighbor automatically eludes the other.
        return repel

    def _alignment(self):
        """Boids try to match velocity with neighboring boids."""
        # If no neighbors, no change.
        if not self.neighbors:
            return np.zeros(self._ndim)

        avg_velocity = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity /= len(self.neighbors)

        return avg_velocity - self.velocity

    def _obstacle_avoidance(self):
        """Boids try to avoid obstacles."""
        # Linear repulsive force model.
        proximity = 10  # Max distance at which the boid starts to react.
        repel = np.zeros(self._ndim)
        for obstacle in self.obstacles:
            distance = obstacle.distance(self.position)
            if distance > proximity:
                continue
            repel += distance ** (-3) * obstacle.direction(self.position)

        return repel

    def _goal_seeking(self, goal):
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
            goal_steering += self._goal_seeking(goal) * goal.priority
            squared_norm += goal.priority ** 2

        goal_steering /= np.sqrt(squared_norm)

        self._acceleration = (c1 * self._cohesion() +
                              c2 * self._seperation() +
                              c3 * self._alignment() +
                              c4 * self._obstacle_avoidance() +
                              g * goal_steering)

    def _regularize(self):
        if self.max_speed:
            speed = np.linalg.norm(self._velocity)
            if speed > self.max_speed:
                self._velocity = self._velocity / speed * self.max_speed

        if self.max_acceleration:
            acceleration = np.linalg.norm(self._acceleration)
            if acceleration > self.max_acceleration:
                self._acceleration = self._acceleration / acceleration * self.max_acceleration

    def move(self, dt):
        self._velocity += self._acceleration * dt
        # Velocity cap
        self._regularize()

        self._position += self._velocity * dt
