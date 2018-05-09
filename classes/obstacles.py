import numpy as np


class Obstacle:
    def __init__(self, position, ndim=None):
        self._ndim = ndim if ndim else 3

        self._position = np.zeros(self._ndim)
        self.position = position

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position[:] = position[:]

    def distance(self, r):
        raise NotImplementedError()

    def direction(self, r):
        raise NotImplementedError()


class Wall(Obstacle):
    def __init__(self, position, direction, ndim=None):
        """
        A plane in space that repels free agents.

        Parameters:
            position: the position of a point the wall passes.
            direction: the normal direction of the plane wall.
        """
        super().__init__(position, ndim)

        self._direction = np.array(direction, dtype=float)
        if self._direction.shape != (self._ndim,):
            raise ValueError('direction must be of shape ({},)'.format(self._ndim))
        self._direction /= np.linalg.norm(self._direction)  # Normalization

    def distance(self, r):
        return np.dot(r - self.position, self.direction(r))

    def direction(self, r):
        return self._direction


class Sphere(Obstacle):
    def __init__(self, position, ndim=None):
        """
        A sphere in ndim space.
        """
        super().__init__(position, ndim)

    def distance(self, r):
        return np.linalg.norm(r - self.position)

    def direction(self, r):
        d = r - self.position
        return d / np.linalg.norm(d)
