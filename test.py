import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from classes import *


def animate(i, scat, env):
    env.update(0.03)

    scat.set_offsets([boid.position for boid in env.population])
    return scat,


def main():

    plt.rcParams['animation.html'] = 'html5'

    fig, ax = plt.subplots()
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_aspect('equal')

    scat = ax.scatter([], [])

    env = Environment2D((0, 100, 0, 100))
    env.population = [Boid.random(100, 10, vision=20, comfort_zone=3, speed_cap=20, ndim=2)
                      for _ in range(100)]

    anim = animation.FuncAnimation(fig, animate,
                                   frames=3000, interval=20, blit=True, fargs=(scat, env))

    anim.save('demo.gif', dpi=80, writer='imagemagick')


if __name__ == '__main__':
    main()
