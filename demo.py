import argparse
import time

import numpy as np

from boids import *


def animate(env, region):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams['animation.html'] = 'html5'

    def animate(i, scat, env):
        env.update(ARGS.dt)

        scat.set_offsets([boid.position for boid in env.population])
        return scat,

    xmin, xmax, ymin, ymax = region

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    scat = ax.scatter([], [])

    for goal in env.goals:
        ax.scatter(*goal.position, color='g')
    for obstacle in env.obstacles:
        if not isinstance(obstacle, Wall):
            circle = plt.Circle(obstacle.position, obstacle.size, color='r', fill=False)
            ax.add_patch(circle)

    anim = animation.FuncAnimation(fig, animate,
                                   fargs=(scat, env),
                                   frames=ARGS.steps, interval=20, blit=True)

    anim.save(ARGS.save_name+'.gif', dpi=80, writer='imagemagick')


def main():
    region = (-100, 100, -100, 100)
    env = Environment2D(region)
    for _ in range(ARGS.agents):
        boid = Boid(ndim=2, comfort=3, max_speed=15, max_acceleration=20)
        boid.initialize(np.random.uniform(10, 100, 2),
                        np.random.uniform(-15, 15, 2))
        env.add_agent(boid)

    goal = Goal(np.random.uniform(-80, -20, 2), ndim=2)
    env.add_goal(goal)
    # Create a sphere obstacle within in +/- 50 of goal's position.
    sphere = Sphere(np.random.uniform(-30, 30, 2), 8, ndim=2)
    env.add_obstacle(sphere)

    animate(env, region)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--agents', type=int, default=100,
                        help='number of agents')
    parser.add_argument('--steps', type=int, default=1000,
                        help='number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='time resolution')
    parser.add_argument('--save-name', type=str,
                        help='name of the save file')

    ARGS = parser.parse_args()

    main()
