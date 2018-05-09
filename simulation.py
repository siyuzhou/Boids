import argparse
import numpy as np


from classes import *


def main(args):
    env = Environment2D((0, 100, 0, 100))
    env.population = [Boid.random(100, 10, vision=20, comfort_zone=3, speed_cap=20, ndim=2)
                      for _ in range(100)]

    if args.animation:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        plt.rcParams['animation.html'] = 'html5'

        def animate(i, scat, env):
            env.update(0.03)

            scat.set_offsets([boid.position for boid in env.population])
            return scat,

        fig, ax = plt.subplots()
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect('equal')

        scat = ax.scatter([], [])

        anim = animation.FuncAnimation(fig, animate,
                                       frames=args.steps, interval=20, blit=True, fargs=(scat, env))

        anim.save(args.save_name+'.gif', dpi=80, writer='imagemagick')

    else:
        data = []
        for _ in range(args.steps):
            env.update(0.03)
            data.append([boid.position for boid in env.population])

        np.save(args.save_name+'.npy', data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--dt', type=float, default=0.03)
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--save-name', type=str)

    args = parser.parse_args()

    main(args)
