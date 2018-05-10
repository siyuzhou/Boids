import argparse
import numpy as np


from classes import *


def main(args):
    env = Environment2D((-100, 200, -100, 200))
    for _ in range(args.num_agents):
        env.add_agent(Boid.random(
            100, 15, comfort_zone=3, speed_cap=20, ndim=2))

    goal = Goal(np.random.rand(2)*100, ndim=2)
    env.add_goal(goal)

    if args.animation:
        import matplotlib.pyplot as plt
        from matplotlib import animation

        plt.rcParams['animation.html'] = 'html5'

        def animate(i, scat, env):
            env.update(args.dt)

            scat.set_offsets([boid.position for boid in env.population])
            return scat,

        fig, ax = plt.subplots()
        ax.set_xlim(-100, 200)
        ax.set_ylim(-100, 200)
        ax.set_aspect('equal')

        scat = ax.scatter([], [])
        ax.scatter(*goal.position)

        anim = animation.FuncAnimation(fig, animate,
                                       fargs=(scat, env),
                                       frames=args.steps, interval=20, blit=True)

        anim.save(args.save_name+'.gif', dpi=80, writer='imagemagick')

    else:
        data = []
        for _ in range(args.steps):
            env.update(0.03)
            data.append([goal.position for goal in env.goals] +
                        [boid.position for boid in env.population])

        np.save(args.save_name+'.npy', data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-agents', type=int, default=100)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--dt', type=float, default=0.03)
    parser.add_argument('--animation', action='store_true')
    parser.add_argument('--save-name', type=str)

    args = parser.parse_args()

    main(args)
