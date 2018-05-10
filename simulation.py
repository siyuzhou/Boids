import argparse
import numpy as np


from classes import *


def main(args):
    env = Environment2D((-100, 200, -100, 200))
    for _ in range(args.agents):
        env.add_agent(Boid.random(
            100, 15, comfort_zone=3, speed_cap=20, ndim=2))

    goal = Goal(np.random.rand(2)*100, ndim=2)
    env.add_goal(goal)

    if args.animation:  # Generate animation
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

    else:  # Generate data
        data_all = []
        for i in range(args.instances):
            if i % 100 == 0:
                print('Simulation {}/{}...'.format(i, args.instances))

            data = []
            for _ in range(args.steps):
                env.update(args.dt)
                data.append([goal.position for goal in env.goals] +
                            [boid.position for boid in env.population])

            data_all.append(data)

        print('All {} simulations completed.'.format(args.instances))

        np.save(args.save_name+'.npy', data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agents', type=int, default=100,
                        help='number of agents')
    parser.add_argument('--steps', type=int, default=1000,
                        help='number of simulation steps')
    parser.add_argument('--instances', type=int, default=1,
                        help='number of simulation instances')
    parser.add_argument('--dt', type=float, default=0.03,
                        help='time resolution')
    parser.add_argument('--animation', action='store_true',
                        help='whether animation is generated')
    parser.add_argument('--save-name', type=str,
                        help='name of the save file')

    args = parser.parse_args()

    main(args)
