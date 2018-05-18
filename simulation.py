import argparse
import numpy as np


from classes import *


def main(args):
    env = Environment2D((-100, 100, -100, 100))
    for _ in range(args.agents):
        env.add_agent(Boid.random(
            100, 15, comfort_zone=3, speed_cap=15, ndim=2))

    goal = Goal(np.random.uniform(-50, 50, 2), ndim=2)
    env.add_goal(goal)
    # Create a sphere obstacle within in +/- 50 of goal's position.
    sphere = Sphere(np.random.uniform(-20, 20, 2) + goal.position, ndim=2)
    env.add_obstacle(sphere)

    if args.animation:  # Generate animation
        import matplotlib.pyplot as plt
        from matplotlib import animation

        plt.rcParams['animation.html'] = 'html5'

        def animate(i, scat, env):
            env.update(args.dt)

            scat.set_offsets([boid.position for boid in env.population])
            return scat,

        fig, ax = plt.subplots()
        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_aspect('equal')

        scat = ax.scatter([], [])
        ax.scatter(*goal.position)
        ax.scatter(*sphere.position)

        anim = animation.FuncAnimation(fig, animate,
                                       fargs=(scat, env),
                                       frames=args.steps, interval=20, blit=True)

        anim.save(args.save_name+'.gif', dpi=80, writer='imagemagick')

    else:  # Generate data
        position_data_all = []
        velocity_data_all = []
        for i in range(args.instances):
            if i % 100 == 0:
                print('Simulation {}/{}...'.format(i, args.instances))

            position_data = []
            velocity_data = []
            for _ in range(args.steps):
                env.update(args.dt)
                position_data.append([goal.position for goal in env.goals] +
                                     [sphere.position] +
                                     [boid.position for boid in env.population])
                velocity_data.append([np.zeros(2) for goal in env.goals] +
                                     [np.zeros(2)] +
                                     [boid.velocity for boid in env.population])

            position_data_all.append(position_data)
            velocity_data_all.append(velocity_data)

        print('All {} simulations completed.'.format(args.instances))

        np.save('data/'+args.save_name+'_position.npy', position_data_all)
        np.save('data/'+args.save_name+'_velocity.npy', velocity_data_all)


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
